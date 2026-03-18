"""
Visualize Spatial Prototype Similarity Heatmaps (Fixed).

Fixes from original:
1. Uses soft attention weights (matching training logic in query_prototypes)
   instead of max similarity — provides much better contrast
2. Shows per-prototype attention maps to reveal specialization
3. Shows attention entropy map (low = confident, high = uncertain)
4. Uses only online path for inference (momentum path not available at inference)
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from tqdm import tqdm
import argparse
import cv2
import math

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

from data import get_dataset, get_loader
from model import get_model
from loss import get_loss_terms
from util.cfg import get_cfg


def load_model_and_loss(config_path, checkpoint_path, device):
    """Load trained model and prototype loss module."""
    cfg = get_cfg(config_path)
    model = get_model(cfg.model)
    model = model.to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'net' in checkpoint:
        model.load_state_dict(checkpoint['net'], strict=False)
    elif isinstance(checkpoint, dict) and 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)

    loss_terms = get_loss_terms(cfg.loss.loss_terms, device=str(device))
    loss_state_path = os.path.join(os.path.dirname(checkpoint_path), 'loss_terms.pth')
    if os.path.exists(loss_state_path):
        loss_state = torch.load(loss_state_path, map_location=device, weights_only=False)
        for name, state in loss_state.items():
            if name in loss_terms:
                loss_terms[name].load_state_dict(state, strict=False)
                print(f"  Loaded loss state: {name}")
    else:
        print(f"  Warning: {loss_state_path} not found, using init prototypes")

    model.eval()
    return model, cfg, loss_terms


def compute_attention_weights(feats, prototypes):
    """
    Compute soft attention weights matching training logic in query_prototypes().

    Args:
        feats: (B, C, H, W) — spatial features (mid from mff_oce)
        prototypes: (H*W, N, C) — learned prototypes

    Returns:
        attention: (B, H*W, N) — soft attention weights (cosine similarity)
        feats_norm: (B, H*W, C) — normalized features
    """
    B, C, H, W = feats.shape

    # Flatten: (B, C, H*W) -> (B, H*W, C)
    feats_flat = feats.view(B, C, -1).transpose(1, 2)

    # Normalize (matching query_prototypes exactly)
    feats_norm = F.normalize(feats_flat, dim=-1, p=2)       # (B, H*W, C)
    proto_norm = F.normalize(prototypes, dim=-1, p=2)        # (H*W, N, C)

    # Cosine similarity at each spatial location
    # This is the SAME einsum used in PrototypeBYOLLoss.query_prototypes()
    attention = torch.einsum('bsc,snc->bsn', feats_norm, proto_norm)  # (B, H*W, N)

    return attention, feats_norm


def overlay_heatmap(img, heatmap, alpha=0.5, colormap=cv2.COLORMAP_JET):
    """Overlay heatmap on RGB image with proper normalization."""
    # Normalize to 0-1 range
    vmin, vmax = heatmap.min(), heatmap.max()
    if vmax - vmin > 1e-8:
        heatmap_norm = (heatmap - vmin) / (vmax - vmin)
    else:
        heatmap_norm = np.zeros_like(heatmap)

    heatmap_uint8 = (heatmap_norm * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    # Resize to image size (bilinear for smooth upsampling)
    heatmap_color = cv2.resize(heatmap_color, (img.shape[1], img.shape[0]),
                               interpolation=cv2.INTER_LINEAR)

    img_uint8 = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
    overlayed = cv2.addWeighted(img_uint8, 1 - alpha, heatmap_color, alpha, 0)
    return overlayed


def denormalize_image(img_tensor):
    """Denormalize ImageNet-normalized image tensor to [0, 1] numpy array."""
    img = img_tensor.permute(1, 2, 0).cpu().numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img * std + mean).clip(0, 1)
    return img


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description='Visualize Prototype Spatial Heatmaps')
    parser.add_argument('--run_dir', type=str, required=True,
                        help='Path to training run directory containing net.pth/ckpt.pth')
    parser.add_argument('--config', type=str, default='configs/rd/rd_byol_mvtec.py')
    parser.add_argument('--num_samples', type=int, default=5)
    parser.add_argument('--save_dir', type=str, default='vis_spatial_proto')
    parser.add_argument('--show_per_proto', action='store_true', default=True,
                        help='Show per-prototype attention maps')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model and prototypes
    ckpt_path = os.path.join(args.run_dir, 'net.pth')
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(args.run_dir, 'ckpt.pth')

    model, cfg, loss_terms = load_model_and_loss(args.config, ckpt_path, device)
    proto_loss = loss_terms.get('proto')

    if proto_loss is None or not hasattr(proto_loss, 'prototypes'):
        print("Error: No prototypes found in loss_terms.")
        return

    prototypes = proto_loss.prototypes  # (H*W, N, C)
    n_prototypes = prototypes.shape[1]
    print(f"Prototypes shape: {prototypes.shape}")
    print(f"  H*W={prototypes.shape[0]}, N={n_prototypes}, C={prototypes.shape[2]}")

    # Load dataset
    train_ds, test_ds = get_dataset(cfg)
    indices = np.random.choice(len(test_ds), min(args.num_samples, len(test_ds)), replace=False)

    for sample_i, idx in enumerate(indices):
        data = test_ds[idx]
        img = data['img'].unsqueeze(0).to(device)
        cls_name = data['cls_name']
        anomaly = data.get('anomaly', 'unknown')

        print(f"\n--- Sample {sample_i+1}/{len(indices)}: {cls_name} (anomaly={anomaly}) ---")

        # === Forward pass (inference mode — online path only) ===
        feats_t = model.net_t(img)
        feats_detached = [f.detach() for f in feats_t]
        feats_proj = model.proj_layer(feats_detached)
        mid = model.mff_oce(feats_proj)  # (1, 2048, H, W)
        print(f"  MFF_OCE output: {mid.shape}")

        # === Compute attention weights (matching training logic) ===
        attention, _ = compute_attention_weights(mid, prototypes)  # (1, H*W, N)
        attention = attention[0]  # (H*W, N) — single image

        B, C, H, W = mid.shape

        # === 1. Mean attention map (better than max!) ===
        mean_attn = attention.mean(dim=-1).view(H, W).cpu().numpy()  # (H, W)

        # === 2. Attention entropy map ===
        # Softmax to get probability distribution over prototypes
        attn_prob = F.softmax(attention, dim=-1)  # (H*W, N)
        entropy = -(attn_prob * torch.log(attn_prob + 1e-8)).sum(dim=-1)  # (H*W,)
        entropy_map = entropy.view(H, W).cpu().numpy()

        # === 3. Dominant prototype map ===
        dominant_proto = attention.argmax(dim=-1).view(H, W).cpu().numpy()  # (H, W)

        # === 4. Per-prototype attention maps ===
        per_proto_maps = attention.view(H, W, n_prototypes).cpu().numpy()  # (H, W, N)

        # === Visualization ===
        orig_img = denormalize_image(data['img'])

        # --- Figure 1: Overview (Original + Mean Attention + Entropy + Dominant) ---
        fig, axes = plt.subplots(1, 4, figsize=(22, 5))

        axes[0].imshow(orig_img)
        axes[0].set_title(f'Original: {cls_name}\n(anomaly={anomaly})', fontsize=11)

        vis_mean = overlay_heatmap(orig_img, mean_attn, alpha=0.5)
        axes[1].imshow(vis_mean)
        axes[1].set_title(f'Mean Attention\n(soft weighted, avg over {n_prototypes} protos)', fontsize=10)

        vis_entropy = overlay_heatmap(orig_img, entropy_map, alpha=0.5, colormap=cv2.COLORMAP_INFERNO)
        axes[2].imshow(vis_entropy)
        axes[2].set_title('Attention Entropy\n(dark=confident, bright=uncertain)', fontsize=10)

        # Dominant prototype as categorical colormap
        cmap = plt.cm.get_cmap('tab10', n_prototypes)
        dom_resized = cv2.resize(dominant_proto.astype(np.float32),
                                  (orig_img.shape[1], orig_img.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)
        axes[3].imshow(orig_img, alpha=0.5)
        im = axes[3].imshow(dom_resized, cmap=cmap, alpha=0.5, vmin=0, vmax=n_prototypes - 1)
        axes[3].set_title('Dominant Prototype\n(which proto has highest sim)', fontsize=10)
        plt.colorbar(im, ax=axes[3], shrink=0.8, label='Proto ID')

        for ax in axes:
            ax.axis('off')
        plt.tight_layout()
        save_path = os.path.join(args.save_dir, f'overview_{idx}_{cls_name}.png')
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {save_path}")

        # --- Figure 2: Per-prototype attention maps ---
        if args.show_per_proto:
            n_cols = min(4, n_prototypes + 1)  # +1 for original image
            n_rows = math.ceil((n_prototypes + 1) / n_cols)
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
            axes = axes.flatten() if n_rows * n_cols > 1 else [axes]

            # First panel: original image
            axes[0].imshow(orig_img)
            axes[0].set_title(f'Original: {cls_name}', fontsize=10)
            axes[0].axis('off')

            # Per-prototype panels
            for p in range(n_prototypes):
                ax = axes[p + 1]
                proto_map = per_proto_maps[:, :, p]  # (H, W)
                vis = overlay_heatmap(orig_img, proto_map, alpha=0.5)
                ax.imshow(vis)

                # Show stats
                sim_min, sim_max = proto_map.min(), proto_map.max()
                ax.set_title(f'Proto {p}\nsim: [{sim_min:.2f}, {sim_max:.2f}]', fontsize=9)
                ax.axis('off')

            # Hide unused panels
            for j in range(n_prototypes + 1, len(axes)):
                axes[j].axis('off')

            plt.suptitle(f'Per-Prototype Attention Maps — {cls_name} (sample {idx})',
                         fontsize=13, fontweight='bold')
            plt.tight_layout()
            save_path = os.path.join(args.save_dir, f'per_proto_{idx}_{cls_name}.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  Saved: {save_path}")

    # === Bonus: Prototype diversity analysis ===
    print("\n=== Prototype Diversity Analysis ===")
    proto_data = prototypes.detach()  # (H*W, N, C)

    # Mean-pool across spatial locations to get global prototypes
    global_protos = proto_data.mean(dim=0)  # (N, C)
    global_protos_norm = F.normalize(global_protos, dim=-1, p=2)

    # Pairwise cosine similarity between prototypes
    cos_sim_matrix = torch.mm(global_protos_norm, global_protos_norm.t()).cpu().numpy()
    print(f"Prototype pairwise cosine similarity (off-diagonal):")
    mask = ~np.eye(n_prototypes, dtype=bool)
    off_diag = cos_sim_matrix[mask]
    print(f"  Mean: {off_diag.mean():.4f}, Std: {off_diag.std():.4f}")
    print(f"  Min:  {off_diag.min():.4f}, Max: {off_diag.max():.4f}")

    # Plot similarity matrix
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    im = ax.imshow(cos_sim_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_title('Prototype Pairwise Cosine Similarity\n(closer to 0 = more diverse)', fontsize=11)
    ax.set_xlabel('Prototype ID')
    ax.set_ylabel('Prototype ID')
    for i in range(n_prototypes):
        for j in range(n_prototypes):
            ax.text(j, i, f'{cos_sim_matrix[i, j]:.2f}',
                    ha='center', va='center', fontsize=8,
                    color='white' if abs(cos_sim_matrix[i, j]) > 0.5 else 'black')
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    save_path = os.path.join(args.save_dir, 'prototype_similarity_matrix.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

    print(f"\nAll results saved to: {args.save_dir}")


if __name__ == '__main__':
    main()
