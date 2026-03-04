"""
Momentum Drift Visualization — Show how features move slowly in representation space.

Key argument: EMA momentum causes feature points to drift slowly,
preserving fine-grained details that would be lost with rapid updates.

Usage:
    python scripts/visualize_momentum_drift.py \
        --run_dir runs/full_model \
        --config configs/rd/rd_byol_mvtec.py \
        --save_dir vis_momentum

Creates:
    1. Feature displacement plot: how much each sample's feature moves between
       online and momentum encoder (small displacement = stability)
    2. Feature consistency heatmap: cosine similarity between online and momentum features
    3. Comparison with/without momentum: feature scatter (online vs target)
    4. Per-class displacement distribution: shows momentum preserves class structure
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm
import argparse

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

from data import get_dataset
from model import MODEL

try:
    from util.cfg import get_cfg
except ImportError:
    from configs import get_cfg


def load_model(config_path, checkpoint_path, device):
    """Load model from config and checkpoint."""
    if hasattr(get_cfg, '__code__') and get_cfg.__code__.co_varnames[0] == 'opt_terminal':
        from argparse import Namespace
        opt = Namespace(
            cfg_path=config_path, mode='test', data_path=None,
            sleep=-1, memory=-1, dist_url='env://', logger_rank=0, opts=[]
        )
        cfg = get_cfg(opt)
    else:
        cfg = get_cfg(config_path)

    model = MODEL.get_module(cfg.model.name)(**cfg.model.kwargs)
    model = model.to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'net' in checkpoint:
        model.load_state_dict(checkpoint['net'], strict=False)
    elif isinstance(checkpoint, dict) and 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)

    model.eval()
    return model, cfg


@torch.no_grad()
def extract_online_vs_momentum_features(model, dataloader, device):
    """
    Extract features from BOTH online and momentum encoders simultaneously.
    This shows the feature displacement caused by momentum EMA.
    """
    online_features = []
    momentum_features = []
    all_labels = []
    all_cls_names = []

    # Multi-scale local features for spatial analysis
    online_local = {0: [], 1: [], 2: []}
    momentum_local = {0: [], 1: [], 2: []}

    for batch in tqdm(dataloader, desc="Extracting online vs momentum features"):
        imgs = batch['img'].to(device)
        labels = batch['label']
        cls_names = batch['cls_name']

        # === Online path ===
        feats_t = model.net_t(imgs)
        feats_proj = model.proj_layer([f.detach() for f in feats_t])
        mid_online = model.mff_oce(feats_proj)
        glo_online = F.adaptive_avg_pool2d(mid_online, 1).squeeze(-1).squeeze(-1)

        # === Momentum path ===
        feats_mom = model.proj_layer_momentum([f.detach() for f in feats_t])
        mid_mom = model.mff_oce(feats_mom)
        glo_mom = F.adaptive_avg_pool2d(mid_mom, 1).squeeze(-1).squeeze(-1)

        online_features.append(glo_online.cpu().numpy())
        momentum_features.append(glo_mom.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_cls_names.extend(cls_names)

        # Local features per scale
        for i in range(min(3, len(feats_proj))):
            f_on = F.adaptive_avg_pool2d(feats_proj[i], 1).squeeze(-1).squeeze(-1)
            f_mom = F.adaptive_avg_pool2d(feats_mom[i], 1).squeeze(-1).squeeze(-1)
            online_local[i].append(f_on.cpu().numpy())
            momentum_local[i].append(f_mom.cpu().numpy())

    result = {
        'online': np.concatenate(online_features, axis=0),
        'momentum': np.concatenate(momentum_features, axis=0),
        'labels': np.array(all_labels),
        'cls_names': all_cls_names,
        'online_local': {k: np.concatenate(v, axis=0) for k, v in online_local.items() if v},
        'momentum_local': {k: np.concatenate(v, axis=0) for k, v in momentum_local.items() if v},
    }
    return result


def plot_feature_displacement(data, save_path):
    """
    Plot 1: Feature displacement between online and momentum encoder.
    Low displacement = momentum successfully slows down feature drift.
    """
    online = data['online']
    momentum = data['momentum']
    labels = data['labels']

    # Compute L2 displacement
    displacement = np.linalg.norm(online - momentum, axis=1)

    # Compute cosine similarity
    online_norm = online / (np.linalg.norm(online, axis=1, keepdims=True) + 1e-8)
    mom_norm = momentum / (np.linalg.norm(momentum, axis=1, keepdims=True) + 1e-8)
    cos_sim = np.sum(online_norm * mom_norm, axis=1)

    # Get class names
    unique_labels = np.unique(labels)
    label_to_cls = {}
    for lbl, name in zip(data['labels'], data['cls_names']):
        if lbl not in label_to_cls:
            label_to_cls[lbl] = name

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: Displacement distribution per class
    ax = axes[0]
    cls_data = []
    cls_names_sorted = []
    for lbl in unique_labels:
        mask = labels == lbl
        cls_data.append(displacement[mask])
        cls_names_sorted.append(label_to_cls[lbl])

    bp = ax.boxplot(cls_data, labels=cls_names_sorted, patch_artist=True,
                    medianprops=dict(color='black', linewidth=1.5))
    cmap = plt.cm.get_cmap('tab20')
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(cmap(i / len(cls_data)))
        patch.set_alpha(0.7)
    ax.set_ylabel('L2 Displacement', fontsize=11)
    ax.set_title('Feature Displacement per Class\n(Online ↔ Momentum)', fontsize=11, fontweight='bold')
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    ax.axhline(y=np.median(displacement), color='red', linestyle='--', alpha=0.5,
               label=f'Median: {np.median(displacement):.4f}')
    ax.legend(fontsize=9)

    # Panel 2: Cosine similarity histogram
    ax = axes[1]
    ax.hist(cos_sim, bins=50, color='steelblue', edgecolor='black', linewidth=0.5, alpha=0.8)
    ax.axvline(x=np.mean(cos_sim), color='red', linestyle='--',
               label=f'Mean: {np.mean(cos_sim):.4f}')
    ax.set_xlabel('Cosine Similarity', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Feature Consistency\n(Online ↔ Momentum)', fontsize=11, fontweight='bold')
    ax.legend(fontsize=10)

    # Panel 3: t-SNE overlay showing online (circle) vs momentum (x)
    ax = axes[2]
    combined = np.concatenate([online, momentum], axis=0)
    n = len(online)
    perp = min(30, n - 1)
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42, n_iter=1000,
                init='pca', learning_rate='auto')
    combined_2d = tsne.fit_transform(combined)

    online_2d = combined_2d[:n]
    mom_2d = combined_2d[n:]

    n_classes = len(unique_labels)
    cmap = plt.cm.get_cmap('tab20' if n_classes <= 20 else 'hsv')
    colors = [cmap(i / n_classes) for i in range(n_classes)]

    for idx, lbl in enumerate(unique_labels):
        mask = labels == lbl
        ax.scatter(online_2d[mask, 0], online_2d[mask, 1], c=[colors[idx]],
                  marker='o', s=30, alpha=0.5, label=f'{label_to_cls[lbl]}')
        ax.scatter(mom_2d[mask, 0], mom_2d[mask, 1], c=[colors[idx]],
                  marker='x', s=30, alpha=0.5)
        # Draw arrows from online to momentum
        for j in np.where(mask)[0][::max(1, mask.sum()//5)]:  # Sample 5 arrows per class
            ax.annotate('', xy=(mom_2d[j, 0], mom_2d[j, 1]),
                       xytext=(online_2d[j, 0], online_2d[j, 1]),
                       arrowprops=dict(arrowstyle='->', color=colors[idx], alpha=0.4, lw=0.8))

    ax.set_title('Online (●) vs Momentum (✕)\n(Arrows show drift)', fontsize=11, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(True, alpha=0.15)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved feature displacement plot to: {save_path}")
    plt.close()


def plot_multiscale_displacement(data, save_path):
    """
    Show displacement at each feature scale (layer 1, 2, 3).
    Argument: momentum preserves fine-grained details at high-resolution scales.
    """
    scales = sorted(data['online_local'].keys())
    if not scales:
        print("No multi-scale features available, skipping.")
        return

    fig, axes = plt.subplots(1, len(scales), figsize=(5 * len(scales), 4))
    if len(scales) == 1:
        axes = [axes]

    scale_names = ['Layer 1\n(High res)', 'Layer 2\n(Mid res)', 'Layer 3\n(Low res)']

    for idx, scale in enumerate(scales):
        online_s = data['online_local'][scale]
        mom_s = data['momentum_local'][scale]

        # Cosine similarity per sample
        on = online_s / (np.linalg.norm(online_s, axis=1, keepdims=True) + 1e-8)
        mo = mom_s / (np.linalg.norm(mom_s, axis=1, keepdims=True) + 1e-8)
        cos_sim = np.sum(on * mo, axis=1)

        ax = axes[idx]
        ax.hist(cos_sim, bins=50, color=['#2196F3', '#4CAF50', '#FF9800'][idx],
                edgecolor='black', linewidth=0.3, alpha=0.8)
        ax.axvline(x=np.mean(cos_sim), color='red', linestyle='--',
                   label=f'Mean: {np.mean(cos_sim):.4f}')
        ax.set_xlabel('Cosine Similarity', fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        ax.set_title(f'{scale_names[idx] if idx < len(scale_names) else f"Scale {scale}"}',
                     fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.set_xlim(0.5, 1.05)

    fig.suptitle('Multi-Scale Feature Consistency (Online ↔ Momentum)\nHigher similarity = slower drift = more detail preserved',
                 fontsize=12, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved multi-scale displacement plot to: {save_path}")
    plt.close()


def plot_momentum_comparison(data_with_mom, data_without_mom, save_path):
    """
    Compare feature stability with vs without momentum encoder.
    data_without_mom: features from a model without momentum (e.g., A1 baseline).
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for idx, (data, title) in enumerate([
        (data_with_mom, 'With Momentum (Proposed)'),
        (data_without_mom, 'Without Momentum (Baseline)')
    ]):
        if data is None:
            axes[idx].text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=16)
            axes[idx].set_title(title, fontsize=11, fontweight='bold')
            continue

        online = data['online']
        labels = data['labels']
        unique_labels = np.unique(labels)
        n_classes = len(unique_labels)
        cmap = plt.cm.get_cmap('tab20' if n_classes <= 20 else 'hsv')
        colors = [cmap(i / n_classes) for i in range(n_classes)]

        perp = min(30, len(online) - 1)
        tsne = TSNE(n_components=2, perplexity=perp, random_state=42,
                    n_iter=1000, init='pca', learning_rate='auto')
        feats_2d = tsne.fit_transform(online)

        ax = axes[idx]
        for lbl_idx, lbl in enumerate(unique_labels):
            mask = labels == lbl
            ax.scatter(feats_2d[mask, 0], feats_2d[mask, 1], c=[colors[lbl_idx]],
                      alpha=0.6, s=20, edgecolors='none')

        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(True, alpha=0.15)

        # Add compactness metric annotation
        from sklearn.metrics import silhouette_score
        if n_classes > 1:
            sil = silhouette_score(online, labels, sample_size=min(2000, len(online)))
            ax.annotate(f'Silhouette: {sil:.3f}', xy=(0.02, 0.02),
                       xycoords='axes fraction', fontsize=10,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    fig.suptitle('Feature Space Comparison: Impact of Momentum Encoder',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved momentum comparison to: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Momentum drift visualization')
    parser.add_argument('--run_dir', type=str, required=True,
                        help='Run directory of the full model (with momentum)')
    parser.add_argument('--baseline_dir', type=str, default=None,
                        help='Run directory of baseline (without momentum, e.g. A1)')
    parser.add_argument('--config', type=str, default='configs/rd/rd_byol_mvtec.py')
    parser.add_argument('--save_dir', type=str, default='vis_momentum')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'test'])
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model with momentum
    ckpt_path = os.path.join(args.run_dir, 'net.pth')
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(args.run_dir, 'ckpt.pth')
    model, cfg = load_model(args.config, ckpt_path, device)

    # Check model has momentum encoder
    if not hasattr(model, 'proj_layer_momentum'):
        print("ERROR: Model does not have a momentum encoder. Only works with rd_lgc_byol.")
        return

    # Create dataloader
    train_dataset, test_dataset = get_dataset(cfg)
    dataset = train_dataset if args.split == 'train' else test_dataset
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True
    )

    # Extract features from both paths
    print("Extracting online vs momentum features...")
    data = extract_online_vs_momentum_features(model, dataloader, device)

    # Plot 1: Feature displacement
    print("\nGenerating feature displacement visualization...")
    plot_feature_displacement(data, os.path.join(args.save_dir, 'momentum_displacement.png'))

    # Plot 2: Multi-scale displacement
    print("Generating multi-scale displacement...")
    plot_multiscale_displacement(data, os.path.join(args.save_dir, 'momentum_multiscale.png'))

    # Plot 3: Comparison with baseline (if provided)
    baseline_data = None
    if args.baseline_dir:
        print(f"\nLoading baseline model from {args.baseline_dir}...")
        bl_ckpt = os.path.join(args.baseline_dir, 'net.pth')
        if not os.path.exists(bl_ckpt):
            bl_ckpt = os.path.join(args.baseline_dir, 'ckpt.pth')
        bl_model, _ = load_model(args.config, bl_ckpt, device)
        if hasattr(bl_model, 'proj_layer_momentum'):
            baseline_data = extract_online_vs_momentum_features(bl_model, dataloader, device)
        else:
            # No momentum — just extract online features
            baseline_data = extract_online_vs_momentum_features(bl_model, dataloader, device)
        del bl_model

    plot_momentum_comparison(data, baseline_data,
                            os.path.join(args.save_dir, 'momentum_comparison.png'))

    print(f"\n✅ All momentum visualizations saved to: {args.save_dir}")


if __name__ == '__main__':
    main()
