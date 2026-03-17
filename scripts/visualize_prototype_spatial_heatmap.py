"""
Visualize Spatial Prototype Similarity Heatmaps.
Calculates cosine similarity between each feature patch and its corresponding spatial prototype.
Displays maps for both Online (Projection) and Target (Momentum) branches.
Upsamples and overlays the maps on the original image.
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import cv2

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

from data import get_dataset, get_loader
from model import get_model
from loss import get_loss_terms
from util.cfg import get_cfg

def load_model_and_loss(config_path, checkpoint_path, device):
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

    model.eval()
    return model, cfg, loss_terms

def get_similarity_map(feats, prototypes):
    """
    Args:
        feats: (B, C, H, W)
        prototypes: (H*W, N, C)
    Returns:
        sim_map: (B, H, W) max similarity with any prototype at that location
    """
    B, C, H, W = feats.shape
    # (B, H*W, C)
    feats_flat = feats.view(B, C, -1).transpose(1, 2)
    feats_norm = F.normalize(feats_flat, dim=-1, p=2)
    
    # (H*W, N, C)
    proto_norm = F.normalize(prototypes, dim=-1, p=2)
    
    # Cosine similarity: (B, H*W, N)
    sim = torch.einsum('bsc,snc->bsn', feats_norm, proto_norm)
    
    # Max similarity across prototypes for each spatial location
    max_sim = sim.max(dim=-1)[0] # (B, H*W)
    return max_sim.view(B, H, W)

def overlay_heatmap(img, heatmap, alpha=0.5):
    """Overlay heatmap on RGB image."""
    # Heatmap to 0-255
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    heatmap = (heatmap * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Resize heatmap to image size
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    
    # Overlay
    img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
    overlayed = cv2.addWeighted(img, 1-alpha, heatmap, alpha, 0)
    return overlayed

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', type=str, required=True)
    parser.add_argument('--config', type=str, default='configs/rd/rd_byol_mvtec.py')
    parser.add_argument('--num_samples', type=int, default=5)
    parser.add_argument('--save_dir', type=str, default='vis_spatial_proto')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt_path = os.path.join(args.run_dir, 'net.pth')
    if not os.path.exists(ckpt_path): ckpt_path = os.path.join(args.run_dir, 'ckpt.pth')
    
    model, cfg, loss_terms = load_model_and_loss(args.config, ckpt_path, device)
    proto_loss = loss_terms.get('proto')
    
    if proto_loss is None or proto_loss.prototypes is None:
        print("Error: No prototypes found.")
        return

    train_ds, test_ds = get_dataset(cfg)
    # Get a few samples from test dataset
    indices = np.random.choice(len(test_ds), args.num_samples, replace=False)
    
    for i, idx in enumerate(indices):
        data = test_ds[idx]
        img = data['img'].unsqueeze(0).to(device)
        cls_name = data['cls_name']
        
        # Forward
        feats_t = model.net_t(img)
        # Online path
        feats_v1 = [f.detach() for f in feats_t]
        feats_t_proj = model.proj_layer(feats_v1)
        mid_online = model.mff_oce(feats_t_proj)
        
        # Momentum path
        feats_t_k_grid = model.proj_layer_momentum(feats_v1)
        mid_momentum = model.mff_oce(feats_t_k_grid)
        
        # Sim maps
        sim_online = get_similarity_map(mid_online, proto_loss.prototypes)[0].cpu().numpy()
        sim_momentum = get_similarity_map(mid_momentum, proto_loss.prototypes)[0].cpu().numpy()
        
        # Visualization
        orig_img = data['img'].permute(1, 2, 0).cpu().numpy()
        # Denormalize image if needed (assuming ImageNet normalization)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        orig_img = (orig_img * std + mean).clip(0, 1)
        
        vis_online = overlay_heatmap(orig_img, sim_online)
        vis_momentum = overlay_heatmap(orig_img, sim_momentum)
        
        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(orig_img); axes[0].set_title(f"Original: {cls_name}")
        axes[1].imshow(vis_online); axes[1].set_title(f"Online Similarity (Projection)")
        axes[2].imshow(vis_momentum); axes[2].set_title(f"Momentum Similarity (Target)")
        
        for ax in axes: ax.axis('off')
        plt.tight_layout()
        save_path = os.path.join(args.save_dir, f"sample_{idx}_{cls_name}.png")
        plt.savefig(save_path, dpi=200)
        plt.close()
        print(f"Saved: {save_path}")

if __name__ == '__main__':
    main()
