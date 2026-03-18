"""
Measure Feature Diversity across multiple stages of the BoRAD model.

Metrics computed:
1. Pairwise Cosine Similarity — mean/std of off-diagonal cos sim (lower = more diverse)
2. Singular Value Entropy — normalized entropy of SVD spectrum (higher = more diverse)
3. Effective Rank — number of significant singular values (higher = more diverse)
4. Feature Std — std across feature dimensions (near 0 = collapsed)
5. Uniformity — log of mean pairwise Gaussian kernel (more negative = more uniform/diverse)

Stages analyzed:
- Encoder features (per scale: layer1, layer2, layer3)
- Projected features (per scale)
- Fused features (MFF_OCE output)
- Global features (GAP)
- Prototypes (from loss_terms)

Usage:
    python scripts/measure_feature_diversity.py --run_dir <checkpoint_dir> --config configs/rd/rd_byol_mvtec.py
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import argparse
from collections import OrderedDict

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

from data import get_dataset
from model import get_model
from loss import get_loss_terms
from configs import get_cfg


# ============================================================================
# Diversity Metrics
# ============================================================================

def cosine_similarity_stats(features):
    """
    Pairwise cosine similarity statistics.
    
    Args:
        features: (N, D) — N vectors of dimension D
    Returns:
        dict with mean, std, min, max of off-diagonal cosine similarities
    """
    features = F.normalize(features, dim=-1, p=2)
    sim_matrix = torch.mm(features, features.t())  # (N, N)
    
    # Off-diagonal elements only
    N = sim_matrix.shape[0]
    mask = ~torch.eye(N, dtype=torch.bool, device=sim_matrix.device)
    off_diag = sim_matrix[mask]
    
    return {
        'cos_sim_mean': off_diag.mean().item(),
        'cos_sim_std': off_diag.std().item(),
        'cos_sim_min': off_diag.min().item(),
        'cos_sim_max': off_diag.max().item(),
    }


def singular_value_entropy(features):
    """
    Normalized entropy of singular value spectrum.
    High entropy = diverse (energy spread across dimensions).
    Low entropy = collapsed (energy concentrated in few dimensions).
    
    Args:
        features: (N, D) — N vectors of dimension D
    Returns:
        dict with sv_entropy (0-1 normalized) and effective_rank
    """
    # Center features
    features = features - features.mean(dim=0, keepdim=True)
    
    # SVD
    try:
        U, S, Vh = torch.linalg.svd(features, full_matrices=False)
    except Exception:
        return {'sv_entropy': 0.0, 'effective_rank': 1.0, 'top1_sv_ratio': 1.0}
    
    # Normalize singular values to get probability distribution
    S = S.clamp(min=1e-10)
    p = S / S.sum()
    
    # Shannon entropy, normalized by log(min(N, D))
    entropy = -(p * torch.log(p)).sum()
    max_entropy = torch.log(torch.tensor(float(min(features.shape))))
    normalized_entropy = (entropy / max_entropy).item()
    
    # Effective rank (exponential of entropy)
    effective_rank = torch.exp(entropy).item()
    
    # Top-1 singular value ratio (how much variance is in the first component)
    top1_ratio = (S[0] / S.sum()).item()
    
    return {
        'sv_entropy': normalized_entropy,
        'effective_rank': effective_rank,
        'top1_sv_ratio': top1_ratio,
    }


def feature_std(features):
    """
    Standard deviation across feature dimensions.
    Near 0 = all features are the same (collapsed).
    
    Args:
        features: (N, D)
    Returns:
        dict with mean_std and min_std
    """
    # Std across samples for each dimension
    per_dim_std = features.std(dim=0)  # (D,)
    
    return {
        'feat_std_mean': per_dim_std.mean().item(),
        'feat_std_min': per_dim_std.min().item(),
        'feat_std_max': per_dim_std.max().item(),
        'dead_dims': (per_dim_std < 1e-6).sum().item(),
        'total_dims': per_dim_std.shape[0],
    }


def uniformity(features, t=2.0):
    """
    Uniformity metric from "Understanding Contrastive Representation Learning".
    L_uniform = log E[exp(-t * ||f_i - f_j||^2)]
    More negative = more uniform distribution on hypersphere = more diverse.
    
    Args:
        features: (N, D)
        t: temperature (default 2.0)
    Returns:
        dict with uniformity value
    """
    features = F.normalize(features, dim=-1, p=2)
    
    # Subsample if too many features (avoid OOM)
    N = features.shape[0]
    if N > 2000:
        idx = torch.randperm(N)[:2000]
        features = features[idx]
        N = 2000
    
    # Pairwise squared distances
    sq_dist = torch.cdist(features, features, p=2).pow(2)  # (N, N)
    
    # Off-diagonal
    mask = ~torch.eye(N, dtype=torch.bool, device=features.device)
    sq_dist_off = sq_dist[mask]
    
    uniform_val = torch.log(torch.exp(-t * sq_dist_off).mean() + 1e-10).item()
    
    return {'uniformity': uniform_val}


def compute_all_metrics(features, name=""):
    """Compute all diversity metrics for a feature matrix."""
    if features.dim() > 2:
        # Flatten spatial dims: (B, C, H, W) -> (B*H*W, C) or (B, C, H, W) -> (B, C)
        if features.dim() == 4:
            B, C, H, W = features.shape
            features = features.permute(0, 2, 3, 1).reshape(-1, C)
        elif features.dim() == 3:
            features = features.reshape(-1, features.shape[-1])
    
    # Subsample if too large
    N = features.shape[0]
    if N > 5000:
        idx = torch.randperm(N)[:5000]
        features = features[idx]
    
    metrics = {}
    metrics.update(cosine_similarity_stats(features))
    metrics.update(singular_value_entropy(features))
    metrics.update(feature_std(features))
    metrics.update(uniformity(features))
    
    return metrics


# ============================================================================
# Main
# ============================================================================

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description='Measure Feature Diversity')
    parser.add_argument('--run_dir', type=str, required=True,
                        help='Path to training run directory')
    parser.add_argument('--config', type=str, default='configs/rd/rd_byol_mvtec.py')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of test samples to use')
    parser.add_argument('--save_dir', type=str, default='vis_diversity')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    class Args:
        def __init__(self, cfg_path):
            self.cfg_path = cfg_path
            self.mode = 'test'
            self.opts = []
            self.sleep = -1
            self.memory = -1
            self.dist_url = 'env://'
            self.logger_rank = 0
            
    cfg = get_cfg(Args(args.config))
    model = get_model(cfg.model)
    model = model.to(device)
    
    ckpt_path = os.path.join(args.run_dir, 'net.pth')
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(args.run_dir, 'ckpt.pth')
    
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and 'net' in ckpt:
        model.load_state_dict(ckpt['net'], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
    model.eval()

    # Load prototypes
    loss_terms = get_loss_terms(cfg.loss.loss_terms, device=str(device))
    loss_state_path = os.path.join(args.run_dir, 'loss_terms.pth')
    if os.path.exists(loss_state_path):
        loss_state = torch.load(loss_state_path, map_location=device, weights_only=False)
        for name, state in loss_state.items():
            if name in loss_terms:
                loss_terms[name].load_state_dict(state, strict=False)

    # Load test dataset
    _, test_ds = get_dataset(cfg)
    num_samples = min(args.num_samples, len(test_ds))
    indices = np.random.choice(len(test_ds), num_samples, replace=False)

    # Collect features from all stages
    encoder_feats = [[], [], []]  # 3 scales
    proj_feats = [[], [], []]     # 3 scales
    fused_feats = []              # MFF_OCE output
    global_feats = []             # GAP features
    class_labels = []             # Class names per sample

    print(f"Collecting features from {num_samples} samples...")
    for i, idx in enumerate(indices):
        data = test_ds[idx]
        img = data['img'].unsqueeze(0).to(device)
        
        # Encoder
        feats_t = model.net_t(img)
        for s in range(3):
            encoder_feats[s].append(feats_t[s].cpu())
        
        # Projector
        feats_det = [f.detach() for f in feats_t]
        feats_proj = model.proj_layer(feats_det)
        for s in range(3):
            proj_feats[s].append(feats_proj[s].cpu())
        
        # MFF_OCE
        mid = model.mff_oce(feats_proj)
        fused_feats.append(mid.cpu())
        
        # Global
        glo = F.adaptive_avg_pool2d(mid, 1).squeeze()
        global_feats.append(glo.cpu())
        
        # Class label
        class_labels.append(data['cls_name'])
        
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{num_samples}")

    # Concatenate
    encoder_feats = [torch.cat(f, dim=0) for f in encoder_feats]
    proj_feats = [torch.cat(f, dim=0) for f in proj_feats]
    fused_feats = torch.cat(fused_feats, dim=0)
    global_feats = torch.stack(global_feats, dim=0)  # (N, C)

    # Compute metrics for each stage
    results = OrderedDict()

    print("\n=== Computing Diversity Metrics ===\n")
    
    stage_names = [
        ('Encoder L1', encoder_feats[0]),
        ('Encoder L2', encoder_feats[1]),
        ('Encoder L3', encoder_feats[2]),
        ('Proj L1', proj_feats[0]),
        ('Proj L2', proj_feats[1]),
        ('Proj L3', proj_feats[2]),
        ('Fused (MFF_OCE)', fused_feats),
        ('Global (GAP)', global_feats),
    ]
    
    # Add prototypes if available
    proto_loss = loss_terms.get('proto')
    if proto_loss is not None and hasattr(proto_loss, 'prototypes'):
        protos = proto_loss.prototypes.detach().cpu()  # (H*W, N, C)
        HW, N, C = protos.shape
        H_proto = int(HW ** 0.5)  # Assuming square spatial grid
        W_proto = HW // H_proto
        
        # Mean-pooled global prototypes: diversity across N prototypes
        global_protos = protos.mean(dim=0)  # (N, C)
        stage_names.append(('Prototypes (global avg)', global_protos))

    for name, feats in stage_names:
        feats_device = feats.to(device) if feats.device.type == 'cpu' else feats
        metrics = compute_all_metrics(feats_device, name)
        results[name] = metrics
        
        print(f"--- {name} (shape: {list(feats.shape)}) ---")
        print(f"  Cosine Sim:  mean={metrics['cos_sim_mean']:.4f}  std={metrics['cos_sim_std']:.4f}  "
              f"[{metrics['cos_sim_min']:.4f}, {metrics['cos_sim_max']:.4f}]")
        print(f"  SV Entropy:  {metrics['sv_entropy']:.4f}  "
              f"Eff Rank: {metrics['effective_rank']:.1f}  "
              f"Top-1 SV ratio: {metrics['top1_sv_ratio']:.4f}")
        print(f"  Feature Std: mean={metrics['feat_std_mean']:.4f}  "
              f"dead_dims={metrics['dead_dims']}/{metrics['total_dims']}")
        print(f"  Uniformity:  {metrics['uniformity']:.4f}")
        print()

    # === Per-location prototype diversity ===
    # This is the KEY metric: at each spatial position, how diverse are the N prototypes?
    if proto_loss is not None and hasattr(proto_loss, 'prototypes'):
        protos = proto_loss.prototypes.detach().cpu()  # (H*W, N, C)
        HW, N, C = protos.shape
        H_proto = int(HW ** 0.5)
        W_proto = HW // H_proto
        
        print("=== Per-Location Prototype Diversity ===")
        print(f"  Prototypes: {HW} locations × {N} prototypes × {C} dims\n")
        
        per_loc_cos_sim = []       # Mean cos sim at each location
        per_loc_effective_rank = [] # Effective rank at each location
        
        for s in range(HW):
            # N prototypes at this spatial location: (N, C)
            local_protos = protos[s]  # (N, C)
            local_norm = F.normalize(local_protos, dim=-1, p=2)
            
            # Pairwise cosine similarity between N prototypes at THIS location
            sim = torch.mm(local_norm, local_norm.t())  # (N, N)
            mask = ~torch.eye(N, dtype=torch.bool)
            off_diag = sim[mask]
            per_loc_cos_sim.append(off_diag.mean().item())
            
            # Effective rank at this location
            local_centered = local_protos - local_protos.mean(dim=0, keepdim=True)
            try:
                _, S, _ = torch.linalg.svd(local_centered, full_matrices=False)
                S = S.clamp(min=1e-10)
                p = S / S.sum()
                ent = -(p * torch.log(p)).sum()
                per_loc_effective_rank.append(torch.exp(ent).item())
            except Exception:
                per_loc_effective_rank.append(1.0)
        
        per_loc_cos_sim = np.array(per_loc_cos_sim)
        per_loc_effective_rank = np.array(per_loc_effective_rank)
        
        print(f"  Per-location cosine sim (between {N} protos at same position):")
        print(f"    Mean: {per_loc_cos_sim.mean():.4f}  Std: {per_loc_cos_sim.std():.4f}")
        print(f"    Min:  {per_loc_cos_sim.min():.4f}  Max: {per_loc_cos_sim.max():.4f}")
        print(f"    → {'COLLAPSED' if per_loc_cos_sim.mean() > 0.8 else 'DIVERSE' if per_loc_cos_sim.mean() < 0.3 else 'MODERATE'}")
        print(f"\n  Per-location effective rank:")
        print(f"    Mean: {per_loc_effective_rank.mean():.2f} / {N}")
        print(f"    Min:  {per_loc_effective_rank.min():.2f}  Max: {per_loc_effective_rank.max():.2f}")
        print()
        
        # Add to results for plotting
        results['Proto per-loc'] = {
            'cos_sim_mean': per_loc_cos_sim.mean(),
            'cos_sim_std': per_loc_cos_sim.std(),
            'cos_sim_min': per_loc_cos_sim.min(),
            'cos_sim_max': per_loc_cos_sim.max(),
            'sv_entropy': 0, 'effective_rank': per_loc_effective_rank.mean(),
            'top1_sv_ratio': 0,
            'feat_std_mean': 0, 'feat_std_min': 0, 'feat_std_max': 0,
            'dead_dims': 0, 'total_dims': C,
            'uniformity': 0,
        }
        
        # Spatial heatmap of prototype diversity
        cos_sim_map = per_loc_cos_sim.reshape(H_proto, W_proto)
        rank_map = per_loc_effective_rank.reshape(H_proto, W_proto)
        
        fig, axes_proto = plt.subplots(1, 2, figsize=(12, 5))
        
        im1 = axes_proto[0].imshow(cos_sim_map, cmap='RdYlGn_r', vmin=0, vmax=1)
        axes_proto[0].set_title(f'Per-Location Cosine Sim\n(closer to 1 = collapsed)')
        plt.colorbar(im1, ax=axes_proto[0], shrink=0.8)
        for i in range(H_proto):
            for j in range(W_proto):
                axes_proto[0].text(j, i, f'{cos_sim_map[i,j]:.2f}',
                                   ha='center', va='center', fontsize=7,
                                   color='white' if cos_sim_map[i,j] > 0.5 else 'black')
        
        im2 = axes_proto[1].imshow(rank_map, cmap='viridis', vmin=1, vmax=N)
        axes_proto[1].set_title(f'Per-Location Effective Rank\n(max={N}, higher = more diverse)')
        plt.colorbar(im2, ax=axes_proto[1], shrink=0.8)
        for i in range(H_proto):
            for j in range(W_proto):
                axes_proto[1].text(j, i, f'{rank_map[i,j]:.1f}',
                                   ha='center', va='center', fontsize=7,
                                   color='white' if rank_map[i,j] < N/2 else 'black')
        
        plt.suptitle('Per-Location Prototype Diversity', fontsize=13, fontweight='bold')
        plt.tight_layout()
        save_path = os.path.join(args.save_dir, 'prototype_per_location_diversity.png')
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")

    # === Visualization ===
    stage_labels = list(results.keys())
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Cosine Similarity
    ax = axes[0, 0]
    means = [results[s]['cos_sim_mean'] for s in stage_labels]
    stds = [results[s]['cos_sim_std'] for s in stage_labels]
    x = np.arange(len(stage_labels))
    bars = ax.bar(x, means, yerr=stds, capsize=4, color='steelblue', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(stage_labels, rotation=35, ha='right', fontsize=9)
    ax.set_ylabel('Mean Cosine Similarity')
    ax.set_title('Pairwise Cosine Similarity\n(lower = more diverse)')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', fontsize=8)

    # 2. SV Entropy & Effective Rank
    ax = axes[0, 1]
    entropies = [results[s]['sv_entropy'] for s in stage_labels]
    ranks = [results[s]['effective_rank'] for s in stage_labels]
    ax2 = ax.twinx()
    bars1 = ax.bar(x - 0.2, entropies, 0.4, label='SV Entropy', color='coral', alpha=0.8)
    bars2 = ax2.bar(x + 0.2, ranks, 0.4, label='Eff Rank', color='seagreen', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(stage_labels, rotation=35, ha='right', fontsize=9)
    ax.set_ylabel('SV Entropy (0-1)', color='coral')
    ax2.set_ylabel('Effective Rank', color='seagreen')
    ax.set_title('Singular Value Diversity\n(higher = more diverse)')
    ax.legend(loc='upper left', fontsize=8)
    ax2.legend(loc='upper right', fontsize=8)

    # 3. Feature Std
    ax = axes[1, 0]
    std_means = [results[s]['feat_std_mean'] for s in stage_labels]
    dead = [results[s]['dead_dims'] for s in stage_labels]
    total = [results[s]['total_dims'] for s in stage_labels]
    dead_pct = [d / t * 100 if t > 0 else 0 for d, t in zip(dead, total)]
    
    bars = ax.bar(x, std_means, color='mediumpurple', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(stage_labels, rotation=35, ha='right', fontsize=9)
    ax.set_ylabel('Mean Feature Std')
    ax.set_title('Feature Variation\n(near 0 = collapsed)')
    for bar, val, dp in zip(bars, std_means, dead_pct):
        label = f'{val:.3f}\n({dp:.0f}% dead)' if dp > 0 else f'{val:.3f}'
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                label, ha='center', fontsize=7)

    # 4. Uniformity
    ax = axes[1, 1]
    uniforms = [results[s]['uniformity'] for s in stage_labels]
    bars = ax.bar(x, uniforms, color='goldenrod', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(stage_labels, rotation=35, ha='right', fontsize=9)
    ax.set_ylabel('Uniformity (log scale)')
    ax.set_title('Uniformity on Hypersphere\n(more negative = more diverse)')
    for bar, val in zip(bars, uniforms):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() - 0.15 if val < 0 else bar.get_height() + 0.05,
                f'{val:.2f}', ha='center', fontsize=8)

    plt.suptitle(f'Feature Diversity Analysis — {os.path.basename(args.run_dir)}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(args.save_dir, 'feature_diversity.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

    # === Save raw metrics as text ===
    txt_path = os.path.join(args.save_dir, 'diversity_metrics.txt')
    with open(txt_path, 'w') as f:
        f.write(f"Feature Diversity Report — {args.run_dir}\n")
        f.write(f"Samples: {num_samples}\n")
        f.write("=" * 80 + "\n\n")
        for name, metrics in results.items():
            f.write(f"--- {name} ---\n")
            for k, v in metrics.items():
                f.write(f"  {k}: {v}\n")
            f.write("\n")
    print(f"Saved: {txt_path}")

    # === Angular Distribution Analysis (Class-Aware) ===
    print("\n=== Angular Distribution Analysis (Class-Aware) ===\n")
    
    unique_classes = sorted(set(class_labels))
    label_to_idx = {c: i for i, c in enumerate(unique_classes)}
    label_ids = np.array([label_to_idx[c] for c in class_labels])  # (num_samples,)
    print(f"  Classes: {unique_classes}")
    print(f"  Samples per class: {', '.join(f'{c}={np.sum(label_ids==i)}' for c, i in label_to_idx.items())}")
    
    # Stages to analyze with class info (only per-sample features, not spatial)
    class_stages = OrderedDict()
    class_stages['Global (GAP)'] = global_feats  # (N, C)
    
    # Also add Fused but GAP'd per sample for fair comparison
    fused_gap = F.adaptive_avg_pool2d(fused_feats, 1).squeeze()  # (N, C)
    class_stages['Fused (GAP)'] = fused_gap
    
    for stage_name, feats in class_stages.items():
        feats_2d = feats
        if feats_2d.dim() > 2:
            feats_2d = feats_2d.reshape(feats_2d.shape[0], -1)
        
        N = feats_2d.shape[0]
        feats_norm = F.normalize(feats_2d, dim=-1, p=2)
        cos_sim_mat = torch.mm(feats_norm, feats_norm.t()).cpu().numpy()  # (N, N)
        
        # Build intra/inter class masks
        label_mat_same = label_ids[:, None] == label_ids[None, :]  # (N, N)
        diag_mask = ~np.eye(N, dtype=bool)
        
        intra_mask = label_mat_same & diag_mask  # same class, not self
        inter_mask = ~label_mat_same & diag_mask  # different class
        
        intra_cos = cos_sim_mat[intra_mask]
        inter_cos = cos_sim_mat[inter_mask]
        
        intra_angles = np.degrees(np.arccos(np.clip(intra_cos, -1, 1)))
        inter_angles = np.degrees(np.arccos(np.clip(inter_cos, -1, 1)))
        all_angles = np.degrees(np.arccos(np.clip(cos_sim_mat[diag_mask], -1, 1)))
        
        print(f"\n--- {stage_name} ---")
        print(f"  Intra-class:  mean={intra_angles.mean():.1f}°  std={intra_angles.std():.1f}°  "
              f"(cos_sim={intra_cos.mean():.4f})  n={len(intra_angles)}")
        print(f"  Inter-class:  mean={inter_angles.mean():.1f}°  std={inter_angles.std():.1f}°  "
              f"(cos_sim={inter_cos.mean():.4f})  n={len(inter_angles)}")
        print(f"  Separation:   {inter_angles.mean() - intra_angles.mean():.1f}° gap")
        
        # Per-class intra stats
        print(f"  Per-class intra-class angles:")
        for cls_name, cls_id in label_to_idx.items():
            cls_mask = label_ids == cls_id
            n_cls = cls_mask.sum()
            if n_cls < 2:
                continue
            cls_feats = feats_norm[cls_mask]
            cls_cos = torch.mm(cls_feats, cls_feats.t()).cpu().numpy()
            cls_diag = ~np.eye(n_cls, dtype=bool)
            cls_angles = np.degrees(np.arccos(np.clip(cls_cos[cls_diag], -1, 1)))
            print(f"    {cls_name:>20s}: mean={cls_angles.mean():.1f}°  "
                  f"std={cls_angles.std():.1f}°  (cos={cls_cos[cls_diag].mean():.3f})  n={n_cls}")
    
    # === Plot: class-aware angular distribution ===
    n_stages = len(class_stages)
    fig, axes_ang = plt.subplots(2, n_stages, figsize=(8 * n_stages, 10))
    if n_stages == 1:
        axes_ang = axes_ang.reshape(-1, 1)
    
    for col, (stage_name, feats) in enumerate(class_stages.items()):
        feats_2d = feats
        if feats_2d.dim() > 2:
            feats_2d = feats_2d.reshape(feats_2d.shape[0], -1)
        
        N = feats_2d.shape[0]
        feats_norm = F.normalize(feats_2d, dim=-1, p=2)
        cos_sim_mat = torch.mm(feats_norm, feats_norm.t()).cpu().numpy()
        
        label_mat_same = label_ids[:, None] == label_ids[None, :]
        diag_mask = ~np.eye(N, dtype=bool)
        intra_mask = label_mat_same & diag_mask
        inter_mask = ~label_mat_same & diag_mask
        
        intra_angles = np.degrees(np.arccos(np.clip(cos_sim_mat[intra_mask], -1, 1)))
        inter_angles = np.degrees(np.arccos(np.clip(cos_sim_mat[inter_mask], -1, 1)))
        all_angles = np.degrees(np.arccos(np.clip(cos_sim_mat[diag_mask], -1, 1)))
        
        # Row 0: All + Intra + Inter overlaid
        ax = axes_ang[0, col]
        ax.hist(all_angles, bins=80, density=True, color='gray', alpha=0.3, label='All')
        ax.hist(intra_angles, bins=80, density=True, color='dodgerblue', alpha=0.6, label=f'Intra-class ({intra_angles.mean():.1f}°)')
        ax.hist(inter_angles, bins=80, density=True, color='tomato', alpha=0.6, label=f'Inter-class ({inter_angles.mean():.1f}°)')
        gap = inter_angles.mean() - intra_angles.mean()
        ax.set_title(f'{stage_name}\nIntra vs Inter (gap={gap:.1f}°)', fontsize=11)
        ax.set_xlabel('Pairwise Angle (degrees)')
        ax.set_ylabel('Density')
        ax.legend(fontsize=8)
        ax.set_xlim(0, 180)
        
        # Row 1: Per-class intra distributions
        ax = axes_ang[1, col]
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_classes)))
        for cls_name, cls_id in label_to_idx.items():
            cls_mask = label_ids == cls_id
            n_cls = cls_mask.sum()
            if n_cls < 2:
                continue
            cls_feats = feats_norm[cls_mask]
            cls_cos = torch.mm(cls_feats, cls_feats.t()).cpu().numpy()
            cls_diag = ~np.eye(n_cls, dtype=bool)
            cls_angles = np.degrees(np.arccos(np.clip(cls_cos[cls_diag], -1, 1)))
            ax.hist(cls_angles, bins=40, density=True, alpha=0.4,
                    color=colors[cls_id], label=f'{cls_name} ({cls_angles.mean():.0f}°)')
        ax.set_title(f'{stage_name}\nPer-Class Intra Distributions', fontsize=11)
        ax.set_xlabel('Pairwise Angle (degrees)')
        ax.set_ylabel('Density')
        ax.legend(fontsize=7, ncol=2, loc='upper right')
        ax.set_xlim(0, 180)
    
    plt.suptitle('Class-Aware Angular Distribution\n'
                 '(blue=intra-class should be small, red=inter-class should be large)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(args.save_dir, 'angular_distribution_class_aware.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {save_path}")


if __name__ == '__main__':
    main()
