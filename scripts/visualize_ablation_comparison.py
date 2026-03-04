"""
Ablation Comparison Visualization — Side-by-side t-SNE + Anomaly Map comparison
across all ablation variants to prove each component's contribution.

Usage:
    python scripts/visualize_ablation_comparison.py \
        --run_dirs runs/ablation_A1 runs/ablation_A2 ... runs/ablation_A7 \
        --labels "CosOnly" "Cos+Dense" "Cos+Proto" "Cos+Dense+Proto" "Pred" "Pred+Dense" "Full" \
        --config configs/rd/rd_byol_mvtec.py \
        --save_dir vis_ablation

Creates:
    1. Multi-panel t-SNE figure: shows how feature clusters improve with each component
    2. Multi-panel anomaly map figure: shows how anomaly detection quality improves
    3. Quantitative bar chart: metrics across ablation variants
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.manifold import TSNE
from tqdm import tqdm
import argparse
import json

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

from data import get_dataset
from model import MODEL

# Try to import get_cfg from util.cfg, fallback to configs.__init__
try:
    from util.cfg import get_cfg
except ImportError:
    from configs import get_cfg


def load_model(config_path, checkpoint_path, device):
    """Load model from config and checkpoint."""
    # Load config
    if hasattr(get_cfg, '__code__') and get_cfg.__code__.co_varnames[0] == 'opt_terminal':
        # configs.__init__.get_cfg expects argparse Namespace
        from argparse import Namespace
        opt = Namespace(
            cfg_path=config_path, mode='test', data_path=None,
            sleep=-1, memory=-1, dist_url='env://', logger_rank=0, opts=[]
        )
        cfg = get_cfg(opt)
    else:
        cfg = get_cfg(config_path)

    # Create model
    model = MODEL.get_module(cfg.model.name)(**cfg.model.kwargs)
    model = model.to(device)

    # Load checkpoint
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
def extract_gap_features(model, dataloader, device):
    """Extract global features after GAP from online branch."""
    all_features = []
    all_labels = []
    all_cls_names = []

    for batch in tqdm(dataloader, desc="Extracting features", leave=False):
        imgs = batch['img'].to(device)
        labels = batch['label']
        cls_names = batch['cls_name']

        feats_t = model.net_t(imgs)
        feats_t_detached = [f.detach() for f in feats_t]
        feats_proj = model.proj_layer(feats_t_detached)
        mid = model.mff_oce(feats_proj)
        glo_feats = F.adaptive_avg_pool2d(mid, 1).squeeze(-1).squeeze(-1)

        all_features.append(glo_feats.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_cls_names.extend(cls_names)

    return np.concatenate(all_features, axis=0), np.array(all_labels), all_cls_names


@torch.no_grad()
def extract_anomaly_maps(model, dataloader, device, num_samples=4):
    """Extract anomaly maps for a few samples."""
    from util.metric import get_evaluator
    samples = []
    count = 0

    for batch in dataloader:
        imgs = batch['img'].to(device)
        masks = batch['img_mask']
        cls_names = batch['cls_name']
        anomaly = batch['anomaly']

        # Only pick anomaly samples
        for i in range(imgs.shape[0]):
            if anomaly[i] == 1 and count < num_samples:
                feats_t = model.net_t(imgs[i:i+1])
                feats_t_d = [f.detach() for f in feats_t]
                feats_proj = model.proj_layer(feats_t_d)
                mid = model.mff_oce(feats_proj)
                feats_s = model.net_s(mid)

                # Compute anomaly map via cosine distance
                ano_map = 0
                for ft, fs in zip(feats_t_d, feats_s):
                    ft_n = F.normalize(ft, dim=1)
                    fs_n = F.normalize(fs, dim=1)
                    cos_dist = 1 - (ft_n * fs_n).sum(dim=1, keepdim=True)
                    cos_dist = F.interpolate(cos_dist, size=(imgs.shape[2], imgs.shape[3]),
                                            mode='bilinear', align_corners=False)
                    ano_map = ano_map + cos_dist

                ano_map = ano_map.squeeze().cpu().numpy()
                # Denormalize image
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img_np = imgs[i].cpu().numpy().transpose(1, 2, 0) * std + mean
                img_np = np.clip(img_np, 0, 1)

                samples.append({
                    'image': img_np,
                    'mask': masks[i].squeeze().numpy(),
                    'anomaly_map': ano_map,
                    'cls_name': cls_names[i],
                })
                count += 1

        if count >= num_samples:
            break

    return samples


def plot_tsne_grid(all_features_dict, save_path, perplexity=30):
    """
    Create a multi-panel t-SNE figure comparing ablation variants.
    Each panel shows the feature distribution for one ablation variant.
    """
    n_variants = len(all_features_dict)
    ncols = min(4, n_variants)
    nrows = (n_variants + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # Compute shared t-SNE on concatenated features for fair comparison
    all_feats_concat = []
    all_labels_concat = []
    boundaries = [0]
    for name, (feats, labels, _) in all_features_dict.items():
        all_feats_concat.append(feats)
        all_labels_concat.append(labels)
        boundaries.append(boundaries[-1] + len(feats))

    all_feats = np.concatenate(all_feats_concat, axis=0)
    all_labels = np.concatenate(all_labels_concat, axis=0)

    print(f"Running shared t-SNE on {all_feats.shape[0]} total samples...")
    effective_perplexity = min(perplexity, all_feats.shape[0] // len(all_features_dict) - 1)
    tsne = TSNE(n_components=2, perplexity=effective_perplexity, random_state=42,
                n_iter=1000, init='pca', learning_rate='auto')
    feats_2d = tsne.fit_transform(all_feats)

    # Determine global axis limits
    x_min, x_max = feats_2d[:, 0].min() - 5, feats_2d[:, 0].max() + 5
    y_min, y_max = feats_2d[:, 1].min() - 5, feats_2d[:, 1].max() + 5

    unique_labels = np.unique(all_labels)
    n_classes = len(unique_labels)
    cmap = plt.cm.get_cmap('tab20' if n_classes <= 20 else 'hsv')
    colors = [cmap(i / n_classes) for i in range(n_classes)]

    for idx, (name, (feats, labels, cls_names)) in enumerate(all_features_dict.items()):
        ax = axes[idx]
        start = boundaries[idx]
        end = boundaries[idx + 1]
        f2d = feats_2d[start:end]

        for lbl_idx, lbl in enumerate(unique_labels):
            mask = labels == lbl
            if mask.sum() > 0:
                ax.scatter(f2d[mask, 0], f2d[mask, 1], c=[colors[lbl_idx]],
                          alpha=0.6, s=15, edgecolors='none')

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_title(name, fontsize=11, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(True, alpha=0.15)

    # Hide unused axes
    for idx in range(n_variants, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle('Feature Space Comparison Across Ablation Variants (t-SNE)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved t-SNE comparison to: {save_path}")
    plt.close()


def plot_anomaly_map_grid(all_samples_dict, save_path):
    """
    Create a grid of anomaly maps: rows = samples, columns = ablation variants.
    Shows how anomaly detection improves with each component.
    """
    variant_names = list(all_samples_dict.keys())
    n_variants = len(variant_names)

    # Find minimum number of samples across all variants
    n_samples = min(len(v) for v in all_samples_dict.values())
    n_samples = min(n_samples, 4)  # Cap at 4 rows

    if n_samples == 0:
        print("No anomaly samples found, skipping anomaly map visualization.")
        return

    # Columns: Input | GT | Variant1 | Variant2 | ...
    ncols = 2 + n_variants
    fig, axes = plt.subplots(n_samples, ncols, figsize=(2.5 * ncols, 2.5 * n_samples))
    if n_samples == 1:
        axes = axes[np.newaxis, :]

    for row in range(n_samples):
        # Input image (from first variant)
        first_variant = all_samples_dict[variant_names[0]]
        axes[row, 0].imshow(first_variant[row]['image'])
        axes[row, 0].set_xticks([])
        axes[row, 0].set_yticks([])
        if row == 0:
            axes[row, 0].set_title('Input', fontsize=9, fontweight='bold')

        # Ground truth mask
        axes[row, 1].imshow(first_variant[row]['mask'], cmap='gray')
        axes[row, 1].set_xticks([])
        axes[row, 1].set_yticks([])
        if row == 0:
            axes[row, 1].set_title('GT', fontsize=9, fontweight='bold')

        # Each variant's anomaly map
        # Find global max for normalization
        global_max = max(all_samples_dict[v][row]['anomaly_map'].max()
                        for v in variant_names if row < len(all_samples_dict[v]))

        for col_idx, variant_name in enumerate(variant_names):
            samples = all_samples_dict[variant_name]
            if row < len(samples):
                amap = samples[row]['anomaly_map']
                amap_norm = amap / (global_max + 1e-8)
                axes[row, col_idx + 2].imshow(amap_norm, cmap='jet', vmin=0, vmax=1)
            axes[row, col_idx + 2].set_xticks([])
            axes[row, col_idx + 2].set_yticks([])
            if row == 0:
                axes[row, col_idx + 2].set_title(variant_name, fontsize=8, fontweight='bold')

    fig.suptitle('Anomaly Map Comparison Across Ablation Variants', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved anomaly map comparison to: {save_path}")
    plt.close()


def plot_cluster_quality(all_features_dict, save_path):
    """
    Bar chart of cluster quality metrics (Silhouette Score, Inter/Intra-class ratio)
    across ablation variants.
    """
    from sklearn.metrics import silhouette_score

    names = []
    silhouette_scores = []
    inter_intra_ratios = []

    for name, (feats, labels, _) in all_features_dict.items():
        names.append(name)
        unique_labels = np.unique(labels)

        if len(unique_labels) > 1 and len(feats) > len(unique_labels):
            sil = silhouette_score(feats, labels, sample_size=min(2000, len(feats)))
            silhouette_scores.append(sil)
        else:
            silhouette_scores.append(0)

        # Inter/Intra ratio
        centroids = []
        intra_dists = []
        for lbl in unique_labels:
            mask = labels == lbl
            cls_feats = feats[mask]
            centroid = cls_feats.mean(axis=0)
            centroids.append(centroid)
            intra_dist = np.mean(np.linalg.norm(cls_feats - centroid, axis=1))
            intra_dists.append(intra_dist)

        centroids = np.array(centroids)
        avg_intra = np.mean(intra_dists)
        if len(centroids) > 1:
            from scipy.spatial.distance import pdist
            avg_inter = np.mean(pdist(centroids))
            inter_intra_ratios.append(avg_inter / (avg_intra + 1e-8))
        else:
            inter_intra_ratios.append(0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    x = np.arange(len(names))
    width = 0.6

    bars1 = ax1.bar(x, silhouette_scores, width, color='steelblue', edgecolor='black', linewidth=0.5)
    ax1.set_ylabel('Silhouette Score', fontsize=11)
    ax1.set_title('Cluster Separation Quality', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=30, ha='right', fontsize=9)
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    for bar, val in zip(bars1, silhouette_scores):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    bars2 = ax2.bar(x, inter_intra_ratios, width, color='coral', edgecolor='black', linewidth=0.5)
    ax2.set_ylabel('Inter/Intra Ratio', fontsize=11)
    ax2.set_title('Class Discriminability (↑ better)', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=30, ha='right', fontsize=9)
    for bar, val in zip(bars2, inter_intra_ratios):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{val:.2f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved cluster quality comparison to: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Ablation comparison visualization')
    parser.add_argument('--run_dirs', nargs='+', required=True,
                        help='Paths to run directories for each ablation variant')
    parser.add_argument('--labels', nargs='+', required=True,
                        help='Labels for each variant (same order as run_dirs)')
    parser.add_argument('--config', type=str, default='configs/rd/rd_byol_mvtec.py',
                        help='Config file (default config for data loading)')
    parser.add_argument('--save_dir', type=str, default='vis_ablation',
                        help='Output directory')
    parser.add_argument('--perplexity', type=int, default=30)
    parser.add_argument('--split', type=str, default='test', choices=['train', 'test'])
    parser.add_argument('--no_anomaly_maps', action='store_true',
                        help='Skip anomaly map visualization (faster)')
    args = parser.parse_args()

    assert len(args.run_dirs) == len(args.labels), \
        f"Number of run_dirs ({len(args.run_dirs)}) must match labels ({len(args.labels)})"

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load dataset once
    print("Loading dataset...")
    first_model, cfg = load_model(args.config,
                                   os.path.join(args.run_dirs[0], 'net.pth'), device)
    train_dataset, test_dataset = get_dataset(cfg)
    dataset = train_dataset if args.split == 'train' else test_dataset
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True
    )
    del first_model

    # Extract features from each variant
    all_features = {}
    all_anomaly_samples = {}

    for run_dir, label in zip(args.run_dirs, args.labels):
        print(f"\n{'='*60}")
        print(f"Processing: {label} ({run_dir})")
        print(f"{'='*60}")

        ckpt_path = os.path.join(run_dir, 'net.pth')
        if not os.path.exists(ckpt_path):
            ckpt_path = os.path.join(run_dir, 'ckpt.pth')
        if not os.path.exists(ckpt_path):
            print(f"  WARNING: No checkpoint found in {run_dir}, skipping.")
            continue

        model, _ = load_model(args.config, ckpt_path, device)

        # Extract t-SNE features
        feats, labels, cls_names = extract_gap_features(model, dataloader, device)
        all_features[label] = (feats, labels, cls_names)
        print(f"  Features: {feats.shape}")

        # Extract anomaly maps
        if not args.no_anomaly_maps and args.split == 'test':
            test_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size=8, shuffle=False, num_workers=4
            )
            samples = extract_anomaly_maps(model, test_loader, device, num_samples=4)
            all_anomaly_samples[label] = samples
            print(f"  Anomaly samples: {len(samples)}")

        del model
        torch.cuda.empty_cache()

    if not all_features:
        print("ERROR: No valid checkpoints found!")
        return

    # Plot t-SNE comparison
    print("\n" + "="*60)
    print("Generating t-SNE comparison...")
    plot_tsne_grid(all_features, os.path.join(args.save_dir, 'ablation_tsne_comparison.png'),
                   perplexity=args.perplexity)

    # Plot cluster quality metrics
    print("Generating cluster quality metrics...")
    plot_cluster_quality(all_features, os.path.join(args.save_dir, 'ablation_cluster_quality.png'))

    # Plot anomaly map comparison
    if all_anomaly_samples:
        print("Generating anomaly map comparison...")
        plot_anomaly_map_grid(all_anomaly_samples,
                             os.path.join(args.save_dir, 'ablation_anomaly_maps.png'))

    print(f"\n✅ All visualizations saved to: {args.save_dir}")


if __name__ == '__main__':
    main()
