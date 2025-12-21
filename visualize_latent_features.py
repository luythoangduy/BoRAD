"""
Standalone script to visualize latent features after GAP layer (online branch).
Run this from the conano directory.

Usage:
    python visualize_latent_features.py --run_dir runs/RDLGCBYOLTrainer_configs_rd_rd_byol_proto_mvtec_ablation/dense_lam_1.0

This script will:
1. Load the trained model from ckpt.pth or net.pth
2. Extract global features after GAP from the online branch
3. Use t-SNE to reduce dimensionality to 2D
4. Plot scatter with colors representing different classes
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
import json

# Ensure we're in the right directory
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from data import get_dataset, get_loader
from model import MODEL
from util.cfg import get_cfg


class FeatureExtractor:
    """Extract latent features from the online branch after GAP."""

    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.eval()

    @torch.no_grad()
    def extract(self, dataloader, include_anomaly_info=False):
        """
        Extract global features after GAP from online branch.

        Returns:
            features: (N, D) numpy array
            labels: (N,) numpy array of class indices
            cls_names: list of class name strings
            anomalies: (N,) numpy array (if include_anomaly_info=True)
        """
        all_features = []
        all_labels = []
        all_cls_names = []
        all_anomalies = []

        for batch in tqdm(dataloader, desc="Extracting latent features"):
            imgs = batch['img'].to(self.device)
            labels = batch['label']
            cls_names = batch['cls_name']

            # Forward through encoder (frozen backbone)
            feats_t = self.model.net_t(imgs)
            feats_t_detached = [f.detach() for f in feats_t]

            # Forward through projection layer
            feats_proj = self.model.proj_layer(feats_t_detached)

            # Feature fusion via MFF_OCE
            mid = self.model.mff_oce(feats_proj)

            # Global Average Pooling (GAP) - this is the latent feature we visualize
            # Shape: (B, C, H, W) -> (B, C)
            glo_feats = F.adaptive_avg_pool2d(mid, 1).squeeze(-1).squeeze(-1)

            all_features.append(glo_feats.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_cls_names.extend(cls_names)

            if include_anomaly_info and 'anomaly' in batch:
                all_anomalies.extend(batch['anomaly'].numpy())

        features = np.concatenate(all_features, axis=0)
        labels = np.array(all_labels)

        if include_anomaly_info:
            anomalies = np.array(all_anomalies)
            return features, labels, all_cls_names, anomalies

        return features, labels, all_cls_names


def create_tsne_plot(features, labels, cls_names, save_path, perplexity=30,
                     title="Latent Features after GAP (Online Branch)"):
    """
    Create t-SNE visualization with class-based coloring.
    """
    print(f"\nRunning t-SNE on {features.shape[0]} samples ({features.shape[1]} dims)...")

    # Handle perplexity for small datasets
    n_samples = features.shape[0]
    effective_perplexity = min(perplexity, n_samples - 1)
    if effective_perplexity < perplexity:
        print(f"Adjusted perplexity from {perplexity} to {effective_perplexity} (dataset size: {n_samples})")

    tsne = TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        random_state=42,
        n_iter=1000,
        init='pca',
        learning_rate='auto'
    )
    features_2d = tsne.fit_transform(features)

    # Get unique classes
    unique_labels = np.unique(labels)
    unique_cls_names = []
    label_to_cls = {}
    for label, cls_name in zip(labels, cls_names):
        if label not in label_to_cls:
            label_to_cls[label] = cls_name
    for label in unique_labels:
        unique_cls_names.append(label_to_cls[label])

    n_classes = len(unique_labels)

    # Choose colormap based on number of classes
    if n_classes <= 10:
        cmap = plt.cm.get_cmap('tab10')
    elif n_classes <= 20:
        cmap = plt.cm.get_cmap('tab20')
    else:
        cmap = plt.cm.get_cmap('hsv')

    colors = [cmap(i / n_classes) for i in range(n_classes)]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot each class
    for idx, (label, cls_name) in enumerate(zip(unique_labels, unique_cls_names)):
        mask = labels == label
        scatter = ax.scatter(
            features_2d[mask, 0],
            features_2d[mask, 1],
            c=[colors[idx]],
            label=cls_name,
            alpha=0.7,
            s=60,
            edgecolors='white',
            linewidth=0.5
        )

    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Legend
    legend = ax.legend(
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),
        fontsize=9,
        framealpha=0.9,
        title='Classes'
    )
    legend.get_title().set_fontweight('bold')

    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved visualization to: {save_path}")
    plt.close()

    return features_2d


def create_tsne_plot_with_anomaly(features, labels, cls_names, anomalies, save_path,
                                   perplexity=30):
    """
    Create t-SNE visualization with class colors and anomaly markers.
    Normal: circle (o), Anomaly: triangle (^)
    """
    print(f"\nRunning t-SNE on {features.shape[0]} samples ({features.shape[1]} dims)...")

    n_samples = features.shape[0]
    effective_perplexity = min(perplexity, n_samples - 1)

    tsne = TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        random_state=42,
        n_iter=1000,
        init='pca',
        learning_rate='auto'
    )
    features_2d = tsne.fit_transform(features)

    unique_labels = np.unique(labels)
    label_to_cls = {}
    for label, cls_name in zip(labels, cls_names):
        if label not in label_to_cls:
            label_to_cls[label] = cls_name

    n_classes = len(unique_labels)
    if n_classes <= 10:
        cmap = plt.cm.get_cmap('tab10')
    elif n_classes <= 20:
        cmap = plt.cm.get_cmap('tab20')
    else:
        cmap = plt.cm.get_cmap('hsv')

    colors = [cmap(i / n_classes) for i in range(n_classes)]

    fig, ax = plt.subplots(figsize=(14, 10))

    for idx, label in enumerate(unique_labels):
        cls_name = label_to_cls[label]
        mask_cls = labels == label

        # Normal samples
        mask_normal = mask_cls & (anomalies == 0)
        if np.any(mask_normal):
            ax.scatter(
                features_2d[mask_normal, 0],
                features_2d[mask_normal, 1],
                c=[colors[idx]],
                label=f'{cls_name} (normal)',
                alpha=0.7,
                s=50,
                marker='o',
                edgecolors='white',
                linewidth=0.5
            )

        # Anomaly samples
        mask_anomaly = mask_cls & (anomalies == 1)
        if np.any(mask_anomaly):
            ax.scatter(
                features_2d[mask_anomaly, 0],
                features_2d[mask_anomaly, 1],
                c=[colors[idx]],
                label=f'{cls_name} (anomaly)',
                alpha=0.9,
                s=80,
                marker='^',
                edgecolors='black',
                linewidth=1
            )

    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax.set_title('Latent Features after GAP - Normal vs Anomaly', fontsize=14, fontweight='bold')

    legend = ax.legend(
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),
        fontsize=8,
        framealpha=0.9,
        ncol=1
    )

    ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved visualization to: {save_path}")
    plt.close()

    return features_2d


def main():
    parser = argparse.ArgumentParser(description='Visualize latent features after GAP layer')
    parser.add_argument('--run_dir', type=str, required=True,
                        help='Path to run directory containing ckpt.pth and config')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file (optional, will try to find automatically)')
    parser.add_argument('--perplexity', type=int, default=30,
                        help='t-SNE perplexity (default: 30)')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'test'],
                        help='Dataset split to use (default: test)')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Directory to save visualizations (default: run_dir/vis_latent)')
    parser.add_argument('--show_anomaly', action='store_true',
                        help='Show normal vs anomaly samples with different markers')
    args = parser.parse_args()

    # Setup paths
    run_dir = args.run_dir
    if not os.path.isabs(run_dir):
        run_dir = os.path.join(script_dir, run_dir)

    save_dir = args.save_dir or os.path.join(run_dir, 'vis_latent')
    os.makedirs(save_dir, exist_ok=True)

    # Find checkpoint
    ckpt_path = os.path.join(run_dir, 'ckpt.pth')
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(run_dir, 'net.pth')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"No checkpoint found in {run_dir}")

    # Find config
    config_path = args.config
    if config_path is None:
        # Try to infer config from run_dir name
        run_name = os.path.basename(run_dir.rstrip('/'))
        parent_name = os.path.basename(os.path.dirname(run_dir.rstrip('/')))

        # Look for config pattern in parent directory name
        if 'configs_rd_rd_byol' in parent_name:
            config_name = parent_name.replace('RDLGCBYOLTrainer_', '').replace('_', '/')
            config_path = os.path.join(script_dir, f'{config_name}.py')

        # Fallback to default
        if config_path is None or not os.path.exists(config_path):
            config_path = os.path.join(script_dir, 'configs/rd/rd_byol_mvtec.py')

    print(f"Run directory: {run_dir}")
    print(f"Config file: {config_path}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Save directory: {save_dir}")

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load config
    cfg = get_cfg(config_path)

    # Create model
    print("\nLoading model...")
    model = MODEL.get_module(cfg.model.name)(**cfg.model.kwargs)
    model = model.to(device)

    # Load checkpoint
    checkpoint = torch.load(ckpt_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'], strict=False)
        print(f"Loaded model from checkpoint (epoch: {checkpoint.get('epoch', 'unknown')})")
    else:
        model.load_state_dict(checkpoint, strict=False)
        print("Loaded model weights")

    # Create dataloader
    print(f"\nLoading {args.split} dataset...")
    train_dataset, test_dataset = get_dataset(cfg)
    dataset = train_dataset if args.split == 'train' else test_dataset

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.trainer.data.batch_size_per_gpu_test,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    print(f"Dataset size: {len(dataset)}")
    print(f"Number of classes: {len(dataset.cls_names)}")
    print(f"Classes: {dataset.cls_names}")

    # Extract features
    extractor = FeatureExtractor(model, device)

    if args.show_anomaly and args.split == 'test':
        features, labels, cls_names, anomalies = extractor.extract(
            dataloader, include_anomaly_info=True
        )
        print(f"\nExtracted features shape: {features.shape}")
        print(f"Normal samples: {np.sum(anomalies == 0)}, Anomaly samples: {np.sum(anomalies == 1)}")

        # Create visualization with anomaly markers
        save_path = os.path.join(save_dir, f'latent_tsne_{args.split}_anomaly.png')
        create_tsne_plot_with_anomaly(
            features, labels, cls_names, anomalies,
            save_path, perplexity=args.perplexity
        )
    else:
        features, labels, cls_names = extractor.extract(dataloader)
        print(f"\nExtracted features shape: {features.shape}")

    # Create standard visualization
    save_path = os.path.join(save_dir, f'latent_tsne_{args.split}.png')
    create_tsne_plot(
        features, labels, cls_names, save_path,
        perplexity=args.perplexity,
        title=f'Latent Features after GAP (Online Branch) - {args.split.upper()}'
    )

    # Save features for further analysis
    np.save(os.path.join(save_dir, f'features_{args.split}.npy'), features)
    np.save(os.path.join(save_dir, f'labels_{args.split}.npy'), labels)
    with open(os.path.join(save_dir, f'cls_names_{args.split}.json'), 'w') as f:
        json.dump(list(set(cls_names)), f)
    print(f"\nSaved features, labels, and class names to {save_dir}")

    print("\nDone!")


if __name__ == '__main__':
    main()
