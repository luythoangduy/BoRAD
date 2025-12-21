"""
Visualization script for latent features after GAP layer (online branch).
Uses t-SNE/UMAP to reduce dimensionality and plots with class-based colors.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import get_dataset, get_loader
from model import MODEL
from util.net import load_checkpoint
from util.cfg import get_cfg


def extract_features(model, dataloader, device):
    """
    Extract global features after GAP from the online branch.

    Returns:
        features: numpy array of shape (N, D) - latent features
        labels: numpy array of shape (N,) - class indices
        cls_names: list of class names
    """
    model.eval()

    all_features = []
    all_labels = []
    all_cls_names = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features"):
            imgs = batch['img'].to(device)
            labels = batch['label']
            cls_names = batch['cls_name']

            # Get features from online branch
            # Forward pass through encoder and projection
            feats_t = model.net_t(imgs)
            feats_t_detached = [f.detach() for f in feats_t]
            feats_proj = model.proj_layer(feats_t_detached)

            # Feature fusion (MFF_OCE)
            mid = model.mff_oce(feats_proj)

            # Global Average Pooling - this is what we want to visualize
            glo_feats = torch.nn.functional.adaptive_avg_pool2d(mid, 1).squeeze(-1).squeeze(-1)

            all_features.append(glo_feats.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_cls_names.extend(cls_names)

    features = np.concatenate(all_features, axis=0)
    labels = np.array(all_labels)

    return features, labels, all_cls_names


def visualize_tsne(features, labels, cls_names, save_path, perplexity=30, title="Latent Features (t-SNE)"):
    """
    Visualize features using t-SNE with class-based coloring.

    Args:
        features: (N, D) array of features
        labels: (N,) array of class indices
        cls_names: list of class names for each sample
        save_path: path to save the figure
        perplexity: t-SNE perplexity parameter
        title: plot title
    """
    print(f"Running t-SNE on {features.shape[0]} samples with {features.shape[1]} dimensions...")

    # Run t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=1000, verbose=1)
    features_2d = tsne.fit_transform(features)

    # Get unique classes
    unique_labels = np.unique(labels)
    unique_cls_names = sorted(set(cls_names))
    n_classes = len(unique_labels)

    # Create color map
    cmap = plt.cm.get_cmap('tab20' if n_classes <= 20 else 'hsv')
    colors = [cmap(i / n_classes) for i in range(n_classes)]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot each class
    for idx, (label, cls_name) in enumerate(zip(unique_labels, unique_cls_names)):
        mask = labels == label
        ax.scatter(
            features_2d[mask, 0],
            features_2d[mask, 1],
            c=[colors[idx]],
            label=cls_name,
            alpha=0.7,
            s=50,
            edgecolors='white',
            linewidth=0.5
        )

    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='best', fontsize=8, ncol=2 if n_classes > 10 else 1)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved figure to {save_path}")
    plt.close()

    return features_2d


def visualize_with_anomaly(features, labels, cls_names, anomalies, save_path, perplexity=30):
    """
    Visualize features with both class coloring and anomaly markers.
    Normal samples: circles, Anomaly samples: triangles
    """
    print(f"Running t-SNE on {features.shape[0]} samples...")

    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=1000, verbose=1)
    features_2d = tsne.fit_transform(features)

    unique_labels = np.unique(labels)
    unique_cls_names = sorted(set(cls_names))
    n_classes = len(unique_labels)

    cmap = plt.cm.get_cmap('tab20' if n_classes <= 20 else 'hsv')
    colors = [cmap(i / n_classes) for i in range(n_classes)]

    fig, ax = plt.subplots(figsize=(14, 10))

    for idx, (label, cls_name) in enumerate(zip(unique_labels, unique_cls_names)):
        mask_cls = labels == label

        # Normal samples (anomaly == 0)
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

        # Anomaly samples (anomaly == 1)
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
    ax.set_title('Latent Features (t-SNE) - Normal vs Anomaly', fontsize=14)
    ax.legend(loc='best', fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved figure to {save_path}")
    plt.close()

    return features_2d


def main():
    parser = argparse.ArgumentParser(description='Visualize latent features after GAP layer')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint file')
    parser.add_argument('--save_dir', type=str, default='./vis_latent', help='Directory to save visualizations')
    parser.add_argument('--perplexity', type=int, default=30, help='t-SNE perplexity')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'test'], help='Dataset split to use')
    parser.add_argument('--use_umap', action='store_true', help='Use UMAP instead of t-SNE')
    args = parser.parse_args()

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Load config
    cfg = get_cfg(args.config)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create model
    model = MODEL.get_module(cfg.model.name)(**cfg.model.kwargs)
    model = model.to(device)

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)

    # Create dataloader
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
    print(f"Classes: {dataset.cls_names}")

    # Extract features
    features, labels, cls_names = extract_features(model, dataloader, device)
    print(f"Extracted features shape: {features.shape}")

    # Visualize with t-SNE
    if args.use_umap:
        try:
            import umap
            print("Using UMAP for dimensionality reduction...")
            reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
            features_2d = reducer.fit_transform(features)

            # Create visualization
            unique_labels = np.unique(labels)
            unique_cls_names = sorted(set(cls_names))
            n_classes = len(unique_labels)
            cmap = plt.cm.get_cmap('tab20' if n_classes <= 20 else 'hsv')
            colors = [cmap(i / n_classes) for i in range(n_classes)]

            fig, ax = plt.subplots(figsize=(12, 10))
            for idx, (label, cls_name) in enumerate(zip(unique_labels, unique_cls_names)):
                mask = labels == label
                ax.scatter(features_2d[mask, 0], features_2d[mask, 1], c=[colors[idx]],
                          label=cls_name, alpha=0.7, s=50, edgecolors='white', linewidth=0.5)

            ax.set_xlabel('UMAP Dimension 1', fontsize=12)
            ax.set_ylabel('UMAP Dimension 2', fontsize=12)
            ax.set_title('Latent Features (UMAP)', fontsize=14)
            ax.legend(loc='best', fontsize=8)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()

            save_path = os.path.join(args.save_dir, f'latent_umap_{args.split}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved UMAP visualization to {save_path}")
            plt.close()

        except ImportError:
            print("UMAP not installed. Falling back to t-SNE.")
            args.use_umap = False

    if not args.use_umap:
        save_path = os.path.join(args.save_dir, f'latent_tsne_{args.split}.png')
        visualize_tsne(features, labels, cls_names, save_path, perplexity=args.perplexity,
                      title=f'Latent Features after GAP (Online Branch) - {args.split}')

    # Save features for further analysis
    np.save(os.path.join(args.save_dir, f'features_{args.split}.npy'), features)
    np.save(os.path.join(args.save_dir, f'labels_{args.split}.npy'), labels)
    print(f"Saved features and labels to {args.save_dir}")


if __name__ == '__main__':
    main()
