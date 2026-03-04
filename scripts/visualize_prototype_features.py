"""
Prototype Features Visualization — Visualize learned prototypes and their role
as gradient absorbers that shield the main branch from excessive updates.

Key arguments:
1. Prototypes absorb gradients → main branch remains stable
2. Prototype vectors form meaningful cluster representatives
3. Features query prototypes to create enhanced representations

Usage:
    python scripts/visualize_prototype_features.py \
        --run_dir runs/full_model \
        --config configs/rd/rd_byol_mvtec.py \
        --save_dir vis_prototype

Creates:
    1. Prototype heatmap: cosine similarity between prototypes (orthogonality)
    2. Feature-prototype assignment: which prototype each sample is closest to
    3. Gradient flow analysis: gradient norms of prototype vs main branch params
    4. Prototype attention map: how prototypes attend to different feature regions
"""

import os
import sys
import torch
import torch.nn as nn
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
from loss import get_loss_terms

try:
    from util.cfg import get_cfg
except ImportError:
    from configs import get_cfg


def load_model_and_loss(config_path, checkpoint_path, device):
    """Load model and reconstruct loss terms (including prototypes)."""
    if hasattr(get_cfg, '__code__') and get_cfg.__code__.co_varnames[0] == 'opt_terminal':
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
    else:
        model.load_state_dict(checkpoint, strict=False)

    # Reconstruct loss terms
    loss_terms = get_loss_terms(cfg.loss.loss_terms, device=str(device))

    # Try to load loss state if saved
    loss_state_path = os.path.join(os.path.dirname(checkpoint_path), 'loss_terms.pth')
    if os.path.exists(loss_state_path):
        loss_state = torch.load(loss_state_path, map_location=device, weights_only=False)
        for name, state in loss_state.items():
            if name in loss_terms:
                try:
                    loss_terms[name].load_state_dict(state, strict=False)
                    print(f"  Loaded loss state for '{name}'")
                except Exception as e:
                    print(f"  Warning: Could not load loss state for '{name}': {e}")

    model.eval()
    return model, cfg, loss_terms


@torch.no_grad()
def extract_features_with_prototype_info(model, loss_terms, dataloader, device):
    """
    Extract global features AND their relationship with prototypes.
    """
    proto_loss = loss_terms.get('proto', None)
    if proto_loss is None:
        raise ValueError("No 'proto' loss term found. This script requires PrototypeInfoNCELoss.")

    all_features = []
    all_labels = []
    all_cls_names = []
    all_proto_sims = []    # cosine similarity with each prototype
    all_proto_enhanced = []  # enhanced features after prototype query

    for batch in tqdm(dataloader, desc="Extracting features with prototype info"):
        imgs = batch['img'].to(device)
        labels = batch['label']
        cls_names = batch['cls_name']

        # Forward through model
        feats_t = model.net_t(imgs)
        feats_t_d = [f.detach() for f in feats_t]
        feats_proj = model.proj_layer(feats_t_d)
        mid = model.mff_oce(feats_proj)
        glo_feats = F.adaptive_avg_pool2d(mid, 1).squeeze(-1).squeeze(-1)

        all_features.append(glo_feats.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_cls_names.extend(cls_names)

        # Compute prototype similarities
        if proto_loss.prototypes is not None:
            feats_norm = F.normalize(glo_feats, dim=1, p=2)
            proto_norm = F.normalize(proto_loss.prototypes, dim=1, p=2)
            cos_sim = torch.matmul(feats_norm, proto_norm.T)  # (B, n_proto)
            all_proto_sims.append(cos_sim.cpu().numpy())

    result = {
        'features': np.concatenate(all_features, axis=0),
        'labels': np.array(all_labels),
        'cls_names': all_cls_names,
    }

    if all_proto_sims:
        result['proto_sims'] = np.concatenate(all_proto_sims, axis=0)

    return result


def compute_gradient_analysis(model, loss_terms, dataloader, device, num_batches=5):
    """
    Compute gradient norms to show that prototypes absorb gradients
    while the main branch (proj_layer, net_s) receives smaller gradients.
    """
    proto_loss = loss_terms.get('proto', None)
    if proto_loss is None or proto_loss.prototypes is None:
        return None

    # Enable gradient for prototypes
    proto_loss.train()
    model.train()

    # Collect gradient norms over a few batches
    grad_records = {
        'proto_params': [],     # Prototype parameters
        'projector_params': [],  # Projector (main branch local)
        'proto_projector': [],  # Prototype projector
        'decoder_params': [],   # Decoder (main branch)
    }

    batch_count = 0
    for batch in dataloader:
        if batch_count >= num_batches:
            break

        imgs = batch['img'].to(device)
        labels = batch['label'].to(device)

        # Forward pass
        outputs = model(imgs, imgs)  # Training mode
        feats_t, feats_s, feats_t_k, feats_t_q_grid, feats_t_k_grid, glo_feats, glo_feats_k = outputs

        # Compute prototype loss only
        loss = proto_loss(glo_feats, glo_feats_k, labels)
        loss.backward(retain_graph=False)

        # Record gradient norms
        if proto_loss.prototypes is not None and proto_loss.prototypes.grad is not None:
            grad_records['proto_params'].append(
                proto_loss.prototypes.grad.norm().item()
            )

        if proto_loss.projector is not None:
            proj_grads = [p.grad.norm().item() for p in proto_loss.projector.parameters()
                         if p.grad is not None]
            if proj_grads:
                grad_records['proto_projector'].append(np.mean(proj_grads))

        # Main branch gradients
        proj_grads = [p.grad.norm().item() for p in model.proj_layer.parameters()
                     if p.grad is not None]
        if proj_grads:
            grad_records['projector_params'].append(np.mean(proj_grads))

        dec_grads = [p.grad.norm().item() for p in model.net_s.parameters()
                    if p.grad is not None]
        if dec_grads:
            grad_records['decoder_params'].append(np.mean(dec_grads))

        # Zero gradients
        model.zero_grad()
        proto_loss.zero_grad()
        batch_count += 1

    model.eval()
    proto_loss.eval()

    # Average over batches
    result = {}
    for key, values in grad_records.items():
        if values:
            result[key] = np.mean(values)

    return result


def plot_prototype_similarity_matrix(proto_loss, save_path):
    """
    Plot cosine similarity matrix between prototypes.
    Should show near-orthogonality (low off-diagonal values).
    """
    if proto_loss.prototypes is None:
        print("Prototypes not initialized, skipping similarity matrix.")
        return

    prototypes = proto_loss.prototypes.detach().cpu()
    proto_norm = F.normalize(prototypes, dim=1, p=2)
    sim_matrix = torch.matmul(proto_norm, proto_norm.T).numpy()

    n_proto = sim_matrix.shape[0]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(sim_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')

    # Add text annotations
    for i in range(n_proto):
        for j in range(n_proto):
            color = 'white' if abs(sim_matrix[i, j]) > 0.5 else 'black'
            ax.text(j, i, f'{sim_matrix[i, j]:.2f}', ha='center', va='center',
                   fontsize=10, color=color, fontweight='bold' if i == j else 'normal')

    ax.set_xticks(range(n_proto))
    ax.set_yticks(range(n_proto))
    ax.set_xticklabels([f'P{i}' for i in range(n_proto)])
    ax.set_yticklabels([f'P{i}' for i in range(n_proto)])
    ax.set_title('Prototype Cosine Similarity Matrix\n(Off-diagonal ≈ 0 means orthogonal)',
                 fontsize=11, fontweight='bold')

    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved prototype similarity matrix to: {save_path}")
    plt.close()


def plot_feature_prototype_tsne(data, proto_loss, save_path):
    """
    t-SNE plot showing features colored by class + prototype vectors as large stars.
    Also shows which prototype each sample is closest to.
    """
    if proto_loss.prototypes is None:
        print("Prototypes not initialized, skipping t-SNE with prototypes.")
        return

    features = data['features']
    labels = data['labels']
    prototypes = proto_loss.prototypes.detach().cpu().numpy()

    n_proto = prototypes.shape[0]

    # Combine features and prototypes for shared t-SNE
    combined = np.concatenate([features, prototypes], axis=0)
    n_feats = len(features)

    perp = min(30, n_feats - 1)
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42,
                n_iter=1000, init='pca', learning_rate='auto')
    combined_2d = tsne.fit_transform(combined)

    feats_2d = combined_2d[:n_feats]
    proto_2d = combined_2d[n_feats:]

    # Assign each feature to nearest prototype
    if 'proto_sims' in data:
        assignments = np.argmax(data['proto_sims'], axis=1)
    else:
        feats_norm = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
        proto_norm = prototypes / (np.linalg.norm(prototypes, axis=1, keepdims=True) + 1e-8)
        sims = feats_norm @ proto_norm.T
        assignments = np.argmax(sims, axis=1)

    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Panel 1: Color by class label
    ax = axes[0]
    cmap_cls = plt.cm.get_cmap('tab20' if n_classes <= 20 else 'hsv')
    colors_cls = [cmap_cls(i / n_classes) for i in range(n_classes)]

    label_to_cls = {}
    for lbl, name in zip(data['labels'], data['cls_names']):
        if lbl not in label_to_cls:
            label_to_cls[lbl] = name

    for lbl_idx, lbl in enumerate(unique_labels):
        mask = labels == lbl
        ax.scatter(feats_2d[mask, 0], feats_2d[mask, 1], c=[colors_cls[lbl_idx]],
                  alpha=0.5, s=15, label=label_to_cls[lbl])

    # Plot prototypes as big stars
    proto_colors = plt.cm.Set1(np.linspace(0, 1, n_proto))
    for i in range(n_proto):
        ax.scatter(proto_2d[i, 0], proto_2d[i, 1], marker='*', s=400,
                  c=[proto_colors[i]], edgecolors='black', linewidth=1.5,
                  label=f'Proto {i}', zorder=10)

    ax.set_title('Features (class-colored) + Prototypes (★)',
                 fontsize=11, fontweight='bold')
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=7, ncol=1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(True, alpha=0.15)

    # Panel 2: Color by nearest prototype
    ax = axes[1]
    for p_idx in range(n_proto):
        mask = assignments == p_idx
        if mask.sum() > 0:
            ax.scatter(feats_2d[mask, 0], feats_2d[mask, 1], c=[proto_colors[p_idx]],
                      alpha=0.4, s=15, label=f'→ Proto {p_idx} ({mask.sum()} samples)')

    # Plot prototypes
    for i in range(n_proto):
        ax.scatter(proto_2d[i, 0], proto_2d[i, 1], marker='*', s=400,
                  c=[proto_colors[i]], edgecolors='black', linewidth=1.5, zorder=10)

    ax.set_title('Features colored by nearest prototype',
                 fontsize=11, fontweight='bold')
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(True, alpha=0.15)

    fig.suptitle('Prototype-Feature Relationship in Latent Space',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved prototype t-SNE to: {save_path}")
    plt.close()


def plot_prototype_attention_per_class(data, save_path):
    """
    Heatmap: rows = classes, columns = prototypes.
    Shows average cosine similarity of each class with each prototype.
    Argues that prototypes learn meaningful class-aware representations.
    """
    if 'proto_sims' not in data:
        print("No prototype similarity data available, skipping.")
        return

    labels = data['labels']
    proto_sims = data['proto_sims']
    unique_labels = np.unique(labels)

    label_to_cls = {}
    for lbl, name in zip(data['labels'], data['cls_names']):
        if lbl not in label_to_cls:
            label_to_cls[lbl] = name

    # Average similarity per class per prototype
    n_proto = proto_sims.shape[1]
    cls_proto_sim = np.zeros((len(unique_labels), n_proto))
    cls_names_ordered = []

    for idx, lbl in enumerate(unique_labels):
        mask = labels == lbl
        cls_proto_sim[idx] = proto_sims[mask].mean(axis=0)
        cls_names_ordered.append(label_to_cls[lbl])

    fig, ax = plt.subplots(figsize=(max(4, n_proto * 1.2), max(6, len(unique_labels) * 0.5)))
    im = ax.imshow(cls_proto_sim, cmap='YlOrRd', aspect='auto')

    # Add text annotations
    for i in range(len(unique_labels)):
        for j in range(n_proto):
            val = cls_proto_sim[i, j]
            color = 'white' if val > 0.5 * cls_proto_sim.max() else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=8, color=color)

    ax.set_xticks(range(n_proto))
    ax.set_yticks(range(len(unique_labels)))
    ax.set_xticklabels([f'Proto {i}' for i in range(n_proto)], fontsize=10)
    ax.set_yticklabels(cls_names_ordered, fontsize=9)
    ax.set_xlabel('Prototypes', fontsize=11)
    ax.set_ylabel('Classes', fontsize=11)
    ax.set_title('Average Prototype Attention per Class\n(Higher = stronger affinity)',
                 fontsize=12, fontweight='bold')

    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved prototype attention heatmap to: {save_path}")
    plt.close()


def plot_gradient_flow(grad_data, save_path):
    """
    Bar chart showing gradient norms: prototype params vs main branch params.
    Argues that prototypes absorb gradients, shielding the main branch.
    """
    if grad_data is None or not grad_data:
        print("No gradient data available, skipping gradient flow plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    labels_display = {
        'proto_params': 'Prototype\nVectors',
        'proto_projector': 'Prototype\nProjector',
        'projector_params': 'Main Branch\nProjector',
        'decoder_params': 'Main Branch\nDecoder',
    }

    names = []
    values = []
    bar_colors = []
    color_map = {
        'proto_params': '#E53935',
        'proto_projector': '#FF7043',
        'projector_params': '#1E88E5',
        'decoder_params': '#42A5F5',
    }

    for key in ['proto_params', 'proto_projector', 'projector_params', 'decoder_params']:
        if key in grad_data:
            names.append(labels_display.get(key, key))
            values.append(grad_data[key])
            bar_colors.append(color_map.get(key, 'gray'))

    x = np.arange(len(names))
    bars = ax.bar(x, values, 0.6, color=bar_colors, edgecolor='black', linewidth=0.5)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
               f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel('Average Gradient Norm', fontsize=11)
    ax.set_title('Gradient Flow: Prototypes Absorb Gradients\n'
                 '(Red = prototype path, Blue = main branch)',
                 fontsize=12, fontweight='bold')

    # Add annotation
    if 'proto_params' in grad_data and 'projector_params' in grad_data:
        ratio = grad_data['proto_params'] / (grad_data['projector_params'] + 1e-8)
        ax.annotate(f'Proto/Main ratio: {ratio:.1f}×',
                   xy=(0.98, 0.95), xycoords='axes fraction',
                   fontsize=11, ha='right', va='top',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                           edgecolor='orange', alpha=0.9))

    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved gradient flow plot to: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Prototype features visualization')
    parser.add_argument('--run_dir', type=str, required=True,
                        help='Run directory of a model with PrototypeInfoNCELoss')
    parser.add_argument('--config', type=str, default='configs/rd/rd_byol_mvtec.py')
    parser.add_argument('--save_dir', type=str, default='vis_prototype')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--compute_gradients', action='store_true',
                        help='Compute gradient flow analysis (requires GPU, takes more time)')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model and loss terms
    ckpt_path = os.path.join(args.run_dir, 'net.pth')
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(args.run_dir, 'ckpt.pth')

    print("Loading model and loss terms...")
    model, cfg, loss_terms = load_model_and_loss(args.config, ckpt_path, device)

    proto_loss = loss_terms.get('proto', None)
    if proto_loss is None:
        print("ERROR: No 'proto' loss found in config. This script requires PrototypeInfoNCELoss.")
        return

    # Create dataloader
    train_dataset, test_dataset = get_dataset(cfg)
    dataset = train_dataset if args.split == 'train' else test_dataset
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True
    )

    # Initialize prototypes by running one forward pass
    print("\nInitializing prototypes by running forward pass...")
    with torch.no_grad():
        sample_batch = next(iter(dataloader))
        imgs = sample_batch['img'].to(device)
        feats_t = model.net_t(imgs)
        feats_proj = model.proj_layer([f.detach() for f in feats_t])
        mid = model.mff_oce(feats_proj)
        glo_feats = F.adaptive_avg_pool2d(mid, 1).squeeze(-1).squeeze(-1)
        # This initializes prototypes if not yet initialized
        _ = proto_loss(glo_feats, glo_feats.clone(), sample_batch['label'].to(device))

    if proto_loss.prototypes is None:
        print("ERROR: Prototypes failed to initialize.")
        return

    print(f"Prototypes shape: {proto_loss.prototypes.shape}")
    print(f"Number of prototypes: {proto_loss.n_prototypes}")

    # Extract features with prototype info
    print("\nExtracting features with prototype info...")
    data = extract_features_with_prototype_info(model, loss_terms, dataloader, device)

    # Plot 1: Prototype similarity matrix (orthogonality)
    print("\nGenerating prototype similarity matrix...")
    plot_prototype_similarity_matrix(proto_loss,
                                     os.path.join(args.save_dir, 'proto_similarity_matrix.png'))

    # Plot 2: t-SNE with prototypes
    print("Generating prototype t-SNE...")
    plot_feature_prototype_tsne(data, proto_loss,
                                os.path.join(args.save_dir, 'proto_tsne.png'))

    # Plot 3: Prototype attention per class
    print("Generating prototype attention heatmap...")
    plot_prototype_attention_per_class(data,
                                       os.path.join(args.save_dir, 'proto_attention_heatmap.png'))

    # Plot 4: Gradient flow analysis (optional, requires training mode)
    if args.compute_gradients:
        print("\nComputing gradient flow analysis...")
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=True
        )
        grad_data = compute_gradient_analysis(model, loss_terms, train_loader, device)
        plot_gradient_flow(grad_data,
                          os.path.join(args.save_dir, 'proto_gradient_flow.png'))

    print(f"\n✅ All prototype visualizations saved to: {args.save_dir}")


if __name__ == '__main__':
    main()
