"""
Visualize Global Feature Space using t-SNE.
Compares two trained model runs to demonstrate lack of feature collapse.
Extracts features after the global predictor (shifted by prototypes).
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

# Add project root to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

from data import get_dataset, get_loader
from model import get_model
from loss import get_loss_terms
from configs import get_cfg

def load_model_and_loss(config_path, checkpoint_dir, device):
    """Load model and loss components from a run directory."""
    # Mock terminal args for get_cfg
    class Args:
        def __init__(self, cfg_path):
            self.cfg_path = cfg_path
            self.mode = 'test'
            self.opts = []
            self.sleep = -1
            self.memory = -1
            self.dist_url = 'env://'
            self.logger_rank = 0
            
    cfg = get_cfg(Args(config_path))
    model = get_model(cfg.model)
    model = model.to(device)

    # Find checkpoint
    ckpt_path = os.path.join(checkpoint_dir, 'net.pth')
    if not os.path.exists(ckpt_path): ckpt_path = os.path.join(checkpoint_dir, 'ckpt.pth')
    
    if not os.path.exists(ckpt_path):
        print(f"Warning: Checkpoint not found in {checkpoint_dir}")
        return None, cfg, None

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'net' in checkpoint:
        model.load_state_dict(checkpoint['net'], strict=False)
    elif isinstance(checkpoint, dict) and 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)

    # Load loss terms (contains global predictor and prototypes)
    loss_terms = get_loss_terms(cfg.loss.loss_terms, device=str(device))
    loss_state_path = os.path.join(checkpoint_dir, 'loss_terms.pth')
    if os.path.exists(loss_state_path):
        loss_state = torch.load(loss_state_path, map_location=device, weights_only=False)
        for name, state in loss_state.items():
            if name in loss_terms:
                loss_terms[name].load_state_dict(state, strict=False)
    else:
        print(f"Warning: loss_terms.pth not found in {checkpoint_dir}")

    model.eval()
    return model, cfg, loss_terms

@torch.no_grad()
def extract_features(model, loss_terms, loader, device, max_samples=100):
    """Extract global predicted features for the given model/loader."""
    proto_loss = loss_terms.get('proto')
    if proto_loss is None:
        print("Error: No 'proto' loss term found.")
        return None, None

    features = []
    labels = []
    
    count = 0
    for batch in tqdm(loader, desc="Extracting features"):
        imgs = batch['img'].to(device)
        cls_names = batch['cls_name']
        
        # Forward pass up to global features
        # Assuming model outputs: feats_t, feats_s, glo_feats, ...
        # Based on RDLGC_BYOL.forward
        feats_t = model.net_t(imgs)
        feats_t_detached = [f.detach() for f in feats_t]
        feats_proj = model.proj_layer(feats_t_detached)
        mid = model.mff_oce(feats_proj)
        
        # Global features
        glo_feats = F.adaptive_avg_pool2d(mid, 1).flatten(1) # (B, C)
        
        # Shift and predict
        shifted = proto_loss.shift_to_prototypes(glo_feats) # (B, N, C)
        B, N, C = shifted.shape
        predicted_global = proto_loss.global_predictor(shifted.reshape(B * N, -1)) # (BN, C)
        
        # We can use the mean predicted feature per image or all prototypes
        # Let's use the mean to have 1 point per image for clarity, 
        # or concatenate if we want to show diversity within image.
        # User asked for "feature space sau predictor của global", usually means the embeddings.
        pred_feat = predicted_global.reshape(B, N, C).mean(dim=1) # (B, C)
        
        features.append(pred_feat.cpu().numpy())
        labels.extend(cls_names)
        
        count += B
        if count >= max_samples:
            break
            
    return np.concatenate(features, axis=0), labels

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir1', type=str, required=True, help="First run directory")
    parser.add_argument('--dir2', type=str, required=True, help="Second run directory")
    parser.add_argument('--config', type=str, default='configs/rd/rd_byol_mvtec.py')
    parser.add_argument('--max_samples', type=int, default=200)
    parser.add_argument('--output', type=str, default='feature_space_comparison.png')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load Run 1
    print(f"Loading Run 1 from {args.dir1}...")
    model1, cfg1, loss1 = load_model_and_loss(args.config, args.dir1, device)
    
    # Load Run 2
    print(f"Loading Run 2 from {args.dir2}...")
    model2, cfg2, loss2 = load_model_and_loss(args.config, args.dir2, device)

    if model1 is None or model2 is None:
        return

    # Use cfg1 for data loading (assumed identical data setup)
    train_ds, test_ds = get_dataset(cfg1)
    loader = torch.utils.data.DataLoader(test_ds, batch_size=8, shuffle=True)

    # Extract features
    feat1, labels1 = extract_features(model1, loss1, loader, device, args.max_samples)
    feat2, labels2 = extract_features(model2, loss2, loader, device, args.max_samples)

    if feat1 is None or feat2 is None:
        return

    # Combine features for t-SNE
    all_features = np.concatenate([feat1, feat2], axis=0)
    
    print(f"Running t-SNE on {len(all_features)} samples...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    embeddings = tsne.fit_transform(all_features)
    
    emb1 = embeddings[:len(feat1)]
    emb2 = embeddings[len(feat1):]

    # Plot
    plt.figure(figsize=(10, 8))
    
    # Plot Run 1
    plt.scatter(emb1[:, 0], emb1[:, 1], c='blue', label=f'Run 1: {os.path.basename(args.dir1)}', alpha=0.6, edgecolors='w')
    
    # Plot Run 2
    plt.scatter(emb2[:, 0], emb2[:, 1], c='red', label=f'Run 2: {os.path.basename(args.dir2)}', alpha=0.6, edgecolors='w')
    
    plt.title("t-SNE Visualization of Global Predicted Features")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(args.output, dpi=300)
    plt.show()
    
    print(f"Visualization saved to {args.output}")

    # Optional: Plot by class
    plt.figure(figsize=(10, 8))
    unique_labels = list(set(labels1 + labels2))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    label_to_color = dict(zip(unique_labels, colors))
    
    for i, label in enumerate(unique_labels):
        # Run 1
        mask1 = [l == label for l in labels1]
        if any(mask1):
            plt.scatter(emb1[mask1, 0], emb1[mask1, 1], color=label_to_color[label], marker='o', label=f'{label} (Run 1)', alpha=0.7)
        
        # Run 2
        mask2 = [l == label for l in labels2]
        if any(mask2):
            plt.scatter(emb2[mask2, 0], emb2[mask2, 1], color=label_to_color[label], marker='x', label=f'{label} (Run 2)', alpha=0.7)
            
    plt.title("t-SNE Visualization colored by Category and Run")
    # Only show legend for categories to avoid clutter
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    # plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(args.output.replace('.png', '_by_class.png'), dpi=300)
    print(f"Visualization by class saved to {args.output.replace('.png', '_by_class.png')}")

if __name__ == "__main__":
    main()
