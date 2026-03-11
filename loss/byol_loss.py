"""
BYOL-style Dense Contrastive Loss for Multi-Class Anomaly Detection

Key features:
1. NO negative samples needed
2. NO diversity loss needed  
3. Predictor + momentum naturally prevents collapse
4. Dense spatial correspondence matching

Reference: BYOL (Bootstrap Your Own Latent)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
# Use relative import to avoid circular dependency
from . import LOSS


@LOSS.register_module
class BYOLDenseLoss(nn.Module):
    """
    BYOL-style dense contrastive loss.
    
    Loss = 2 - 2 * cosine_similarity(online_output, stop_gradient(target_output))
    
    No negative samples needed because:
    1. Predictor creates asymmetry
    2. Target is a moving target (momentum update)
    3. Online must predict target, cannot just copy
    
    Args:
        lam: Loss weight multiplier
        use_spatial_matching: Whether to use spatial correspondence matching
    """
    
    def __init__(self, lam=1.0, use_spatial_matching=False):
        super(BYOLDenseLoss, self).__init__()
        self.lam = lam
        self.use_spatial_matching = use_spatial_matching
    
    def byol_loss(self, online, target):
        """
        Basic BYOL loss: negative cosine similarity
        
        Args:
            online: Online network output (with predictor) - gradients flow
            target: Target network output (no predictor) - no gradients
        
        Returns:
            loss: 2 - 2 * cos_sim(online, target)
        """
        online = F.normalize(online, dim=1, p=2)
        target = F.normalize(target, dim=1, p=2)
        
        # Mean over spatial dimensions and batch
        loss = 2 - 2 * (online * target).sum(dim=1).mean()
        return loss
    
    def byol_dense_loss(self, q_grid, k_grid, q_b, k_b):
        """
        Dense BYOL loss with spatial correspondence.
        
        1. Find spatial correspondence using backbone features (q_b, k_b)
        2. Match online features (q_grid) with corresponding target features (k_grid)
        3. Compute BYOL loss on matched pairs
        
        Args:
            q_grid: Online output features (B, C, H, W) - after predictor
            k_grid: Target output features (B, C, H, W) - no predictor, detached
            q_b: Online backbone features for matching (B, C, H, W)
            k_b: Target backbone features for matching (B, C, H, W)
        
        Returns:
            loss: Dense BYOL loss value
        """
        # Normalize all features
        q_grid = F.normalize(q_grid, p=2, dim=1)
        k_grid = F.normalize(k_grid, p=2, dim=1)  # Already detached from momentum encoder
        q_b = F.normalize(q_b, p=2, dim=1)
        k_b = F.normalize(k_b, p=2, dim=1)
        
        B, C, H, W = q_grid.shape
        
        if self.use_spatial_matching:
            # === Find spatial correspondence using backbone features ===
            q_b_flat = q_b.view(B, q_b.size(1), -1)  # (B, C_b, H*W)
            k_b_flat = k_b.view(B, k_b.size(1), -1)  # (B, C_b, H*W)
            
            # Compute similarity matrix between all positions
            sim_matrix = torch.einsum('bci,bcj->bij', q_b_flat, k_b_flat)  # (B, H*W, H*W)
            
            # Find best matching position in target for each online position
            max_sim_idx = torch.argmax(sim_matrix, dim=-1)  # (B, H*W)
            
            # === Gather matched target features ===
            q_grid_flat = q_grid.view(B, C, -1)  # (B, C, H*W)
            k_grid_flat = k_grid.view(B, C, -1)  # (B, C, H*W)
            
            # Expand indices for gathering
            indices = max_sim_idx.unsqueeze(1).expand(-1, C, -1)  # (B, C, H*W)
            k_matched = k_grid_flat.gather(2, indices)  # (B, C, H*W)
            
            # === BYOL loss on matched pairs ===
            # Loss = 2 - 2 * cos_sim
            loss = 2 - 2 * (q_grid_flat * k_matched).sum(dim=1).mean()
        else:
            # Simple BYOL loss without spatial matching (same position)
            loss = 2 - 2 * (q_grid * k_grid).sum(dim=1).mean()
        
        return loss

    def forward(self, q_b, k_b, q_grid, k_grid, labels=None):
        """
        Forward pass.
        
        Args:
            q_b: Online backbone features (list of tensors or single tensor)
            k_b: Target backbone features (list of tensors or single tensor)
            q_grid: Online output after predictor (list of tensors or single tensor)
            k_grid: Target output, no predictor (list of tensors or single tensor)
            labels: Class labels (optional, for compatibility but not used in BYOL)
        
        Returns:
            loss: Weighted BYOL dense loss
        """
        # Handle both list and single tensor inputs
        if not isinstance(q_grid, list):
            q_grid = [q_grid]
            k_grid = [k_grid]
            q_b = [q_b]
            k_b = [k_b]
        
        total_loss = 0.0
        
        for qg, kg, qb, kb in zip(q_grid, k_grid, q_b, k_b):
            loss = self.byol_dense_loss(qg, kg, qb, kb)
            total_loss += loss
        
        return total_loss / len(q_grid) * self.lam


@LOSS.register_module
class ClassAwareBYOLDenseLoss(nn.Module):
    """
    Class-aware BYOL dense loss.
    
    Only computes BYOL loss within the same class, ensuring that
    features from the same class are pulled together while maintaining
    class separation through the global SCL loss.
    
    Args:
        lam: Loss weight multiplier
        use_spatial_matching: Whether to use spatial correspondence matching
    """
    
    def __init__(self, lam=1.0, use_spatial_matching=False):
        super(ClassAwareBYOLDenseLoss, self).__init__()
        self.lam = lam
        self.use_spatial_matching = use_spatial_matching
    
    def class_aware_byol_loss(self, q_grid, k_grid, q_b, k_b, labels):
        """
        Compute BYOL loss only within samples of the same class.
        """
        q_grid = F.normalize(q_grid, p=2, dim=1)
        k_grid = F.normalize(k_grid, p=2, dim=1)
        q_b = F.normalize(q_b, p=2, dim=1)
        k_b = F.normalize(k_b, p=2, dim=1)
        
        unique_labels = torch.unique(labels)
        total_loss = 0.0
        num_valid_classes = 0
        
        for label in unique_labels:
            mask = labels == label
            if mask.sum() < 1:
                continue
            
            # Get features for this class
            q_cls = q_grid[mask]
            k_cls = k_grid[mask]
            qb_cls = q_b[mask]
            kb_cls = k_b[mask]
            
            B_cls, C, H, W = q_cls.shape
            
            if self.use_spatial_matching:
                # Find spatial correspondence within class
                qb_flat = qb_cls.view(B_cls, qb_cls.size(1), -1)
                kb_flat = kb_cls.view(B_cls, kb_cls.size(1), -1)
                
                sim = torch.einsum('bci,bcj->bij', qb_flat, kb_flat)
                max_idx = torch.argmax(sim, dim=-1)
                
                q_flat = q_cls.view(B_cls, C, -1)
                k_flat = k_cls.view(B_cls, C, -1)
                
                indices = max_idx.unsqueeze(1).expand(-1, C, -1)
                k_matched = k_flat.gather(2, indices)
                
                loss = 2 - 2 * (q_flat * k_matched).sum(dim=1).mean()
            else:
                loss = 2 - 2 * (q_cls * k_cls).sum(dim=1).mean()
            
            total_loss += loss
            num_valid_classes += 1
        
        if num_valid_classes == 0:
            return torch.tensor(0.0, device=q_grid.device)
        
        return total_loss / num_valid_classes

    def forward(self, q_b, k_b, q_grid, k_grid, labels):
        """
        Forward pass with class-aware BYOL loss.
        """
        if not isinstance(q_grid, list):
            q_grid = [q_grid]
            k_grid = [k_grid]
            q_b = [q_b]
            k_b = [k_b]
        
        total_loss = 0.0
        
        for qg, kg, qb, kb in zip(q_grid, k_grid, q_b, k_b):
            loss = self.class_aware_byol_loss(qg, kg, qb, kb, labels)
            total_loss += loss
        
        return total_loss / len(q_grid) * self.lam


def generate_orthonormal_vectors(n, dim):
    """
    Generate n orthonormal vectors of dimension dim using SVD.
    Same as PAPN implementation.

    Args:
        n: Number of prototypes (e.g., 5)
        dim: Dimension of each prototype (e.g., 2048)

    Returns:
        Tensor of shape (n, dim) with orthonormal rows
    """
    A = torch.randn(dim, n)
    U, S, Vt = torch.svd(A)
    return U[:, :n].T  # (n, dim)


@LOSS.register_module
class PrototypeBYOLLoss(nn.Module):
    """
    Prototype-guided BYOL loss for global features.

    For each prototype p_i, shift origin to p_i:
        shifted_i = feature - p_i

    Then compute BYOL loss per prototype and average:
        loss = mean_over_prototypes(2 - 2 * cos_sim(predictor(shifted_online_i), shifted_target_i))

    Args:
        lam: Loss weight multiplier
        n_prototypes: Number of prototypes (default: 10)
        feat_dim: Feature dimension (REQUIRED, e.g. 2048 for mff_oce output)
    """

    def __init__(self, lam=1.0, n_prototypes=10, feat_dim=2048):
        super(PrototypeBYOLLoss, self).__init__()
        self.lam = lam
        self.n_prototypes = n_prototypes
        self.feat_dim = feat_dim

        # === Prototypes (learnable, orthonormal init) ===
        self.prototypes = nn.Parameter(
            generate_orthonormal_vectors(n_prototypes, feat_dim)
        )

        # === Predictor head (online only — creates asymmetry for BYOL) ===
        # Operates per-prototype: input (B*N, C) → output (B*N, C)
        self.predictor = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.InstanceNorm1d(feat_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim),
            nn.InstanceNorm1d(feat_dim),
            nn.LeakyReLU(inplace=True),
        )

    def shift_to_prototypes(self, features):
        """
        Shift coordinate origin to each prototype.

        Args:
            features: (B, C) global features

        Returns:
            shifted: (B, N, C) — features shifted to each prototype's coordinate system
                     shifted[:, i, :] = features - prototype_i
        """
        prototypes_norm = F.normalize(self.prototypes, dim=1, p=2)  # (N, C)

        # features: (B, C) → (B, 1, C)
        # prototypes: (N, C) → (1, N, C)
        # shifted: (B, N, C) — feature_b shifted to prototype_n's origin
        shifted = features.unsqueeze(1) - prototypes_norm.unsqueeze(0)  # (B, N, C)

        return shifted

    def forward(self, glo_feats, glo_feats_k, labels=None):
        """
        Compute per-prototype BYOL loss.

        For each prototype i:
            online_shifted_i = glo_feats - prototype_i
            target_shifted_i = glo_feats_k - prototype_i
            loss_i = 2 - 2 * cos_sim(predictor(online_shifted_i), target_shifted_i)
        Total loss = mean(loss_i)
        """
        B = glo_feats.shape[0]
        N = self.n_prototypes

        # === Shift to each prototype's coordinate system ===
        shifted_q = self.shift_to_prototypes(glo_feats)      # (B, N, C)
        with torch.no_grad():
            shifted_k = self.shift_to_prototypes(glo_feats_k)  # (B, N, C)

        # === Online path: predictor (reshape to (B*N, C) for Linear layers) ===
        predicted = self.predictor(shifted_q.reshape(B * N, -1))  # (B*N, C)
        predicted = predicted.reshape(B, N, -1)                    # (B, N, C)

        # === BYOL loss per prototype, then average ===
        predicted = F.normalize(predicted, dim=2, p=2)   # normalize along C
        target = F.normalize(shifted_k, dim=2, p=2)      # normalize along C

        # cos_sim per (batch, prototype) → mean over both
        per_proto_loss = 2 - 2 * (predicted * target).sum(dim=2)  # (B, N)
        loss = per_proto_loss.mean()  # average over batch and prototypes

        return loss * self.lam