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


def generate_orthonormal_vectors(hw, n, dim):
    """
    Generate n orthonormal vectors of dimension dim for EACH spatial location.
    
    Args:
        hw: Number of spatial locations (H * W)
        n: Number of prototypes per location (e.g., 5)
        dim: Dimension of each prototype (e.g., 2048)
    """
    A = torch.randn(hw, dim, n)
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    # Return shape: (hw, n, dim)
    return U.transpose(1, 2)


@LOSS.register_module
class PrototypeBYOLLoss(nn.Module):
    """
    Spatial Prototype-guided BYOL loss for dense features + Global BYOL.
    
    Args:
        lam: Loss weight multiplier
        n_prototypes: Number of prototypes per spatial location (default: 10)
        feat_dim: Feature dimension (REQUIRED, e.g. 2048)
        H: Feature map height (REQUIRED to init spatial parameters)
        W: Feature map width (REQUIRED to init spatial parameters)
    """

    def __init__(self, lam=1.0, n_prototypes=10, feat_dim=2048, H=8, W=8, lam_spatial=1.0, lam_global=1.0):
        super(PrototypeBYOLLoss, self).__init__()
        self.lam = lam
        self.n_prototypes = n_prototypes
        self.feat_dim = feat_dim
        self.H = H
        self.W = W
        self.lam_spatial = lam_spatial
        self.lam_global = lam_global

        # === Prototypes (learnable, orthonormal init) ===
        # Shape: (H*W, n_prototypes, feat_dim)
        self.prototypes = nn.Parameter(
            generate_orthonormal_vectors(H * W, n_prototypes, feat_dim)
        )

        # === Linear merge: Use 1x1 Conv to handle (B, C, H, W) natively ===
        self.linear_merge = nn.Sequential(
            nn.Conv2d(feat_dim * 2, feat_dim, kernel_size=1),
            nn.InstanceNorm2d(feat_dim),
            nn.LeakyReLU(inplace=True),
        )

        # === Predictor head (online only — creates asymmetry for BYOL) ===
        self.predictor = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim, kernel_size=1),
            nn.InstanceNorm2d(feat_dim),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(feat_dim, feat_dim, kernel_size=1),
            nn.InstanceNorm2d(feat_dim),
            nn.LeakyReLU(inplace=True)
        )

        # === Global Predictor head ===
        self.global_predictor = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.InstanceNorm1d(feat_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim),
            nn.InstanceNorm1d(feat_dim),
            nn.LeakyReLU(inplace=True)
        )

    def shift_to_prototypes(self, features):
        """
        Shift coordinate origin to each prototype for global features.

        Args:
            features: (B, C) global features

        Returns:
            shifted: (B, N, C) — features shifted to each prototype's origin
        """
        # self.prototypes has shape (H*W, n_prototypes, C)
        # We mean-pool across spatial locations to get global prototypes
        global_prototypes = self.prototypes.mean(dim=0)  # (N, C)
        prototypes_norm = F.normalize(global_prototypes, dim=1, p=2)  # (N, C)
        # Shift: features (B, 1, C) - prototypes (1, N, C) = (B, N, C)
        shifted = features.unsqueeze(1) - prototypes_norm.unsqueeze(0)
        return shifted

    def query_prototypes(self, spatial_feats):
        """
        Query prototypes with spatial features and create enhanced features.

        Args:
            spatial_feats: (B, C, H, W) dense features before GAP

        Returns:
            merged: (B, C, H, W) merged features
        """
        B, C, H, W = spatial_feats.shape
        
        # Flatten spatial dims: (B, C, H, W) -> (B, H*W, C)
        features_flat = spatial_feats.view(B, C, -1).transpose(1, 2)

        # Normalize features and prototypes
        features_norm = F.normalize(features_flat, dim=-1, p=2)    # (B, H*W, C)
        prototypes_norm = F.normalize(self.prototypes, dim=-1, p=2) # (H*W, n_prototypes, C)

        # Compute cosine similarity with prototypes AT THE SAME SPATIAL LOCATION
        cosine_sim = torch.einsum('bsc,snc->bsn', features_norm, prototypes_norm) # (B, H*W, n_prototypes)

        # Soft attention: weighted combination of prototypes
        proto_features = torch.einsum('bsn,snc->bsc', cosine_sim, self.prototypes) # (B, H*W, C)

        # Reshape back to spatial grid: (B, H*W, C) -> (B, C, H, W)
        proto_features = proto_features.transpose(1, 2).view(B, C, H, W)

        # Concatenate: [original_features, prototype_features]
        concat_features = torch.cat([spatial_feats, proto_features], dim=1)  # (B, 2*C, H, W)

        # Linear merge: 2*C → C
        merged = self.linear_merge(concat_features)  # (B, C, H, W)

        return merged

    def forward(self, glo_feats, glo_feats_k, spatial_feats, spatial_feats_k, labels=None):
        """
        Compute Spatial Prototype BYOL loss + Global BYOL loss.

        Args:
            glo_feats: Online GAP features (B, C)
            glo_feats_k: Target GAP features (B, C)
            spatial_feats: Online dense features before GAP (B, C, H, W)
            spatial_feats_k: Target dense features before GAP (B, C, H, W)
            labels: Class labels (optional, not used)
        """
        # ==========================================
        # 1. SPATIAL PROTOTYPE BYOL (on spatial_feats)
        # ==========================================
        # Online path
        merged_q = self.query_prototypes(spatial_feats)
        predicted_spatial = self.predictor(merged_q)  

        # Target path
        with torch.no_grad():
            merged_k = self.query_prototypes(spatial_feats_k)

        # Loss computation
        predicted_spatial = F.normalize(predicted_spatial, dim=1, p=2)
        target_spatial = F.normalize(merged_k, dim=1, p=2)
        loss_spatial = 2 - 2 * (predicted_spatial * target_spatial).sum(dim=1).mean()


        # ==========================================
        # 2. GLOBAL PROTOTYPE BYOL (on glo_feats)
        # ==========================================
        B = glo_feats.shape[0]
        N = self.n_prototypes

        # Shift to each prototype's coordinate system
        shifted_q = self.shift_to_prototypes(glo_feats)      # (B, N, C)
        with torch.no_grad():
            shifted_k = self.shift_to_prototypes(glo_feats_k)  # (B, N, C)

        # Online predictor
        predicted_global = self.global_predictor(shifted_q.reshape(B * N, -1))  # (B*N, C)
        predicted_global = predicted_global.reshape(B, N, -1)                   # (B, N, C)

        # BYOL loss per prototype
        predicted_global = F.normalize(predicted_global, dim=2, p=2)
        target_global = F.normalize(shifted_k, dim=2, p=2)
        
        per_proto_loss = 2 - 2 * (predicted_global * target_global).sum(dim=2)  # (B, N)
        loss_global = per_proto_loss.mean()

        # ==========================================
        # 3. TOTAL LOSS
        # ==========================================
        total_loss = self.lam_spatial * loss_spatial + self.lam_global * loss_global

        return total_loss * self.lam