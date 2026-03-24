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

    def __init__(self, lam=1.0, n_prototypes=7, feat_dim=2048, H=8, W=8, lam_spatial=1.0, lam_global=1.0, lam_align=0.1):
        super(PrototypeBYOLLoss, self).__init__()
        self.lam = lam
        self.n_prototypes = n_prototypes
        self.feat_dim = feat_dim
        self.H = H
        self.W = W
        self.lam_spatial = lam_spatial
        self.lam_global = lam_global
        self.lam_align = lam_align

        # === Prototypes (learnable, orthonormal init) ===
        # Shape: (H*W, n_prototypes, feat_dim)
        self.prototypes = nn.Parameter(
            generate_orthonormal_vectors(H * W, n_prototypes, feat_dim)
        )


        # === Predictor head (online only — creates asymmetry for BYOL) ===
        self.predictor = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim, kernel_size=1),
            nn.InstanceNorm2d(feat_dim),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(feat_dim, feat_dim, kernel_size=1),
            nn.InstanceNorm2d(feat_dim),
        )

        # === Global Predictor head ===
        self.global_predictor = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.InstanceNorm1d(feat_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim),
            nn.InstanceNorm1d(feat_dim),
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
        
        # Normalize features to unit sphere before shift
        features_norm = F.normalize(features, dim=1, p=2)  # (B, C)
        
        # Shift: features (B, 1, C) - prototypes (1, N, C) = (B, N, C)
        shifted = features_norm.unsqueeze(1) - prototypes_norm.unsqueeze(0)
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

        return proto_features

    @torch.no_grad()
    def compute_diagnostics(self, per_proto_loss, glo_feats, glo_feats_k):
        """
        Compute diagnostic metrics for prototype monitoring.

        Hyperplane geometry:
        - Normalize f_q, f_k, prototypes → all on unit sphere
        - f_q and f_k span a 2D plane through origin
        - Hyperplane H passes through f_q and f_k, perpendicular to this 2D plane
        - H's normal lies IN the 2D plane, perpendicular to (f_k - f_q)
        - Check if O (origin) and each prototype are same/different side of H

        Args:
            per_proto_loss: (B, N) per-prototype loss values
            glo_feats: (B, C) online global features
            glo_feats_k: (B, C) target global features
        """
        diagnostics = {}

        # === 1. Pairwise cosine similarity between global prototypes ===
        global_prototypes = self.prototypes.mean(dim=0)  # (N, C)
        proto_norm = F.normalize(global_prototypes, dim=1, p=2)  # (N, C)
        cos_matrix = torch.mm(proto_norm, proto_norm.t())  # (N, N)

        N = cos_matrix.shape[0]
        mask = ~torch.eye(N, dtype=torch.bool, device=cos_matrix.device)
        off_diag = cos_matrix[mask]

        diagnostics['cosine_sim_mean'] = off_diag.mean().item()
        diagnostics['cosine_sim_max'] = off_diag.max().item()
        diagnostics['cosine_sim_min'] = off_diag.min().item()

        # === 2. Hyperplane case analysis ===
        # All on unit sphere
        B, C = glo_feats.shape
        f_q = F.normalize(glo_feats, dim=1, p=2)    # (B, C)
        f_k = F.normalize(glo_feats_k, dim=1, p=2)  # (B, C)
        # proto_norm: (N, C) already normalized

        # Direction along positive pair (within 2D plane)
        d = f_k - f_q  # (B, C)
        d_dot_d = (d * d).sum(dim=-1, keepdim=True).clamp(min=1e-10)  # (B, 1)

        # For O = (0,...,0):  v_O = O - f_q = -f_q
        v_O = -f_q  # (B, C)
        proj_coeff_O = (v_O * d).sum(dim=-1, keepdim=True) / d_dot_d  # (B, 1)
        perp_O = v_O - proj_coeff_O * d  # (B, C) — component ⊥ d, lies in 2D plane

        # For each prototype p_i:  v_p = p_i - f_q
        # (B, 1, C) - (1, N, C) → (B, N, C)
        v_p = proto_norm.unsqueeze(0) - f_q.unsqueeze(1)  # (B, N, C)
        d_exp = d.unsqueeze(1)  # (B, 1, C)
        d_dot_d_exp = d_dot_d.unsqueeze(1)  # (B, 1, 1)
        proj_coeff_p = (v_p * d_exp).sum(dim=-1, keepdim=True) / d_dot_d_exp  # (B, N, 1)
        perp_p = v_p - proj_coeff_p * d_exp  # (B, N, C) — component ⊥ d

        # Same/different side: sign of dot(perp_O, perp_p)
        # perp_O: (B, C) → (B, 1, C)
        dot_sign = (perp_O.unsqueeze(1) * perp_p).sum(dim=-1)  # (B, N)

        is_same = dot_sign >= 0  # O and p_i same side of H
        is_diff = dot_sign < 0   # O and p_i different side of H

        n_same = is_same.sum().item()
        n_diff = is_diff.sum().item()
        total = n_same + n_diff

        diagnostics['case_same_ratio'] = n_same / max(total, 1)
        diagnostics['case_opposite_ratio'] = n_diff / max(total, 1)
        diagnostics['loss_case_same'] = per_proto_loss[is_same].mean().item() if n_same > 0 else 0.0
        diagnostics['loss_case_opposite'] = per_proto_loss[is_diff].mean().item() if n_diff > 0 else 0.0
        diagnostics['n_pairs_same'] = n_same
        diagnostics['n_pairs_opposite'] = n_diff
        
        # Add raw dot_sign stats for debugging why is_diff is 0
        diagnostics['dot_sign_max'] = dot_sign.max().item()
        diagnostics['dot_sign_min'] = dot_sign.min().item()
        diagnostics['dot_sign_mean'] = dot_sign.mean().item()

        return diagnostics

    def forward(self, glo_feats, glo_feats_k, spatial_feats, spatial_feats_k, labels=None):
        """
        Compute Spatial Prototype BYOL loss + Global BYOL loss.

        Args:
            glo_feats: Online GAP features (B, C)
            glo_feats_k: Target GAP features (B, C)
            spatial_feats: Online dense features before GAP (B, C, H, W)
            spatial_feats_k: Target dense features before GAP (B, C, H, W)
            labels: Class labels (optional, not used)

        Returns:
            total_loss: scalar loss
            diagnostics: dict with prototype monitoring metrics (detached, no grad)
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

        # loss_spatial = F.mse_loss(predicted_spatial, merged_k, reduction='mean')


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

        per_proto_loss = F.mse_loss(predicted_global, shifted_k, reduction='none').mean(dim=2)  # (B, N)


        # ==========================================
        # 3. PROTOTYPE ALIGNMENT LOSS (Pull prototypes to features)
        # ==========================================
        # glo_feats: (B, C), global_prototypes: (N, C)
        glo_feats_norm = F.normalize(glo_feats, dim=1, p=2)
        global_prototypes = self.prototypes.mean(dim=0)  # (N, C)
        prototypes_norm = F.normalize(global_prototypes, dim=1, p=2)
        
        # Max cosine similarity for each sample with its nearest prototype
        sims = torch.mm(glo_feats_norm, prototypes_norm.t())  # (B, N)
        max_sims = sims.max(dim=1).values
        loss_align = 1.0 - max_sims.mean()

        # ==========================================
        # 4. DIAGNOSTICS (no grad, no overhead on backward)
        # ==========================================
        diagnostics = self.compute_diagnostics(
            per_proto_loss.detach(), glo_feats.detach(), glo_feats_k.detach()
        )
        diagnostics['max_cos_sim_mean'] = max_sims.detach().mean().item()
        diagnostics['loss_align'] = loss_align.detach().item()

        # ==========================================
        # 5. FINAL COMBINATION
        # ==========================================
        total_loss = self.lam_spatial * loss_spatial + \
                     self.lam_global * loss_global 
        
        return total_loss * self.lam, diagnostics