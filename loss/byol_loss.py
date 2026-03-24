"""
Prototype Orthogonal Residual Learning (PORL) for Multi-Class Anomaly Detection

Key idea: Instead of shift (f - p_i) which cancels out in MSE,
project OUT the prototype component and compute MSE on the RESIDUAL:
  r_i(f) = f - (f · p_i) * p_i

This creates anisotropic penalty:
- Tolerant along prototype directions (reduces mis-reconstruction)
- Strict perpendicular to prototypes (prevents identical shortcut)

Reference: BYOL (Bootstrap Your Own Latent) + Orthogonal Projection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
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
    Prototype Orthogonal Residual Learning (PORL) loss.
    
    Replaces shift prototype (subtraction, cancels in MSE) with orthogonal
    projection (removes prototype component, MSE on residual).
    
    Args:
        lam: Loss weight multiplier
        n_prototypes: Number of prototypes per spatial location (default: 7)
        feat_dim: Feature dimension (REQUIRED, e.g. 2048)
        H: Feature map height
        W: Feature map width
        lam_spatial: Weight for PORL spatial loss
    """

    def __init__(self, lam=1.0, n_prototypes=7, feat_dim=2048, H=8, W=8, 
                 lam_spatial=1.0, lam_global=1.0, lam_align=0.1):
        super(PrototypeBYOLLoss, self).__init__()
        self.lam = lam
        self.n_prototypes = n_prototypes
        self.feat_dim = feat_dim
        self.H = H
        self.W = W
        self.lam_spatial = lam_spatial
        # lam_global and lam_align kept for config compatibility but unused

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
        )

    def project_out_prototypes(self, spatial_feats):
        """
        PORL: Project out prototype component, keep orthogonal residual.
        
        For each spatial location s and prototype p_{s,i}:
            r_{s,i}(f) = f_s - (f_s · p_{s,i}) * p_{s,i}
        
        Args:
            spatial_feats: (B, C, H, W) dense features
            
        Returns:
            residuals: (B, H*W, N, C) orthogonal residuals per prototype
            proj_coeffs: (B, H*W, N) projection coefficients (for diagnostics)
        """
        B, C, H, W = spatial_feats.shape
        
        # Flatten spatial: (B, C, H*W) → (B, H*W, C)
        features_flat = spatial_feats.view(B, C, -1).transpose(1, 2)
        
        # Normalize features and prototypes
        features_norm = F.normalize(features_flat, dim=-1, p=2)     # (B, H*W, C)
        protos_norm = F.normalize(self.prototypes, dim=-1, p=2)     # (H*W, N, C)
        
        # Projection coefficients: (f · p_i) for each spatial location and prototype
        # (B, H*W, C) x (H*W, N, C) → (B, H*W, N)
        proj_coeffs = torch.einsum('bsc,snc->bsn', features_norm, protos_norm)
        
        # Residual: f - (f·p)*p
        # features_norm: (B, H*W, 1, C)
        # proj_coeffs:   (B, H*W, N, 1)  
        # protos_norm:   (1, H*W, N, C)
        residuals = features_norm.unsqueeze(2) - \
                    proj_coeffs.unsqueeze(-1) * protos_norm.unsqueeze(0)
        # Shape: (B, H*W, N, C)
        
        return residuals, proj_coeffs

    @torch.no_grad()
    def compute_diagnostics(self, per_proto_loss, residual_q, residual_k, 
                            proj_coeffs_q, features_norm_q):
        """
        Compute PORL-specific diagnostics for monitoring.
        
        Metrics:
        1. Prototype pairwise cosine similarity (diversity)
        2. PORL vs raw MSE ratio (prototype effectiveness)
        3. Variance parallel vs perpendicular (channeling effect)
        4. Per-prototype loss distribution
        """
        diagnostics = {}

        # === 1. Pairwise cosine similarity between prototypes ===
        # Mean-pool across spatial locations for global view
        global_prototypes = self.prototypes.mean(dim=0)  # (N, C)
        proto_norm = F.normalize(global_prototypes, dim=1, p=2)
        cos_matrix = torch.mm(proto_norm, proto_norm.t())  # (N, N)
        
        N = cos_matrix.shape[0]
        mask = ~torch.eye(N, dtype=torch.bool, device=cos_matrix.device)
        off_diag = cos_matrix[mask]
        
        diagnostics['cosine_sim_mean'] = off_diag.mean().item()
        diagnostics['cosine_sim_max'] = off_diag.max().item()
        diagnostics['cosine_sim_min'] = off_diag.min().item()

        # === 2. PORL vs Raw MSE (verify prototype is working) ===
        # Raw MSE: ||f_q - f_k||² (without projection)
        # residual_q, residual_k: (B, H*W, N, C)
        # We need original features — reconstruct from residuals + projections
        # d_raw = f_q - f_k, d_porl = r_i(f_q) - r_i(f_k) = d - (d·p_i)*p_i
        # So ||d_porl||² = ||d||² - (d·p_i)²
        # We can get this from per_proto_loss and the full MSE
        
        # features_norm_q has shape info we need
        # per_proto_loss: (B, N) — mean over spatial and channels
        loss_porl = per_proto_loss.mean().item()
        
        # Raw MSE between residuals (includes projected-out component)
        # ||d||² per sample: reconstruct from residual loss + projection diff
        d_residual = residual_q - residual_k  # (B, H*W, N, C)
        d_residual_sq = (d_residual ** 2).sum(dim=-1).mean(dim=1)  # (B, N)
        
        # Approximate raw MSE from the view of each prototype:
        # ||d||² = ||d_perp||² + (d·p_i)²
        protos_norm = F.normalize(self.prototypes, dim=-1, p=2)  # (H*W, N, C)
        
        diagnostics['loss_porl'] = loss_porl
        diagnostics['per_proto_loss_std'] = per_proto_loss.mean(dim=0).std().item()
        diagnostics['per_proto_loss_min'] = per_proto_loss.mean(dim=0).min().item()
        diagnostics['per_proto_loss_max'] = per_proto_loss.mean(dim=0).max().item()

        # === 3. Variance channeling: parallel vs perpendicular ===
        # proj_coeffs_q: (B, H*W, N) — how much feature lies along each prototype
        # Var of proj_coeff = variance along prototype direction
        var_parallel = proj_coeffs_q.var(dim=0).mean().item()  # avg over spatial, prototypes
        
        # Variance of residual norm = variance perpendicular to prototypes
        residual_norms = residual_q.norm(dim=-1)  # (B, H*W, N)
        var_perpendicular = residual_norms.var(dim=0).mean().item()
        
        diagnostics['var_parallel'] = var_parallel
        diagnostics['var_perpendicular'] = var_perpendicular
        diagnostics['var_ratio'] = var_parallel / max(var_perpendicular, 1e-8)
        
        # === 4. Mean projection magnitude (how aligned features are with prototypes) ===
        diagnostics['proj_coeff_mean'] = proj_coeffs_q.abs().mean().item()
        diagnostics['proj_coeff_std'] = proj_coeffs_q.std().item()

        return diagnostics

    def forward(self, glo_feats, glo_feats_k, spatial_feats, spatial_feats_k, labels=None):
        """
        Compute PORL loss on spatial features.
        
        Architecture:
            Online: spatial_feats → predictor → project_out → residuals
            Target: spatial_feats_k → project_out → residuals (no predictor)
            Loss: MSE(online_residuals, target_residuals)

        Args:
            glo_feats: Online GAP features (B, C) — unused, kept for interface compat
            glo_feats_k: Target GAP features (B, C) — unused
            spatial_feats: Online dense features (B, C, H, W)
            spatial_feats_k: Target dense features (B, C, H, W)
            labels: Class labels (optional, not used)

        Returns:
            total_loss: scalar loss
            diagnostics: dict with PORL monitoring metrics
        """
        # ==========================================
        # 1. ONLINE PATH: predictor → PORL
        # ==========================================
        predicted = self.predictor(spatial_feats)  # (B, C, H, W) — asymmetry
        residual_q, proj_coeffs_q = self.project_out_prototypes(predicted)  # (B, H*W, N, C)

        # ==========================================
        # 2. TARGET PATH: PORL only (no predictor)
        # ==========================================
        with torch.no_grad():
            residual_k, proj_coeffs_k = self.project_out_prototypes(spatial_feats_k)

        # ==========================================
        # 3. PORL LOSS: MSE on orthogonal residuals
        # ==========================================
        # Per-prototype loss: MSE averaged over spatial and channels
        # residual shape: (B, H*W, N, C)
        per_proto_loss = F.mse_loss(
            residual_q, residual_k, reduction='none'
        ).mean(dim=(1, 3))  # (B, N) — per sample, per prototype
        
        loss = per_proto_loss.mean()

        # ==========================================
        # 4. DIAGNOSTICS (no grad)
        # ==========================================
        with torch.no_grad():
            # Get normalized features for diagnostics
            B, C, H, W = predicted.shape
            features_norm_q = F.normalize(
                predicted.view(B, C, -1).transpose(1, 2), dim=-1, p=2
            )
            
            diagnostics = self.compute_diagnostics(
                per_proto_loss.detach(),
                residual_q.detach(),
                residual_k.detach(),
                proj_coeffs_q.detach(),
                features_norm_q.detach()
            )

        return loss * self.lam * self.lam_spatial, diagnostics