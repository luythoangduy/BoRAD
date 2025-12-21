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
    
    def __init__(self, lam=1.0, use_spatial_matching=True):
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
    
    def __init__(self, lam=1.0, use_spatial_matching=True):
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
class PrototypeInfoNCELoss(nn.Module):
    """
    Prototype-based InfoNCE loss for global features.

    Flow:
    1. Features query prototypes → cosine similarity
    2. Multiply cosine sim with features → concat with original features
    3. Pass through projector → compute InfoNCE loss

    Prototypes are initialized as orthogonal vectors.

    Args:
        lam: Loss weight multiplier
        n_prototypes: Number of prototypes (default: 5)
        feat_dim: Feature dimension (will be auto-detected from input)
        temperature: Temperature for InfoNCE loss
    """

    def __init__(self, lam=1.0, n_prototypes=5, feat_dim=None, temperature=0.07):
        super(PrototypeInfoNCELoss, self).__init__()
        self.lam = lam
        self.n_prototypes = n_prototypes
        self.temperature = temperature
        self.feat_dim = feat_dim

        # Prototypes will be initialized on first forward pass
        self.prototypes = None
        self.projector = None

    def _initialize_prototypes(self, feat_dim, device):
        """Initialize prototypes as orthonormal vectors"""
        if self.prototypes is None:
            self.prototypes = nn.Parameter(
                generate_orthonormal_vectors(self.n_prototypes, feat_dim).to(device)
            )
            self.feat_dim = feat_dim

    def _initialize_projector(self, input_dim, device):
        """
        Initialize projector for proto-enhanced features.
        Input: concat(features, cosine_sim * features) has dim = feat_dim + feat_dim = 2 * feat_dim
        Output: same as feat_dim for consistency
        """
        if self.projector is None:
            self.projector = nn.Sequential(
                nn.Linear(input_dim, input_dim // 2),
                nn.BatchNorm1d(input_dim // 2),
                nn.ReLU(inplace=True),
                nn.Linear(input_dim // 2, self.feat_dim),
            ).to(device)

    def query_prototypes(self, features):
        """
        Query prototypes with features and create enhanced features.

        Args:
            features: (B, C) global features

        Returns:
            enhanced_features: (B, 2*C) concatenated features
        """
        # Normalize features and prototypes
        features_norm = F.normalize(features, dim=1, p=2)  # (B, C)
        prototypes_norm = F.normalize(self.prototypes, dim=1, p=2)  # (n_proto, C)

        # Compute cosine similarity with all prototypes
        cosine_sim = torch.matmul(features_norm, prototypes_norm.T)  # (B, n_proto)

        # Average cosine similarities across all prototypes to get a scalar weight per sample
        # Then expand to match feature dimension
        avg_cosine_sim = cosine_sim.mean(dim=1, keepdim=True)  # (B, 1)

        # Multiply original features with average cosine similarity
        weighted_features = features * avg_cosine_sim  # (B, C)

        # Concatenate: [original_features, weighted_features]
        enhanced_features = torch.cat([features, weighted_features], dim=1)  # (B, 2*C)

        return enhanced_features

    def info_nce_loss(self, q, k):
        """
        InfoNCE loss for prototype-enhanced features.

        Args:
            q: Online features after projector (B, C)
            k: Target features after projector (B, C) - detached

        Returns:
            loss: InfoNCE loss value
        """
        # Normalize
        q = F.normalize(q, dim=1, p=2)
        k = F.normalize(k, dim=1, p=2)

        # Positive pairs: each sample with itself
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)  # (B, 1)

        # Negative pairs: each sample with all other samples in the batch
        l_neg = torch.einsum('nc,mc->nm', [q, k])  # (B, B)

        # Concatenate positive and negative logits
        logits = torch.cat([l_pos, l_neg], dim=1)  # (B, 1+B)
        logits /= self.temperature

        # Labels: positive pair is at index 0
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=q.device)

        # Cross entropy loss
        loss = F.cross_entropy(logits, labels)

        return loss

    def forward(self, glo_feats, glo_feats_k, labels=None):
        """
        Compute Prototype InfoNCE loss.

        Args:
            glo_feats: Online global features (B, C)
            glo_feats_k: Target global features (B, C) - detached
            labels: Class labels (optional, not used)

        Returns:
            loss: InfoNCE loss with prototypes
        """
        B, C = glo_feats.shape
        device = glo_feats.device

        # Initialize prototypes and projector on first forward pass
        if self.prototypes is None:
            self._initialize_prototypes(C, device)
        if self.projector is None:
            self._initialize_projector(C * 2, device)

        # Query prototypes for online features
        enhanced_q = self.query_prototypes(glo_feats)

        # Query prototypes for target features (detached)
        with torch.no_grad():
            enhanced_k = self.query_prototypes(glo_feats_k)

        # Pass through projector
        proj_q = self.projector(enhanced_q)

        with torch.no_grad():
            proj_k = self.projector(enhanced_k)

        # Compute InfoNCE loss
        loss = self.info_nce_loss(proj_q, proj_k)

        return loss * self.lam