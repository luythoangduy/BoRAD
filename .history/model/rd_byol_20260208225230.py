"""
RDLGC with BYOL-style architecture for Multi-Class Anomaly Detection

Key changes from original:
1. Added Predictor layer (creates asymmetry - KEY to prevent collapse)
2. Online path: encoder → projector → predictor
3. Target path: encoder → projector (NO predictor)
4. No negative samples needed - momentum + asymmetry prevents collapse

Reference: BYOL (Bootstrap Your Own Latent)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
# Use relative import to avoid circular dependency
from . import get_model, MODEL
from functools import partial


# ============================================================================
# POSITIONAL NORMALIZATION LAYERS
# ============================================================================

class PositionalNorm2d(nn.Module):
    """
    Standard Positional Normalization (PONO).
    Formula: y = (x - mean) / std
    Output centered at 0, unit variance.
    """
    def __init__(self, num_features, affine=False, eps=1e-5):
        super(PositionalNorm2d, self).__init__()
        # LayerNorm applied on the last dimension (Channel)
        self.norm = nn.LayerNorm(num_features, elementwise_affine=affine, eps=eps)

    def forward(self, x):
        # x: B, C, H, W -> B, H, W, C
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        # -> B, C, H, W
        x = x.permute(0, 3, 1, 2)
        return x


class PositionalScaleNorm2d(nn.Module):
    """
    Positional Normalization WITHOUT Mean Subtraction.
    Formula: y = x / std
    
    * Note: std is still calculated relative to the mean.
    * Preserves the original offset (brightness) of the feature vector.
    """
    def __init__(self, num_features, affine=False, eps=1e-5):
        super(PositionalScaleNorm2d, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps
        
        if affine:
            # Learnable scale (gamma) and bias (beta)
            self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1))
            self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        # x: B, C, H, W
        
        # 1. Tính Std (Standard Deviation) dọc theo chiều Channel (dim=1)
        # unbiased=False để giống mặc định của LayerNorm/BatchNorm
        std = torch.std(x, dim=1, unbiased=False, keepdim=True)
        
        # 2. Chỉ chia cho Std (Scaling), KHÔNG trừ Mean
        x_norm = x / (std + self.eps)

        # 3. Affine transformation (nếu có)
        if self.affine:
            x_norm = x_norm * self.weight + self.bias
            
        return x_norm

# ============================================================================
# Predictor Layers (NEW - KEY component for BYOL)
# ============================================================================

class PredictorLayer(nn.Module):
    """
    BYOL-style predictor: creates asymmetry between online and target networks.
    This is the KEY component that prevents collapse without negative samples.
    
    Architecture: Conv1x1 → BN → ReLU → Conv1x1
    """
    def __init__(self, in_c, hidden_c=None, out_c=None):
        super(PredictorLayer, self).__init__()
        hidden_c = hidden_c or in_c // 2
        out_c = out_c or in_c
        
        self.predictor = nn.Sequential(
            nn.Conv2d(in_c, hidden_c, kernel_size=1, bias=False),
            PositionalNorm2d(hidden_c, affine=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_c, out_c, kernel_size=1, bias=False),
        )
    
    def forward(self, x):
        return self.predictor(x)


class MultiPredictorLayer(nn.Module):
    """
    Multi-scale predictor for all feature levels.
    Matches the structure of MultiProjectionLayer.
    """
    def __init__(self, base=64):
        super(MultiPredictorLayer, self).__init__()
        # Match the output channels of MultiProjectionLayer
        self.pred_a = PredictorLayer(base * 4, hidden_c=base * 2, out_c=base * 4)
        self.pred_b = PredictorLayer(base * 8, hidden_c=base * 4, out_c=base * 8)
        self.pred_c = PredictorLayer(base * 16, hidden_c=base * 8, out_c=base * 16)
    
    def forward(self, features):
        """
        Args:
            features: list of 3 feature maps from projector
        Returns:
            list of 3 predicted feature maps
        """
        return [
            self.pred_a(features[0]),
            self.pred_b(features[1]),
            self.pred_c(features[2])
        ]


# ============================================================================
# Existing layers (copied from rd.py for completeness)
# ============================================================================

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = PositionalNorm2d
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, stride=1, bias=False)
        self.bn1 = norm_layer(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False,
                               dilation=dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = nn.Conv2d(width, planes * self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class MFF_OCE(nn.Module):
    """Multi-scale Feature Fusion with OCE"""
    def __init__(self, block, layers, width_per_group=64, norm_layer=None):
        super(MFF_OCE, self).__init__()
        if norm_layer is None:
            norm_layer = PositionalNorm2d
        self._norm_layer = norm_layer
        self.base_width = width_per_group

        # Fixed: inplanes should be 256 * block.expansion (256 * 4 = 1024)
        self.inplanes = 256 * block.expansion
        self.dilation = 1
        # bn_layer needs to be created AFTER concat, which has inplanes*3 channels
        self.bn_layer = self._make_layer(block, 512, layers, stride=2)

        # Convolution layers for feature processing
        self.conv1 = nn.Conv2d(64 * block.expansion, 128 * block.expansion, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = norm_layer(128 * block.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(128 * block.expansion, 256 * block.expansion, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = norm_layer(256 * block.expansion)
        self.conv3 = nn.Conv2d(128 * block.expansion, 256 * block.expansion, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn3 = norm_layer(256 * block.expansion)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        # Fixed: After concat, input has inplanes*3 channels (256+512+1024=1792 for expansion=4)
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes * 3, planes * block.expansion, 1, stride, bias=False),
                norm_layer(planes * block.expansion),
            )
        layers = []
        # First layer receives concatenated features with inplanes*3 channels
        layers.append(block(self.inplanes * 3, planes, stride, downsample, base_width=self.base_width,
                            dilation=previous_dilation, norm_layer=norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, base_width=self.base_width,
                                dilation=self.dilation, norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def forward(self, x):
        l1 = self.relu(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x[0]))))))
        l2 = self.relu(self.bn3(self.conv3(x[1])))
        feature = torch.cat([l1, l2, x[2]], 1)
        output = self.bn_layer(feature)
        return output.contiguous()


class ProjLayer(nn.Module):
    """Projection layer for feature transformation"""
    def __init__(self, in_c, out_c):
        super(ProjLayer, self).__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_c, in_c // 2, kernel_size=3, stride=1, padding=1),
            PositionalNorm2d(in_c // 2),
            nn.LeakyReLU(),
            nn.Conv2d(in_c // 2, in_c // 4, kernel_size=3, stride=1, padding=1),
            PositionalNorm2d(in_c // 4),
            nn.LeakyReLU(),
            nn.Conv2d(in_c // 4, in_c // 2, kernel_size=3, stride=1, padding=1),
            PositionalNorm2d(in_c // 2, affine=False),
            nn.LeakyReLU(),
            nn.Conv2d(in_c // 2, out_c, kernel_size=3, stride=1, padding=1),
            PositionalNorm2d(out_c),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.proj(x)


class SparseProjLayer(nn.Module):
    """Sparse projection layer using depthwise separable convolutions"""
    def __init__(self, in_c, out_c):
        super(SparseProjLayer, self).__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_c, in_c, kernel_size=3, stride=1, padding=1, groups=in_c),
            nn.Conv2d(in_c, in_c // 2, kernel_size=1, stride=1),
            PositionalNorm2d(in_c // 2),
            nn.LeakyReLU(),
            nn.Conv2d(in_c // 2, in_c // 2, kernel_size=3, stride=1, padding=1, groups=in_c // 2),
            nn.Conv2d(in_c // 2, in_c // 4, kernel_size=1, stride=1),
            PositionalNorm2d(in_c // 4),
            nn.LeakyReLU(),
            nn.Conv2d(in_c // 4, in_c // 4, kernel_size=3, stride=1, padding=1, groups=in_c // 4),
            nn.Conv2d(in_c // 4, in_c // 2, kernel_size=1, stride=1),
            PositionalNorm2d(in_c // 2, affine=False),
            nn.LeakyReLU(),
            nn.Conv2d(in_c // 2, in_c // 2, kernel_size=3, stride=1, padding=1, groups=in_c // 2),
            nn.Conv2d(in_c // 2, out_c, kernel_size=1, stride=1),
            PositionalNorm2d(out_c),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.proj(x)


class MultiProjectionLayer(nn.Module):
    """Multi-scale projection layer"""
    def __init__(self, base=64, dp=False):
        super(MultiProjectionLayer, self).__init__()
        self.proj_a = SparseProjLayer(base * 4, base * 4) if dp else ProjLayer(base * 4, base * 4)
        self.proj_b = SparseProjLayer(base * 8, base * 8) if dp else ProjLayer(base * 8, base * 8)
        self.proj_c = SparseProjLayer(base * 16, base * 16) if dp else ProjLayer(base * 16, base * 16)

    def forward(self, features, features_noise=False):
        if features_noise is not False:
            return (
                [self.proj_a(features_noise[0]), self.proj_b(features_noise[1]), self.proj_c(features_noise[2])],
                [self.proj_a(features[0]), self.proj_b(features[1]), self.proj_c(features[2])]
            )
        else:
            return [self.proj_a(features[0]), self.proj_b(features[1]), self.proj_c(features[2])]


# ============================================================================
# RDLGC with BYOL-style architecture (Main Model)
# ============================================================================

class RDLGC_BYOL(nn.Module):
    """
    RDLGC with BYOL-style architecture for preventing feature collapse.
    
    Architecture:
        Online Network:  encoder → projector → predictor → output
        Target Network:  encoder → projector → output (NO predictor)
    
    The asymmetry created by the predictor is KEY to preventing collapse
    without needing negative samples or diversity losses.
    
    Args:
        model_t: Teacher/encoder model config
        model_s: Student/decoder model config  
        dp: Use depthwise separable projections
        momentum: Momentum coefficient for target network update (default: 0.99)
        momentum_schedule: 'constant', 'cosine', 'linear' (default: 'cosine')
        momentum_start: Starting momentum for scheduled updates
        momentum_end: Ending momentum for scheduled updates
    """
    
    def __init__(self, model_t, model_s, dp=False, 
                 momentum=0.99,
                 momentum_schedule='cosine',
                 momentum_start=0.9,
                 momentum_end=0.999):
        super(RDLGC_BYOL, self).__init__()
        
        # === Encoder (Teacher - frozen) ===
        self.net_t = get_model(model_t)
        
        # === Feature fusion ===
        norm = PositionalNorm2d
        norm_layer = partial(norm, affine=False)
        self.mff_oce = MFF_OCE(Bottleneck, 3, norm_layer=norm_layer)
        
        # === Online Network ===
        self.proj_layer = MultiProjectionLayer(base=64, dp=dp)
        self.predictor = MultiPredictorLayer(base=64)  # KEY: Predictor creates asymmetry
        
        # === Target Network (Momentum) ===
        self.proj_layer_momentum = MultiProjectionLayer(base=64, dp=dp)
        # Initialize target with online weights
        self.proj_layer_momentum.load_state_dict(self.proj_layer.state_dict())
        # Freeze target network (only updated via EMA)
        for param in self.proj_layer_momentum.parameters():
            param.requires_grad = False
        
        # === Decoder (Student) ===
        self.net_s = get_model(model_s)
        
        # === Momentum settings ===
        self.momentum = momentum
        self.momentum_schedule = momentum_schedule
        self.momentum_start = momentum_start
        self.momentum_end = momentum_end
        self.current_step = 0
        self.total_steps = 1  # Will be set by trainer
        
        # === Frozen layers ===
        self.frozen_layers = ['net_t']

    def set_total_steps(self, total_steps):
        """Set total training steps for momentum scheduling"""
        self.total_steps = total_steps

    def get_current_momentum(self):
        """Get momentum based on schedule"""
        if self.momentum_schedule == 'constant':
            return self.momentum
        
        progress = min(self.current_step / max(self.total_steps, 1), 1.0)
        
        if self.momentum_schedule == 'cosine':
            # Cosine annealing from start to end
            return self.momentum_end - (self.momentum_end - self.momentum_start) * \
                   (1 + torch.cos(torch.tensor(progress * 3.14159))) / 2
        elif self.momentum_schedule == 'linear':
            return self.momentum_start + (self.momentum_end - self.momentum_start) * progress
        else:
            return self.momentum

    def freeze_layer(self, module):
        """Freeze a module"""
        module.eval()
        for param in module.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def update_momentum_encoder(self):
        """
        Update target network using Exponential Moving Average (EMA).
        θ_target = m * θ_target + (1 - m) * θ_online
        """
        m = self.get_current_momentum()
        if isinstance(m, torch.Tensor):
            m = m.item()
        
        for param_online, param_target in zip(
            self.proj_layer.parameters(), 
            self.proj_layer_momentum.parameters()
        ):
            param_target.data = param_target.data * m + param_online.data * (1 - m)
        
        self.current_step += 1

    def train(self, mode=True):
        """Set training mode"""
        self.training = mode
        for mname, module in self.named_children():
            if mname in self.frozen_layers:
                self.freeze_layer(module)
            else:
                module.train(mode)
        return self

    def train_forward(self, imgs, aug_imgs=None):
        """
        Forward pass during training (BYOL-style with momentum).

        Architecture:
        - Online path: imgs → encoder → projector → predictor → q_grid
        - Target path: imgs → encoder → momentum_projector → k_grid (NO predictor!)

        Key design choices:
        1. Both paths use SAME image (not augmented views)
        2. Difference comes from momentum parameters (EMA updated)
        3. Predictor asymmetry prevents collapse
        4. aug_imgs parameter kept for compatibility but NOT used

        Why same image for both paths?
        - Target should represent stable features of the SAME content
        - Momentum network provides smooth, stable targets
        - Augmentation diversity handled by data pipeline (if needed)
        """
        # === Extract features from encoder ===
        feats_t = self.net_t(imgs)  # Online: original image

        # === Online path: projector → predictor ===
        feats_t_proj = self.proj_layer(feats_t)
        feats_t_q_grid = self.predictor(feats_t_proj)  # With predictor (for BYOL loss)

        # === Target path: SAME image through momentum network ===
        with torch.no_grad():
            feats_k = self.net_t(imgs)  # Target: SAME image, same frozen encoder
            feats_t_k_grid = self.proj_layer_momentum(feats_k)  # Momentum projector (NO predictor!)
            feats_t_k = [f.clone() for f in feats_k]  # Detach target backbone features

        # === Optional: Add noise for regularization ===
        # Add noise to projected features BEFORE passing to mff_oce
        if self.training and torch.rand(1) > 0.5:
            for i in range(len(feats_t_proj)):
                noise = torch.randn_like(feats_t_proj[i]) * 0.1
                B, C, H, W = feats_t_proj[i].shape
                mask = torch.randint(0, 2, (B, 1, H, W), device=imgs.device).float()
                feats_t_proj[i] = feats_t_proj[i] + noise * mask

        # === Feature fusion and decoding ===
        # Use projected features (NOT predicted) for reconstruction
        mid = self.mff_oce(feats_t_proj)
        mid_k = self.mff_oce(feats_t_k_grid)
        feats_s = self.net_s(mid)

        # === Global features for SCL ===
        glo_feats = F.adaptive_avg_pool2d(mid, 1).squeeze()
        glo_feats_k = F.adaptive_avg_pool2d(mid_k, 1).squeeze()

        # Return features WITH gradient for losses (DO NOT detach feats_t!)
        # feats_t: Online backbone features (has gradient) - for cos loss and dense loss spatial matching
        # feats_s: Decoder output (has gradient) - for cos loss
        # feats_t_k: Target backbone features (detached) - for dense loss spatial matching
        # feats_t_q_grid: Online predictor output (has gradient) - for dense loss
        # feats_t_k_grid: Target projector output (detached) - for dense loss
        return feats_t, feats_s, feats_t_k, feats_t_q_grid, feats_t_k_grid, glo_feats, glo_feats_k

    def forward(self, imgs, aug_imgs=None):
        """Main forward pass"""
        if self.training and aug_imgs is not None:
            return self.train_forward(imgs, aug_imgs)

        # === Inference mode ===
        feats_t = self.net_t(imgs)
        feats_t_detached = [f.detach() for f in feats_t]  # Detach for inference
        feats = self.proj_layer(feats_t_detached)
        mid = self.mff_oce(feats)
        feats_s = self.net_s(mid)

        return feats_t_detached, feats_s, None, None, None, None, None


# ============================================================================
# Model registration
# ============================================================================

@MODEL.register_module
def rd_lgc_byol(pretrained=False, **kwargs):
    """Create RDLGC model with BYOL-style architecture"""
    model = RDLGC_BYOL(**kwargs)
    return model


# ============================================================================
# Test
# ============================================================================

if __name__ == '__main__':
    from argparse import Namespace as _Namespace
    
    # Test configuration
    bs = 2
    reso = 256
    x = torch.randn(bs, 3, reso, reso).cuda()
    x_aug = torch.randn(bs, 3, reso, reso).cuda()

    model_t = _Namespace()
    model_t.name = 'timm_wide_resnet50_2'
    model_t.kwargs = dict(
        pretrained=False,
        checkpoint_path='',
        strict=False, 
        features_only=True, 
        out_indices=[1, 2, 3]
    )
    
    model_s = _Namespace()
    model_s.name = 'de_wide_resnet50_2'
    model_s.kwargs = dict(
        pretrained=False,
        checkpoint_path='',
        strict=False
    )

    net = RDLGC_BYOL(model_t, model_s, momentum_schedule='cosine').cuda()
    net.train()
    
    # Test forward
    outputs = net(x, x_aug)
    print(f"Number of outputs: {len(outputs)}")
    print(f"feats_t_q shapes: {[f.shape for f in outputs[0]]}")
    print(f"feats_s shapes: {[f.shape for f in outputs[1]]}")
    print(f"q_grid shapes: {[f.shape for f in outputs[3]]}")
    print(f"k_grid shapes: {[f.shape for f in outputs[4]]}")
    
    # Test momentum update
    net.update_momentum_encoder()
    print(f"Current momentum: {net.get_current_momentum()}")
    
    print("\n✅ RDLGC_BYOL test passed!")
