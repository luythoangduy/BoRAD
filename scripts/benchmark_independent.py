import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import sys
import os
import timm
from argparse import Namespace
from tabulate import tabulate

try:
    from fvcore.nn import FlopCountAnalysis
except ImportError:
    print("Error: fvcore not found. Please install with 'pip install fvcore'")
    sys.exit(1)

# ============================================================================
# Core Components (Merged from project files)
# ============================================================================

# --- Utilities ---
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False,
                     dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def deconv2x2(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=2, stride=stride, groups=groups, bias=False,
                              dilation=dilation)

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None: norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, stride=1, bias=False)
        self.bn1 = norm_layer(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = nn.Conv2d(width, planes * self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None: identity = self.downsample(x)
        out += identity
        return self.relu(out)

# --- MFF-OCE ---
class MFF_OCE(nn.Module):
    def __init__(self, block, layers, width_per_group=64, norm_layer=None):
        super(MFF_OCE, self).__init__()
        if norm_layer is None: norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.base_width = width_per_group
        self.inplanes = 256 * block.expansion
        self.dilation = 1
        self.bn_layer = self._make_layer(block, 512, layers, stride=2)
        self.conv1 = conv3x3(64 * block.expansion, 128 * block.expansion, 2)
        self.bn1 = norm_layer(128 * block.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(128 * block.expansion, 256 * block.expansion, 2)
        self.bn2 = norm_layer(256 * block.expansion)
        self.conv3 = conv3x3(128 * block.expansion, 256 * block.expansion, 2)
        self.bn3 = norm_layer(256 * block.expansion)
    def _make_layer(self, block, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.inplanes * 3, planes * block.expansion, stride), norm_layer(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes * 3, planes, stride, downsample, base_width=self.base_width, norm_layer=norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, base_width=self.base_width, norm_layer=norm_layer))
        return nn.Sequential(*layers)
    def forward(self, x):
        l1 = self.relu(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x[0]))))))
        l2 = self.relu(self.bn3(self.conv3(x[1])))
        feature = torch.cat([l1, l2, x[2]], 1)
        return self.bn_layer(feature).contiguous()

# --- Proj & Predictor ---
class ProjLayer(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_c, in_c // 2, kernel_size=3, padding=1), nn.InstanceNorm2d(in_c // 2), nn.LeakyReLU(),
            nn.Conv2d(in_c // 2, in_c // 4, kernel_size=3, padding=1), nn.InstanceNorm2d(in_c // 4), nn.LeakyReLU(),
            nn.Conv2d(in_c // 4, in_c // 2, kernel_size=3, padding=1), nn.InstanceNorm2d(in_c // 2), nn.LeakyReLU(),
            nn.Conv2d(in_c // 2, out_c, kernel_size=3, padding=1), nn.InstanceNorm2d(out_c), nn.LeakyReLU(),
        )
    def forward(self, x): return self.proj(x)

class MultiProjectionLayer(nn.Module):
    def __init__(self, base=64):
        super().__init__()
        self.proj_a = ProjLayer(base * 4, base * 4)
        self.proj_b = ProjLayer(base * 8, base * 8)
        self.proj_c = ProjLayer(base * 16, base * 16)
    def forward(self, features):
        return [self.proj_a(features[0]), self.proj_b(features[1]), self.proj_c(features[2])]

class PredictorLayer(nn.Module):
    def __init__(self, in_c, hidden_c=None, out_c=None):
        super().__init__()
        hidden_c = hidden_c or in_c // 2
        out_c = out_c or in_c
        self.predictor = nn.Sequential(
            nn.Conv2d(in_c, hidden_c, kernel_size=1, bias=False), nn.InstanceNorm2d(hidden_c), nn.LeakyReLU(),
            nn.Conv2d(hidden_c, out_c, kernel_size=1, bias=False), nn.InstanceNorm2d(out_c), nn.LeakyReLU(),
        )
    def forward(self, x): return self.predictor(x)

class MultiPredictorLayer(nn.Module):
    def __init__(self, base=64):
        super().__init__()
        self.pred_a = PredictorLayer(base * 4, hidden_c=base * 2, out_c=base * 4)
        self.pred_b = PredictorLayer(base * 8, hidden_c=base * 4, out_c=base * 8)
        self.pred_c = PredictorLayer(base * 16, hidden_c=base * 8, out_c=base * 16)
    def forward(self, features):
        return [self.pred_a(features[0]), self.pred_b(features[1]), self.pred_c(features[2])]

# --- Decoder (ResNet-based) ---
class DeBottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, upsample=None, groups=1, base_width=64, dilation=1, norm_layer=None):
        super().__init__()
        if norm_layer is None: norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = deconv2x2(width, width, stride) if stride == 2 else conv3x3(width, width, stride)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = upsample
    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.upsample is not None: identity = self.upsample(x)
        out += identity
        return self.relu(out)

class ResNetDecoder(nn.Module):
    def __init__(self, block, layers):
        super().__init__()
        self.inplanes = 512 * block.expansion
        self.layer1 = self._make_layer(block, 256, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
    def _make_layer(self, block, planes, blocks, stride=1):
        upsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            upsample = nn.Sequential(deconv2x2(self.inplanes, planes * block.expansion, stride), nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, upsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks): layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)
    def forward(self, x):
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        return [f3, f2, f1]

# --- Prototype Loss ---
def generate_orthonormal_vectors(hw, n, dim):
    A = torch.randn(hw, dim, n)
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    return U.transpose(1, 2)

class PrototypeModule(nn.Module):
    def __init__(self, n_prototypes=7, feat_dim=2048, H=8, W=8):
        super().__init__()
        self.n_prototypes = n_prototypes
        self.prototypes = nn.Parameter(generate_orthonormal_vectors(H * W, n_prototypes, feat_dim))
        self.linear_merge = nn.Sequential(nn.Conv2d(feat_dim * 2, feat_dim, kernel_size=1), nn.InstanceNorm2d(feat_dim), nn.LeakyReLU(inplace=True))
        self.spatial_predictor = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim, kernel_size=1), nn.InstanceNorm2d(feat_dim), nn.LeakyReLU(inplace=True),
            nn.Conv2d(feat_dim, feat_dim, kernel_size=1), nn.InstanceNorm2d(feat_dim), nn.LeakyReLU(inplace=True)
        )
        self.global_predictor = nn.Sequential(
            nn.Linear(feat_dim, feat_dim), nn.InstanceNorm1d(feat_dim), nn.LeakyReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim), nn.InstanceNorm1d(feat_dim), nn.LeakyReLU(inplace=True)
        )
    def query_prototypes(self, spatial_feats):
        B, C, H, W = spatial_feats.shape
        features_flat = spatial_feats.view(B, C, -1).transpose(1, 2)
        features_norm = F.normalize(features_flat, dim=-1, p=2)
        prototypes_norm = F.normalize(self.prototypes, dim=-1, p=2)
        cosine_sim = torch.einsum('bsc,snc->bsn', features_norm, prototypes_norm)
        proto_features = torch.einsum('bsn,snc->bsc', cosine_sim, self.prototypes)
        proto_features = proto_features.transpose(1, 2).view(B, C, H, W)
        merged = self.linear_merge(torch.cat([spatial_feats, proto_features], dim=1))
        return merged
    def shift_to_prototypes(self, features):
        B = features.shape[0]
        global_prototypes = self.prototypes.mean(dim=0)
        prototypes_norm = F.normalize(global_prototypes, dim=1, p=2)
        return features.unsqueeze(1) - prototypes_norm.unsqueeze(0)

# ============================================================================
# Wrappers for Benchmarking
# ============================================================================

class UnifiedModelWrapper(nn.Module):
    def __init__(self, net_t, proj, mff, net_s, proto_mod, proj_momentum, train_pred, mode='test', segment='full'):
        super().__init__()
        self.net_t = net_t
        self.proj = proj
        self.mff = mff
        self.net_s = net_s
        self.proto = proto_mod
        self.proj_momentum = proj_momentum
        self.train_pred = train_pred
        self.mode = mode  # 'train' or 'test'
        self.segment = segment # 'cos', 'global', 'spatial', 'full'

    def forward(self, x):
        # 1. Inference Path (Used in both Train and Test)
        feats_t = self.net_t(x)
        proj_feats = self.proj(feats_t)
        mid = self.mff(proj_feats)
        out = self.net_s(mid)
        
        if self.mode == 'test':
            return out # Minimal inference path
            
        # 2. Training Path (Additional components)
        with torch.no_grad():
            target_proj = self.proj_momentum(feats_t)
            
        res = [out]
        
        # Segment logic
        if self.segment in ['global', 'full']:
            glo_feats = F.adaptive_avg_pool2d(mid, 1).flatten(1)
            shifted = self.proto.shift_to_prototypes(glo_feats)
            B, N, C = shifted.shape
            pred_glo = self.proto.global_predictor(shifted.reshape(B * N, -1))
            res.append(pred_glo)
            
        if self.segment in ['spatial', 'full']:
            merged = self.proto.query_prototypes(mid)
            pred_spa = self.proto.spatial_predictor(merged)
            res.append(pred_spa)
            
        # BYOL Predator (Online predictor)
        pred_online = self.train_pred(proj_feats)
        res.append(pred_online)
        
        return res

# ============================================================================
# Benchmark Logic
# ============================================================================

def get_params_count(model):
    """Calculate parameters based on mode and segment"""
    # Base online path (Inference)
    # Online path uses: net_t, proj, mff, net_s
    online_params = sum(p.numel() for p in model.net_t.parameters())
    online_params += sum(p.numel() for p in model.proj.parameters())
    online_params += sum(p.numel() for p in model.mff.parameters())
    online_params += sum(p.numel() for p in model.net_s.parameters())
    
    if model.mode == 'test':
        return online_params / 1e6
        
    # Training adds momentum projectors and BYOL online predictor
    train_core = online_params
    train_core += sum(p.numel() for p in model.proj_momentum.parameters())
    train_core += sum(p.numel() for p in model.train_pred.parameters())
    
    if model.segment == 'cos':
        return train_core / 1e6
    elif model.segment == 'global':
        # Adds Global predictor + Global Prototypes
        count = train_core
        count += sum(p.numel() for p in model.proto.global_predictor.parameters())
        count += model.proto.prototypes.numel() # Prototype bank
        return count / 1e6
    elif model.segment == 'spatial':
        # Adds Linear Merge + Spatial predictor + Spatial Prototypes
        count = train_core
        count += sum(p.numel() for p in model.proto.linear_merge.parameters())
        count += sum(p.numel() for p in model.proto.spatial_predictor.parameters())
        count += model.proto.prototypes.numel() # Prototype bank
        return count / 1e6
    else: # full
        return sum(p.numel() for p in model.parameters()) / 1e6

def benchmark_unified(name, model, input_tensor, iterations=50):
    model.eval() if model.mode == 'test' else model.train()
    
    # 1. Params (Refined calculation)
    params_m = get_params_count(model)
    
    # 2. GFLOPS
    # fvcore's FlopCountAnalysis tracks what's called in forward()
    flops_g = FlopCountAnalysis(model, input_tensor).total() / 1e9
    
    # 3. FPS
    for _ in range(5): 
        with torch.no_grad(): model(input_tensor)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    start = time.time()
    # During training benchmark, we don't use no_grad because we want to see training speed
    with torch.set_grad_enabled(model.mode == 'train'):
        for _ in range(iterations):
            model(input_tensor)
            
    if torch.cuda.is_available(): torch.cuda.synchronize()
    fps = iterations / (time.time() - start)
    
    return params_m, flops_g, fps

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Benchmarking on {device}...")
    
    # --- Shared Components ---
    net_t = timm.create_model('wide_resnet50_2', features_only=True, out_indices=[1, 2, 3], pretrained=False).to(device)
    proj = MultiProjectionLayer(base=64).to(device)
    mff = MFF_OCE(Bottleneck, 3).to(device)
    net_s = ResNetDecoder(DeBottleneck, [3, 4, 6, 3]).to(device)
    proto_mod = PrototypeModule(feat_dim=2048).to(device)
    proj_mom = MultiProjectionLayer(base=64).to(device) # Target Network
    train_pred = MultiPredictorLayer(base=64).to(device) # Online predictor
    
    x = torch.randn(1, 3, 256, 256).to(device)
    
    segments = ['cos', 'global', 'spatial', 'full']
    segment_names = {
        'cos': 'Cos Loss Only',
        'global': 'Global Loss (+Cos)',
        'spatial': 'Spatial Loss (+Cos)',
        'full': 'Overall (Global+Spatial+Cos)'
    }
    
    results = []
    
    for seg in segments:
        name = segment_names[seg]
        print(f"Benchmarking {name}...")
        
        # Test Mode
        model_test = UnifiedModelWrapper(net_t, proj, mff, net_s, proto_mod, proj_mom, train_pred, mode='test', segment=seg).to(device)
        p_tst, f_tst, fps_tst = benchmark_unified(name, model_test, x)
        
        # Train Mode
        model_train = UnifiedModelWrapper(net_t, proj, mff, net_s, proto_mod, proj_mom, train_pred, mode='train', segment=seg).to(device)
        p_trn, f_trn, fps_trn = benchmark_unified(name, model_train, x)
        
        results.append([
            name, 
            f"{p_trn:.2f} M", f"{p_tst:.2f} M", 
            f"{f_trn:.2f} G", f"{f_tst:.2f} G",
            f"{fps_trn:.2f}", f"{fps_tst:.2f}"
        ])
    
    headers = ["Part", "Train Params", "Test Params", "Train GFLOPS", "Test GFLOPS", "Train FPS", "Test FPS"]
    print("\n" + tabulate(results, headers=headers, tablefmt="grid"))
    print("\nNote: Train Params includes Target net and Loss predictors. Test Params only includes Online inference path.")

if __name__ == "__main__":
    main()
