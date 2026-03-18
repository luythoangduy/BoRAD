import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import sys
import os
from argparse import Namespace
from tabulate import tabulate

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.rd_byol import RDLGC_BYOL
from loss.byol_loss import PrototypeBYOLLoss

try:
    from fvcore.nn import FlopCountAnalysis, parameter_count_table
except ImportError:
    print("Error: fvcore not found. Please install with 'pip install fvcore'")
    sys.exit(1)

class PartCosOnly(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, x):
        # Full RDLGC pipeline minus prototype loss components
        feats_t = self.model.net_t(x)
        feats_proj = self.model.proj_layer(feats_t)
        mid = self.model.mff_oce(feats_proj)
        feats_s = self.model.net_s(mid)
        return feats_s

class PartGlobal(nn.Module):
    def __init__(self, model, loss_module):
        super().__init__()
        self.model = model
        self.loss_module = loss_module
        
    def forward(self, x):
        # Part 1: Reconstruction path
        feats_t = self.model.net_t(x)
        feats_proj = self.model.proj_layer(feats_t)
        mid = self.model.mff_oce(feats_proj)
        feats_s = self.model.net_s(mid)
        
        # Part 2: Global Prototype path
        glo_feats = F.adaptive_avg_pool2d(mid, 1).squeeze()
        shifted_q = self.loss_module.shift_to_prototypes(glo_feats)
        B, N, C = shifted_q.shape
        predicted_global = self.loss_module.global_predictor(shifted_q.reshape(B * N, -1))
        
        return feats_s, predicted_global

class PartSpatial(nn.Module):
    def __init__(self, model, loss_module):
        super().__init__()
        self.model = model
        self.loss_module = loss_module
        
    def forward(self, x):
        # Part 1: Reconstruction path
        feats_t = self.model.net_t(x)
        feats_proj = self.model.proj_layer(feats_t)
        mid = self.model.mff_oce(feats_proj)
        feats_s = self.model.net_s(mid)
        
        # Part 3: Spatial Prototype path
        # Note: query_prototypes includes the linear merge and spatial predictor is separate in loss
        merged_q = self.loss_module.query_prototypes(mid)
        predicted_spatial = self.loss_module.predictor(merged_q)
        
        return feats_s, predicted_spatial

def benchmark(name, model, input_tensor, iterations=100):
    model.eval()
    device = input_tensor.device
    
    # 1. Parameter count
    params = sum(p.numel() for p in model.parameters())
    params_m = params / 1e6
    
    # 2. FLOPs calculation
    flops = FlopCountAnalysis(model, input_tensor)
    total_flops = flops.total()
    gflops = total_flops / 1e9
    
    # 3. FPS calculation
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            model(input_tensor)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
        
    start_time = time.time()
    for _ in range(iterations):
        with torch.no_grad():
            model(input_tensor)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
        
    end_time = time.time()
    avg_time = (end_time - start_time) / iterations
    fps = 1.0 / avg_time
    
    return params_m, gflops, fps

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Benchmarking on device: {device}")
    
    # Configuration (matching typical BoRAD setup)
    model_t = Namespace(
        name='timm_wide_resnet50_2',
        kwargs=dict(pretrained=False, features_only=True, out_indices=[1, 2, 3])
    )
    model_s = Namespace(
        name='de_wide_resnet50_2',
        kwargs=dict(pretrained=False)
    )
    
    # Initialize base model
    base_model = RDLGC_BYOL(model_t, model_s).to(device)
    
    # Initialize loss module (contains prototype parameters and predictors)
    # Match default dims (usually C=2048 for ResNet50 at bottleneck, but MFF output is 2048)
    loss_module = PrototypeBYOLLoss(feat_dim=2048, H=8, W=8).to(device)
    
    # Input tensor
    input_tensor = torch.randn(1, 3, 256, 256).to(device)
    
    # Define parts
    parts = {
        "Cos Loss Only": PartCosOnly(base_model),
        "Global Loss": PartGlobal(base_model, loss_module),
        "Spatial Loss": PartSpatial(base_model, loss_module)
    }
    
    results = []
    for name, model in parts.items():
        print(f"Running benchmark for {name}...")
        p_m, g_flops, fps = benchmark(name, model, input_tensor)
        results.append([name, f"{p_m:.2f} M", f"{g_flops:.2f} G", f"{fps:.2f}"])
    
    # Also calculate for Backbone only for reference
    print("Running benchmark for Backbone (WideResNet50)...")
    backbone = base_model.net_t
    p_m_b, g_flops_b, fps_b = benchmark("Backbone", backbone, input_tensor)
    results.append(["Backbone Only", f"{p_m_b:.2f} M", f"{g_flops_b:.2f} G", f"{fps_b:.2f}"])

    print("\n" + "="*60)
    print("BoRAD Model Benchmark Results")
    print("="*60)
    print(tabulate(results, headers=["Component", "Params (M)", "TFLOPS (G)", "FPS"], tablefmt="grid"))
    print("="*60)
    print("Note: GFLOPS (G) = GigaFLOPs (10^9 FLOPs)")

if __name__ == "__main__":
    main()
