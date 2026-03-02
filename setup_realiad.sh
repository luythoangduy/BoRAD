#!/bin/bash

# ============================================================================
# Setup Script for RDPP Noising Experiments - Real-IAD Dataset (realiad_256)
# ============================================================================
# This script will:
# 1. Verify directory structure
# 2. Install required packages
# 3. Download Real-IAD realiad_256 into data/realiad_256
# 4. Verify RealIAD structure (per-class JSON + images)
# 5. Install additional dependencies
# 6. Setup Weights & Biases (optional)
# ============================================================================
# Downloads Real-IAD realiad_256 (256x256, smaller than 1024) from Hugging Face.
# Pass your token: HF_TOKEN=your_token ./setup_realiad.sh
# ============================================================================

set -e  # Exit on error

# Hugging Face token for dataset download (use env: HF_TOKEN=xxx or HUGGING_FACE_HUB_TOKEN=xxx)
HF_TOKEN="${HF_TOKEN:-$HUGGING_FACE_HUB_TOKEN}"

REALIAD_ROOT="data/realiad_256"
REALIAD_SUBDIR="realiad_256"
HF_REALIAD_REPO="Real-IAD/Real-IAD"

echo "============================================================================"
echo "RDPP Noising Experiment Setup - Real-IAD (realiad_256)"
echo "Started at: $(date)"
echo "============================================================================"
echo ""

# ============================================================================
# Step 1: Verify Directory Structure
# ============================================================================
if [ ! -f "run.py" ]; then
    echo "Error: run.py not found. Make sure you're in the ADer directory."
    exit 1
fi

echo "[Step 1/7] Verifying directory structure..."
echo "Current directory: $(pwd)"
echo ""

# ============================================================================
# Step 2: Install Core Python Packages
# ============================================================================
echo "[Step 2/7] Installing core Python packages..."
pip install --upgrade pip

# pandas required for data processing
pip install pandas

# Install perlin noise library for experiments
pip install perlin-numpy

echo "Core packages installed."
echo ""

# ============================================================================
# Step 3: Download realiad_256 Dataset
# ============================================================================
echo "[Step 3/7] Downloading Real-IAD realiad_256 into $REALIAD_ROOT..."
echo ""

if [ -d "$REALIAD_ROOT" ] && [ "$(ls -A $REALIAD_ROOT 2>/dev/null)" ]; then
    echo "Found existing $REALIAD_ROOT - skipping download."
else
    if [ -z "$HF_TOKEN" ]; then
        echo "Error: HF_TOKEN not set. Run with: HF_TOKEN=your_token ./setup_realiad.sh"
        echo "Get a token at https://huggingface.co/settings/tokens"
        echo "Accept dataset access at https://huggingface.co/datasets/$HF_REALIAD_REPO"
        exit 1
    fi
    mkdir -p data
    pip install -q "huggingface_hub"
    dl_dir="data/realiad_256_dl"
    rm -rf "$dl_dir"
    mkdir -p "$dl_dir"
    echo "Downloading Real-IAD realiad_256 from Hugging Face (class zips + realiad_jsons.zip)..."
    echo "Using Python huggingface_hub API (no CLI required)."
    python - "$HF_REALIAD_REPO" "$HF_TOKEN" "$dl_dir" "$REALIAD_SUBDIR" << 'PYDOWNLOAD'
import os, sys
from huggingface_hub import list_repo_files, hf_hub_download

repo_id = sys.argv[1]
token = sys.argv[2]
local_dir = sys.argv[3]
subdir = sys.argv[4]  # e.g. realiad_256
os.makedirs(local_dir, exist_ok=True)
# List files under subdir/ (each class is a .zip)
try:
    files = list_repo_files(repo_id, repo_type="dataset", token=token)
except Exception as e:
    print("List repo failed:", e, file=sys.stderr)
    sys.exit(1)
zips = [f for f in files if f.startswith(subdir + "/") and f.endswith(".zip")]
if not zips:
    print("No zip files under %s/. Repo files: %s" % (subdir, files[:20]), file=sys.stderr)
    sys.exit(1)
print("Found", len(zips), "class zip(s). Downloading...")
for i, path in enumerate(sorted(zips)):
    fname = os.path.basename(path)
    print("  [%d/%d] %s" % (i + 1, len(zips), fname))
    try:
        path_local = hf_hub_download(
            repo_id=repo_id,
            filename=path,
            repo_type="dataset",
            token=token,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
        )
    except Exception as e:
        print("  Download failed:", e, file=sys.stderr)
        sys.exit(1)
print("Class zips download done.")

# Download realiad_jsons.zip (JSON metadata for all classes), if available at repo root
json_zip = None
for f in files:
    if f.endswith("realiad_jsons.zip"):
        json_zip = f
        break
if json_zip is not None:
    print("Downloading", json_zip, "...")
    try:
        path_local = hf_hub_download(
            repo_id=repo_id,
            filename=json_zip,
            repo_type="dataset",
            token=token,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
        )
    except Exception as e:
        print("  Download realiad_jsons.zip failed:", e, file=sys.stderr)
        sys.exit(1)
    print("realiad_jsons.zip download done.")
else:
    print("Warning: realiad_jsons.zip not found in repo file list.", file=sys.stderr)
PYDOWNLOAD
    if [ $? -ne 0 ]; then
        echo "Download failed. Ensure you accepted the dataset terms at https://huggingface.co/datasets/$HF_REALIAD_REPO"
        exit 1
    fi
    echo "Extracting class zip files into $REALIAD_ROOT..."
    mkdir -p "$REALIAD_ROOT"
    for z in "$dl_dir/$REALIAD_SUBDIR"/*.zip; do
        [ -f "$z" ] || continue
        echo "  Extracting $(basename "$z")..."
        unzip -q -o "$z" -d "$REALIAD_ROOT" 2>/dev/null || true
    done
    # Extract JSON metadata archive if present (creates realiad_jsons/ subfolder)
    if [ -f "$dl_dir/realiad_jsons.zip" ]; then
        echo "Extracting realiad_jsons.zip into $REALIAD_ROOT..."
        unzip -q -o "$dl_dir/realiad_jsons.zip" -d "$REALIAD_ROOT" 2>/dev/null || true
    fi
    # If JSONs are inside class folders (e.g. audiojack/audiojack.json), copy to root
    for d in "$REALIAD_ROOT"/*/; do
        [ -d "$d" ] || continue
        cls=$(basename "$d")
        if [ -f "$d${cls}.json" ] && [ ! -f "$REALIAD_ROOT/${cls}.json" ]; then
            cp "$d${cls}.json" "$REALIAD_ROOT/${cls}.json"
        fi
    done
    rm -rf "$dl_dir"
    echo "Done. Data is in $REALIAD_ROOT"
fi

echo ""

# ============================================================================
# Step 4: Verify RealIAD Structure
# ============================================================================
echo "[Step 4/7] Verifying RealIAD structure..."
# RealIAD uses per-class JSON files (e.g. audiojack.json) and image dirs under each class
json_count=$(find "$REALIAD_ROOT" -maxdepth 1 -name "*.json" 2>/dev/null | wc -l)
if [ "$json_count" -gt 0 ]; then
    echo "Found $json_count class JSON file(s) in $REALIAD_ROOT."
else
    echo "Note: No .json files found in $REALIAD_ROOT. Ensure you have Real-IAD format (per-class .json + images)."
fi
echo ""

# ============================================================================
# Step 5: Install Additional Dependencies
# ============================================================================
echo "[Step 5/7] Installing additional dependencies..."

# Install PyTorch dependencies (adjust based on your CUDA version)
# Uncomment the appropriate line for your system:

# For CUDA 11.8
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CPU only
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other required packages
pip install adeval \
    FrEIA \
    geomloss \
    ninja \
    faiss-cpu \
    einops \
    numba \
    imgaug \
    scikit-image \
    opencv-python \
    fvcore \
    tensorboardX \
    timm
pip install numpy==1.26.4 scikit-learn wandb

conda install -c conda-forge accimage -y
echo "Additional dependencies installed."
echo ""

# ============================================================================
# Step 6: Setup Weights & Biases (Optional)
# ============================================================================
echo "[Step 6/7] Setting up Weights & Biases (optional)..."
echo ""
echo "Weights & Biases (wandb) is used for experiment tracking and visualization."
echo "wandb is enabled by default in the config."
echo ""
read -p "Do you want to setup wandb now? (y/n): " setup_wandb

if [ "$setup_wandb" = "y" ] || [ "$setup_wandb" = "Y" ]; then
    echo ""
    echo "Starting wandb setup..."
    chmod +x wandb_setup.sh 2>/dev/null || true
    ./wandb_setup.sh 2>/dev/null || echo "Run ./wandb_setup.sh manually if needed."
else
    echo ""
    echo "Skipping wandb setup."
    echo "You can run './wandb_setup.sh' later to configure wandb."
    echo ""
    echo "Note: wandb is enabled by default. To disable it for a run:"
    echo "  python run.py -c configs/rdpp_noising/rdpp_noising_256_100e_realiad.py wandb.enable=False"
fi

echo ""

# ============================================================================
# Step 7: Verify Installation
# ============================================================================
echo "[Step 7/7] Verifying installation..."
echo "============================================================================"
echo "Verifying installation..."
echo "============================================================================"
echo ""

# Check if data exists
if [ -d "$REALIAD_ROOT" ] && [ "$(ls -A $REALIAD_ROOT 2>/dev/null)" ]; then
    echo "✓ Real-IAD realiad_256: OK ($REALIAD_ROOT)"
else
    echo "✗ Real-IAD realiad_256: NOT FOUND or empty (expected: $REALIAD_ROOT)"
fi

# Check for class JSON files
json_count=$(find "$REALIAD_ROOT" -maxdepth 1 -name "*.json" 2>/dev/null | wc -l)
if [ "$json_count" -gt 0 ]; then
    echo "✓ RealIAD class JSON files: OK ($json_count classes)"
else
    echo "✗ RealIAD class JSON files: NONE (add per-class .json from Real-IAD)"
fi

# Check if model directory exists
if [ -d "model" ]; then
    echo "✓ Model directory: OK"
else
    echo "✗ Model directory: NOT FOUND"
fi

# Test Python imports
echo ""
echo "Testing Python package imports..."
python -c "import torch; print('✓ PyTorch:', torch.__version__)" || echo "✗ PyTorch: FAILED"
python -c "import timm; print('✓ timm:', timm.__version__)" || echo "✗ timm: FAILED"
python -c "import pandas; print('✓ pandas: OK')" || echo "✗ pandas: FAILED"
python -c "from perlin_numpy import generate_perlin_noise_2d; print('✓ perlin-numpy: OK')" || echo "✗ perlin-numpy: FAILED"
python -c "import geomloss; print('✓ geomloss: OK')" || echo "✗ geomloss: FAILED"
python -c "import faiss; print('✓ faiss: OK')" || echo "✗ faiss: FAILED"

echo ""
echo "============================================================================"
echo "Setup Complete (Real-IAD realiad_256)!"
echo "============================================================================"
echo ""
echo "Next steps:"
echo "1. Download pretrained weights (if needed):"
echo "   - Wide ResNet50: model/pretrain/wide_resnet50_racm-8234f177.pth"
echo ""
echo "2. Use a config that points to realiad_256, e.g.:"
echo "   configs/rdpp_noising/rdpp_noising_256_100e_readiad.py"
echo "   (self.data.root = 'data/realiad_256')"
echo ""
echo "3. Run training on Real-IAD (realiad_256):"
echo "   python run.py -c configs/rdpp_noising/rdpp_noising_256_100e_readiad.py"
echo ""
echo "4. Request Real-IAD access if you have not: realiad4ad@outlook.com"
echo "   Project: https://realiad4ad.github.io/Real-IAD/"
echo ""
echo "For more information, see: data/README.md (Real-IAD section)"
echo ""
echo "Finished at: $(date)"
echo "============================================================================"
