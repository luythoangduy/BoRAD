#!/bin/bash

# ============================================================================
# Setup Script for RDPP Noising Experiments
# ============================================================================
# This script will:
# 1. Clone the repository (if needed)
# 2. Install required packages
# 3. Download and prepare MVTec AD dataset
# 4. Generate benchmark metadata
# ============================================================================

set -e  # Exit on error

echo "============================================================================"
echo "RDPP Noising Experiment Setup"
echo "Started at: $(date)"
echo "============================================================================"
echo ""

# ============================================================================
# Step 1: Clone Repository (Optional - if running from scratch)
# ============================================================================
# Uncomment if you need to clone the repo
# echo "[Step 1/6] Cloning repository..."
# git clone https://github.com/luythoangduy/ANo4AD.git
# cd ANo4AD/ADer

# If already in ADer directory, just continue
if [ ! -f "run.py" ]; then
    echo "Error: run.py not found. Make sure you're in the ADer directory."
    exit 1
fi

echo "[Step 1/6] Verifying directory structure..."
echo "Current directory: $(pwd)"
echo ""

# ============================================================================
# Step 2: Install Core Python Packages
# ============================================================================
echo "[Step 2/6] Installing core Python packages..."
pip install --upgrade pip

# Install gdown for Google Drive downloads
pip install gdown

# Install perlin noise library for experiments
pip install perlin-numpy

echo "Core packages installed."
echo ""

# ============================================================================
# Step 3: Download MVTec AD Dataset
# ============================================================================
echo "[Step 3/6] Downloading MVTec AD dataset..."
echo "This may take several minutes depending on your internet connection..."

# Create temporary download directory
mkdir -p temp_download
cd temp_download

# Download dataset from Google Drive
gdown --fuzzy "https://drive.google.com/file/d/1JhhA36qmH8lKCgiX9lU6v8D7B1Y3Xa7r/view?usp=drive_link" -O datasets.zip

echo "Download completed."
echo ""

# ============================================================================
# Step 4: Extract and Organize Dataset
# ============================================================================
echo "[Step 4/6] Extracting dataset..."
unzip -q datasets.zip -d .

echo "Creating data directory structure..."
cd ..
mkdir -p data/mvtec

echo "Moving dataset files..."
mv temp_download/datasets/mvtec_anomaly_detection/* data/mvtec/

echo "Cleaning up temporary files..."
rm -rf temp_download

echo "Dataset setup completed."
echo ""

# ============================================================================
# Step 5: Generate Benchmark Metadata
# ============================================================================
echo "[Step 5/6] Generating benchmark metadata..."
python data/gen_benchmark/mvtec.py

echo "Benchmark metadata generated."
echo ""

# ============================================================================
# Step 6: Install Additional Dependencies
# ============================================================================
echo "[Step 6/6] Installing additional dependencies..."

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
pip install numpy==1.26.4 scikit-learn wandb tensorboard

conda install -c conda-forge accimage -y
echo "Additional dependencies installed."
echo ""

# ============================================================================
# Step 7: Setup Weights & Biases (Optional)
# ============================================================================
echo "[Step 7/8] Setting up Weights & Biases (optional)..."
echo ""
echo "Weights & Biases (wandb) is used for experiment tracking and visualization."
echo "wandb is enabled by default in the config."
echo ""
read -p "Do you want to setup wandb now? (y/n): " setup_wandb

if [ "$setup_wandb" = "y" ] || [ "$setup_wandb" = "Y" ]; then
    echo ""
    echo "Starting wandb setup..."
    chmod +x wandb_setup.sh
    ./wandb_setup.sh
else
    echo ""
    echo "Skipping wandb setup."
    echo "You can run './wandb_setup.sh' later to configure wandb."
    echo ""
    echo "Note: wandb is enabled by default. To disable it for a run:"
    echo "  python run.py -c ... wandb.enable=False"
fi

echo ""

# ============================================================================
# Step 8: Verify Installation
# ============================================================================
echo "============================================================================"
echo "Verifying installation..."
echo "============================================================================"
echo ""

# Check if data exists
if [ -d "data/mvtec" ] && [ "$(ls -A data/mvtec)" ]; then
    echo "✓ MVTec dataset: OK"
else
    echo "✗ MVTec dataset: NOT FOUND"
fi

# Check if meta.json exists
if [ -f "data/mvtec/meta.json" ]; then
    echo "✓ Benchmark metadata: OK"
else
    echo "✗ Benchmark metadata: NOT FOUND"
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
python -c "from perlin_numpy import generate_perlin_noise_2d; print('✓ perlin-numpy: OK')" || echo "✗ perlin-numpy: FAILED"
python -c "import geomloss; print('✓ geomloss: OK')" || echo "✗ geomloss: FAILED"
python -c "import faiss; print('✓ faiss: OK')" || echo "✗ faiss: FAILED"

echo ""
echo "============================================================================"
echo "Setup Complete!"
echo "============================================================================"
echo ""
echo "Next steps:"
echo "1. Download pretrained weights (if needed):"
echo "   - Wide ResNet50: model/pretrain/wide_resnet50_racm-8234f177.pth"
echo ""
echo "2. Make experiment scripts executable:"
echo "   chmod +x run_all_rdpp_experiments.sh"
echo "   chmod +x run_rdpp_single.sh"
echo "   chmod +x run_rdpp_by_group.sh"
echo ""
echo "3. Test with a quick experiment:"
echo "   ./run_rdpp_single.sh none none none"
echo ""
echo "4. Or run all experiments:"
echo "   ./run_all_rdpp_experiments.sh"
echo ""
echo "For more information, see: RDPP_EXPERIMENTS_README.md"
echo ""
echo "Finished at: $(date)"
echo "============================================================================"
