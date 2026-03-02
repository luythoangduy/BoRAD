#!/bin/bash

# ============================================================================
# Setup Script for RDPP Noising Experiments - ViSA Dataset
# ============================================================================
# This script will:
# 1. Verify directory structure
# 2. Install required packages
# 3. Download and prepare ViSA (Visual Anomaly) dataset from AWS
# 4. Download split_csv/1cls.csv from amazon-science/spot-diff
# 5. Generate benchmark metadata (data/visa/meta.json)
# 6. Install additional dependencies
# ============================================================================
# ViSA: https://github.com/amazon-science/spot-diff#data-preparation
# ============================================================================

set -e  # Exit on error

VISA_URL="https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar"
SPOT_DIFF_1CLS_CSV="https://raw.githubusercontent.com/amazon-science/spot-diff/main/split_csv/1cls.csv"

echo "============================================================================"
echo "RDPP Noising Experiment Setup - ViSA Dataset"
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

# pandas required for data/gen_benchmark/visa.py
pip install pandas

# Install perlin noise library for experiments
pip install perlin-numpy

echo "Core packages installed."
echo ""

# ============================================================================
# Step 3: Download ViSA Dataset from AWS
# ============================================================================
echo "[Step 3/7] Downloading ViSA dataset from AWS..."
echo "URL: $VISA_URL"
echo "This may take a while (dataset is large)."

mkdir -p temp_download
cd temp_download

if command -v wget &> /dev/null; then
    wget -q --show-progress -O VisA_20220922.tar "$VISA_URL" || {
        echo "wget failed, trying curl..."
        curl -L -o VisA_20220922.tar "$VISA_URL"
    }
elif command -v curl &> /dev/null; then
    curl -L -o VisA_20220922.tar "$VISA_URL"
else
    echo "Error: Neither wget nor curl found. Please install one or download manually:"
    echo "  $VISA_URL"
    echo "  Extract into data/visa/"
    exit 1
fi

echo "Download completed."
echo ""

# ============================================================================
# Step 4: Extract and Organize Dataset
# ============================================================================
echo "[Step 4/7] Extracting dataset..."
tar -xf VisA_20220922.tar

cd ..
mkdir -p data/visa

# Tar may extract as "VisA" or "VisA_20220922" (per spot-diff) - move contents to data/visa
if [ -d "temp_download/VisA" ]; then
    echo "Moving VisA/* to data/visa/..."
    mv temp_download/VisA/* data/visa/
elif [ -d "temp_download/VisA_20220922" ]; then
    echo "Moving VisA_20220922/* to data/visa/..."
    mv temp_download/VisA_20220922/* data/visa/
else
    echo "Contents of temp_download:"
    ls -la temp_download/
    echo "Moving extracted items to data/visa/ (excluding tar)..."
    for f in temp_download/*; do
        [ -e "$f" ] || continue
        [ "$f" = "temp_download/VisA_20220922.tar" ] && continue
        mv "$f" data/visa/
    done
fi
rm -f data/visa/VisA_20220922.tar 2>/dev/null || true

echo "Downloading split_csv/1cls.csv from spot-diff (required for meta.json)..."
mkdir -p data/visa/split_csv
if command -v wget &> /dev/null; then
    wget -q -O data/visa/split_csv/1cls.csv "$SPOT_DIFF_1CLS_CSV"
elif command -v curl &> /dev/null; then
    curl -sL -o data/visa/split_csv/1cls.csv "$SPOT_DIFF_1CLS_CSV"
fi

echo "Cleaning up temporary files..."
rm -rf temp_download

echo "Dataset setup completed."
echo ""

# ============================================================================
# Step 5: Generate Benchmark Metadata
# ============================================================================
echo "[Step 5/7] Generating benchmark metadata (data/visa/meta.json)..."
python data/gen_benchmark/visa.py

echo "Benchmark metadata generated."
echo ""

# ============================================================================
# Step 6: Install Additional Dependencies
# ============================================================================
echo "[Step 6/7] Installing additional dependencies..."

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
# Step 7: Setup Weights & Biases (Optional)
# ============================================================================
echo "[Step 7/7] Setting up Weights & Biases (optional)..."
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
    echo "  python run.py -c configs/rdpp_noising/rdpp_noising_256_100e.py wandb.enable=False"
fi

echo ""

# ============================================================================
# Verify Installation
# ============================================================================
echo "============================================================================"
echo "Verifying installation..."
echo "============================================================================"
echo ""

# Check if data exists
if [ -d "data/visa" ] && [ "$(ls -A data/visa 2>/dev/null)" ]; then
    echo "✓ ViSA dataset: OK"
else
    echo "✗ ViSA dataset: NOT FOUND or empty"
fi

# Check if split_csv exists
if [ -f "data/visa/split_csv/1cls.csv" ]; then
    echo "✓ ViSA split_csv/1cls.csv: OK"
else
    echo "✗ ViSA split_csv/1cls.csv: NOT FOUND"
fi

# Check if meta.json exists
if [ -f "data/visa/meta.json" ]; then
    echo "✓ Benchmark metadata (data/visa/meta.json): OK"
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
python -c "import pandas; print('✓ pandas: OK')" || echo "✗ pandas: FAILED"
python -c "from perlin_numpy import generate_perlin_noise_2d; print('✓ perlin-numpy: OK')" || echo "✗ perlin-numpy: FAILED"
python -c "import geomloss; print('✓ geomloss: OK')" || echo "✗ geomloss: FAILED"
python -c "import faiss; print('✓ faiss: OK')" || echo "✗ faiss: FAILED"

echo ""
echo "============================================================================"
echo "Setup Complete (ViSA)!"
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
echo "3. Run training on ViSA (config already points to data/visa):"
echo "   python run.py -c configs/rdpp_noising/rdpp_noising_256_100e.py"
echo ""
echo "4. Or use your experiment runner with ViSA config:"
echo "   CONFIG_FILE=configs/rdpp_noising/rdpp_noising_256_100e.py ./run_all_rdpp_experiments.sh"
echo ""
echo "For more information, see: data/README.md (VisA section)"
echo ""
echo "Finished at: $(date)"
echo "============================================================================"
