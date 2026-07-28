#!/usr/bin/env bash
# Setup script for the crowd safety testbed.
# Installs Python deps and checks for GPU availability.

set -e

PIP_INSTALL="pip install --break-system-packages"

echo "== Detecting NVIDIA GPU =="
CUDA_TAG=""
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    echo "nvidia-smi found — GPU detected:"
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
    CUDA_TAG="${CUDA_TAG:-cu121}"
else
    echo "No NVIDIA GPU / driver detected (nvidia-smi not found or failed)."
    echo "Proceeding with CPU-only install. If this machine DOES have a GPU,"
    echo "install/fix the NVIDIA driver first, then re-run this script."
fi

echo ""
if [ -n "$CUDA_TAG" ]; then
    echo "== Installing torch/torchvision from the CUDA ($CUDA_TAG) index =="
    $PIP_INSTALL torch torchvision --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"
else
    echo "== Installing CPU torch/torchvision =="
    $PIP_INSTALL torch torchvision
fi

echo ""
echo "== Installing remaining Python dependencies =="
$PIP_INSTALL -r requirements.txt

echo ""
echo "== Checking for ffmpeg (required by yt-dlp for format merging) =="
if ! command -v ffmpeg &> /dev/null; then
    echo "ffmpeg not found. Install it via your system package manager, e.g.:"
    echo "  sudo apt-get install ffmpeg"
else
    echo "ffmpeg found: $(ffmpeg -version | head -n1)"
fi

echo ""
echo "== Verifying device availability =="
python3 -m pipeline.device

if [ -n "$CUDA_TAG" ]; then
    python3 -c "
import torch, sys
if not torch.cuda.is_available():
    print()
    print('WARNING: an NVIDIA GPU was detected by nvidia-smi, but torch.cuda.is_available()')
    print('is False after install. The CUDA build/driver versions likely mismatch.')
    print('Check nvidia-smi for your driver/CUDA version and re-run with e.g.:')
    print('  CUDA_TAG=cu118 ./scripts/setup.sh')
    sys.exit(1)
"
fi

echo ""
echo "Setup complete."