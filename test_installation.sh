#!/bin/bash
# Test installation script for Graph of Marks
# This script creates a clean conda environment and tests the installation

set -e  # Exit on error

echo "=========================================="
echo "Graph of Marks - Installation Test"
echo "=========================================="
echo ""

# Configuration
ENV_NAME="gom_test"
PYTHON_VERSION="3.10"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found. Please install Miniconda or Anaconda first."
    echo "   Download from: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

echo "✅ Conda found: $(conda --version)"
echo ""

# Remove existing environment if it exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "🗑️  Removing existing environment: ${ENV_NAME}"
    conda env remove -n ${ENV_NAME} -y
    echo ""
fi

# Create new conda environment
echo "🔧 Creating new conda environment: ${ENV_NAME} (Python ${PYTHON_VERSION})"
conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y
echo ""

# Activate environment
echo "🔄 Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ${ENV_NAME}
echo ""

# Install PyTorch with CUDA support (adjust for your CUDA version)
echo "🔥 Installing PyTorch with CUDA 11.8..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
echo ""

# Install the package
echo "📦 Installing Graph of Marks from source..."
pip install -e .
echo ""

# Test basic import
echo "🧪 Testing basic import..."
python -c "
import gom
print(f'✅ Successfully imported gom version {gom.__version__}')
print(f'   Available classes: {list(gom.__all__)}')
"
echo ""

# Test PyTorch and CUDA
echo "🔬 Testing PyTorch and CUDA..."
python -c "
import torch
print(f'✅ PyTorch version: {torch.__version__}')
print(f'   CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'   CUDA version: {torch.version.cuda}')
    print(f'   GPU device: {torch.cuda.get_device_name(0)}')
else:
    print('   ⚠️  CUDA not available (CPU mode only)')
"
echo ""

# Test high-level API
echo "🎯 Testing high-level API..."
python -c "
from gom import GraphOfMarks
import numpy as np
from PIL import Image
import tempfile
import os

# Create a test image
test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
    test_path = f.name
    Image.fromarray(test_image).save(test_path)

try:
    # Initialize pipeline (this will test model loading)
    print('   Creating GraphOfMarks pipeline...')
    gom = GraphOfMarks(
        detectors=['yolov8'],  # Use single detector for quick test
        output_folder='test_output'
    )
    print('   ✅ Pipeline created successfully!')

    # Note: Actual processing requires model downloads
    # print('   Processing test image...')
    # result = gom.process_image(test_path)
    # print(f'   ✅ Image processed! Detected {len(result[\"detections\"])} objects')

finally:
    # Cleanup
    if os.path.exists(test_path):
        os.remove(test_path)
"
echo ""

# Test CLI entry points
echo "🖥️  Testing CLI entry points..."
if command -v gom-preprocess &> /dev/null; then
    echo "   ✅ gom-preprocess command available"
else
    echo "   ⚠️  gom-preprocess command not found"
fi

if command -v gom-vqa &> /dev/null; then
    echo "   ✅ gom-vqa command available"
else
    echo "   ⚠️  gom-vqa command not found"
fi
echo ""

# Summary
echo "=========================================="
echo "✅ Installation Test Complete!"
echo "=========================================="
echo ""
echo "Environment: ${ENV_NAME}"
echo "Python: $(python --version)"
echo ""
echo "Next steps:"
echo "  1. Activate the environment:"
echo "     conda activate ${ENV_NAME}"
echo ""
echo "  2. Download model checkpoints:"
echo "     bash download_ckpt.sh"
echo ""
echo "  3. Run the demo notebook:"
echo "     jupyter notebook examples/demo_notebook.ipynb"
echo ""
echo "  4. Process your first image:"
echo "     python -c \"from gom import GraphOfMarks; gom = GraphOfMarks(); result = gom.process_image('your_image.jpg')\""
echo ""
echo "To remove this test environment later:"
echo "  conda env remove -n ${ENV_NAME}"
echo ""
