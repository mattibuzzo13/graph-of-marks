#!/bin/bash

# ===============================================================================
# download_refcoco.sh
# Downloads the RefCOCO dataset (COCO 2014 images and RefCOCO annotations)
# ===============================================================================

# Get output directory from parameter, default to "refcoco" if not provided
OUTPUT_DIR="${1:-refcoco}"

echo "Downloading RefCOCO dataset to $OUTPUT_DIR"

# Create directories
mkdir -p "$OUTPUT_DIR/images"
mkdir -p "$OUTPUT_DIR/annotations"

# Navigate to the images directory
cd "$OUTPUT_DIR/images"

# Download COCO 2014 images (RefCOCO uses COCO 2014 images)
echo "Downloading COCO 2014 training images (train2014)..."
wget -c http://images.cocodataset.org/zips/train2014.zip
echo "Downloading COCO 2014 validation images (val2014)..."
wget -c http://images.cocodataset.org/zips/val2014.zip

# Unzip the downloaded files
echo "Extracting image archives..."
unzip -q train2014.zip
unzip -q val2014.zip

# Remove zip files to conserve space
echo "Cleaning up image archives..."
rm train2014.zip
rm val2014.zip

# Navigate to the annotations directory
cd "../annotations"

# Download RefCOCO annotations
echo "Downloading RefCOCO annotations from UNC..."
# Note: The annotations are hosted on external repositories; ensure you have access.
wget -c http://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco.zip

# Unzip the annotations
echo "Extracting annotation archives..."
unzip -q refcoco.zip

# Remove zip files to conserve space
echo "Cleaning up annotation archives..."
rm refcoco.zip

echo "RefCOCO dataset download completed successfully"
