#!/bin/bash

# ===============================================================================
# download_coco.sh
# Downloads the COCO dataset (images and annotations)
# ===============================================================================

# Get output directory from parameter, default to "coco" if not provided
OUTPUT_DIR="${1:-coco}"

echo "Downloading COCO dataset to $OUTPUT_DIR"

# Create directories
mkdir -p "$OUTPUT_DIR/images"
cd "$OUTPUT_DIR/images"

# Download image datasets
echo "Downloading training images (train2017)..."
wget -c http://images.cocodataset.org/zips/train2017.zip
echo "Downloading validation images (val2017)..."
wget -c http://images.cocodataset.org/zips/val2017.zip
echo "Downloading test images (test2017)..."
wget -c http://images.cocodataset.org/zips/test2017.zip

# Unzip the downloaded files
echo "Extracting image archives..."
unzip -q train2017.zip
unzip -q val2017.zip
unzip -q test2017.zip

# Remove zip files to conserve space
echo "Cleaning up image archives..."
rm train2017.zip
rm val2017.zip
rm test2017.zip

# Navigate back to the dataset directory
cd ..

# Download annotation files
echo "Downloading annotation files..."
wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip

# Unzip annotation files
echo "Extracting annotation archives..."
unzip -q annotations_trainval2017.zip

# Remove zip files to conserve space
echo "Cleaning up annotation archives..."
rm annotations_trainval2017.zip

echo "COCO dataset download completed successfully"
