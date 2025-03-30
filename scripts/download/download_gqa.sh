#!/bin/bash

# ===============================================================================
# download_gqa.sh
# Downloads the GQA dataset (images and annotations)
# ===============================================================================

# Get output directory from parameter, default to "gqa" if not provided
OUTPUT_DIR="${1:-gqa}"

echo "Downloading GQA dataset to $OUTPUT_DIR"

# Create directories
mkdir -p "$OUTPUT_DIR/images"
mkdir -p "$OUTPUT_DIR/annotations"

# Navigate to the images directory
cd "$OUTPUT_DIR/images"

# Download GQA images
echo "Downloading GQA images (this may take a while)..."
wget -c https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip

# Unzip the downloaded files
echo "Extracting image archives..."
unzip -q images.zip

# Remove zip files to conserve space
echo "Cleaning up image archives..."
rm images.zip

# Navigate to the annotations directory
cd "../annotations"

# Download GQA annotations
echo "Downloading GQA question annotations (v1.2)..."
wget -c https://downloads.cs.stanford.edu/nlp/data/gqa/questions1.2.zip

# Unzip the annotations
echo "Extracting annotation archives..."
unzip -q questions1.2.zip

# Remove zip files to conserve space
echo "Cleaning up annotation archives..."
rm questions1.2.zip

echo "GQA dataset download completed successfully"
