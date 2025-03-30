#!/bin/bash

# ===============================================================================
# download_vqa.sh
# Downloads the VQA v2 dataset (COCO images and VQA v2 annotations)
# ===============================================================================

# Get output directory from parameter, default to "vqa" if not provided
OUTPUT_DIR="${1:-vqa}"

echo "Downloading VQA dataset to $OUTPUT_DIR"

# Create directories
mkdir -p "$OUTPUT_DIR/images"
mkdir -p "$OUTPUT_DIR/annotations"

# Navigate to the images directory
cd "$OUTPUT_DIR/images"

# Download COCO images (VQA uses COCO images)
echo "Downloading COCO 2014 training images (train2014)..."
wget -c http://images.cocodataset.org/zips/train2014.zip
echo "Downloading COCO 2014 validation images (val2014)..."
wget -c http://images.cocodataset.org/zips/val2014.zip
echo "Downloading COCO 2015 test images (test2015)..."
wget -c http://images.cocodataset.org/zips/test2015.zip

# Unzip the downloaded files
echo "Extracting image archives..."
unzip -q train2014.zip
unzip -q val2014.zip
unzip -q test2015.zip

# Remove zip files to conserve space
echo "Cleaning up image archives..."
rm train2014.zip
rm val2014.zip
rm test2015.zip

# Navigate to the annotations directory
cd "../annotations"

# Download VQA annotations
echo "Downloading VQA v2 annotation files..."
wget -c https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip
wget -c https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip
wget -c https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip
wget -c https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip
wget -c https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Test_mscoco.zip

# Unzip the annotations
echo "Extracting annotation archives..."
unzip -q v2_Annotations_Train_mscoco.zip
unzip -q v2_Annotations_Val_mscoco.zip
unzip -q v2_Questions_Train_mscoco.zip
unzip -q v2_Questions_Val_mscoco.zip
unzip -q v2_Questions_Test_mscoco.zip

# Remove zip files to conserve space
echo "Cleaning up annotation archives..."
rm v2_Annotations_Train_mscoco.zip
rm v2_Annotations_Val_mscoco.zip
rm v2_Questions_Train_mscoco.zip
rm v2_Questions_Val_mscoco.zip
rm v2_Questions_Test_mscoco.zip

echo "VQA dataset download completed successfully"
