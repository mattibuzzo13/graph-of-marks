#!/bin/bash

# ===============================================================================
# download_textvqa.sh
# Downloads the TextVQA dataset (images and annotations)
# ===============================================================================

# Get output directory from parameter, default to "textvqa" if not provided
OUTPUT_DIR="${1:-textvqa}"

echo "Downloading TextVQA dataset to $OUTPUT_DIR"

# Create directories
mkdir -p "$OUTPUT_DIR/images"
mkdir -p "$OUTPUT_DIR/annotations"

# Navigate to the annotations directory
cd "$OUTPUT_DIR/annotations"

# Download TextVQA annotations
echo "Downloading TextVQA question and answer annotations..."
wget -c https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_train.json
wget -c https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_val.json
wget -c https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_test.json

# Download OCR annotations
# echo "Downloading TextVQA OCR annotations..."
# wget -c https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_train_ocr_tokens.json
# wget -c https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_val_ocr_tokens.json
# wget -c https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_test_ocr_tokens.json

# Go back to images directory
cd "../images"
echo "Downloading TextVQA images..."
wget -c https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip
wget -c https://dl.fbaipublicfiles.com/textvqa/images/test_images.zip


echo "Extracting image archives..."
unzip -q train_val_images.zip
unzip -q test_images.zip

# Remove zip files to conserve space
echo "Cleaning up image archives..."
rm train_val_images.zip
rm test_images.zip

echo "TextVQA dataset download completed successfully"