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
wget -c https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_test_annotations.json

# Download OCR annotations
echo "Downloading TextVQA OCR annotations..."
wget -c https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_train_ocr_tokens.json
wget -c https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_val_ocr_tokens.json
wget -c https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_test_ocr_tokens.json

# Go back to images directory
cd "../images"

# Download OpenImages train images (note: this will be a large download)
echo "Downloading TextVQA images from OpenImages (this may take a while)..."
python3 -c '
import json, os, urllib.request, concurrent.futures
from pathlib import Path

# Load annotations to get image IDs
def download_images(split):
    anno_path = Path(f"../annotations/TextVQA_0.5.1_{split}.json")
    if not anno_path.exists():
        print(f"Cannot find {anno_path}")
        return
        
    with open(anno_path) as f:
        data = json.load(f)
    
    # Extract unique image IDs and URLs
    images = {}
    for img_info in data["data"]:
        img_id = img_info["image_id"]
        img_path = img_info["image_path"]
        images[img_id] = img_path
    
    print(f"Found {len(images)} images for {split} split")
    
    # Function to download a single image
    def download_image(img_id, url):
        try:
            img_url = f"https://cs.stanford.edu/people/rak248/{url}"
            dest_path = f"{img_id}.jpg"
            
            if os.path.exists(dest_path):
                return f"Skipping {img_id}, already exists"
                
            urllib.request.urlretrieve(img_url, dest_path)
            return f"Downloaded {img_id}"
        except Exception as e:
            return f"Error downloading {img_id}: {str(e)}"
    
    # Download images in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(download_image, img_id, url) for img_id, url in images.items()]
        for future in concurrent.futures.as_completed(futures):
            print(future.result())

# Download images for each split
for split in ["train", "val", "test"]:
    download_images(split)
'

echo "TextVQA dataset download completed successfully"