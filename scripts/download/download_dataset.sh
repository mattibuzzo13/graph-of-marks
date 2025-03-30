#!/bin/bash

# ===============================================================================
# download_dataset.sh
# A comprehensive dataset downloader script for Graph-of-Marks project
# 
# Usage: ./download_dataset.sh -d [dataset_name] -o [output_dir]
#
# Parameters:
#   -d, --dataset : Dataset name to download (coco, gqa, refcoco, vqa)
#   -o, --output  : Output directory (optional, defaults to dataset name)
#   -h, --help    : Show this help message
# ===============================================================================

set -e  # Exit on error

# Default values
DATASET=""
OUTPUT_DIR=""
SCRIPT_DIR=$(dirname "$(realpath "$0")")
PROJECT_ROOT=$(realpath "$SCRIPT_DIR/../..")

# Display help message
show_help() {
    echo "Usage: $0 -d [dataset_name] -o [output_dir]"
    echo ""
    echo "A comprehensive dataset downloader for Graph-of-Marks project"
    echo ""
    echo "Parameters:"
    echo "  -d, --dataset  Dataset name to download (coco, gqa, refcoco, vqa)"
    echo "  -o, --output   Output directory (optional, defaults to dataset name)"
    echo "  -h, --help     Show this help message"
    echo ""
    echo "Example:"
    echo "  $0 -d coco                # Download COCO dataset to coco/"
    echo "  $0 -d vqa -o data/vqa     # Download VQA dataset to data/vqa/"
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -d|--dataset)
            DATASET="$2"
            shift
            shift
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift
            shift
            ;;
        -h|--help)
            show_help
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            ;;
    esac
done

# Validate dataset name
if [[ -z "$DATASET" ]]; then
    echo "Error: Dataset name is required"
    show_help
fi

# Convert dataset name to lowercase
DATASET=$(echo "$DATASET" | tr '[:upper:]' '[:lower:]')

# Check if dataset is supported
case $DATASET in
    coco|gqa|refcoco|vqa|textvqa)
        # Valid dataset
        ;;
    *)
        echo "Error: Unsupported dataset '$DATASET'"
        echo "Supported datasets: coco, gqa, refcoco, vqa, textvqa"
        exit 1
        ;;
esac

# Set output directory if not specified
if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="$DATASET"
fi

# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"
cd "$PROJECT_ROOT"  # Change to project root

echo "======================================================================"
echo "Downloading $DATASET dataset to $OUTPUT_DIR"
echo "======================================================================"

# Execute the appropriate download script based on dataset name
case $DATASET in
    coco)
        echo "Downloading COCO dataset..."
        bash "$SCRIPT_DIR/download_coco.sh" "$OUTPUT_DIR"
        ;;
    gqa)
        echo "Downloading GQA dataset..."
        bash "$SCRIPT_DIR/download_gqa.sh" "$OUTPUT_DIR"
        ;;
    refcoco)
        echo "Downloading RefCOCO dataset..."
        bash "$SCRIPT_DIR/download_refcoco.sh" "$OUTPUT_DIR"
        ;;
    vqa)
        echo "Downloading VQA dataset..."
        bash "$SCRIPT_DIR/download_vqa.sh" "$OUTPUT_DIR"
        ;;
    textvqa)
        echo "Downloading TextVQA dataset..."
        bash "$SCRIPT_DIR/download_textvqa.sh" "$OUTPUT_DIR"
        ;;
esac

echo "======================================================================"
echo "Download completed successfully!"
echo "Dataset available at: $OUTPUT_DIR"
echo "======================================================================"