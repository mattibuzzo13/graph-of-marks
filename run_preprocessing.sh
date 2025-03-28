#!/bin/bash

# Default values
INPUT_PATH=""
OUTPUT_FOLDER="output_images"
DETECTORS="owlvit,yolov8,detectron2"
RELATIONSHIP_TYPE="all"
MAX_RELATIONS=8

# Help function
function show_help {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -i, --input      Input path (file or directory) [required]"
    echo "  -o, --output     Output folder [default: output_images]"
    echo "  -d, --detectors  Comma-separated list of detectors to use [default: owlvit,yolov8,detectron2]"
    echo "                   Available detectors: owlvit, yolov8, detectron2"
    echo "  -r, --relations  Relationship types to extract [default: all]"
    echo "                   Options: all, above, below, left_of, right_of"
    echo "  -m, --max        Maximum number of relationships to extract [default: 8]"
    echo "  -h, --help       Show this help message"
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -i|--input)
            INPUT_PATH="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_FOLDER="$2"
            shift 2
            ;;
        -d|--detectors)
            DETECTORS="$2"
            shift 2
            ;;
        -r|--relations)
            RELATIONSHIP_TYPE="$2"
            shift 2
            ;;
        -m|--max)
            MAX_RELATIONS="$2"
            shift 2
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

# Check if input path is provided
if [ -z "$INPUT_PATH" ]; then
    echo "Error: Input path is required."
    show_help
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_FOLDER"

# Convert detectors to Python list format
DETECTORS_LIST=$(echo "$DETECTORS" | sed 's/,/", "/g')
DETECTORS_LIST="[\"$DETECTORS_LIST\"]"

# Run the Python script
python -m src.image_graph_preprocessor \
    --input_path "$INPUT_PATH" \
    --output_folder "$OUTPUT_FOLDER" \
    --detectors "$DETECTORS_LIST" \
    --relationship_type "$RELATIONSHIP_TYPE" \
    --max_relations "$MAX_RELATIONS"

echo "Processing completed. Results saved to $OUTPUT_FOLDER"
