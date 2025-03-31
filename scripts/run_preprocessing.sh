#!/bin/bash

# Default values
INPUT_PATH=""
OUTPUT_FOLDER="output_images"
DETECTORS="owlvit,yolov8,detectron2"
RELATIONSHIP_TYPE="all"
MAX_RELATIONS=8
START_INDEX=-1
END_INDEX=-1
NUM_INSTANCES=-1

# Detection thresholds
OWL_THRESHOLD=0.15
YOLO_THRESHOLD=0.3
DETECTRON_THRESHOLD=0.3

# NMS parameters
LABEL_NMS_THRESHOLD=0.5
SEG_IOU_THRESHOLD=0.8

# Relationship inference parameters
OVERLAP_THRESH=0.3
MARGIN=20
MIN_DISTANCE=90
MAX_DISTANCE=20000

# SAM parameters
POINTS_PER_SIDE=32
PRED_IOU_THRESH=0.9
STABILITY_SCORE_THRESH=0.95
MIN_MASK_REGION_AREA=100

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
    echo "  -s, --start      Start index (0-based) for processing instances [default: process all]"
    echo "  -e, --end        End index (inclusive) for processing instances [default: process all]"
    echo "  -n, --num        Absolute number of instances to process [default: process all]"
    echo ""
    echo "Detection parameters:"
    echo "  --owl-thresh     Confidence threshold for OWL-ViT detector [default: 0.15]"
    echo "  --yolo-thresh    Confidence threshold for YOLOv8 detector [default: 0.3]"
    echo "  --d2-thresh      Confidence threshold for Detectron2 detector [default: 0.3]"
    echo ""
    echo "NMS parameters:"
    echo "  --label-nms-thresh   IoU threshold for label-based NMS [default: 0.5]"
    echo "  --seg-iou-thresh     IoU threshold for segmentation filtering [default: 0.8]"
    echo ""
    echo "Relationship parameters:"
    echo "  --overlap-thresh     Horizontal overlap threshold [default: 0.3]"
    echo "  --margin             Margin in pixels [default: 20]"
    echo "  --min-dist           Minimum distance between centers [default: 90]"
    echo "  --max-dist           Maximum distance between centers [default: 20000]"
    echo ""
    echo "SAM parameters:"
    echo "  --points-per-side    Points per side for SAM [default: 32]"
    echo "  --pred-iou-thresh    Predicted IoU threshold for SAM [default: 0.9]"
    echo "  --stability-thresh   Stability score threshold for SAM [default: 0.95]"
    echo "  --min-mask-area      Minimum mask region area for SAM [default: 100]"
    echo ""
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
        -s|--start)
            START_INDEX="$2"
            shift 2
            ;;
        -e|--end)
            END_INDEX="$2"
            shift 2
            ;;
        -n|--num)
            NUM_INSTANCES="$2"
            shift 2
            ;;
        --owl-thresh)
            OWL_THRESHOLD="$2"
            shift 2
            ;;
        --yolo-thresh)
            YOLO_THRESHOLD="$2"
            shift 2
            ;;
        --d2-thresh)
            DETECTRON_THRESHOLD="$2"
            shift 2
            ;;
        --label-nms-thresh)
            LABEL_NMS_THRESHOLD="$2"
            shift 2
            ;;
        --seg-iou-thresh)
            SEG_IOU_THRESHOLD="$2"
            shift 2
            ;;
        --overlap-thresh)
            OVERLAP_THRESH="$2"
            shift 2
            ;;
        --margin)
            MARGIN="$2"
            shift 2
            ;;
        --min-dist)
            MIN_DISTANCE="$2"
            shift 2
            ;;
        --max-dist)
            MAX_DISTANCE="$2"
            shift 2
            ;;
        --points-per-side)
            POINTS_PER_SIDE="$2"
            shift 2
            ;;
        --pred-iou-thresh)
            PRED_IOU_THRESH="$2"
            shift 2
            ;;
        --stability-thresh)
            STABILITY_SCORE_THRESH="$2"
            shift 2
            ;;
        --min-mask-area)
            MIN_MASK_REGION_AREA="$2"
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

# Build the command with optional parameters
CMD="python -m src.image_graph_preprocessor \
    --input_path \"$INPUT_PATH\" \
    --output_folder \"$OUTPUT_FOLDER\" \
    --detectors \"$DETECTORS_LIST\" \
    --relationship_type \"$RELATIONSHIP_TYPE\" \
    --max_relations \"$MAX_RELATIONS\" \
    --owl_threshold \"$OWL_THRESHOLD\" \
    --yolo_threshold \"$YOLO_THRESHOLD\" \
    --detectron_threshold \"$DETECTRON_THRESHOLD\" \
    --label_nms_threshold \"$LABEL_NMS_THRESHOLD\" \
    --seg_iou_threshold \"$SEG_IOU_THRESHOLD\" \
    --overlap_thresh \"$OVERLAP_THRESH\" \
    --margin \"$MARGIN\" \
    --min_distance \"$MIN_DISTANCE\" \
    --max_distance \"$MAX_DISTANCE\" \
    --points_per_side \"$POINTS_PER_SIDE\" \
    --pred_iou_thresh \"$PRED_IOU_THRESH\" \
    --stability_score_thresh \"$STABILITY_SCORE_THRESH\" \
    --min_mask_region_area \"$MIN_MASK_REGION_AREA\""

# Add indexing parameters if provided
if [ "$START_INDEX" -ge 0 ]; then
    CMD="$CMD --start_index $START_INDEX"
fi

if [ "$END_INDEX" -ge 0 ]; then
    CMD="$CMD --end_index $END_INDEX"
fi

if [ "$NUM_INSTANCES" -ge 0 ]; then
    CMD="$CMD --num_instances $NUM_INSTANCES"
fi

# Run the command
eval "$CMD"

echo "Processing completed. Results saved to $OUTPUT_FOLDER"
