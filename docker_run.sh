#!/usr/bin/env bash
set -e

# Pre-built Docker image name
IMAGE_NAME="gom"
# Hugging Face token (environment overrideable)
HF_TOKEN="Insert your Huggingface token"
# Host-side HF cache mount
HOST_HF_CACHE="$HOME/.cache/huggingface"
mkdir -p "$HOST_HF_CACHE"

# Show CUDA device binding (for local debug; outside SLURM)
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# Use SLURM-provided CUDA_VISIBLE_DEVICES inside the container
GPU_FLAG="--gpus device=$CUDA_VISIBLE_DEVICES"

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64

echo "Starting Docker container with GPU flag: $GPU_FLAG"
echo "Using modular igp pipeline"

#------------------------------------------------------------------------------
# Examples for the modular system
#------------------------------------------------------------------------------

# PREPROCESSING EXAMPLES (image_preprocessor.py + igp/)

# Single-image preprocessing with question filtering
#docker run --rm ${GPU_FLAG} \
#  -v "$(pwd)":/workdir \
#  -v "$(pwd)/images/test.jpg":/workdir/test.jpg \
#  -v "$(pwd)/output_single":/output_images \
#  -w /workdir \
#  "$IMAGE_NAME" \
#  preprocess \
#    INPUT_PATH=test.jpg \
#    QUESTION="What objects are in this image?" \
#    ENABLE_Q_FILTER=true \
#    OUTPUT_FOLDER=/output_images

# Batch preprocessing with multiple detectors
#docker run --rm ${GPU_FLAG} \
#  -v "$(pwd)":/workdir \
#  -v "/datasets/VisualQA_Datasets/coco/images/train2017":/input_data \
#  -v "$(pwd)/output_images_COCO":/output_images \
#  "$IMAGE_NAME" \
#  batch_preprocess \
#    INPUT_PATH=/input_data \
#    OUTPUT_FOLDER=/output_images \
#    DETECTORS=owlvit,yolov8,detectron2 \
#    NUM_INSTANCES=1000

# VQA INFERENCE EXAMPLES (vqa.py + igp/)

# Full VQA pipeline with preprocessing
#docker run --rm ${GPU_FLAG} --memory=30g \
#  -e CUDA_LAUNCH_BLOCKING=1 \
#  -e HF_TOKEN=$HF_TOKEN \
#  -e HF_HOME=/root/.cache/huggingface \
#  -e TRANSFORMERS_CACHE=/root/.cache/huggingface/transformers \
#  -v "$(pwd)":/workdir \
#  -v "/datasets/VisualQA_Datasets/Preprocessing/VQAV1/GoM":/input_images \
#  -v "$HOST_HF_CACHE":/root/.cache/huggingface \
#  "$IMAGE_NAME" \
#  run_vqa \
#    VQA_INPUT_FILE=/workdir/VQAV1.json \
#    IMAGE_DIR=/input_images \
#    VQA_OUTPUT_FILE=/workdir/VQAV1_results.json \
#    MODEL_NAME=llava-hf/llava-1.5-7b-hf \
#    USE_VLLM=false \
#    MAX_IMAGES=-1 \
#    MAX_QUESTIONS_PER_IMAGE=-1 \
#    SKIP_PREPROCESSING=true

# Preprocessing + VQA inference
docker run --rm ${GPU_FLAG} --memory=30g \
  -e CUDA_LAUNCH_BLOCKING=1 \
  -e HF_HOME=/root/.cache/huggingface \
  -e TRANSFORMERS_CACHE=/root/.cache/huggingface/transformers \
  -e HF_TOKEN=$HF_TOKEN \
  -e PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64 \
  -v "$(pwd)":/workdir \
  -v "/datasets/VisualQA_Datasets/Preprocessing/VQAV2/original_VQAV2/vqav2_imgs_1000":/input_images \
  -v "/datasets/VisualQA_Datasets/VQAV2/GoM_rels_num_related":/output_preprocessed \
  -v "$HOST_HF_CACHE":/root/.cache/huggingface \
  "$IMAGE_NAME" \
  run_vqa \
    VQA_INPUT_FILE=/workdir/VQAV2.json \
    IMAGE_DIR=/input_images \
    MAX_IMAGES=3 \
    MAX_QUESTIONS_PER_IMAGE=-1 \
    PREPROC_FOLDER=/output_preprocessed \
    OUTPUT_FOLDER=/output_preprocessed \
    SKIP_PREPROCESSING=false \
    ENABLE_Q_FILTER=true \
    PREPROCESS_ONLY=false \
    SAVE_IMAGE_ONLY=true \
    SKIP_GRAPH=true \
    SKIP_PROMPT=true \
    DETECTORS=owlvit,yolov8,detectron2 \
    LABEL_MODE=original \
    DISPLAY_RELATION_LABELS=true \
    DISPLAY_RELATIONSHIPS=true \
    DISPLAY_LABELS=true \
    NO_LEGEND=true \
    VQA_OUTPUT_FILE=/output_preprocessed/results.json \
    MODEL_NAME=Qwen/Qwen2.5-VL-7B-Instruct \
    USE_VLLM=false \
    TEMPERATURE=0.3 \
    MAX_LENGTH=512 \
    TOP_P=0.9 \
    PROMPT_TEMPLATE="Answer with only one word.\nQuestion: {question}\nAnswer:"

# VQA inference only (skip preprocessing)
#docker run --rm ${GPU_FLAG} --memory=30g \
#  -e CUDA_LAUNCH_BLOCKING=1 \
#  -e HF_TOKEN=$HF_TOKEN \
#  -e HF_HOME=/root/.cache/huggingface \
#  -e TRANSFORMERS_CACHE=/root/.cache/huggingface/transformers \
#  -v "$(pwd)":/workdir \
#  -v "/datasets/VisualQA_Datasets/Preprocessing/VQAV1/GoM":/input_images \
#  -v "$HOST_HF_CACHE":/root/.cache/huggingface \
#  "$IMAGE_NAME" \
#  run_vqa \
#    VQA_INPUT_FILE=/workdir/VQAV1.json \
#    IMAGE_DIR=/input_images \
#    VQA_OUTPUT_FILE=/workdir/VQAV1_results.json \
#    MODEL_NAME=llava-hf/llava-1.5-7b-hf \
#    USE_VLLM=false \
#    MAX_IMAGES=-1 \
#    MAX_QUESTIONS_PER_IMAGE=-1 \
#    SKIP_PREPROCESSING=true

# Preprocessing only (no inference)
#docker run --rm ${GPU_FLAG} --memory=30g \
#  -e CUDA_LAUNCH_BLOCKING=1 \
#  -e HF_HOME=/root/.cache/huggingface \
#  -e TRANSFORMERS_CACHE=/root/.cache/huggingface/transformers \
#  -e HF_TOKEN=$HF_TOKEN \
#  -e PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64 \
#  -v "$(pwd)":/workdir \
#  -v "/datasets/VisualQA_Datasets/Preprocessing/VQAV2/original_VQAV2/vqav2_imgs_1000":/input_images \
#  -v "/datasets/VisualQA_Datasets/VQAV2/GoM_rels_num":/output_preprocessed \
#  -v "$HOST_HF_CACHE":/root/.cache/huggingface \
#  "$IMAGE_NAME" \
#  run_vqa \
#    VQA_INPUT_FILE=/workdir/VQAV2.json \
#    IMAGE_DIR=/input_images \
#    MAX_IMAGES=3 \
#    MAX_QUESTIONS_PER_IMAGE=-1 \
#    PREPROC_FOLDER=/output_preprocessed \
#    OUTPUT_FOLDER=/output_preprocessed \
#    SKIP_PREPROCESSING=false \
#    ENABLE_Q_FILTER=true \
#    PREPROCESS_ONLY=true \
#    SAVE_IMAGE_ONLY=true \
#    SKIP_GRAPH=true \
#    SKIP_PROMPT=true \
#    DETECTORS=owlvit,yolov8,detectron2 \
#    LABEL_MODE=numeric \
#    DISPLAY_RELATION_LABELS=true \
#    DISPLAY_RELATIONSHIPS=true \
#    DISPLAY_LABELS=true \
#    NO_LEGEND=true

# VQA with scene graph input
#docker run --rm ${GPU_FLAG} --memory=30g \
#  -e CUDA_LAUNCH_BLOCKING=1 \
#  -e HF_HOME=/root/.cache/huggingface \
#  -e TRANSFORMERS_CACHE=/root/.cache/huggingface/transformers \
#  -e HF_TOKEN=$HF_TOKEN \
#  -e PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64 \
#  -v "$(pwd)":/workdir \
#  -v "/datasets/VisualQA_Datasets/VQAV2/original_VQAV2/vqav2_imgs_1000":/input_images \
#  -v "/datasets/VisualQA_Datasets/VQAV2/preprocessing_output":/output_preprocessed \
#  -v "$HOST_HF_CACHE":/root/.cache/huggingface \
#  "$IMAGE_NAME" \
#  run_vqa \
#    VQA_INPUT_FILE=/workdir/VQAV2.json \
#    IMAGE_DIR=/input_images \
#    OUTPUT_FOLDER=/output_preprocessed \
#    VQA_OUTPUT_FILE=/workdir/VQAV2_results_with_scene_graph.json \
#    MODEL_NAME=google/gemma-2-2b-it \
#    USE_VLLM=false \
#    TEMPERATURE=0.3 \
#    MAX_LENGTH=512 \
#    TOP_P=0.9 \
#    MAX_IMAGES=100 \
#    MAX_QUESTIONS_PER_IMAGE=1 \
#    PREPROC_FOLDER=/output_preprocessed \
#    SKIP_PREPROCESSING=false \
#    ENABLE_Q_FILTER=true \
#    INCLUDE_SCENE_GRAPH=true

echo "Docker container with modular igp pipeline has completed."
