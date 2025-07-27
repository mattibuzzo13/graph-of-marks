#!/usr/bin/env bash
set -e

# Configurazione base
IMAGE_NAME="gom"
HF_TOKEN="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"
HOST_HF_CACHE="$HOME/.cache/huggingface"
GPU_FLAG="--gpus device=$CUDA_VISIBLE_DEVICES"

# Dataset e modello
DATASET="RefCOCOg"
MODEL_NAME="omkarthawakar/LlamaV-o1"
INPUT_IMAGES="/datasets/VisualQA_Datasets/Preprocessing/RefCOCOg/GoM"
OUTPUT_BASE="/datasets/VisualQA_Datasets/Preprocessing/RefCOCOg"

# Parametri di sampling per diversi run
declare -a TEMPERATURES=(0.1 0.3 0.5)
declare -a TOP_P_VALUES=(0.7 0.9 0.95)
declare -a MAX_LENGTHS=(256 512 768)

# Seeds per riproducibilità
declare -a SEEDS=(42 123 456)

echo "=== STARTING MULTIPLE EXPERIMENTS ==="
echo "Model: $MODEL_NAME"
echo "Dataset: $DATASET"
echo "Number of runs: ${#TEMPERATURES[@]} x ${#TOP_P_VALUES[@]} configurations"

# Loop per ogni combinazione di parametri
for i in "${!TEMPERATURES[@]}"; do
    TEMP="${TEMPERATURES[$i]}"
    TOP_P="${TOP_P_VALUES[$i]}"
    MAX_LEN="${MAX_LENGTHS[$i]}"
    SEED="${SEEDS[$i]}"
    
    # Nome univoco per questo run
    RUN_NAME="${DATASET}_${MODEL_NAME//\//_}_temp${TEMP}_top${TOP_P}_len${MAX_LEN}_seed${SEED}"
    OUTPUT_DIR="${OUTPUT_BASE}/${RUN_NAME}_output"
    OUTPUT_FILE="/workdir/${RUN_NAME}_results.json"
    
    echo ""
    echo "=== RUN $((i+1))/${#TEMPERATURES[@]}: $RUN_NAME ==="
    echo "Temperature: $TEMP, Top-p: $TOP_P, Max length: $MAX_LEN, Seed: $SEED"
    
    # Esegui il container
    docker run --rm ${GPU_FLAG} --memory=30g \
      -e CUDA_LAUNCH_BLOCKING=1 \
      -e HF_HOME=/root/.cache/huggingface \
      -e TRANSFORMERS_CACHE=/root/.cache/huggingface/transformers \
      -e HF_TOKEN=$HF_TOKEN \
      -e PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64 \
      -e PYTHONHASHSEED=$SEED \
      -v "$(pwd)":/workdir \
      -v "$INPUT_IMAGES":/input_images \
      -v "$HOST_HF_CACHE":/root/.cache/huggingface \
      "$IMAGE_NAME" \
      run_vqa \
        VQA_INPUT_FILE=/workdir/$DATASET.json \
        MAX_IMAGES=-1 \
        MAX_QUESTIONS_PER_IMAGE=-1 \
        PREPROC_FOLDER=/input_images \
        VQA_OUTPUT_FILE="$OUTPUT_FILE" \
        USE_VLLM=false \
        MODEL_NAME="$MODEL_NAME" \
        TEMPERATURE="$TEMP" \
        MAX_LENGTH="$MAX_LEN" \
        TOP_P="$TOP_P" \
        SKIP_PREPROCESSING=true

#  run_vqa \
#    VQA_INPUT_FILE=/workdir/VQAV1.json \
#    MAX_IMAGES=-1 \
#    MAX_QUESTIONS_PER_IMAGE=-1 \
#    PREPROC_FOLDER=/input_images \
#    VQA_OUTPUT_FILE=/workdir/VQAV1_LlamaV-o1_GoM_relation_labeled.json \
#    USE_VLLM=false \
#    MODEL_NAME=omkarthawakar/LlamaV-o1 \
#    SKIP_PREPROCESSING=true
    
    echo "Run $((i+1)) completed. Results saved to: $OUTPUT_FILE"
done

echo ""
echo "=== ALL EXPERIMENTS COMPLETED ==="
echo "Results files:"
ls -la *_results.json