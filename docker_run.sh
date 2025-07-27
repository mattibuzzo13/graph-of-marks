#!/usr/bin/env bash
set -e

# Nome dell'immagine costruita in precedenza
IMAGE_NAME="gom"
HF_TOKEN="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"
HOST_HF_CACHE="$HOME/.cache/huggingface"
mkdir -p "$HOST_HF_CACHE"

# Usa GPU 2 direttamente per debugging (senza SLURM)
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# Usa la variabile CUDA_VISIBLE_DEVICES assegnata da SLURM
GPU_FLAG="--gpus device=$CUDA_VISIBLE_DEVICES"

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64

echo "Avvio del container Docker con GPU flag: $GPU_FLAG"

docker run --rm ${GPU_FLAG} \
  -v "$(pwd)":/workdir \
  "$IMAGE_NAME" \
  python -c "
try:
    from src.image_preprocessor import ImageGraphPreprocessor
    print('Import OK')
except Exception as e:
    print(f'Import failed: {e}')
    import traceback
    traceback.print_exc()
"