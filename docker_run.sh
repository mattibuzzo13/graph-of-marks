#!/usr/bin/env bash
set -e

# Nome dell'immagine costruita in precedenza
IMAGE_NAME="gom"


echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# Usa la variabile CUDA_VISIBLE_DEVICES assegnata da SLURM
GPU_FLAG="--gpus device=$CUDA_VISIBLE_DEVICES"

echo "Avvio del container Docker con GPU flag: $GPU_FLAG"

# Avvia il container in modo 'usa e getta' (--rm) e lancia batch_preprocess con i parametri specificati
# COCO
# docker run --rm ${GPU_FLAG} -v "$(pwd)":/workdir -v /datasets/VisualQA_Datasets/coco/images/train2017:/input_data -v "$(pwd)/output_images_COCO":/output_images "$IMAGE_NAME" batch_preprocess INPUT_PATH=/input_data OUTPUT_FOLDER=/output_images DETECTORS=owl,yolov8,detectron2
# RefCOCOg
# docker run --rm ${GPU_FLAG} -v "$(pwd)":/workdir -v /datasets/VisualQA_Datasets/refcoco/images/train2014:/input_data -v "$(pwd)/output_images_RefCOCOg":/output_images "$IMAGE_NAME" batch_preprocess INPUT_PATH=/input_data OUTPUT_FOLDER=/output_images DETECTORS=owl,yolov8,detectron2
# VQA
# docker run --rm ${GPU_FLAG} -v "$(pwd)":/workdir -v /datasets/VisualQA_Datasets/vqa/images/train2014:/input_data -v "$(pwd)/output_images_VQA":/output_images "$IMAGE_NAME" batch_preprocess INPUT_PATH=/input_data OUTPUT_FOLDER=/output_images DETECTORS=owl,yolov8,detectron2
# TextVQA
# docker run --rm ${GPU_FLAG} -v "$(pwd)":/workdir -v /datasets/VisualQA_Datasets/textvqa/images/train_images:/input_data -v "$(pwd)/output_images_TextVQA":/output_images "$IMAGE_NAME" batch_preprocess INPUT_PATH=/input_data OUTPUT_FOLDER=/output_images DETECTORS=owl,yolov8,detectron2
# GQA
# docker run --rm ${GPU_FLAG} -v "$(pwd)":/workdir -v /datasets/VisualQA_Datasets/gqa/images/images:/input_data -v "$(pwd)/output_images_GQA":/output_images "$IMAGE_NAME" batch_preprocess INPUT_PATH=/input_data OUTPUT_FOLDER=/output_images DETECTORS=owl,yolov8,detectron2

# Avvia il container per inference
docker run --rm ${GPU_FLAG} -v "$(pwd)":/workdir -v "$(pwd)/vqav2_train2014_streamed.json":/workdir/vqav2_train2014_streamed.json -v "$(pwd)/Makefile":/workdir/Makefile -v "$(pwd)/output_images_VQA":/input_images -v "$(pwd)/vqa_results.json":/workdir/vqa_results.json "$IMAGE_NAME" run_vqa VQA_INPUT_FILE=vqav2_train2014_streamed.json VQA_OUTPUT_FILE=vqa_results.json IMAGE_DIR=/input_images


echo "Container Docker terminato."
