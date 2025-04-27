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
docker run --rm ${GPU_FLAG} -v "$(pwd)":/workdir -v "$(pwd)/images":/input_data -v "$(pwd)/output_images_VQA_rel_filled":/output_images "$IMAGE_NAME" batch_preprocess INPUT_PATH=/input_data OUTPUT_FOLDER=/output_images DETECTORS=owl,yolov8,detectron2 NUM_INSTANCES=1000
# TextVQA
# docker run --rm ${GPU_FLAG} -v "$(pwd)":/workdir -v /datasets/VisualQA_Datasets/textvqa/images/train_images:/input_data -v "$(pwd)/output_images_TextVQA":/output_images "$IMAGE_NAME" batch_preprocess INPUT_PATH=/input_data OUTPUT_FOLDER=/output_images DETECTORS=owl,yolov8,detectron2
# GQA
# docker run --rm ${GPU_FLAG} -v "$(pwd)":/workdir -v /datasets/VisualQA_Datasets/gqa/images/images:/input_data -v "$(pwd)/output_images_GQA":/output_images "$IMAGE_NAME" batch_preprocess INPUT_PATH=/input_data OUTPUT_FOLDER=/output_images DETECTORS=owl,yolov8,detectron2

# Avvia il container per inference

#docker run --rm ${GPU_FLAG} \
#    -e CUDA_LAUNCH_BLOCKING=1 \
#    -e HF_TOKEN=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 \
#    -v "$(pwd)":/workdir \
#    -v "$(pwd)/vqa_data_normalized_filtered_output.json":/workdir/vqa_data_normalized_filtered_output.json \
#    -v "$(pwd)/Makefile":/workdir/Makefile \
#    -v "$(pwd)/VQA_SoM":/input_images \
#    "$IMAGE_NAME" run_vqa VQA_INPUT_FILE=vqa_data_normalized_filtered_output.json \
#    USE_VLLM=false VQA_OUTPUT_FILE=vqa_result_qwen_SoM.json IMAGE_DIR=/input_images \
#    MODEL_NAME=Qwen/Qwen2.5-VL-7B-Instruct TEMPERATURE=0.7 MAX_LENGTH=32 TOP_P=0.9 \
#    MAX_IMAGES=1000 MAX_QUESTIONS_PER_IMAGE=1 

# docker run --rm ${GPU_FLAG} \
#     -e CUDA_LAUNCH_BLOCKING=1 \
#     -e HF_TOKEN=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 \
#     -v "$(pwd)":/workdir \
#     -v "$(pwd)/vqav2_train2014_filtered.json":/workdir/vqav2_train2014_filtered.json \
#     -v "$(pwd)/Makefile":/workdir/Makefile \
#     -v "$(pwd)/data":/input_images \
#     "$IMAGE_NAME" run_vqa VQA_INPUT_FILE=vqav2_train2014_filtered.json \
#     USE_VLLM=false VQA_OUTPUT_FILE=vqa_result_llava_GoM_num_rel.json IMAGE_DIR=/input_images \
#     MODEL_NAME=llava-hf/llava-1.5-7b-hf TEMPERATURE=0.7 MAX_LENGTH=32 TOP_P=0.9 \
#     MAX_IMAGES=1000 MAX_QUESTIONS_PER_IMAGE=1



echo "Container Docker terminato."
