# Makefile for graph-visual-reasoning preprocessing

# Default values
INPUT_PATH ?= 
OUTPUT_FOLDER ?= output_images
DETECTORS ?= owlvit,yolov8,detectron2
RELATIONSHIP_TYPE ?= all
MAX_RELATIONS ?= 10
START_INDEX ?= -1
END_INDEX ?= -1
NUM_INSTANCES ?= -1

# Detection thresholds
OWL_THRESHOLD ?= 0.5
YOLO_THRESHOLD ?= 0.8
DETECTRON_THRESHOLD ?= 0.8

# NMS parameters
LABEL_NMS_THRESHOLD ?= 0.5
SEG_IOU_THRESHOLD ?= 0.8

# Relationship inference parameters
OVERLAP_THRESH ?= 0.3
MARGIN ?= 30
MIN_DISTANCE ?= 60
MAX_DISTANCE ?= 20000

# SAM parameters
POINTS_PER_SIDE ?= 32
PRED_IOU_THRESH ?= 0.9
STABILITY_SCORE_THRESH ?= 0.95
MIN_MASK_REGION_AREA ?= 100

# VQA defaults
VQA_INPUT_FILE ?=
VQA_OUTPUT_FILE ?= vqa_results.json
MODEL_NAME ?= llava-hf/llava-1.5-7b-hf
IMAGE_DIR ?=
MAX_LENGTH ?= 512
TEMPERATURE ?= 0.2
TOP_P ?= 0.9
PROMPT_TEMPLATE ?= 'Answer with only one word.\nQuestion: {question}\nAnswer:'
BATCH_SIZE ?= 1
TENSOR_PARALLEL_SIZE ?= 1
USE_VLLM ?= true
MAX_IMAGES ?= -1
MAX_QUESTIONS_PER_IMAGE ?= -1
# DTYPE ?= float32

# Dataset download defaults
DATASET ?=
DATASET_DIR ?=

# Check if input path is provided
check_input:
ifndef INPUT_PATH
	$(error INPUT_PATH is required. Use make target INPUT_PATH=/path/to/input)
endif

# Main preprocessing target
preprocess: check_input
	./scripts/run_preprocessing.sh \
		--input "$(INPUT_PATH)" \
		--output "$(OUTPUT_FOLDER)" \
		--detectors "$(DETECTORS)" \
		--relations "$(RELATIONSHIP_TYPE)" \
		--max "$(MAX_RELATIONS)" \
		$(if $(filter-out -1,$(START_INDEX)),--start "$(START_INDEX)") \
		$(if $(filter-out -1,$(END_INDEX)),--end "$(END_INDEX)") \
		$(if $(filter-out -1,$(NUM_INSTANCES)),--num "$(NUM_INSTANCES)") \
		--owl-thresh "$(OWL_THRESHOLD)" \
		--yolo-thresh "$(YOLO_THRESHOLD)" \
		--d2-thresh "$(DETECTRON_THRESHOLD)" \
		--label-nms-thresh "$(LABEL_NMS_THRESHOLD)" \
		--seg-iou-thresh "$(SEG_IOU_THRESHOLD)" \
		--overlap-thresh "$(OVERLAP_THRESH)" \
		--margin "$(MARGIN)" \
		--min-dist "$(MIN_DISTANCE)" \
		--max-dist "$(MAX_DISTANCE)" \
		--points-per-side "$(POINTS_PER_SIDE)" \
		--pred-iou-thresh "$(PRED_IOU_THRESH)" \
		--stability-thresh "$(STABILITY_SCORE_THRESH)" \
		--min-mask-area "$(MIN_MASK_REGION_AREA)"

# Specialized preprocessing targets
preprocess_owlvit: check_input
	./scripts/run_preprocessing.sh \
		--input "$(INPUT_PATH)" \
		--output "$(OUTPUT_FOLDER)" \
		--detectors "owlvit" \
		--relations "$(RELATIONSHIP_TYPE)" \
		--max "$(MAX_RELATIONS)" \
		$(if $(filter-out -1,$(START_INDEX)),--start "$(START_INDEX)") \
		$(if $(filter-out -1,$(END_INDEX)),--end "$(END_INDEX)") \
		$(if $(filter-out -1,$(NUM_INSTANCES)),--num "$(NUM_INSTANCES)") \
		--owl-thresh "$(OWL_THRESHOLD)" \
		--label-nms-thresh "$(LABEL_NMS_THRESHOLD)" \
		--seg-iou-thresh "$(SEG_IOU_THRESHOLD)" \
		--overlap-thresh "$(OVERLAP_THRESH)" \
		--margin "$(MARGIN)" \
		--min-dist "$(MIN_DISTANCE)" \
		--max-dist "$(MAX_DISTANCE)" \
		--points-per-side "$(POINTS_PER_SIDE)" \
		--pred-iou-thresh "$(PRED_IOU_THRESH)" \
		--stability-thresh "$(STABILITY_SCORE_THRESH)" \
		--min-mask-area "$(MIN_MASK_REGION_AREA)"

preprocess_yolo: check_input
	./scripts/run_preprocessing.sh \
		--input "$(INPUT_PATH)" \
		--output "$(OUTPUT_FOLDER)" \
		--detectors "yolov8" \
		--relations "$(RELATIONSHIP_TYPE)" \
		--max "$(MAX_RELATIONS)" \
		$(if $(filter-out -1,$(START_INDEX)),--start "$(START_INDEX)") \
		$(if $(filter-out -1,$(END_INDEX)),--end "$(END_INDEX)") \
		$(if $(filter-out -1,$(NUM_INSTANCES)),--num "$(NUM_INSTANCES)") \
		--yolo-thresh "$(YOLO_THRESHOLD)" \
		--label-nms-thresh "$(LABEL_NMS_THRESHOLD)" \
		--seg-iou-thresh "$(SEG_IOU_THRESHOLD)" \
		--overlap-thresh "$(OVERLAP_THRESH)" \
		--margin "$(MARGIN)" \
		--min-dist "$(MIN_DISTANCE)" \
		--max-dist "$(MAX_DISTANCE)" \
		--points-per-side "$(POINTS_PER_SIDE)" \
		--pred-iou-thresh "$(PRED_IOU_THRESH)" \
		--stability-thresh "$(STABILITY_SCORE_THRESH)" \
		--min-mask-area "$(MIN_MASK_REGION_AREA)"

preprocess_detectron2: check_input
	./scripts/run_preprocessing.sh \
		--input "$(INPUT_PATH)" \
		--output "$(OUTPUT_FOLDER)" \
		--detectors "detectron2" \
		--relations "$(RELATIONSHIP_TYPE)" \
		--max "$(MAX_RELATIONS)" \
		$(if $(filter-out -1,$(START_INDEX)),--start "$(START_INDEX)") \
		$(if $(filter-out -1,$(END_INDEX)),--end "$(END_INDEX)") \
		$(if $(filter-out -1,$(NUM_INSTANCES)),--num "$(NUM_INSTANCES)") \
		--d2-thresh "$(DETECTRON_THRESHOLD)" \
		--label-nms-thresh "$(LABEL_NMS_THRESHOLD)" \
		--seg-iou-thresh "$(SEG_IOU_THRESHOLD)" \
		--overlap-thresh "$(OVERLAP_THRESH)" \
		--margin "$(MARGIN)" \
		--min-dist "$(MIN_DISTANCE)" \
		--max-dist "$(MAX_DISTANCE)" \
		--points-per-side "$(POINTS_PER_SIDE)" \
		--pred-iou-thresh "$(PRED_IOU_THRESH)" \
		--stability-thresh "$(STABILITY_SCORE_THRESH)" \
		--min-mask-area "$(MIN_MASK_REGION_AREA)"

# Target for preprocessing a batch of images
batch_preprocess:
	python3 src/image_graph_preprocessor.py \
		--input_path "$(INPUT_PATH)" \
		--output_folder "$(OUTPUT_FOLDER)"

# Visual Question Answering targets
run_vqa: check_vqa_input
	python3 src/qa_generation.py \
		--input_file "$(VQA_INPUT_FILE)" \
		--output_file "$(VQA_OUTPUT_FILE)" \
		--image_dir "$(IMAGE_DIR)" \
		--model_name "$(MODEL_NAME)" \
		--device cuda \
		$(if $(filter true,$(USE_VLLM)),--use_vllm) \
    	--max_length "$(MAX_LENGTH)" \
    	--temperature "$(TEMPERATURE)" \
    	--top_p "$(TOP_P)" \
    	--prompt_template "$(PROMPT_TEMPLATE)" \
    	--batch_size "$(BATCH_SIZE)" \
    	--tensor_parallel_size "$(TENSOR_PARALLEL_SIZE)" \
		--max_images "$(MAX_IMAGES)" \
		--max_questions_per_image "$(MAX_QUESTIONS_PER_IMAGE)" 

check_vqa_input:
ifndef VQA_INPUT_FILE
	$(error VQA_INPUT_FILE is required. Use make run_vqa VQA_INPUT_FILE=/path/to/input.json)
endif

# Dataset download targets
download_dataset: check_dataset
	bash scripts/download/download_dataset.sh -d "$(DATASET)" $(if $(DATASET_DIR),-o "$(DATASET_DIR)")

download_coco:
	make download_dataset DATASET=coco $(if $(DATASET_DIR),DATASET_DIR=$(DATASET_DIR))

download_gqa:
	make download_dataset DATASET=gqa $(if $(DATASET_DIR),DATASET_DIR=$(DATASET_DIR))

download_refcoco:
	make download_dataset DATASET=refcoco $(if $(DATASET_DIR),DATASET_DIR=$(DATASET_DIR))

download_vqa:
	make download_dataset DATASET=vqa $(if $(DATASET_DIR),DATASET_DIR=$(DATASET_DIR))

download_textvqa:
	make download_dataset DATASET=textvqa $(if $(DATASET_DIR),DATASET_DIR=$(DATASET_DIR))

check_dataset:
ifndef DATASET
	$(error DATASET is required. Use make download_dataset DATASET=coco)
endif

# Install dependencies
install_deps:
	python3 -m pip install --no-cache-dir numpy==1.24.4 scipy==1.10.1
	python3 -m pip install --no-cache-dir wrapt --upgrade --ignore-installed
	python3 -m pip install --no-cache-dir spacy==3.5.0
	python3 -m spacy download en_core_web_md
	python3 -m pip install nltk
	python3 -m nltk.downloader wordnet

# Install VQA dependencies
install_vqa_deps:
	python3 -m pip install --upgrade transformers sentence-transformers huggingface_hub torch torchvision timm
	pip install vllm --no-deps
	pip install transformers pillow tqdm

# Help target
help:
	@echo "Available targets:"
	@echo "  make preprocess INPUT_PATH=/path/to/input [OPTIONS]"
	@echo "  make preprocess_owlvit INPUT_PATH=/path/to/input [OPTIONS]"
	@echo "  make preprocess_yolo INPUT_PATH=/path/to/input [OPTIONS]"
	@echo "  make preprocess_detectron2 INPUT_PATH=/path/to/input [OPTIONS]"
	@echo "  make batch_preprocess INPUT_PATH=/path/to/directory"
	@echo "  make run_vqa VQA_INPUT_FILE=/path/to/input.json [VQA_OPTIONS]"
	@echo "  make download_dataset DATASET=<dataset_name> [DATASET_DIR=<output_dir>]"
	@echo "  make download_coco|download_gqa|download_refcoco|download_vqa|download_textvqa [DATASET_DIR=<output_dir>]"
	@echo "  make install_deps"
	@echo "  make install_vqa_deps"
	@echo ""
	@echo "Options:"
	@echo "  OUTPUT_FOLDER=folder_name       Output folder [default: output_images]"
	@echo "  DETECTORS=d1,d2,...             Comma-separated list of detectors [default: owlvit,yolov8,detectron2]"
	@echo "  RELATIONSHIP_TYPE=type          Relationship types to extract [default: all]"
	@echo "  MAX_RELATIONS=n                 Maximum number of relationships to extract [default: 8]"
	@echo "  START_INDEX=n                   Start index (0-based) for processing [default: process all]"
	@echo "  END_INDEX=n                     End index (inclusive) for processing [default: process all]"
	@echo "  NUM_INSTANCES=n                 Absolute number of instances to process [default: process all]"
	@echo ""
	@echo "Detection thresholds:"
	@echo "  OWL_THRESHOLD=n                 Confidence threshold for OWL-ViT [default: 0.15]"
	@echo "  YOLO_THRESHOLD=n                Confidence threshold for YOLOv8 [default: 0.3]"
	@echo "  DETECTRON_THRESHOLD=n           Confidence threshold for Detectron2 [default: 0.3]"
	@echo ""
	@echo "NMS parameters:"
	@echo "  LABEL_NMS_THRESHOLD=n           IoU threshold for label-based NMS [default: 0.5]"
	@echo "  SEG_IOU_THRESHOLD=n             IoU threshold for segmentation filtering [default: 0.8]"
	@echo ""
	@echo "Relationship parameters:"
	@echo "  OVERLAP_THRESH=n                Horizontal overlap threshold [default: 0.3]"
	@echo "  MARGIN=n                        Margin in pixels [default: 20]"
	@echo "  MIN_DISTANCE=n                  Minimum distance between centers [default: 90]"
	@echo "  MAX_DISTANCE=n                  Maximum distance between centers [default: 20000]"
	@echo ""
	@echo "SAM parameters:"
	@echo "  POINTS_PER_SIDE=n               Points per side for SAM [default: 32]"
	@echo "  PRED_IOU_THRESH=n               Predicted IoU threshold for SAM [default: 0.9]"
	@echo "  STABILITY_SCORE_THRESH=n        Stability score threshold for SAM [default: 0.95]"
	@echo "  MIN_MASK_REGION_AREA=n          Minimum mask region area for SAM [default: 100]"
	@echo ""
	@echo "VQA Options:"
	@echo "  VQA_OUTPUT_FILE=file.json       Output file for VQA results [default: vqa_results.json]"
	@echo "  MODEL_NAME=model                Model name or path [default: llava-hf/llava-1.5-7b-hf]"
	@echo "  IMAGE_DIR=directory             Directory containing images [optional]"
	@echo "  MAX_LENGTH=n                    Maximum length of generated text [default: 512]"
	@echo "  TEMPERATURE=n                   Temperature for sampling [default: 0.2]"
	@echo "  TOP_P=n                         Top-p probability threshold [default: 0.9]"
	@echo "  PROMPT_TEMPLATE=template        Template for formatting prompts [default: 'Question: {question}\nAnswer:']"
	@echo "  BATCH_SIZE=n                    Batch size for processing [default: 1]"
	@echo "  TENSOR_PARALLEL_SIZE=n          Number of GPUs for tensor parallelism [default: 1]"
	@echo ""
	@echo "Dataset Options:"
	@echo "  DATASET=name                    Dataset to download (coco, gqa, refcoco, vqa, textvqa)"
	@echo "  DATASET_DIR=directory           Output directory for downloaded dataset [optional]"

.PHONY: preprocess preprocess_owlvit preprocess_yolo preprocess_detectron2 batch_preprocess install_deps help check_input run_vqa check_vqa_input install_vqa_deps download_dataset download_coco download_gqa download_refcoco download_vqa download_textvqa check_dataset
