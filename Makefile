# Makefile for graph-visual-reasoning preprocessing

# Default values
INPUT_PATH ?= 
OUTPUT_FOLDER ?= output_images
DETECTORS ?= owlvit,yolov8,detectron2
RELATIONSHIP_TYPE ?= all
MAX_RELATIONS ?= 8

# VQA defaults
VQA_INPUT_FILE ?=
VQA_OUTPUT_FILE ?= vqa_results.json
MODEL_NAME ?= llava-hf/llava-1.5-7b-hf
IMAGE_DIR ?=
MAX_LENGTH ?= 512
TEMPERATURE ?= 0.2
TOP_P ?= 0.9
PROMPT_TEMPLATE ?= "Question: {question}\nAnswer:"
BATCH_SIZE ?= 1
TENSOR_PARALLEL_SIZE ?= 1

# Check if input path is provided
check_input:
ifndef INPUT_PATH
	$(error INPUT_PATH is required. Use make target INPUT_PATH=/path/to/input)
endif

# Main preprocessing target
preprocess: check_input
	./run_preprocessing.sh \
		--input "$(INPUT_PATH)" \
		--output "$(OUTPUT_FOLDER)" \
		--detectors "$(DETECTORS)" \
		--relations "$(RELATIONSHIP_TYPE)" \
		--max "$(MAX_RELATIONS)"

# Specialized preprocessing targets
preprocess_owlvit: check_input
	./run_preprocessing.sh \
		--input "$(INPUT_PATH)" \
		--output "$(OUTPUT_FOLDER)" \
		--detectors "owlvit" \
		--relations "$(RELATIONSHIP_TYPE)" \
		--max "$(MAX_RELATIONS)"

preprocess_yolo: check_input
	./run_preprocessing.sh \
		--input "$(INPUT_PATH)" \
		--output "$(OUTPUT_FOLDER)" \
		--detectors "yolov8" \
		--relations "$(RELATIONSHIP_TYPE)" \
		--max "$(MAX_RELATIONS)"

preprocess_detectron2: check_input
	./run_preprocessing.sh \
		--input "$(INPUT_PATH)" \
		--output "$(OUTPUT_FOLDER)" \
		--detectors "detectron2" \
		--relations "$(RELATIONSHIP_TYPE)" \
		--max "$(MAX_RELATIONS)"

# Target for preprocessing a batch of images
batch_preprocess:
	python -c 'import os; [os.system(f"make preprocess INPUT_PATH={os.path.join(root, f)} OUTPUT_FOLDER=output_images/{f.split(\".\")[0]}") for root, _, files in os.walk("$(INPUT_PATH)") for f in files if f.endswith((".jpg", ".png", ".jpeg"))]'

# Visual Question Answering targets
run_vqa: check_vqa_input
	python src/qa_generation.py \
		--input_file "$(VQA_INPUT_FILE)" \
		--output_file "$(VQA_OUTPUT_FILE)" \
		--image_dir "$(IMAGE_DIR)" \
		--model_name "$(MODEL_NAME)" \
		--device cuda \
		--max_length "$(MAX_LENGTH)" \
		--temperature "$(TEMPERATURE)" \
		--top_p "$(TOP_P)" \
		--prompt_template "$(PROMPT_TEMPLATE)" \
		--batch_size "$(BATCH_SIZE)" \
		--tensor_parallel_size "$(TENSOR_PARALLEL_SIZE)"

check_vqa_input:
ifndef VQA_INPUT_FILE
	$(error VQA_INPUT_FILE is required. Use make run_vqa VQA_INPUT_FILE=/path/to/input.json)
endif

# Install dependencies
install_deps:
	python3 -m spacy download en_core_web_md
	python3 -m nltk.downloader wordnet

# Install VQA dependencies
install_vqa_deps:
	pip install vllm transformers pillow tqdm

# Help target
help:
	@echo "Available targets:"
	@echo "  make preprocess INPUT_PATH=/path/to/input [OPTIONS]"
	@echo "  make preprocess_owlvit INPUT_PATH=/path/to/input [OPTIONS]"
	@echo "  make preprocess_yolo INPUT_PATH=/path/to/input [OPTIONS]"
	@echo "  make preprocess_detectron2 INPUT_PATH=/path/to/input [OPTIONS]"
	@echo "  make batch_preprocess INPUT_PATH=/path/to/directory"
	@echo "  make run_vqa VQA_INPUT_FILE=/path/to/input.json [VQA_OPTIONS]"
	@echo "  make install_deps"
	@echo "  make install_vqa_deps"
	@echo ""
	@echo "Options:"
	@echo "  OUTPUT_FOLDER=folder_name       Output folder [default: output_images]"
	@echo "  DETECTORS=d1,d2,...             Comma-separated list of detectors [default: owlvit,yolov8,detectron2]"
	@echo "  RELATIONSHIP_TYPE=type          Relationship types to extract [default: all]"
	@echo "  MAX_RELATIONS=n                 Maximum number of relationships to extract [default: 8]"
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

.PHONY: preprocess preprocess_owlvit preprocess_yolo preprocess_detectron2 batch_preprocess install_deps help check_input run_vqa check_vqa_input install_vqa_deps
