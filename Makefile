# Makefile for graph-visual-reasoning preprocessing (with optional QA filtering & optional inference)

# JSON file containing QA pairs (if set, loops over entries)
JSON_FILE               ?=

# Optional single-image question (used if JSON_FILE is empty)
QUESTION                ?=

# Toggle question-based filtering (true/false)
ENABLE_Q_FILTER         ?= true

# Input path (directory or single image)
INPUT_PATH              ?=

# Output folder for preprocessed images
OUTPUT_FOLDER           ?= output_images
PREPROC_FOLDER          ?= vqa_out

# Detectors to use (comma-separated)
DETECTORS               ?= owlvit,yolov8,detectron2

# Relationship extraction settings
RELATIONSHIP_TYPE       ?= all
MAX_RELATIONS           ?= 10
MAX_RELATIONS_PER_OBJECT ?= 1
MIN_RELATIONS_PER_OBJECT ?= 1
START_INDEX             ?= -1
END_INDEX               ?= -1
NUM_INSTANCES           ?= -1
LABEL_MODE			 ?= original

# Detection thresholds
OWL_THRESHOLD           ?= 0.3
YOLO_THRESHOLD          ?= 0.5
DETECTRON_THRESHOLD     ?= 0.5

# NMS parameters
LABEL_NMS_THRESHOLD     ?= 0.5
SEG_IOU_THRESHOLD       ?= 0.8

# Relationship inference parameters
OVERLAP_THRESH          ?= 0.3
MARGIN                  ?= 30
MIN_DISTANCE            ?= 60
MAX_DISTANCE            ?= 20000

# SAM parameters
POINTS_PER_SIDE         ?= 32
PRED_IOU_THRESH         ?= 0.9
STABILITY_SCORE_THRESH  ?= 0.95
MIN_MASK_REGION_AREA    ?= 100
SAM_VERSION             ?= hq
SAM_HQ_MODEL_TYPE       ?= vit_h

# Inference toggle: if true, run `make run_vqa` after preprocessing
RUN_INFERENCE           ?= false

# VQA defaults (used only if RUN_INFERENCE=true)
VQA_INPUT_FILE          ?=
VQA_OUTPUT_FILE         ?= vqa_results.json
MODEL_NAME              ?= llava-hf/llava-1.5-7b-hf
IMAGE_DIR               ?= $(OUTPUT_FOLDER)
USE_VLLM                ?= true
PROMPT_TEMPLATE         ?= 'Answer with only one word.\nQuestion: {question}\nAnswer:'
BATCH_SIZE              ?= 1
MAX_LENGTH              ?= 512
TEMPERATURE             ?= 0.2
TOP_P                   ?= 0.9
TENSOR_PARALLEL_SIZE    ?= 1
MAX_IMAGES              ?= -1
MAX_QUESTIONS_PER_IMAGE ?= 3
SKIP_PREPROCESSING       ?= false

# Dataset download defaults
DATASET                 ?=
DATASET_DIR             ?=

.PHONY: all preprocess preprocess_owlvit preprocess_yolo preprocess_detectron2 batch_preprocess run_vqa download_dataset download_coco download_gqa download_refcoco download_vqa download_textvqa install_deps install_vqa_deps clean help

all: preprocess

#------------------------------------------------------------------------------
# Preconditions
#------------------------------------------------------------------------------
check_input:
ifeq ($(strip $(JSON_FILE))$(strip $(INPUT_PATH)),)
	$(error You must set either JSON_FILE or INPUT_PATH)
endif

check_vqa_input:
ifndef VQA_INPUT_FILE
	$(error VQA_INPUT_FILE is required for run_vqa)
endif

check_dataset:
ifndef DATASET
	$(error DATASET is required for download targets)
endif

#------------------------------------------------------------------------------
# Main preprocessing: JSON-driven if JSON_FILE set, else single INPUT_PATH
#------------------------------------------------------------------------------
preprocess:
ifeq ($(strip $(JSON_FILE)),)
	@[ -n "$(INPUT_PATH)" ] || (echo "ERROR: serve INPUT_PATH o JSON_FILE"; exit 1)
	@echo "[INFO] Preprocessing single image $(INPUT_PATH)…"
	python3 src/image_graph_preprocessor.py \
		--input_path "$(INPUT_PATH)" \
		--output_folder "$(OUTPUT_FOLDER)" \
		$(if $(strip $(QUESTION)),--question "$(QUESTION)") \
		$(if $(filter false,$(ENABLE_Q_FILTER)),--disable_question_filter) \
		$(if $(strip $(DATASET)),--dataset "$(DATASET)") \
		$(if $(strip $(SPLIT)),--split "$(SPLIT)") \
		$(if $(strip $(IMAGE_COLUMN)),--image_column "$(IMAGE_COLUMN)")
else
	@echo "[INFO] Preprocessing JSON batch $(JSON_FILE)…"
	python3 src/image_graph_preprocessor.py \
		--json_file "$(JSON_FILE)" \
		--output_folder "$(OUTPUT_FOLDER)" \
		$(if $(strip $(QUESTION)),--question "$(QUESTION)") \
		$(if $(filter false,$(ENABLE_Q_FILTER)),--disable_question_filter) \
		$(if $(strip $(DATASET)),--dataset "$(DATASET)") \
		$(if $(strip $(SPLIT)),--split "$(SPLIT)") \
		$(if $(strip $(IMAGE_COLUMN)),--image_column "$(IMAGE_COLUMN)")
endif


#------------------------------------------------------------------------------
# Specialized preprocessing shortcuts (use same logic as 'preprocess' but override detectors)
#------------------------------------------------------------------------------
preprocess_owlvit: check_input
	$(MAKE) preprocess DETECTORS=owlvit

preprocess_yolo: check_input
	$(MAKE) preprocess DETECTORS=yolov8

preprocess_detectron2: check_input
	$(MAKE) preprocess DETECTORS=detectron2

#------------------------------------------------------------------------------
# Legacy batch preprocessing (no QA filtering)
#------------------------------------------------------------------------------
batch_preprocess:
	@[ -n "$(INPUT_PATH)" ] || (echo "ERROR: serve INPUT_PATH"; exit 1)
	@echo "[INFO] Batch preprocessing $(INPUT_PATH) (num_instances=$(NUM_INSTANCES))…"
	python3 src/image_graph_preprocessor.py \
		--input_path "$(INPUT_PATH)" \
		--output_folder "$(OUTPUT_FOLDER)" \
		--num_instances $(NUM_INSTANCES) \
		$(if $(strip $(DATASET)),--dataset "$(DATASET)") \
		$(if $(strip $(SPLIT)),--split "$(SPLIT)")

#------------------------------------------------------------------------------
# VQA target
#------------------------------------------------------------------------------
run_vqa:
	python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB' if torch.cuda.is_available() else 'No CUDA')"
	python3 src/qa_generation.py \
	  --input_file $(VQA_INPUT_FILE) \
	  --image_dir $(IMAGE_DIR) \
	  --output_file $(VQA_OUTPUT_FILE) \
	  --preproc_folder $(PREPROC_FOLDER) \
	  --model_name $(MODEL_NAME) \
	  --max_images $(MAX_IMAGES) \
	  --max_questions_per_image $(MAX_QUESTIONS_PER_IMAGE) \
	  --prompt_template "$(PROMPT_TEMPLATE)" \
	  $(if $(filter true,$(SKIP_PREPROCESSING)),--skip-preprocessing) \
	  $(if $(filter true,$(USE_VLLM)),--use_vllm) \
	  $(if $(filter false,$(ENABLE_Q_FILTER)),--disable_question_filter) \
	  --owl_threshold $(OWL_THRESHOLD) \
	  --yolo_threshold $(YOLO_THRESHOLD) \
	  --detectron_threshold $(DETECTRON_THRESHOLD) \
	  --max_relations $(MAX_RELATIONS) \
	  --max_relations_per_object $(MAX_RELATIONS_PER_OBJECT) \
	  --min_relations_per_object $(MIN_RELATIONS_PER_OBJECT) \
	  --fill_segmentation \
	  --label_mode "$(LABEL_MODE)" \
	  --show_segmentation \
	  --sam_version $(SAM_VERSION) \
	  --sam_hq_model_type $(SAM_HQ_MODEL_TYPE) \
	  --no_legend \
	  --aggressive_pruning \
	  --save_image_only \
	  $(if $(filter true,$(PREPROCESS_ONLY)),--preprocess_only)


#------------------------------------------------------------------------------
# Dataset download targets
#------------------------------------------------------------------------------
download_dataset: check_dataset
	bash scripts/download/download_dataset.sh -d "$(DATASET)" $(if $(DATASET_DIR),-o "$(DATASET_DIR)")

download_coco:
	$(MAKE) download_dataset DATASET=coco $(if $(DATASET_DIR),DATASET_DIR=$(DATASET_DIR))

download_gqa:
	$(MAKE) download_dataset DATASET=gqa $(if $(DATASET_DIR),DATASET_DIR=$(DATASET_DIR))

download_refcoco:
	$(MAKE) download_dataset DATASET=refcoco $(if $(DATASET_DIR),DATASET_DIR=$(DATASET_DIR))

download_vqa:
	$(MAKE) download_dataset DATASET=vqa $(if $(DATASET_DIR),DATASET_DIR=$(DATASET_DIR))

download_textvqa:
	$(MAKE) download_dataset DATASET=textvqa $(if $(DATASET_DIR),DATASET_DIR=$(DATASET_DIR))

#------------------------------------------------------------------------------
# Install dependencies
#------------------------------------------------------------------------------
install_deps:
	python3 -m pip install --no-cache-dir numpy==1.24.4 scipy==1.10.1
	python3 -m pip install --no-cache-dir wrapt --upgrade --ignore-installed
	python3 -m pip install --no-cache-dir spacy==3.5.0
	python3 -m spacy download en_core_web_md
	python3 -m pip install nltk
	python3 -m nltk.downloader wordnet

install_vqa_deps:
	python3 -m pip install --upgrade transformers sentence-transformers huggingface_hub torch torchvision timm
	pip install vllm --no-deps
	pip install transformers pillow tqdm

#------------------------------------------------------------------------------
# Clean & help
#------------------------------------------------------------------------------
clean:
	@echo "[INFO] Cleaning up..."
	@rm -rf $(OUTPUT_FOLDER) $(VQA_OUTPUT_FILE) $(basename $(VQA_OUTPUT_FILE))_metrics.json

help:
	@echo "Available targets:"
	@echo "  make preprocess [JSON_FILE=..|INPUT_PATH=..] [QUESTION=..] [ENABLE_Q_FILTER=true|false] [RUN_INFERENCE=true|false]"
	@echo "  make preprocess_owlvit/preprocess_yolo/preprocess_detectron2"
	@echo "  make batch_preprocess"
	@echo "  make run_vqa VQA_INPUT_FILE=.. [ARGS]"
	@echo "  make download_coco/download_gqa/download_refcoco/download_vqa/download_textvqa"
	@echo "  make install_deps/install_vqa_deps"
	@echo "  make clean"
