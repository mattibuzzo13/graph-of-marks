# Makefile for graph-visual-reasoning preprocessing and VQA inference (modular igp version)

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

# Detection thresholds
OWL_THRESHOLD           ?= 0.4
YOLO_THRESHOLD          ?= 0.5
DETECTRON_THRESHOLD     ?= 0.5

# SAM parameters
SAM_VERSION             ?= 1

# Visualization options
DISPLAY_RELATION_LABELS ?= false
DISPLAY_RELATIONSHIPS   ?= false
DISPLAY_LABELS         ?= false
NO_LEGEND               ?= false

# Output configurations
SAVE_IMAGE_ONLY         ?= false
SKIP_GRAPH              ?= false
SKIP_PROMPT             ?= false

# Dataset parameters
DATASET                 ?=
SPLIT                   ?= train
IMAGE_COLUMN            ?= image
NUM_INSTANCES           ?= -1

# VQA parameters
VQA_INPUT_FILE          ?=
VQA_OUTPUT_FILE         ?= vqa_results.json
MODEL_NAME              ?= llava-hf/llava-1.5-7b-hf
IMAGE_DIR               ?= $(OUTPUT_FOLDER)
USE_VLLM                ?= true
PROMPT_TEMPLATE         ?= 'Answer with only one word. Answer directly without saying anything else. Consider to describe the object in the image. \nQuestion: {question}\nAnswer:'
SINGLE_QUESTION         ?=
BATCH_SIZE              ?= 1
MAX_LENGTH              ?= 512
TEMPERATURE             ?= 0.2
TOP_P                   ?= 0.9
TENSOR_PARALLEL_SIZE    ?= 1
MAX_IMAGES              ?= -1
MAX_QUESTIONS_PER_IMAGE ?= 3
SKIP_PREPROCESSING      ?= false
INCLUDE_SCENE_GRAPH     ?= false
PREPROCESS_ONLY         ?= false

.PHONY: all preprocess preprocess_owlvit preprocess_yolo preprocess_detectron2 batch_preprocess run_vqa run_vqa_folder download_dataset clean help

all: preprocess

#------------------------------------------------------------------------------
# Preconditions
#------------------------------------------------------------------------------
check_input:
ifeq ($(strip $(JSON_FILE))$(strip $(INPUT_PATH)),)
	$(error You must set either JSON_FILE or INPUT_PATH)
endif

check_vqa_input:
ifeq ($(strip $(VQA_INPUT_FILE))$(strip $(IMAGE_DIR)),)
	$(error You must set either VQA_INPUT_FILE or IMAGE_DIR for run_vqa)
endif

#------------------------------------------------------------------------------
# ✅ PREPROCESSING: Usa il nuovo image_preprocessor.py modulare
#------------------------------------------------------------------------------
preprocess: check_input
ifeq ($(strip $(JSON_FILE)),)
	@echo "[INFO] Preprocessing single image $(INPUT_PATH) with igp..."
	PYTHONPATH=/workdir/src:/workdir:$$PYTHONPATH python3 src/image_preprocessor.py \
	    --input_path "$(INPUT_PATH)" \
	    --output_folder "$(OUTPUT_FOLDER)" \
	    --detectors "$(DETECTORS)" \
	    --owl_threshold $(OWL_THRESHOLD) \
	    --yolo_threshold $(YOLO_THRESHOLD) \
	    --detectron_threshold $(DETECTRON_THRESHOLD) \
	    --sam_version $(SAM_VERSION) \
	    $(if $(strip $(QUESTION)),--question "$(QUESTION)") \
	    $(if $(filter false,$(ENABLE_Q_FILTER)),--disable_question_filter) \
	    $(if $(strip $(DATASET)),--dataset "$(DATASET)") \
	    $(if $(strip $(SPLIT)),--split "$(SPLIT)") \
	    $(if $(strip $(IMAGE_COLUMN)),--image_column "$(IMAGE_COLUMN)") \
	    $(if $(filter-out -1,$(NUM_INSTANCES)),--num_instances $(NUM_INSTANCES)) \
		$(if $(filter true,$(DISPLAY_RELATION_LABELS)),--display_relation_labels) \
        $(if $(filter true,$(DISPLAY_RELATIONSHIPS)),--display_relationships) \
        $(if $(filter true,$(DISPLAY_LABELS)),--display_labels) \
        $(if $(filter true,$(NO_LEGEND)),--no_legend) \
        $(if $(filter true,$(SAVE_IMAGE_ONLY)),--save_image_only) \
        $(if $(filter true,$(SKIP_GRAPH)),--skip_graph) \
        $(if $(filter true,$(SKIP_PROMPT)),--skip_prompt)
else
	@echo "[INFO] Preprocessing JSON batch $(JSON_FILE) with igp..."
	PYTHONPATH=/workdir/src:/workdir:$$PYTHONPATH python3 src/image_preprocessor.py \
	    --json_file "$(JSON_FILE)" \
	    --output_folder "$(OUTPUT_FOLDER)" \
	    --detectors "$(DETECTORS)" \
	    --owl_threshold $(OWL_THRESHOLD) \
	    --yolo_threshold $(YOLO_THRESHOLD) \
	    --detectron_threshold $(DETECTRON_THRESHOLD) \
	    --sam_version $(SAM_VERSION) \
	    $(if $(strip $(QUESTION)),--question "$(QUESTION)") \
	    $(if $(filter false,$(ENABLE_Q_FILTER)),--disable_question_filter) \
	    $(if $(strip $(DATASET)),--dataset "$(DATASET)") \
	    $(if $(strip $(SPLIT)),--split "$(SPLIT)") \
	    $(if $(strip $(IMAGE_COLUMN)),--image_column "$(IMAGE_COLUMN)") \
	    $(if $(filter-out -1,$(NUM_INSTANCES)),--num_instances $(NUM_INSTANCES)) \
		$(if $(filter true,$(DISPLAY_RELATION_LABELS)),--display_relation_labels) \
        $(if $(filter true,$(DISPLAY_RELATIONSHIPS)),--display_relationships) \
        $(if $(filter true,$(DISPLAY_LABELS)),--display_labels) \
        $(if $(filter true,$(NO_LEGEND)),--no_legend) \
        $(if $(filter true,$(SAVE_IMAGE_ONLY)),--save_image_only) \
        $(if $(filter true,$(SKIP_GRAPH)),--skip_graph) \
        $(if $(filter true,$(SKIP_PROMPT)),--skip_prompt)
endif

#------------------------------------------------------------------------------
# Specialized preprocessing shortcuts
#------------------------------------------------------------------------------
preprocess_owlvit: check_input
	$(MAKE) preprocess DETECTORS=owlvit

preprocess_yolo: check_input
	$(MAKE) preprocess DETECTORS=yolov8

preprocess_detectron2: check_input
	$(MAKE) preprocess DETECTORS=detectron2

#------------------------------------------------------------------------------
# Legacy batch preprocessing (mantieni per backward compatibility)
#------------------------------------------------------------------------------
batch_preprocess: check_input
	@echo "[INFO] Batch preprocessing $(INPUT_PATH) with igp (num_instances=$(NUM_INSTANCES))..."
	PYTHONPATH=/workdir/src:/workdir:$$PYTHONPATH python3 src/image_preprocessor.py \
	    --input_path "$(INPUT_PATH)" \
	    --output_folder "$(OUTPUT_FOLDER)" \
	    --detectors "$(DETECTORS)" \
	    --owl_threshold $(OWL_THRESHOLD) \
	    --yolo_threshold $(YOLO_THRESHOLD) \
	    --detectron_threshold $(DETECTRON_THRESHOLD) \
	    --sam_version $(SAM_VERSION) \
	    $(if $(filter-out -1,$(NUM_INSTANCES)),--num_instances $(NUM_INSTANCES)) \
	    $(if $(strip $(DATASET)),--dataset "$(DATASET)") \
	    $(if $(strip $(SPLIT)),--split "$(SPLIT)")
		$(if $(filter true,$(DISPLAY_RELATION_LABELS)),--display_relation_labels) \
        $(if $(filter true,$(DISPLAY_RELATIONSHIPS)),--display_relationships) \
        $(if $(filter true,$(DISPLAY_LABELS)),--display_labels) \
		$(if $(filter true,$(NO_LEGEND)),--no_legend) \
        $(if $(filter true,$(SAVE_IMAGE_ONLY)),--save_image_only) \
        $(if $(filter true,$(SKIP_GRAPH)),--skip_graph) \
        $(if $(filter true,$(SKIP_PROMPT)),--skip_prompt)


#------------------------------------------------------------------------------
# ✅ VQA: Usa il nuovo vqa.py modulare
#------------------------------------------------------------------------------
run_vqa: check_vqa_input
	@echo "[INFO] Running VQA inference with igp..."
	python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB' if torch.cuda.is_available() else 'No CUDA')"
	PYTHONPATH=/workdir/src:/workdir:$$PYTHONPATH python3 src/vqa.py \
	    $(if $(strip $(VQA_INPUT_FILE)),--input_file $(VQA_INPUT_FILE)) \
	    $(if $(strip $(IMAGE_DIR)),--image_dir $(IMAGE_DIR)) \
	    --output_file $(VQA_OUTPUT_FILE) \
	    --model_name $(MODEL_NAME) \
	    --max_images $(MAX_IMAGES) \
	    --max_questions_per_image $(MAX_QUESTIONS_PER_IMAGE) \
	    --batch_size $(BATCH_SIZE) \
	    --max_length $(MAX_LENGTH) \
	    --temperature $(TEMPERATURE) \
	    --top_p $(TOP_P) \
	    $(if $(strip $(SINGLE_QUESTION)),--single_question "$(SINGLE_QUESTION)") \
	    $(if $(filter true,$(SKIP_PREPROCESSING)),--skip_preprocessing) \
	    $(if $(filter true,$(USE_VLLM)),--use_vllm) \
	    $(if $(filter false,$(ENABLE_Q_FILTER)),--disable_question_filter) \
	    $(if $(filter true,$(INCLUDE_SCENE_GRAPH)),--include_scene_graph) \
	    $(if $(filter true,$(PREPROCESS_ONLY)),--preprocess_only) \
	    --preproc_folder $(PREPROC_FOLDER) \
	    --output_folder $(OUTPUT_FOLDER) \
	    --detectors "$(DETECTORS)" \
	    --owl_threshold $(OWL_THRESHOLD) \
	    --yolo_threshold $(YOLO_THRESHOLD) \
	    --detectron_threshold $(DETECTRON_THRESHOLD) \
	    --sam_version $(SAM_VERSION) \
		$(if $(filter true,$(DISPLAY_RELATION_LABELS)),--display_relation_labels) \
        $(if $(filter true,$(DISPLAY_RELATIONSHIPS)),--display_relationships) \
        $(if $(filter true,$(DISPLAY_LABELS)),--display_labels) \
        $(if $(filter true,$(NO_LEGEND)),--no_legend) \
        $(if $(filter true,$(SAVE_IMAGE_ONLY)),--save_image_only) \
        $(if $(filter true,$(SKIP_GRAPH)),--skip_graph) \
        $(if $(filter true,$(SKIP_PROMPT)),--skip_prompt)

#------------------------------------------------------------------------------
# VQA su cartella di immagini
#------------------------------------------------------------------------------
run_vqa_folder:
	@[ -n "$(IMAGE_FOLDER)" ] || (echo "ERROR: IMAGE_FOLDER è richiesto"; exit 1)
	$(MAKE) run_vqa IMAGE_DIR=$(IMAGE_FOLDER) VQA_INPUT_FILE= SINGLE_QUESTION="$(FIXED_PROMPT)"

#------------------------------------------------------------------------------
# Dataset download (mantieni come prima)
#------------------------------------------------------------------------------
download_dataset:
ifndef DATASET
	$(error DATASET is required for download targets)
endif
	bash scripts/download/download_dataset.sh -d "$(DATASET)" $(if $(DATASET_DIR),-o "$(DATASET_DIR)")

download_coco:
	$(MAKE) download_dataset DATASET=coco

download_gqa:
	$(MAKE) download_dataset DATASET=gqa

download_refcoco:
	$(MAKE) download_dataset DATASET=refcoco

download_vqa:
	$(MAKE) download_dataset DATASET=vqa

download_textvqa:
	$(MAKE) download_dataset DATASET=textvqa

#------------------------------------------------------------------------------
# Install dependencies
#------------------------------------------------------------------------------
install_deps:
	python3 -m pip install --no-cache-dir -r build/requirements.txt
	python3 -m spacy download en_core_web_md
	python3 -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"

#------------------------------------------------------------------------------
# Clean & help
#------------------------------------------------------------------------------
clean:
	@echo "[INFO] Cleaning up..."
	@rm -rf $(OUTPUT_FOLDER) $(VQA_OUTPUT_FILE) $(basename $(VQA_OUTPUT_FILE))_metrics.json

help:
	@echo "🚀 Graph-of-Marks Modular Pipeline (igp)"
	@echo ""
	@echo "Available targets:"
	@echo "  📊 PREPROCESSING:"
	@echo "    make preprocess INPUT_PATH=path/to/image [QUESTION='...'] [ENABLE_Q_FILTER=true|false]"
	@echo "    make preprocess JSON_FILE=data.json [QUESTION='...']"
	@echo "    make batch_preprocess INPUT_PATH=path/to/folder [NUM_INSTANCES=100]"
	@echo "    make preprocess_owlvit|preprocess_yolo|preprocess_detectron2"
	@echo ""
	@echo "  🤖 VQA INFERENCE:"
	@echo "    make run_vqa VQA_INPUT_FILE=data.json MODEL_NAME=llava-hf/llava-1.5-7b-hf"
	@echo "    make run_vqa_folder IMAGE_FOLDER=path/to/images [FIXED_PROMPT='Describe this']"
	@echo ""
	@echo "  📥 DATASET DOWNLOAD:"
	@echo "    make download_coco|download_gqa|download_refcoco|download_vqa|download_textvqa"
	@echo ""
	@echo "  🔧 UTILITIES:"
	@echo "    make install_deps"
	@echo "    make clean"
	@echo ""
	@echo "Examples:"
	@echo "  # Preprocessing single image"
	@echo "  make preprocess INPUT_PATH=test.jpg QUESTION='What is this?'"
	@echo ""
	@echo "  # Full VQA pipeline"
	@echo "  make run_vqa VQA_INPUT_FILE=data.json MODEL_NAME=llava-hf/llava-1.5-7b-hf"
	@echo ""
	@echo "  # Preprocessing only"
	@echo "  make run_vqa VQA_INPUT_FILE=data.json PREPROCESS_ONLY=true"