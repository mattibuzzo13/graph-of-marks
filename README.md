# 🔍 Graph of Marks - Visual Reasoning Toolkit

A powerful toolkit for visual reasoning on images using object detection, relation extraction, and vision-language models.

## 📋 Table of Contents

- [Overview](#-overview)
- [Repository Structure](#-repository-structure)
- [Prerequisites](#-prerequisites)
- [Image Preprocessing](#-image-preprocessing)
  - [Features](#features)
  - [Basic Usage](#basic-usage)
  - [Advanced Options](#advanced-options)
  - [Output](#output)
- [Visual Question Answering (VQA)](#-visual-question-answering-vqa)
  - [Features](#features-1)
  - [Input Format](#input-format)
  - [Basic Usage](#basic-usage-1)
  - [Advanced Options](#advanced-options-1)
- [Dataset Download](#-dataset-download)
  - [Available Datasets](#available-datasets)
  - [Basic Usage](#basic-usage-for-datasets)
  - [Advanced Options](#advanced-options-for-datasets)
- [Customization](#-customization)
- [Evaluation](#-evaluation)

## 🌟 Overview

This repository provides tools for comprehensive visual understanding of images through:

1. **Object Detection and Segmentation**: Identifies and segments objects using multiple detection methods
2. **Relation Extraction**: Determines spatial and semantic relationships between detected objects
3. **Visual Question Answering**: Answers natural language questions about image content using vision-language models

The toolkit is designed for researchers and developers working on visual reasoning, scene understanding, and multimodal AI applications.

## 📂 Repository Structure

```
graph-of-marks/
├── src/
│   ├── image_preprocessor.py       # Main script for image preprocessing
│   ├── vqa.py                     # Main script for Visual Question Answering
│   └── igp/                       # IGP modular package (Image Graph Processing)
│       ├── detectors/             # Object detectors (OWL-ViT, YOLOv8, Detectron2)
│       ├── segmentation/          # Segmentation modules (SAM, SAM2, SAM-HQ)
│       ├── relations/             # Spatial and semantic relation extraction
│       ├── fusion/                # Detection fusion (NMS, WBF)
│       ├── graph/                 # Scene graph and prompt generation
│       ├── viz/                   # Visualization and rendering
│       ├── vqa/                   # Integrated VQA pipeline
│       └── utils/                 # Various utilities
├── build/
│   ├── Dockerfile                 # Containerization
│   └── requirements.txt           # Python dependencies
└── Makefile                       # Command automation
```

## 🛠️ Prerequisites

Before using this repository, install the required dependencies:

```bash
make install_deps
```

This will install the necessary Python packages and download models for spaCy, NLTK, and vision-language processing.

## 🖼️ Image Preprocessing

The image preprocessor creates structured representations of images by identifying objects, generating segmentation masks, and extracting relationships.

### Features

- ✅ **Multiple Detection Methods**: Choose from OWL-ViT, YOLOv8, or Detectron2
- ✅ **Automatic Segmentation**: Generates precise object masks using SAM (Segment Anything Model)
- ✅ **Spatial Relations**: Extracts relative positions (above, below, left_of, right_of)
- ✅ **Rich Visualizations**: Generates annotated images showing detections and relationships
- ✅ **Question-Guided Filtering**: Filters relevant objects for specific questions
- ✅ **Multi-Detector Fusion**: Combines results from multiple detectors using NMS or WBF
- ✅ **Scene Graph Generation**: Creates structured scene graphs

### Basic Usage

Preprocess a single image:

```bash
make preprocess INPUT_PATH=/path/to/image.jpg
```

Preprocess with a specific question:

```bash
make preprocess INPUT_PATH=/path/to/image.jpg QUESTION="What color is the car?"
```

Preprocess from a JSON file:

```bash
make preprocess JSON_FILE=/path/to/data.json
```

### Advanced Options

#### Using Specific Detectors

```bash
make preprocess INPUT_PATH=/path/to/image.jpg DETECTORS=owlvit,yolov8
```

Or use specialized targets:

```bash
make preprocess_owlvit INPUT_PATH=/path/to/image.jpg
make preprocess_yolo INPUT_PATH=/path/to/image.jpg
make preprocess_detectron2 INPUT_PATH=/path/to/image.jpg
```

#### Controlling Detection Thresholds

```bash
make preprocess INPUT_PATH=/path/to/image.jpg \
    OWL_THRESHOLD=0.3 YOLO_THRESHOLD=0.7 DETECTRON_THRESHOLD=0.6
```

#### Visualization Options

```bash
make preprocess INPUT_PATH=/path/to/image.jpg \
    DISPLAY_LABELS=true DISPLAY_RELATIONSHIPS=true DISPLAY_RELATION_LABELS=true
```

#### Processing from Datasets

```bash
make preprocess DATASET=coco SPLIT=train NUM_INSTANCES=100
```

### Output

The preprocessor generates:

1. 📊 **Detection Files**: JSON with object coordinates, labels, and confidence scores
2. 🎯 **Visualization Images**: Images with bounding boxes or segmentation masks
3. 🔗 **Relation Data**: JSON files with spatial relationships between objects
4. 📈 **Scene Graph**: Structured representations of objects and relations
5. 🤖 **Generated Prompts**: Textual descriptions for VQA models

## 🤖 Visual Question Answering (VQA)

The VQA system enables natural language queries of image content using state-of-the-art vision-language models.

### Features

- ✅ **Multiple Model Support**: Compatible with LLaVA, BLIP, Pixtral, Qwen2.5-VL, and others
- ✅ **High-Performance Inference**: VLLM integration for efficient processing
- ✅ **Model Flexibility**: Support for both Hugging Face and VLLM backends
- ✅ **Batch Processing**: Efficiently processes multiple questions and images
- ✅ **Integrated Pipeline**: Automatic preprocessing and VQA in a single command
- ✅ **Scene Graph Enhancement**: Response enrichment with structural information

### Input Format

Create a JSON file with your VQA examples:

```json
[
  {
    "image_path": "/path/to/image1.jpg",
    "question": "What color is the car?",
    "answer": "red"
  },
  {
    "image_path": "https://example.com/image2.jpg",
    "question": "How many people are in the picture?",
    "answer": "3"
  }
]
```

### Basic Usage

Run VQA on a set of examples:

```bash
make run_vqa VQA_INPUT_FILE=/path/to/examples.json
```

VQA with automatic preprocessing:

```bash
make run_vqa VQA_INPUT_FILE=/path/to/examples.json MODEL_NAME=Qwen/Qwen2.5-VL-7B-Instruct
```

VQA on image folder only:

```bash
make run_vqa_folder IMAGE_FOLDER=/path/to/images FIXED_PROMPT="Describe this image"
```

### Advanced Options

#### Model Selection

```bash
make run_vqa VQA_INPUT_FILE=/path/to/examples.json MODEL_NAME=mistralai/Pixtral-12B-2409
```

#### Generation Parameters

```bash
make run_vqa VQA_INPUT_FILE=/path/to/examples.json \
    TEMPERATURE=0.7 TOP_P=0.95 MAX_LENGTH=1024
```

#### Multi-GPU Inference

```bash
make run_vqa VQA_INPUT_FILE=/path/to/examples.json TENSOR_PARALLEL_SIZE=2
```

#### Preprocessing Only

```bash
make run_vqa VQA_INPUT_FILE=/path/to/examples.json PREPROCESS_ONLY=true
```

#### With Scene Graph

```bash
make run_vqa VQA_INPUT_FILE=/path/to/examples.json INCLUDE_SCENE_GRAPH=true
```

## 📥 Dataset Download

The toolkit provides convenient methods for downloading standard benchmark datasets for visual reasoning tasks.

### Available Datasets

- **COCO**: Common Objects in Context dataset
- **GQA**: Visual question answering dataset based on Visual Genome
- **RefCOCO**: Dataset for referring expression comprehension
- **VQA**: Visual Question Answering dataset
- **TextVQA**: VQA dataset focused on text in images

### Basic Usage for Datasets

```bash
make download_dataset DATASET=coco
```

Or use specialized targets:

```bash
make download_coco
make download_gqa
make download_refcoco
make download_vqa
make download_textvqa
```

### Advanced Options for Datasets

Specify a custom output directory:

```bash
make download_coco DATASET_DIR=/path/to/data/coco
```

## 🧩 Customization

### Adding Custom Detectors

To add custom object detectors:

1. Create a new class in `src/igp/detectors/` inheriting from `BaseDetector`
2. Implement the required methods (`detect()`, `get_name()`)
3. Register the detector in the configuration system
4. Your detector will be available in the `--detectors` parameter

### Adding Custom VQA Models

The VQA system supports two types of custom models:

1. **VLLM Models**: Update the `VLLMModel` class in `src/igp/vqa/models.py`
2. **Hugging Face Models**: Update the `HFVisionLanguageModel` class

### Customizing Relations

Add new relation types in `src/igp/relations/` by implementing the appropriate interface.

## 📊 Evaluation

The system generates comprehensive evaluation metrics:

1. **Complete Results**: JSON files with all questions, images, generated answers, and processing times
2. **Performance Metrics**: 
   - Exact match scores against ground truth
   - Average processing time per question
   - Customizable evaluation metrics
3. **Error Analysis**: Logging of failures and processing errors for debugging

## 🐳 Docker Support

The project includes containerization support:

```bash
# Build the image
docker build -f build/Dockerfile -t graph-of-marks .

# Run the container
docker run --gpus all -v $(pwd):/workdir graph-of-marks make preprocess INPUT_PATH=test.jpg
```

## 📖 Complete Examples

### Full VQA Pipeline

```bash
# 1. Preprocessing + VQA in one command
make run_vqa VQA_INPUT_FILE=data.json MODEL_NAME=Qwen/Qwen2.5-VL-7B-Instruct

# 2. Preprocessing only
make run_vqa VQA_INPUT_FILE=data.json PREPROCESS_ONLY=true

# 3. VQA with scene graph enhancement
make run_vqa VQA_INPUT_FILE=data.json INCLUDE_SCENE_GRAPH=true SKIP_PREPROCESSING=true
```

### Advanced Preprocessing

```bash
# Preprocessing with question filtering
make preprocess INPUT_PATH=image.jpg QUESTION="What animals are visible?" ENABLE_Q_FILTER=true

# Batch preprocessing with full visualizations
make preprocess INPUT_PATH=images/ DISPLAY_LABELS=true DISPLAY_RELATIONSHIPS=true NUM_INSTANCES=50
```