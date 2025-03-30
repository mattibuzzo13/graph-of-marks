# 🔍 Graph Visual Reasoning

A powerful toolkit for visual reasoning on images using object detection, relationship extraction, and vision-language models.

## 📋 Table of Contents

- [Overview](#-overview)
- [Repository Structure](#-repository-structure)
- [Prerequisites](#-prerequisites)
- [Image Graph Preprocessing](#-image-graph-preprocessing)
  - [Features](#features)
  - [Basic Usage](#basic-usage)
  - [Advanced Usage](#advanced-usage)
  - [Output](#output)
- [Visual Question Answering (VQA)](#-visual-question-answering-vqa)
  - [Features](#features-1)
  - [Installation](#installation)
  - [Input Format](#input-format)
  - [Basic Usage](#basic-usage-1)
  - [Advanced Options](#advanced-options)
  - [Direct Usage](#direct-script-usage)
- [Customization](#-customization)
  - [Custom Detectors](#adding-custom-detectors)
  - [Custom VQA Models](#adding-custom-models)
- [Evaluation](#-evaluation)

## 🌟 Overview

This repository provides tools for comprehensive visual understanding of images through:

1. **Object Detection & Segmentation**: Identify and segment objects in images using multiple detection methods
2. **Relationship Extraction**: Determine spatial and semantic relationships between detected objects
3. **Visual Question Answering**: Answer natural language questions about image content using vision-language models

The toolkit is designed for researchers and developers working on visual reasoning, scene understanding, and multimodal AI applications.

## 📂 Repository Structure

```
graph-of-marks/
├── src/                        # Source code directory
│   ├── qa_generation.py        # Visual Question Answering module
│   └── image_graph_preprocessor.py  # Image processing and graph creation
├── Makefile                    # Command automation for all features
├── run_preprocessing.sh        # Script for image preprocessing
└── bench_vllm.py               # Benchmarking script for VLLM models
```

## 🛠️ Prerequisites

Before using this repository, install the required dependencies:

```bash
make install_deps     # Core dependencies for preprocessing
make install_vqa_deps # Dependencies for Visual QA functionality
```

This will install Python packages and download necessary models for spaCy, NLTK, and vision-language processing.

## 🖼️ Image Graph Preprocessing

The image graph preprocessor creates structured representations of images by identifying objects, generating segmentation masks, and extracting relationships.

### Features

- ✅ **Multiple Detection Methods**: Choose from OWL-ViT, YOLOv8, or Detectron2
- ✅ **Automatic Segmentation**: Generate precise object masks using SAM (Segment Anything Model)
- ✅ **Spatial Relationships**: Extract relative positions (above, below, left_of, right_of)
- ✅ **Rich Visualizations**: Generate annotated images showing detections and relationships

### Basic Usage

Process a single image:

```bash
make preprocess INPUT_PATH=/path/to/image.jpg
```

Process a directory of images:

```bash
make preprocess INPUT_PATH=/path/to/image/directory
```

Specify an output directory:

```bash
make preprocess INPUT_PATH=/path/to/image.jpg OUTPUT_FOLDER=my_results
```

### Advanced Usage

#### Using Specific Detectors

You can specify which detectors to use:

```bash
make preprocess INPUT_PATH=/path/to/image.jpg DETECTORS=owlvit,yolov8
```

Or use specialized targets for individual detectors:

```bash
make preprocess_owlvit INPUT_PATH=/path/to/image.jpg
make preprocess_yolo INPUT_PATH=/path/to/image.jpg
make preprocess_detectron2 INPUT_PATH=/path/to/image.jpg
```

#### Relationship Extraction

Specify which spatial relationships to extract:

```bash
make preprocess INPUT_PATH=/path/to/image.jpg RELATIONSHIP_TYPE=above,below
```

Limit the maximum number of relationships:

```bash
make preprocess INPUT_PATH=/path/to/image.jpg MAX_RELATIONS=5
```

#### Batch Processing

Process all images in a directory, creating separate output folders for each:

```bash
make batch_preprocess INPUT_PATH=/path/to/image/directory
```

### Direct Script Usage

If you prefer to use the script directly:

```bash
./run_preprocessing.sh --input /path/to/image.jpg --output my_results --detectors owlvit,yolov8 --relations all --max 8
```

Use the help option to see all available parameters:

```bash
./run_preprocessing.sh --help
```

### Output

The preprocessor generates:

1. 📊 **Detection Files**: JSON files with object coordinates, labels, and confidence scores
2. 🎯 **Visualization Images**: Images with bounding boxes or segmentation masks
3. 🔗 **Relationship Data**: JSON files detailing spatial relationships between objects
4. 📈 **Combined Graphs**: Visual representations of objects and their relationships

## 🤖 Visual Question Answering (VQA)

The VQA system enables natural language querying of image content using state-of-the-art vision-language models.

### Features

- ✅ **Multiple Model Support**: Compatible with LLaVA, BLIP, Pixtral, and more
- ✅ **High-Performance Inference**: Integration with VLLM for efficient processing
- ✅ **Model Flexibility**: Support for both Hugging Face and VLLM backends
- ✅ **Batch Processing**: Process multiple questions and images efficiently
- ✅ **Evaluation Tools**: Compare generated answers against ground truth

### Installation

Install the required dependencies for VQA:

```bash
make install_vqa_deps
```

### Input Format

Create a JSON file with your VQA examples:

```json
[
  {
    "image_path": "/path/to/image1.jpg",
    "question": "What color is the car?",
    "answer": "red"  // Optional ground truth
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

Specify an output file:

```bash
make run_vqa VQA_INPUT_FILE=/path/to/examples.json VQA_OUTPUT_FILE=my_results.json
```

### Advanced Options

#### Model Selection

Choose a different vision-language model:

```bash
make run_vqa VQA_INPUT_FILE=/path/to/examples.json MODEL_NAME=mistralai/Pixtral-12B-2409
```

#### Image Directory

Specify a base directory for image paths (useful when image paths in the JSON are relative):

```bash
make run_vqa VQA_INPUT_FILE=/path/to/examples.json IMAGE_DIR=/path/to/images
```

#### Generation Parameters

Adjust generation parameters:

```bash
make run_vqa VQA_INPUT_FILE=/path/to/examples.json TEMPERATURE=0.7 TOP_P=0.95 MAX_LENGTH=1024
```

#### Multi-GPU Inference

Use multiple GPUs for tensor parallelism:

```bash
make run_vqa VQA_INPUT_FILE=/path/to/examples.json TENSOR_PARALLEL_SIZE=2
```

### Direct Script Usage

If you prefer to use the script directly:

```bash
python src/qa_generation.py \
  --input_file /path/to/examples.json \
  --output_file results.json \
  --model_name llava-hf/llava-1.5-7b-hf \
  --temperature 0.2 \
  --top_p 0.9 \
  --max_length 512
```

## 🧩 Customization

### Adding Custom Detectors

To add custom object detectors:

1. Modify the `image_graph_preprocessor.py` file
2. Implement your detector class following the existing pattern
3. Update the `DETECTORS_TO_USE` list in the main function
4. Your detector will be available as an option in the `--detectors` parameter

### Adding Custom Models

The VQA system supports two types of custom models:

1. **VLLM Models**: 
   - Update the `VLLMWrapper` class in `src/qa_generation.py`
   - Add model-specific logic for handling image inputs

2. **Hugging Face Models**:
   - Update the `HFVisionLanguageModel` class
   - Implement custom processing for your model architecture

## 📊 Evaluation

The VQA system generates comprehensive evaluation metrics:

1. **Complete Results**: JSON file with all questions, images, generated answers, and processing times
2. **Performance Metrics**: 
   - Exact match scores against ground truth answers
   - Average processing time per question
   - Customizable evaluation metrics
3. **Error Analysis**: Logging of failures and processing errors for debugging