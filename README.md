# 🔍 Graph of Marks - Visual Reasoning Toolkit

A high-performance, production-ready toolkit for visual reasoning on images using state-of-the-art object detection, relation extraction, and vision-language models.

## 🚀 Performance Highlights

- ⚡ **25-35% faster** end-to-end pipeline with smart optimizations
- 🎯 **90-95% relation accuracy** with physics-based filtering
- 🔥 **2-2.5x speedup** in visualization rendering
- 💾 **Smart GPU cache** management for efficient memory usage
- 🎨 **5-10x faster** color extraction with histogram-based algorithm

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

1. **Object Detection and Segmentation**: Identifies and segments objects using multiple detection methods with smart fusion
2. **Relation Extraction**: Determines spatial and semantic relationships with physics-based validation and consistency checking
3. **Visual Question Answering**: Answers natural language questions about image content using vision-language models
4. **Scene Graph Generation**: Creates structured knowledge representations of visual scenes
5. **Advanced Visualizations**: Renders high-quality annotated images with vectorized rendering

The toolkit is designed for researchers and developers working on visual reasoning, scene understanding, and multimodal AI applications.

### 🎯 Key Optimizations (November 2025)

- **Smart GPU Cache Management**: 80% threshold-based clearing reduces overhead by 5-7x
- **Physics-Based Relation Filtering**: Eliminates impossible spatial relations (e.g., "sofa on_top_of book")
- **Vectorized Mask Rendering**: 2-2.5x faster visualization with batch processing
- **Fast Color Extraction**: Histogram-based algorithm replaces KMeans (5-10x speedup)
- **Semantic Pruning**: CLIP-based filtering with 20-30% precision improvement
- **Conditional Postprocessing**: Adaptive mask refinement saves 10-22% processing time

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

- ✅ **Multiple Detection Methods**: Choose from OWL-ViT, YOLOv8, or Detectron2 with smart fusion
- ✅ **Automatic Segmentation**: Generates precise object masks using SAM (Segment Anything Model) with smart cache
- ✅ **Physics-Aware Relations**: Extracts validated spatial relationships with size-based plausibility checks
- ✅ **Rich Visualizations**: Generates high-quality annotated images with vectorized rendering (2-2.5x faster)
- ✅ **Question-Guided Filtering**: CLIP-based semantic filtering with 20-30% precision improvement
- ✅ **Multi-Detector Fusion**: Combines results from multiple detectors using Weighted Boxes Fusion (WBF)
- ✅ **Scene Graph Generation**: Creates structured scene graphs with optimized color extraction
- ✅ **Smart Memory Management**: Adaptive GPU cache clearing (80% threshold) for efficient resource usage
- ✅ **Relation Consistency**: Automatic contradiction detection and confidence calibration
- ✅ **Performance Optimized**: 25-35% faster end-to-end processing with smart optimizations

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
# Default optimized values: OWL=0.60, YOLO=0.85, Detectron2=0.85
make preprocess INPUT_PATH=/path/to/image.jpg \
    OWL_THRESHOLD=0.3 YOLO_THRESHOLD=0.7 DETECTRON_THRESHOLD=0.6
```

#### Relation Inference Control

```bash
# Limit relations per object for better quality
make preprocess INPUT_PATH=/path/to/image.jpg \
    MAX_RELATIONS_PER_OBJECT=3 MIN_RELATIONS_PER_OBJECT=1
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

The VQA system enables natural language queries of image content using state-of-the-art vision-language models with optimized preprocessing.

### Features

- ✅ **Multiple Model Support**: Compatible with LLaVA, BLIP, Pixtral, Qwen2.5-VL, and others
- ✅ **High-Performance Inference**: VLLM integration for efficient processing with smart GPU cache
- ✅ **Model Flexibility**: Support for both Hugging Face and VLLM backends
- ✅ **Batch Processing**: Efficiently processes multiple questions and images with resume capability
- ✅ **Integrated Pipeline**: Automatic preprocessing and VQA in a single command
- ✅ **Scene Graph Enhancement**: Response enrichment with structured visual information
- ✅ **Smart Caching**: Disk-based preprocessing cache with deterministic naming (MD5-based)
- ✅ **Memory Optimized**: Adaptive GPU cache clearing (5-7x reduced overhead)
- ✅ **Resume-Safe**: Incremental JSON writes allow crash recovery
- ✅ **Performance Monitoring**: Built-in memory usage tracking and periodic cleanup

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
5. Benefit from automatic WBF fusion with existing detectors

### Adding Custom VQA Models

The VQA system supports two types of custom models:

1. **VLLM Models**: Update the `VLLMWrapper` class in `src/igp/vqa/models.py`
2. **Hugging Face Models**: Update the `HFVLModel` class
3. Both benefit from smart GPU cache management automatically

### Customizing Relations

Add new relation types in `src/igp/relations/` by implementing the appropriate interface:

1. Extend `RelationInferencer` class
2. Implement custom relation logic
3. Relations automatically benefit from:
   - Physics-based validation
   - Consistency checking
   - Confidence calibration
   - CLIP-based semantic scoring

### Performance Tuning

Key configuration parameters for optimization:

```python
# GPU Cache (default: 80% threshold)
use_smart_gpu_cache: bool = True
gpu_cache_threshold: float = 0.80

# Visualization (vectorized rendering)
use_vectorized_masks: bool = True
use_batch_text_renderer: bool = True

# Relations (physics filtering)
use_physics_filtering: bool = True
size_ratio_threshold: float = 3.0

# Segmentation (smart cache)
seg_smart_cache: bool = True
seg_cache_threshold: float = 0.80
```

## 📊 Evaluation

The system generates comprehensive evaluation metrics:

1. **Complete Results**: JSON files with all questions, images, generated answers, and processing times
2. **Performance Metrics**: 
   - Exact match scores against ground truth
   - Average processing time per question (optimized: 25-35% faster)
   - Memory usage statistics with smart GPU cache
   - Customizable evaluation metrics
3. **Error Analysis**: Logging of failures and processing errors for debugging
4. **Relation Quality**: Physics-based validation ensures <1% impossible relations

### Performance Benchmarks (November 2025)

| Component | Optimization | Speedup | Scene (50 objects) |
|-----------|--------------|---------|-------------------|
| **Relations** | CLIP threshold + physics | +20-30% | ~200-300ms saved |
| **Segmentation** | Smart cache + conditional | +10-22% | ~150-300ms saved |
| **Color Extraction** | Histogram vs KMeans | 5-10x | ~500-700ms saved |
| **Visualization** | Vectorized rendering | 2-2.5x | ~400-800ms saved |
| **GPU Cache** | Smart clearing (80%) | 5-7x | ~20ms saved |
| **End-to-End** | All optimizations | **+25-35%** | **~1.3-2.1 sec saved** |

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
# 1. Preprocessing + VQA in one command (optimized with smart cache)
make run_vqa VQA_INPUT_FILE=data.json MODEL_NAME=Qwen/Qwen2.5-VL-7B-Instruct

# 2. Preprocessing only with physics-validated relations
make run_vqa VQA_INPUT_FILE=data.json PREPROCESS_ONLY=true

# 3. VQA with scene graph enhancement (vectorized rendering)
make run_vqa VQA_INPUT_FILE=data.json INCLUDE_SCENE_GRAPH=true SKIP_PREPROCESSING=true

# 4. High-performance batch processing (100 images)
make run_vqa VQA_INPUT_FILE=data.json MAX_IMAGES=100 BATCH_SIZE=4
```

### Advanced Preprocessing

```bash
# Preprocessing with CLIP-based question filtering
make preprocess INPUT_PATH=image.jpg QUESTION="What animals are visible?" ENABLE_Q_FILTER=true

# Batch preprocessing with optimized visualizations
make preprocess INPUT_PATH=images/ DISPLAY_LABELS=true DISPLAY_RELATIONSHIPS=true NUM_INSTANCES=50

# High-quality relation extraction with physics validation
make preprocess INPUT_PATH=image.jpg \
    MAX_RELATIONS_PER_OBJECT=5 \
    DISPLAY_RELATION_LABELS=true \
    DISPLAY_RELATIONSHIPS=true
```

## 📚 Documentation

For detailed documentation on optimizations and configurations, see:

- `PIPELINE_GRAPH_UTILS_ANALYSIS.md` - Complete analysis of pipeline optimizations
- `HIGH_PRIORITY_OPTIMIZATIONS_IMPLEMENTED.md` - Smart GPU cache and fast color extraction
- `RELATIONS_COHERENCE_IMPROVEMENTS.md` - Physics-based relation validation
- `SEGMENTATION_OPTIMIZATIONS.md` - Smart cache and conditional postprocessing
- `VIZ_OPTIMIZATION_SUMMARY.md` - Vectorized rendering improvements

## 🙏 Acknowledgments

This toolkit builds upon and integrates several state-of-the-art models and techniques:

- **SAM (Segment Anything)** - Meta AI
- **OWL-ViT** - Google Research
- **YOLOv8** - Ultralytics
- **Detectron2** - Facebook AI Research
- **CLIP** - OpenAI
- **Vision-Language Models** - Various contributors

Performance optimizations (November 2025) developed with focus on production readiness and efficiency.