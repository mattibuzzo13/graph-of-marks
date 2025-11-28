# Graph of Marks (GoM)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

## Overview

**Graph of Marks (GoM)** transforms images into structured semantic graphs for visual scene understanding. It combines multiple state-of-the-art detection and segmentation models to extract objects, relationships, and scene graphs, enabling both visual analysis and Visual Question Answering (VQA).

## Installation

### Quick Install

```bash
# Clone repository
git clone https://github.com/disi-unibo-nlp/graph-of-marks.git
cd graph-of-marks

# Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r build/requirements.txt

# Download pretrained models
bash download_ckpt.sh
```

### Docker Install

```bash
docker build -f build/Dockerfile -t gom:latest .
```

## How It Works

GoM processes images through a 7-stage pipeline:

```
Input Image
    ↓
1. DETECTION
    Multiple detectors (YOLOv8, OWL-ViT, etc.) find objects
    → Weighted Box Fusion (WBF) combines results
    ↓
2. SEGMENTATION  
    SAM models generate precise masks for each object
    → Smart GPU cache management
    ↓
3. RELATION EXTRACTION
    Compute spatial (left/right/above), semantic (CLIP),
    physical (support/occlusion), and depth (3D) relations
    ↓
4. SEMANTIC FILTERING
    CLIP filters irrelevant objects based on question context
    → Physics validation removes impossible relations
    ↓
5. SCENE GRAPH GENERATION
    Build NetworkX graph: nodes = objects, edges = relations
    ↓
6. VISUALIZATION
    Render annotated images (SVG/PNG/JPG) with labels,
    masks, and relationship arrows
    ↓
7. VQA (Optional)
    Pass scene graph + image to vision-language models
    → Generate answers to questions
```

### Output Files

For each image, GoM generates:
- `{id}_detections.json` - Bounding boxes, labels, confidence scores
- `{id}_relations.json` - All detected relationships
- `{id}_scene_graph.json` - Graph structure (nodes + edges)
- `{id}_viz.{svg,png,jpg}` - Annotated visualization

## Quick Start

### Basic Image Preprocessing

```bash
python src/image_preprocessor.py \
  --input_file data.json \
  --image_dir images/ \
  --preproc_folder output/
```

### Visual Question Answering

```bash
python src/vqa.py \
  --input_file vqa_data.json \
  --model_name llava-hf/llava-1.5-7b-hf \
  --include_scene_graph
```

### Docker Usage

```bash
docker run --rm --gpus all \
  -v $(pwd):/workdir \
  gom:latest python src/image_preprocessor.py \
    --input_file data.json
```

## 📖 Advanced Usage

### Custom Detectors & Fusion

```bash
# Use specific detectors with custom IoU threshold
python src/image_preprocessor.py \
  --input_file data.json \
  --detectors owlvit yolov8 \
  --fusion_iou_threshold 0.45
```

### High-Quality Segmentation

```bash
# SAM-HQ with smart caching
python src/image_preprocessor.py \
  --input_file data.json \
  --sam_version sam_hq \
  --seg_smart_cache
```

### CLIP Semantic Filtering

```bash
# Filter objects by question relevance
python src/image_preprocessor.py \
  --input_file data.json \
  --enable_q_filter \
  --clip_pruning_threshold 0.25
```

### Custom Visualizations

```bash
# Transparent SVG with segmentation masks only
python src/image_preprocessor.py \
  --input_file data.json \
  --show_segmentation \
  --no_display_relationships \
  --save_without_background \
  --output_format svg
```

## Components Overview

### Detection Models
- **YOLOv8**: Fast real-time detection
- **OWL-ViT**: Open-vocabulary detection
- **Detectron2**: High-accuracy instance detection
- **GroundingDINO**: Text-guided detection

**Fusion:** Weighted Box Fusion (WBF) combines predictions with confidence-based weights

### Segmentation Models
- **SAM 1/2**: Meta's Segment Anything
- **SAM-HQ**: High-quality segmentation
- **FastSAM**: Lightweight alternative

**Optimization:** Smart GPU cache with 80% threshold, adaptive memory management

### Relationship Types
- **Spatial (2D)**: left, right, above, below, near, far
- **Semantic**: CLIP-based object similarity
- **Physical**: support, containment, occlusion (size-validated)
- **Depth (3D)**: front, behind (Depth Anything V2)

### VQA Models
- **LLaVA** 1.5/1.6 (7B-34B)
- **BLIP-2** (2.7B-6.7B)
- **Qwen2.5-VL** (7B-72B)
- **Gemma-3** (4B)
- **LLaMA-v-o1** (11B)
- **Others**: Gemma-2, Pixtral, Llama-3.2-Vision

## Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--detectors` | Detection models to use | `owlvit yolov8` |
| `--sam_version` | Segmentation model | `sam1` |
| `--fusion_iou_threshold` | IoU for WBF fusion | `0.45` |
| `--enable_q_filter` | CLIP semantic filtering | `False` |
| `--clip_pruning_threshold` | Relevance threshold | `0.25` |
| `--seg_smart_cache` | Adaptive GPU cache | `False` |
| `--output_format` | Output format | `svg` |
| `--display_labels` | Show object labels | `True` |
| `--display_relationships` | Show relations | `True` |
| `--show_segmentation` | Show masks | `True` |

**Full documentation:** `python src/image_preprocessor.py --help`

## Python API

```python
from gom.pipeline.preprocessor import ImagePreprocessor
from gom.config import GoMConfig

# Configure pipeline
config = GoMConfig(
    detectors=["owlvit", "yolov8"],
    sam_version="sam_hq",
    enable_q_filter=True,
    clip_pruning_threshold=0.25
)

# Process image
preprocessor = ImagePreprocessor(config)
result = preprocessor.process_image(
    image_path="room.jpg",
    question="What objects are in the room?"
)

# Access results
print(f"Detected {len(result.objects)} objects")
print(f"Found {len(result.relations)} relationships")
print(f"Scene graph: {result.scene_graph}")
```

## Benchmarks & Datasets

### Supported Datasets
Download scripts available for: **COCO**, **GQA**, **RefCOCO/+/g**, **VQA v2**, **TextVQA**

```bash
bash scripts/download/download_coco.sh /path/to/data
bash scripts/download/download_gqa.sh /path/to/data
bash scripts/download/download_vqa.sh /path/to/data
```
