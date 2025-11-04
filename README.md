# Graph of Marks (GoM)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

An optimized pipeline for **visual scene understanding** that combines object detection, segmentation, relationship extraction, and Visual Question Answering (VQA) into a unified system.

## 🚀 Key Features

- ⚡ **25-35% faster** with advanced optimizations
- 🎯 **20-30% better precision** with CLIP semantic filtering
- 🧠 **<1% impossible relations** with physics validation
- 📊 **Multi-detector fusion** (YOLOv8, OWL-ViT, Detectron2, GroundingDINO)
- 🎨 **Publication-ready outputs** (SVG/PNG/JPG with transparent backgrounds)
- 💾 **5-7x less GPU overhead** with adaptive cache management

## 📦 Installation

```bash
# Clone repository
git clone https://github.com/disi-unibo-nlp/graph-of-marks.git
cd graph-of-marks

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r build/requirements.txt

# Download models
bash download_ckpt.sh
```

## 🚀 Quick Start

### Image Preprocessing

```bash
python src/image_preprocessor.py \
  --input_file room.json \
  --image_dir room_image/ \
  --preproc_folder room_image_output/ \
  --preprocess_only
```

### Visual Question Answering

```bash
python src/vqa.py \
  --input_file vqa_data.json \
  --model_name llava-hf/llava-1.5-7b-hf \
  --output_file results.json \
  --include_scene_graph
```

### Docker

```bash
# Build
docker build -f build/Dockerfile -t gom:latest .

# Run
docker run --rm --gpus all \
  -v $(pwd):/workdir \
  gom:latest python src/image_preprocessor.py --help
```

## 🏗️ Pipeline (7 Stages)

```
Input Image
    ↓
1. DETECTION → Multi-model fusion (WBF)
    ↓
2. SEGMENTATION → Adaptive SAM with smart cache
    ↓
3. RELATIONS → 10+ types (geometric, semantic, physical, depth)
    ↓
4. FILTERING → CLIP-based semantic pruning
    ↓
5. SCENE GRAPH → NetworkX structure
    ↓
6. VISUALIZATION → Optimized vectorial rendering
    ↓
7. VQA → Vision-language models integration
```

## 📖 Usage Examples

### 1. Preprocessing with Specific Detectors

```bash
python src/image_preprocessor.py \
  --input_file data.json \
  --detectors owlvit yolov8 \
  --fusion_iou_threshold 0.45 \
  --enable_q_filter
```

### 2. High-Quality Segmentation

```bash
python src/image_preprocessor.py \
  --input_file data.json \
  --sam_version sam_hq \
  --seg_smart_cache \
  --output_format svg
```

### 3. Custom Visualization

```bash
# Segmentation only (transparent SVG)
python src/image_preprocessor.py \
  --input_file data.json \
  --display_labels \
  --no_display_relationships \
  --show_segmentation \
  --save_without_background \
  --output_format svg

# Relations only (PNG)
python src/image_preprocessor.py \
  --input_file data.json \
  --no_display_labels \
  --display_relationships \
  --no_show_segmentation \
  --output_format png
```

### 4. VQA Multi-GPU con vLLM

```bash
python src/vqa.py \
  --input_file large_dataset.json \
  --model_name Qwen/Qwen2.5-VL-7B-Instruct \
  --use_vllm \
  --tensor_parallel_size 2 \
  --batch_size 8
```

## 🎯 Main Components

### Detection
- **4 Models**: YOLOv8, OWL-ViT, Detectron2, GroundingDINO
- **WBF Fusion**: Intelligent combination with confidence-based weights
- **Non-Competing Recovery**: Low-score detection recovery when no spatial competition exists

### Segmentation
- **4 SAM Versions**: SAM 1, SAM 2, SAM-HQ, FastSAM
- **Smart Cache**: Adaptive GPU memory management (80% threshold)
- **Refinement**: Prompt-based (box, point, mask)

### Relations
- **Geometric 2D**: left, right, above, below, near, far
- **Semantic**: CLIP-based similarity between objects
- **Physical**: support, containment, occlusion (with size-ratio validation)
- **Depth 3D**: front, behind (Depth Anything V2)

### VQA Models
- LLaVA 1.5/1.6 (7B-34B)
- BLIP-2 (2.7B-6.7B)
- Qwen2.5-VL (7B-72B)
- Gemma-2, Pixtral, Llama-3.2-Vision

## 📊 Performance

| Component | Optimization | Speedup |
|------------|----------------|---------|
| Relations | CLIP threshold + physics | +20-30% |
| Segmentation | Smart cache | +10-22% |
| Visualization | Vectorial rendering | 2-2.5x |
| Color extraction | KMeans | 5-10x |
| **End-to-End** | **All** | **+25-35%** |

## 🔧 Main CLI Parameters

### Input/Output
- `--input_file`: JSON file with images and questions
- `--image_dir`: Images folder
- `--preproc_folder`: Preprocessing output folder
- `--output_format`: svg, png, jpg

### Detection
- `--detectors`: Detector list (owlvit, yolov8, detectron2, grounding_dino)
- `--fusion_iou_threshold`: IoU threshold for WBF fusion (default: 0.45)
- `--non_competing_iou_threshold`: Recovery threshold (default: 0.30)

### Segmentation
- `--sam_version`: sam1, sam2, sam_hq, fastsam
- `--seg_smart_cache`: Enable adaptive GPU cache
- `--seg_cache_threshold`: Cache threshold (default: 0.80)

### Filtering
- `--enable_q_filter`: Enable CLIP semantic filter
- `--clip_pruning_threshold`: Relevance threshold (default: 0.25)
- `--use_physics_filtering`: Physical relations validation

### Visualization
- `--display_labels`: Show labels
- `--display_relationships`: Show relationships
- `--show_segmentation`: Show masks
- `--save_without_background`: Transparent background

## 📁 Output Structure

```
room_image_output/
├── {image_id}_detections.json      # Detections with bbox and confidence
├── {image_id}_relations.json       # Spatial/semantic relations
├── {image_id}_scene_graph.json     # Scene graph (NetworkX)
└── {image_id}_viz.{svg,png,jpg}   # Annotated visualization
```

## 🐳 Docker

```bash
# Build
docker build -f build/Dockerfile -t gom:latest .

# Preprocessing
docker run --rm --gpus all \
  -v $(pwd):/workdir \
  gom:latest python src/image_preprocessor.py \
    --input_file room.json \
    --preprocess_only

# Full VQA
docker run --rm --gpus device=0 \
  -v $(pwd):/workdir \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  gom:latest python src/vqa.py \
    --input_file data.json \
    --model_name llava-hf/llava-1.5-7b-hf
```

## 🔍 Python API

```python
from igp.pipeline.preprocessor import ImagePreprocessor
from igp.config import GoMConfig

# Configuration
config = GoMConfig(
    detectors=["owlvit", "yolov8"],
    sam_version="sam_hq",
    enable_q_filter=True,
    clip_pruning_threshold=0.25
)

# Preprocessing
preprocessor = ImagePreprocessor(config)
result = preprocessor.process_image(
    image_path="room.jpg",
    question="What objects are in the room?"
)

# Results
print(f"Objects: {len(result.objects)}")
print(f"Relations: {len(result.relations)}")
print(f"Scene graph: {result.scene_graph}")
```

## 📚 Supported Datasets

```bash
# Download dataset
bash scripts/download/download_coco.sh /path/to/data
bash scripts/download/download_gqa.sh /path/to/data
bash scripts/download/download_vqa.sh /path/to/data
```

**Available Datasets**: COCO, GQA, RefCOCO/+/g, VQA v2, TextVQA

## 🛠️ Requirements

- Python 3.8+
- CUDA 11.8+
- GPU with 8GB+ VRAM (12GB+ recommended for SAM-HQ)
- ~15GB disk space for models


## 🙏 Acknowledgments

This project integrates:
- **SAM/SAM-HQ** (Meta AI)
- **YOLOv8** (Ultralytics)
- **OWL-ViT** (Google Research)
- **Detectron2** (FAIR)
- **GroundingDINO** (IDEA Research)
- **Depth Anything V2** (Depth Estimation)
- **LLaVA, BLIP-2, Qwen2.5-VL** (Vision-Language Models)

