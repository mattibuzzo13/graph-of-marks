# Graph of Marks (GoM)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4+-red.svg)](https://pytorch.org/)
[![PyPI version](https://badge.fury.io/py/graph-of-marks.svg)](https://badge.fury.io/py/graph-of-marks)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Graph of Marks (GoM)** transforms images into structured semantic graphs for visual scene understanding. It combines state-of-the-art detection, segmentation, depth estimation, and relationship extraction models to build comprehensive scene graphs.

## 📚 Paper

This work has been accepted at **AAAI 2026**. The paper and supplementary material are available in the `paper/` directory.

## 🛠️ Installation

### From PyPI

```bash
pip install graph-of-marks
```

With optional dependencies:

```bash
# Install with all features
pip install "graph-of-marks[all]"

# Or install specific extras
pip install "graph-of-marks[detection,segmentation,vqa,depth]"
```

### From Source

```bash
git clone https://github.com/disi-unibo-nlp/graph-of-marks.git
cd graph-of-marks

# Install with all dependencies
pip install -e ".[all]"
```

## 🚀 Quick Start

### Python API

```python
from gom import GoM

# Create pipeline with default models
gom = GoM(output_dir="output")

# Process an image
result = gom.process("scene.jpg", question="What objects are in the room?")

# Access results
print(f"Detected {len(result['boxes'])} objects")
print(f"Found {len(result['relationships'])} relationships")
```

### GoM Visual Prompting Styles (AAAI 2026 Paper)

The library supports all visual prompting configurations from the paper:

```python
from gom import GoM

# Use predefined style presets matching the paper's experiments
gom = GoM(style="gom_text_labeled")  # Best for VQA tasks
# or
gom = GoM(style="gom_numeric_labeled")  # Best for RefCOCO tasks

# Available styles:
# - "som_text": Set-of-Mark with textual IDs (oven_1, chair_2) - no relations
# - "som_numeric": Set-of-Mark with numeric IDs (1, 2, 3) - no relations
# - "gom_text": GoM with textual IDs + relation arrows (no labels)
# - "gom_numeric": GoM with numeric IDs + relation arrows (no labels)
# - "gom_text_labeled": GoM with textual IDs + labeled relations
# - "gom_numeric_labeled": GoM with numeric IDs + labeled relations

result = gom.process("scene.jpg")

# For Visual + Textual SG prompting (multimodal), access:
print(result["scene_graph_text"])    # Triples format for LLM prompts
print(result["scene_graph_prompt"])  # Compact inline format
```

You can also configure manually:

```python
gom = GoM(
    label_mode="numeric",           # "original" (text IDs) or "numeric"
    display_relationships=True,      # Show relation arrows
    display_relation_labels=True,    # Show labels on arrows
)
```

### CLI Usage

```bash
# Image preprocessing
gom-preprocess --input_file data.json --image_dir images/ --output_folder output/

# Visual Question Answering
gom-vqa --input_file vqa_data.json --model_name llava-hf/llava-1.5-7b-hf
```

### Custom Models

GoM supports plugging in your own detection, segmentation, and depth models:

```python
from gom import GoM

def my_detector(image):
    # Your detection logic
    boxes = [[100, 100, 200, 200]]
    labels = ["person"]
    scores = [0.95]
    return boxes, labels, scores

def my_segmenter(image, boxes):
    # Your segmentation logic
    import numpy as np
    h, w = image.size[1], image.size[0]
    masks = [np.ones((h, w), dtype=np.uint8) for _ in boxes]
    return masks

gom = GoM(
    detect_fn=my_detector,
    segment_fn=my_segmenter,
    output_dir="custom_output"
)
result = gom.process("scene.jpg")
```

## 📁 Repository Structure

```
graph-of-marks/
├── src/gom/                    # Main package
│   ├── api.py                  # High-level API (GoM class)
│   ├── config.py               # Configuration management
│   ├── cli/                    # Command-line interface
│   │   ├── preprocess.py       # gom-preprocess command
│   │   └── vqa.py              # gom-vqa command
│   ├── detectors/              # Object detection models
│   │   ├── yolov8.py           # YOLOv8 detector
│   │   ├── owlvit.py           # OWL-ViT detector
│   │   ├── grounding_dino.py   # GroundingDINO detector
│   │   └── detectron2.py       # Detectron2 detector
│   ├── segmentation/           # Segmentation models
│   │   ├── sam1.py             # SAM 1 segmentation
│   │   ├── sam2.py             # SAM 2 segmentation
│   │   ├── samhq.py            # SAM-HQ segmentation
│   │   └── fastsam.py          # FastSAM segmentation
│   ├── fusion/                 # Detection fusion strategies
│   │   ├── wbf.py              # Weighted Box Fusion
│   │   ├── nms.py              # Non-Maximum Suppression
│   │   └── cascade.py          # Cascade fusion
│   ├── relations/              # Relationship extraction
│   │   ├── inference.py        # Main relationship inferencer
│   │   ├── clip_rel.py         # CLIP-based semantic relations
│   │   ├── physics.py          # Physical relations
│   │   ├── spatial_3d.py       # 3D spatial relations
│   │   └── geometry/           # Geometric computations
│   ├── graph/                  # Scene graph construction
│   │   ├── scene_graph.py      # Graph building utilities
│   │   └── prompt.py           # Graph-to-text prompts
│   ├── viz/                    # Visualization
│   │   └── visualizer.py       # Image rendering
│   ├── vqa/                    # Visual Question Answering
│   │   ├── models.py           # VLM model support
│   │   └── runner.py           # VQA inference runner
│   ├── utils/                  # Utilities
│   │   ├── depth.py            # Depth estimation (v1)
│   │   ├── depth_v2.py         # Depth Anything V2
│   │   ├── clip_utils.py       # CLIP utilities
│   │   └── gpu_memory.py       # GPU memory management
│   └── pipeline/               # Processing pipeline
│       └── preprocessor.py     # Main preprocessing pipeline
├── examples/                   # Usage examples
│   ├── README.md               # Examples documentation
│   ├── quickstart.py           # Quick start script
│   └── demo.ipynb              # Jupyter notebook demo
├── scripts/                    # Inference scripts
│   ├── run_vqa_inference.py    # VQA inference script
│   └── run_ref_inference.py    # Referring expression script
├── external_libs/              # External dependencies
│   └── sam2/                   # SAM2 library
├── images/                     # Sample images
├── paper/                      # AAAI 2026 paper PDFs
├── pyproject.toml              # Package configuration
├── setup.py                    # Setup script
└── Makefile                    # Build/development commands
```

## 🔧 Pipeline Overview

The GoM pipeline processes images through the following stages:

1. **Detection** → Multiple detectors (YOLOv8, OWL-ViT, GroundingDINO, Detectron2)
2. **Fusion** → Weighted Box Fusion (WBF) combines predictions
3. **Segmentation** → SAM/SAM2/SAM-HQ/FastSAM generates masks
4. **Depth Estimation** → Depth Anything V2 for 3D understanding
5. **Relationship Extraction** → Spatial, semantic, and physical relations
6. **Scene Graph** → NetworkX graph with nodes (objects) and edges (relations)
7. **Visualization** → Annotated output images (PNG/SVG/JPG)

### Output Files

For each processed image, GoM generates:
- `{name}_01_detections.png` - Bounding boxes visualization
- `{name}_02_segmentation.png` - Segmentation masks overlay
- `{name}_03_depth.png` - Depth map
- `{name}_04_output.png` - Final composite with relations
- `{name}_05_graph.json` - Scene graph structure

### Output Dictionary

The `process()` method returns a dictionary with:

```python
result = {
    "boxes": [[x1, y1, x2, y2], ...],     # Bounding boxes
    "labels": ["person", "chair", ...],    # Object labels
    "scores": [0.95, 0.87, ...],           # Confidence scores
    "masks": [np.ndarray, ...],            # Segmentation masks
    "depth": np.ndarray,                   # Depth map
    "relationships": [...],                 # Spatial relations
    "scene_graph": nx.DiGraph,             # NetworkX graph
    "scene_graph_text": "Triples:...",     # T^SG for LLM prompts
    "scene_graph_prompt": "0:person...",   # Compact format
    "processing_time": 12.5,               # Time in seconds
}
```

## 🤖 Supported Models

### Detection
| Model | Type | Key Feature |
|-------|------|-------------|
| YOLOv8 | Real-time | Fast inference |
| OWL-ViT | Open-vocabulary | Text-guided detection |
| GroundingDINO | Open-vocabulary | Text-guided detection |
| Detectron2 | Instance | High accuracy |

### Segmentation
| Model | Quality | Speed |
|-------|---------|-------|
| SAM-HQ | Highest | Slow |
| SAM 2 | High | Medium |
| SAM 1 | Good | Medium |
| FastSAM | Lower | Fast |

### VQA Models
- LLaVA 1.5/1.6 (7B-34B)
- BLIP-2 (2.7B-6.7B)
- Qwen2.5-VL (7B-72B)
- Gemma-3 (4B)
- LLaMA-v-o1 (11B)

## ⚙️ Key Configuration Options

### GoM Visual Prompting Styles (Paper Table 2)

| Style Preset | Label Mode | Relations | Relation Labels | Use Case |
|--------------|------------|-----------|-----------------|----------|
| `som_text` | Textual (oven_1) | ❌ | ❌ | Set-of-Mark baseline |
| `som_numeric` | Numeric (1, 2) | ❌ | ❌ | Set-of-Mark baseline |
| `gom_text` | Textual | ✅ | ❌ | GoM with arrows only |
| `gom_numeric` | Numeric | ✅ | ❌ | GoM with arrows only |
| `gom_text_labeled` | Textual | ✅ | ✅ | **Best for VQA** |
| `gom_numeric_labeled` | Numeric | ✅ | ✅ | **Best for RefCOCO** |

### Pipeline Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `detectors_to_use` | Detection models | `("yolov8",)` |
| `sam_version` | Segmentation model | `"hq"` |
| `wbf_iou_threshold` | IoU for WBF fusion | `0.55` |
| `label_mode` | Label format | `"original"` |
| `display_labels` | Show object labels | `True` |
| `display_relationships` | Show relations | `True` |
| `display_relation_labels` | Show labels on arrows | `True` |
| `show_segmentation` | Show masks | `True` |
| `output_format` | Output format | `"png"` |

Full configuration options available in `src/gom/config.py`.

## 📖 Examples

See the [`examples/`](examples/) directory for:
- **`quickstart.py`** - Basic usage and installation verification
- **`demo.ipynb`** - Comprehensive Jupyter notebook with all features
- **Custom model integration** - Templates for plugging in your own models

## 🐳 Docker

```bash
# Build
docker build -f build/Dockerfile -t gom:latest .

# Run
docker run --rm --gpus all -v $(pwd):/workdir gom:latest \
    gom-preprocess --input_file data.json
```

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 📝 Citation

If you use Graph of Marks in your research, please cite our AAAI 2026 paper.

## 🔗 Links

- [GitHub Repository](https://github.com/disi-unibo-nlp/graph-of-marks)
- [PyPI Package](https://pypi.org/project/graph-of-marks/)
- [Bug Reports](https://github.com/disi-unibo-nlp/graph-of-marks/issues)
