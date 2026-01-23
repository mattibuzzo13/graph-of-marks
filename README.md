# Graph of Marks (GoM)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![PyPI version](https://badge.fury.io/py/graph-of-marks.svg)](https://badge.fury.io/py/graph-of-marks)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Graph of Marks (GoM)** is a visual prompting framework that transforms images into structured semantic graphs for enhanced visual scene understanding. The system integrates state-of-the-art object detection, instance segmentation, depth estimation, and relationship extraction models to construct comprehensive scene graphs that can be used as visual prompts for Multimodal Language Models (MLMs).

<p align="center">
  <img src="https://raw.githubusercontent.com/disi-unibo-nlp/graph-of-marks/main/assets/gom_lab_obj_lab_rel.png" alt="Graph of Marks Output Example" width="600"/>
</p>
<p align="center"><em>Example output showing detected objects with segmentation masks and spatial relationships.</em></p>

---

## Publication

This work has been accepted at the **40th Annual AAAI Conference on Artificial Intelligence (AAAI 2026)**. The paper and supplementary materials are available in the [`paper/`](paper/) directory.

If you use Graph of Marks in your research, please cite:

```bibtex
@inproceedings{gom2026aaai,
  title={Graph-of-Mark: Promote Spatial Reasoning in Multimodal Language Models with Graph-Based Visual Prompting},
  author={Giacomo Frisoni, Lorenzo Molfetta, Mattia Buzzoni, Gianluca Moro},
  booktitle    = {AAAI-26, Sponsored by the Association for the Advancement of Artificial Intelligence},
  year={2026},
  publisher    = {{AAAI} Press},
  year         = {2026},
}
```

Visit our research group website at: https://disi-unibo-nlp.github.io

---

## Installation

### From PyPI

```bash
pip install graph-of-marks
```

With optional dependencies:

```bash
# Install with all features
pip install "graph-of-marks[all]"

# Or install specific components
pip install "graph-of-marks[detection,segmentation,vqa,depth]"
```

### From Source

```bash
git clone https://github.com/disi-unibo-nlp/graph-of-marks.git
cd graph-of-marks
pip install -e ".[all]"
```

---

## Quick Start

### Python API

```python
from gom import GoM

# Initialize the pipeline
gom = GoM(output_dir="output")

# Process an image
result = gom.process("scene.jpg", question="What objects are in the room?")

# Access results
print(f"Detected {len(result['boxes'])} objects")
print(f"Found {len(result['relationships'])} relationships")
```

### Visual Prompting Styles

The library implements all visual prompting configurations presented in the paper:

```python
from gom import GoM

# Use predefined style presets
gom = GoM(style="gom_text_labeled")    # Recommended for VQA tasks
gom = GoM(style="gom_numeric_labeled") # Recommended for RefCOCO tasks

# Available styles:
# - "som_text": Set-of-Mark with textual IDs (baseline, no relations)
# - "som_numeric": Set-of-Mark with numeric IDs (baseline, no relations)
# - "gom_text": GoM with textual IDs and relation arrows
# - "gom_numeric": GoM with numeric IDs and relation arrows
# - "gom_text_labeled": GoM with textual IDs and labeled relations
# - "gom_numeric_labeled": GoM with numeric IDs and labeled relations

result = gom.process("scene.jpg")

# Access scene graph representations for VLM prompting
print(result["scene_graph_text"])    # Triple format for LLM prompts
print(result["scene_graph_prompt"])  # Compact inline format
```

Manual configuration is also supported:

```python
gom = GoM(
    label_mode="numeric",
    display_relationships=True,
    display_relation_labels=True,
)
```

### Command-Line Interface

```bash
# Image preprocessing
gom-preprocess --input_file data.json --image_dir images/ --output_folder output/

# Visual Question Answering
gom-vqa --input_file vqa_data.json --model_name llava-hf/llava-1.5-7b-hf
```

---

## Pipeline Overview

The GoM pipeline processes images through the following stages:

| Stage | Description | Models |
|-------|-------------|--------|
| Detection | Object localization | YOLOv8, OWL-ViT, GroundingDINO, Detectron2 |
| Fusion | Prediction aggregation | Weighted Box Fusion (WBF), NMS |
| Segmentation | Instance mask generation | SAM, SAM2, SAM-HQ, FastSAM |
| Depth Estimation | 3D scene understanding | Depth Anything V2 |
| Relationship Extraction | Spatial/semantic relations | CLIP-based, physics-based |
| Graph Construction | Scene graph generation | NetworkX |

<p align="center">
  <img src="https://raw.githubusercontent.com/disi-unibo-nlp/graph-of-marks/main/assets/gqa_sample_01_detections.png" alt="Detection Stage" width="280"/>
  <img src="https://raw.githubusercontent.com/disi-unibo-nlp/graph-of-marks/main/assets/gqa_sample_03_depth.png" alt="Depth Estimation" width="280"/>
  <img src="https://raw.githubusercontent.com/disi-unibo-nlp/graph-of-marks/main/assets/gqa_sample_02_segmentation.png" alt="Segmentation Stage" width="280"/>
  <img src="https://raw.githubusercontent.com/disi-unibo-nlp/graph-of-marks/main/assets/gqa_sample_04_output.png" alt="Final GoM output" width="280"/>
</p>
<p align="center"><em>Pipeline stages: object detection, instance segmentation, depth estimation.</em></p>


### Return Dictionary

The `process()` method returns:

```python
result = {
    "boxes": [[x1, y1, x2, y2], ...],     # Bounding boxes
    "labels": ["person", "chair", ...],    # Object labels
    "scores": [0.95, 0.87, ...],           # Confidence scores
    "masks": [np.ndarray, ...],            # Segmentation masks
    "depth": np.ndarray,                   # Depth map
    "relationships": [...],                 # Extracted relations
    "scene_graph": nx.DiGraph,             # NetworkX graph
    "scene_graph_text": "...",             # Triple format for prompts
    "scene_graph_prompt": "...",           # Compact format
    "processing_time": 12.5,               # Processing time (seconds)
}
```

---

## Supported Models

### Object Detection

| Model | Type | Description |
|-------|------|-------------|
| YOLOv8 | Real-time | Fast inference, suitable for most applications |
| OWL-ViT | Open-vocabulary | Text-guided detection for custom categories |
| GroundingDINO | Open-vocabulary | Text-guided detection with grounding |
| Detectron2 | Instance | High accuracy for benchmark evaluation |

### Instance Segmentation

| Model | Quality | Speed |
|-------|---------|-------|
| SAM-HQ | Highest | Slow |
| SAM 2 | High | Medium |
| SAM 1 | Good | Medium |
| FastSAM | Lower | Fast |

### Vision-Language Models

Supported models for VQA inference include LLaVA 1.5/1.6 (7B-34B), BLIP-2 (2.7B-6.7B), Qwen2.5-VL (7B-72B), Gemma-3 (4B), and LLaMA-v-o1 (11B).

---

## Configuration

### Visual Prompting Styles (Paper Table 2)

| Style Preset | Label Mode | Relations | Relation Labels | Recommended Use |
|--------------|------------|-----------|-----------------|-----------------|
| `som_text` | Textual | No | No | Set-of-Mark baseline |
| `som_numeric` | Numeric | No | No | Set-of-Mark baseline |
| `gom_text` | Textual | Yes | No | GoM with arrows |
| `gom_numeric` | Numeric | Yes | No | GoM with arrows |
| `gom_text_labeled` | Textual | Yes | Yes | VQA tasks |
| `gom_numeric_labeled` | Numeric | Yes | Yes | RefCOCO tasks |

### Pipeline Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `detectors_to_use` | Detection models to employ | `("yolov8",)` |
| `sam_version` | Segmentation model version | `"hq"` |
| `wbf_iou_threshold` | IoU threshold for WBF fusion | `0.55` |
| `label_mode` | Label format (`"original"` or `"numeric"`) | `"original"` |
| `display_labels` | Render object labels | `True` |
| `display_relationships` | Render relationship arrows | `True` |
| `display_relation_labels` | Render labels on arrows | `True` |
| `show_segmentation` | Render segmentation masks | `True` |
| `output_format` | Output image format | `"png"` |

Complete configuration options are documented in [`src/gom/config.py`](src/gom/config.py).

---

## Custom Model Integration

GoM supports integration of custom detection, segmentation, and depth models:

```python
from gom import GoM

def custom_detector(image):
    # Custom detection logic
    boxes = [[100, 100, 200, 200]]
    labels = ["person"]
    scores = [0.95]
    return boxes, labels, scores

def custom_segmenter(image, boxes):
    # Custom segmentation logic
    import numpy as np
    h, w = image.size[1], image.size[0]
    masks = [np.ones((h, w), dtype=np.uint8) for _ in boxes]
    return masks

gom = GoM(
    detect_fn=custom_detector,
    segment_fn=custom_segmenter,
    output_dir="output"
)
result = gom.process("scene.jpg")
```

---

## Examples

The [`examples/`](examples/) directory contains:

- **`quickstart.py`**: Basic usage and installation verification
- **`demo.ipynb`**: Comprehensive Jupyter notebook demonstrating all features

---

## Docker

```bash
# Build the container
docker build -f build/Dockerfile -t gom:latest .

# Run with GPU support
docker run --rm --gpus all -v $(pwd):/workdir gom:latest \
    gom-preprocess --input_file data.json
```

---

## Repository Structure

```
graph-of-marks/
├── src/gom/                    # Main package
│   ├── api.py                  # High-level API (GoM class)
│   ├── config.py               # Configuration management
│   ├── cli/                    # Command-line interface
│   ├── detectors/              # Object detection models
│   ├── segmentation/           # Segmentation models
│   ├── fusion/                 # Detection fusion strategies
│   ├── relations/              # Relationship extraction
│   ├── graph/                  # Scene graph construction
│   ├── viz/                    # Visualization utilities
│   ├── vqa/                    # VQA inference
│   └── utils/                  # Utility functions
├── examples/                   # Usage examples
├── scripts/                    # Inference scripts
├── external_libs/              # External dependencies (SAM2)
├── paper/                      # AAAI 2026 paper
├── pyproject.toml              # Package configuration
└── Makefile                    # Build commands
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Links

- [GitHub Repository](https://github.com/disi-unibo-nlp/graph-of-marks)
- [PyPI Package](https://pypi.org/project/graph-of-marks/)
- [Issue Tracker](https://github.com/disi-unibo-nlp/graph-of-marks/issues)
