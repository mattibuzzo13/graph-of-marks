# Graph of Mark (GoM)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![PyPI version](https://img.shields.io/pypi/v/graph-of-mark.svg)](https://pypi.org/project/graph-of-marks/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Graph of Mark (GoM)** is a visual prompting framework that transforms images into structured semantic graphs for enhanced visual scene understanding. The system integrates state-of-the-art object detection, instance segmentation, depth estimation, and relationship extraction models to construct comprehensive scene graphs that can be used as visual prompts for Multimodal Language Models (MLMs).

<p align="center">
  <img src="https://raw.githubusercontent.com/disi-unibo-nlp/graph-of-marks/main/assets/gom_lab_obj_lab_rel.png" alt="Graph of Marks Output Example" width="600"/>
</p>
<p align="center"><em>Example output showing detected objects with segmentation masks and spatial relationships.</em></p>

---

## Publication

This work has been accepted at the **40th Annual AAAI Conference on Artificial Intelligence (AAAI 2026)**. The paper and supplementary materials are available in the [`paper/`](paper/) directory.

If you use Graph of Mark in your research, please cite:

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
pip install graph-of-mark
```

With optional dependencies:

```bash
# Install with all features
pip install "graph-of-mark[all]"
```

### From Source

```bash
git clone https://github.com/disi-unibo-nlp/graph-of-marks.git
cd graph-of-marks
pip install -e ".[all]"
```

---

## Quick Start

Check out `examples/demo_gom.ipynb` for detailed examples on how to use GoM.

### Python API

```python
from gom import GoM, ProcessingConfig

# Initialize the pipeline
pipeline = GoM(device="cuda")  # or "mps" for Apple Silicon, "cpu" for CPU

# Process an image with a question
config = ProcessingConfig(
    question="What objects are in the room?",
    style="gom_text_labeled",
)
result = pipeline.process("scene.jpg", config=config, save=False)

# Access results
print(f"Detected {len(result['boxes'])} objects")
print(f"Found {len(result['relationships'])} relationships")

# Display the output image
result["output_image"].show()  # PIL Image
```

### Visual Prompting Styles

The library implements all visual prompting configurations presented in the paper:

```python
from gom import GoM, ProcessingConfig, GOM_STYLE_PRESETS

pipeline = GoM(device="cuda")

# Use predefined style presets via ProcessingConfig
config = ProcessingConfig(
    question="Where is the bowl?",
    style="gom_text_labeled",      # Recommended for VQA tasks
    apply_question_filter=True,    # Filter objects by question relevance
)
result = pipeline.process("scene.jpg", config=config, save=False)

# Available styles:
# - "som_text": Set-of-Mark with textual IDs (baseline, no relations)
# - "som_numeric": Set-of-Mark with numeric IDs (baseline, no relations)
# - "gom_text": GoM with textual IDs and relation arrows
# - "gom_numeric": GoM with numeric IDs and relation arrows
# - "gom_text_labeled": GoM with textual IDs and labeled relations
# - "gom_numeric_labeled": GoM with numeric IDs and labeled relations

# Access scene graph representations for VLM prompting
print(result["scene_graph_text"])    # Triple format for LLM prompts
print(result["scene_graph_prompt"])  # Compact inline format
```

Manual configuration is also supported:

```python
config = ProcessingConfig(
    question="What is near the table?",
    label_mode="numeric",           # "original", "numeric", or "alphabetic"
    display_relationships=True,
    display_relation_labels=True,
    aggressive_pruning=True,        # Keep only question-relevant objects
)
result = pipeline.process("scene.jpg", config=config, save=False)
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
    "output_image": PIL.Image.Image,       # Rendered visualization as PIL Image
    "processing_time": 12.5,               # Processing time (seconds)
}
```


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
from gom import GoM, ProcessingConfig
import numpy as np

def custom_detector(image):
    # Custom detection logic
    # Returns: boxes, labels, scores
    boxes = [[100, 100, 200, 200]]
    labels = ["person"]
    scores = [0.95]
    return boxes, labels, scores

def custom_segmenter(image, boxes):
    # Custom segmentation logic
    # Returns: list of boolean masks (H, W)
    h, w = image.size[1], image.size[0]
    masks = [np.ones((h, w), dtype=bool) for _ in boxes]
    return masks

def custom_depth(image):
    # Custom depth estimation
    # Returns: depth map (H, W) normalized to [0, 1]
    h, w = image.size[1], image.size[0]
    return np.zeros((h, w), dtype=np.float32)

# Create GoM with custom functions
pipeline = GoM(
    detect_fn=custom_detector,
    segment_fn=custom_segmenter,
    depth_fn=custom_depth,
    device="cuda"
)

config = ProcessingConfig(
    question="What objects are visible?",
    style="gom_text_labeled",
)
result = pipeline.process("scene.jpg", config=config, save=False)
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
- [PyPI Package](https://pypi.org/project/graph-of-mark/)
- [Issue Tracker](https://github.com/disi-unibo-nlp/graph-of-marks/issues)
