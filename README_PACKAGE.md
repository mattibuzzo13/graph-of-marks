# Graph of Marks - Python Package

**Complete, production-ready visual scene understanding library**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

---

## 🎯 What's New

✅ **Pip Installable Package** - Standard Python package with `pip install -e .`
✅ **Renamed to `gom`** - Clean, consistent package name (Graph of Marks)
✅ **GoM Features** - Numeric labels (SoM baseline), alphabetic labels
✅ **Custom Model Support** - Easy integration of your own models
✅ **Real Dataset Examples** - COCO, GQA, VQA v2 in demo notebook
✅ **Production Ready** - Complete documentation and working examples

---

## 📦 Installation

### Quick Install

```bash
# Clone repository
git clone https://github.com/disi-unibo-nlp/graph-of-marks.git
cd graph-of-marks

# Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install Graph of Marks
pip install -e .

# Or install with all features
pip install -e ".[all]"
```

### Verify Installation

```bash
# Quick test
python -c "from gom import GraphOfMarks; print('✅ Success!')"

# Run demo
python examples/quickstart.py

# Full demo notebook
jupyter notebook examples/demo_notebook.ipynb
```

---

## 🚀 Quick Start

### Basic Usage

```python
from gom import GraphOfMarks

# Create pipeline
gom = GraphOfMarks()

# Process image
result = gom.process_image("photo.jpg")

# Results
print(f"Objects: {len(result['detections'])}")
print(f"Relations: {len(result['relations'])}")
```

### Set-of-Mark (SoM) Baseline

```python
from gom import GraphOfMarks

# Use numeric labels (1, 2, 3, ...) for VQA
gom = GraphOfMarks(label_mode="numeric")

result = gom.process_image(
    "photo.jpg",
    question="What is object 2 doing?"
)

# Objects labeled: 1, 2, 3, 4, ...
# Scene graph: "1 --holding--> 3"
```

---

## 🎨 GoM Features

### Label Modes

**Original** (Default) - Class names:
```python
gom = GraphOfMarks(label_mode="original")
# Labels: "person", "car", "dog"
```

**Numeric** (SoM Baseline) - Numbers:
```python
gom = GraphOfMarks(label_mode="numeric")
# Labels: "1", "2", "3", "4"
# Perfect for VQA and Set-of-Mark papers
```

**Alphabetic** - Letters:
```python
gom = GraphOfMarks(label_mode="alphabetic")
# Labels: "A", "B", "C", "D"
```

### Visual Comparison

```python
# Side-by-side comparison
configs = [
    {"name": "Original", "label_mode": "original"},
    {"name": "Numeric (SoM)", "label_mode": "numeric"},
    {"name": "Alphabetic", "label_mode": "alphabetic"}
]

for cfg in configs:
    gom = GraphOfMarks(**cfg)
    result = gom.process_image("photo.jpg")
    # See examples/demo_notebook.ipynb for visualization
```

---

## 🔧 Custom Models

### Custom Segmentation (MobileSAM Example)

```python
from mobile_sam import SamPredictor
from gom import GraphOfMarks

# Load MobileSAM
predictor = SamPredictor(mobile_sam_model)

def mobile_sam_segmenter(image, boxes, **kwargs):
    """
    Custom segmentation function.

    Args:
        image: np.ndarray (H, W, 3) RGB
        boxes: List of [x1, y1, x2, y2]

    Returns:
        {'masks': List[np.ndarray]}
    """
    predictor.set_image(image)
    masks = []
    for box in boxes:
        mask, score, _ = predictor.predict(box=np.array(box))
        masks.append(mask[0])
    return {'masks': masks}

# Use MobileSAM
gom = GraphOfMarks(custom_segmenter=mobile_sam_segmenter)
result = gom.process_image("photo.jpg")
```

### Custom Detection (Detectron2 Example)

```python
from detectron2.engine import DefaultPredictor
from gom import GraphOfMarks, Detection

# Setup Detectron2
predictor = DefaultPredictor(cfg)

def detectron2_detector(image, **kwargs):
    """
    Custom detection function.

    Args:
        image: np.ndarray (H, W, 3) RGB

    Returns:
        List[Detection]
    """
    outputs = predictor(image)
    instances = outputs["instances"].to("cpu")

    detections = []
    for box, cls, score in zip(instances.pred_boxes,
                                instances.pred_classes,
                                instances.scores):
        detections.append(Detection(
            box=box.tensor[0].tolist(),
            label=class_names[cls],
            score=float(score)
        ))
    return detections

# Use Detectron2
gom = GraphOfMarks(custom_detector=detectron2_detector)
result = gom.process_image("photo.jpg")
```

---

## 📚 Complete API

```python
from gom import GraphOfMarks

gom = GraphOfMarks(
    # Detection
    detectors=["yolov8", "owlvit", "detectron2", "grounding_dino"],

    # Segmentation
    sam_version="sam1",  # "sam1", "sam2", "hq", "fast"

    # Features
    use_depth=True,                 # Depth-aware 3D relations
    use_clip_relations=True,        # Semantic relations

    # GoM Settings
    label_mode="numeric",           # "original", "numeric", "alphabetic"
    show_masks=True,                # Show segmentation
    show_relationships=True,        # Show relation arrows

    # Output
    output_folder="results/",

    # Custom Models (Optional)
    custom_detector=None,           # Your detector
    custom_segmenter=None,          # Your segmenter
    custom_depth_estimator=None,    # Your depth model
    custom_relation_extractor=None, # Your relation extractor

    # Advanced (kwargs passed to PreprocessorConfig)
    threshold_yolo=0.3,
    wbf_iou_threshold=0.55,
    # ... any other config parameter
)

# Process image
result = gom.process_image(
    "photo.jpg",
    question="What is in the image?",  # Optional VQA question
    save_visualization=True
)

# Process batch
results = gom.process_batch(
    ["img1.jpg", "img2.jpg"],
    questions=["Q1?", "Q2?"]
)

# Process directory
results = gom.process_directory(
    "images/",
    pattern="*.jpg",
    recursive=True
)
```

---

## 📓 Demo Notebook

**Complete production demo:** `examples/demo_notebook.ipynb`

### What's Included

1. ✅ **Real Datasets**
   - COCO images from HuggingFace
   - GQA/VQA v2 questions
   - Actual data loading and processing

2. ✅ **GoM Features**
   - All label modes (original, numeric, alphabetic)
   - Side-by-side comparisons
   - Set-of-Mark baseline usage

3. ✅ **Default Model Signatures**
   - Exact function definitions
   - Input/output specifications
   - How to match the interface

4. ✅ **Real Custom Models**
   - MobileSAM (60x smaller, 10x faster)
   - Detectron2 (alternative detector)
   - Working installation and usage

5. ✅ **VQA Examples**
   - Question-aware processing
   - Scene graph generation
   - Real GQA dataset

6. ✅ **Model Comparison**
   - Performance benchmarks
   - Speed vs quality tradeoffs
   - Configuration recommendations

7. ✅ **Production Ready**
   - Complete working code
   - Error handling
   - Best practices

### Run the Notebook

```bash
pip install jupyter
jupyter notebook examples/demo_notebook.ipynb
```

---

## 📖 Documentation

### Main Guides

- **[INSTALLATION.md](INSTALLATION.md)** - Complete installation guide
- **[PACKAGE_USAGE.md](PACKAGE_USAGE.md)** - Full API reference
- **[GOM_FEATURES_SUMMARY.md](GOM_FEATURES_SUMMARY.md)** - GoM-specific features
- **[examples/README.md](examples/README.md)** - Example usage
- **[FINAL_SUMMARY.md](FINAL_SUMMARY.md)** - Complete overview

### Quick References

- **Default Models**: YOLOv8, OWL-ViT, Detectron2, SAM1/2, Depth Anything V2
- **Custom Models**: Clear function signatures in demo notebook
- **CLI Commands**: `gom-preprocess`, `gom-vqa`
- **Package Structure**: `src/gom/` with modular organization

---

## 🎯 Use Cases

### 1. Visual Question Answering (VQA)

```python
gom = GraphOfMarks(
    label_mode="numeric",  # SoM-style
    show_relationships=True
)

result = gom.process_image(
    "photo.jpg",
    question="What is object 3 holding?"
)

# Scene graph for VQA model
scene_graph = result['scene_graph']
```

### 2. Scene Understanding

```python
gom = GraphOfMarks(
    detectors=["owlvit", "yolov8"],
    sam_version="sam2",
    use_depth=True
)

result = gom.process_image("scene.jpg")

# Rich scene graph with:
# - Objects with masks
# - Spatial relationships
# - Depth ordering
```

### 3. Dataset Annotation

```python
gom = GraphOfMarks()

results = gom.process_directory(
    "dataset/images/",
    pattern="*.jpg",
    recursive=True
)

# Generate annotations for training
for result in results:
    save_annotations(result['scene_graph_json'])
```

---

## 🔬 Model Configurations

### Fast (Prototyping)

```python
gom = GraphOfMarks(
    detectors=["yolov8"],
    sam_version="fast",
    use_depth=False
)
# ~2-3 seconds per image
```

### Balanced (Recommended)

```python
gom = GraphOfMarks(
    detectors=["yolov8", "owlvit"],
    sam_version="sam1",
    use_depth=False,
    use_clip_relations=True
)
# ~5-8 seconds per image
```

### High Quality (Production)

```python
gom = GraphOfMarks(
    detectors=["owlvit", "yolov8", "detectron2"],
    sam_version="sam2",
    use_depth=True,
    use_clip_relations=True
)
# ~10-15 seconds per image
```

---

## 🛠️ CLI Tools

After installation, two CLI commands are available:

### Image Preprocessing

```bash
gom-preprocess \
  --input_path image.jpg \
  --output_folder results/ \
  --detectors yolov8,owlvit \
  --sam_version sam2
```

### Visual Question Answering

```bash
gom-vqa \
  --input_file vqa_data.json \
  --model_name llava-hf/llava-1.5-7b-hf \
  --include_scene_graph
```

---

## 📊 Results Format

```python
result = gom.process_image("photo.jpg")

# Result dictionary contains:
{
    'detections': [
        {
            'box': [x1, y1, x2, y2],
            'label': 'person',  # or "1" if label_mode="numeric"
            'score': 0.95,
            'id': 0
        },
        ...
    ],
    'relations': [
        {
            'source_id': 0,
            'target_id': 1,
            'relation_type': 'left_of',
            'confidence': 0.9
        },
        ...
    ],
    'scene_graph': <NetworkX DiGraph>,
    'scene_graph_json': {...},
    'output_path': 'results/photo_viz.png',
    'depth_map': np.ndarray,  # if use_depth=True
    'processing_time': 5.23
}
```

---

## 🧪 Testing

```bash
# Automated test (creates conda env)
bash test_installation.sh

# Quick test
python examples/quickstart.py

# Demo notebook
jupyter notebook examples/demo_notebook.ipynb
```

---

## 📦 Package Structure

```
graph-of-marks/
├── src/gom/                      # Main package
│   ├── __init__.py               # GraphOfMarks, types, config
│   ├── api.py                    # High-level API
│   ├── config.py                 # Configuration
│   ├── types.py                  # Detection, Relationship
│   ├── pipeline/                 # Core pipeline
│   ├── detectors/                # YOLOv8, OWL-ViT, etc.
│   ├── segmentation/             # SAM variants
│   ├── relations/                # Relationship extraction
│   ├── viz/                      # Visualization
│   └── cli/                      # CLI commands
├── examples/
│   ├── demo_notebook.ipynb       # Comprehensive demo
│   ├── quickstart.py             # Quick test
│   └── README.md                 # Examples guide
├── setup.py                      # Package setup
├── pyproject.toml                # Modern config
└── [Documentation files...]
```

---

## 🌟 Key Features

### ✨ Unified API
- Single `GraphOfMarks` class
- Clean, intuitive interface
- Full backward compatibility

### 🎯 GoM Features
- Numeric labels (SoM baseline)
- Alphabetic labels
- Configurable visualization
- VQA-aware filtering

### 🔧 Customizable
- Custom detection models
- Custom segmentation models
- Custom depth estimation
- Custom relationship extraction

### 📚 Well Documented
- Complete installation guide
- API reference
- Working examples
- Production-ready notebook

### 🚀 Production Ready
- Real dataset examples
- Performance benchmarks
- Error handling
- Best practices

---

## 💡 Examples

### Example 1: Basic Processing

```python
from gom import GraphOfMarks

gom = GraphOfMarks()
result = gom.process_image("photo.jpg")
```

### Example 2: SoM Baseline for VQA

```python
gom = GraphOfMarks(label_mode="numeric")
result = gom.process_image("photo.jpg", question="What is object 2 doing?")
```

### Example 3: Custom MobileSAM

```python
gom = GraphOfMarks(custom_segmenter=mobile_sam_function)
result = gom.process_image("photo.jpg")
```

### Example 4: Batch Processing

```python
gom = GraphOfMarks()
results = gom.process_batch(["img1.jpg", "img2.jpg", "img3.jpg"])
```

---

## 🎓 Learn More

- **Demo Notebook**: `examples/demo_notebook.ipynb` - Complete tutorial
- **Quick Start**: `examples/quickstart.py` - Simple example
- **Documentation**: See docs listed above
- **Issues**: [GitHub Issues](https://github.com/disi-unibo-nlp/graph-of-marks/issues)

---

## 📝 Summary

**Package Name**: `graph-of-marks`
**Import**: `from gom import GraphOfMarks`
**CLI**: `gom-preprocess`, `gom-vqa`
**Demo**: `examples/demo_notebook.ipynb`

**Get Started**:
```bash
pip install -e .
python examples/quickstart.py
jupyter notebook examples/demo_notebook.ipynb
```

🎉 **Graph of Marks is ready for production use!**
