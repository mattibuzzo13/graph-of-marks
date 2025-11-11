# Graph of Marks - Examples

This directory contains examples demonstrating how to use the Graph of Marks library with **real datasets and models**.

## Quick Start

### 1. Run the Quick Start Script

The fastest way to verify your installation and see basic usage:

```bash
# Run with a test image (auto-generated)
python quickstart.py

# Run with your own image
python quickstart.py path/to/your/image.jpg
```

This script will:
- ✅ Test that Graph of Marks is installed correctly
- ✅ Create a simple pipeline
- ✅ Process an image
- ✅ Show detected objects and relationships
- ✅ Display next steps

### 2. Run the Demo Notebook (Recommended)

**The comprehensive demo showing all features with real datasets and models:**

```bash
# Install Jupyter if needed
pip install jupyter

# Start Jupyter
jupyter notebook demo_notebook.ipynb
```

**The notebook includes 7 sections covering:**
1. **Real Datasets** - COCO, GQA, VQA v2 examples
2. **GoM Features** - Numeric labels (SoM baseline), Alphabetic labels
3. **Default Model Signatures** - How to implement compatible custom functions
4. **Real Custom Models**:
   - MobileSAM (segmentation alternative)
   - Detectron2 (detection alternative)
5. **VQA with Real Data** - Question-aware processing with GQA dataset
6. **Model Comparison** - Performance benchmarks
7. **Production Examples** - Ready-to-use configurations

## Example Files

### `quickstart.py`
- **Purpose**: Simple script to test installation and basic functionality
- **Features**:
  - Tests imports
  - Creates test image if needed
  - Demonstrates basic pipeline usage
  - Shows results and next steps
- **Usage**: `python quickstart.py [image_path]`

### `demo_notebook.ipynb`
- **Purpose**: Comprehensive Jupyter notebook with all features
- **Sections**: 12 sections covering all aspects of the library
- **Best for**: Learning and experimentation
- **Usage**: `jupyter notebook demo_notebook.ipynb`

## Custom Function Examples

The library allows you to replace default models with your own. Here are templates:

### Custom Segmentation

```python
from gom import GraphOfMarks
import numpy as np

def my_segmenter(image, boxes, **kwargs):
    """
    Custom segmentation function.

    Args:
        image: np.ndarray (H, W, 3) - RGB image
        boxes: List of [x1, y1, x2, y2] bounding boxes

    Returns:
        Dict with 'masks': List of binary masks (H, W)
    """
    masks = []
    h, w = image.shape[:2]

    for box in boxes:
        x1, y1, x2, y2 = map(int, box)

        # Your segmentation model here
        # Example: simple rectangular mask
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[y1:y2, x1:x2] = 1

        masks.append(mask)

    return {'masks': masks}

# Use custom segmentation
gom = GraphOfMarks(custom_segmenter=my_segmenter)
result = gom.process_image("image.jpg")
```

### Custom Detection

```python
from gom import GraphOfMarks, Detection

def my_detector(image, **kwargs):
    """
    Custom detection function.

    Args:
        image: np.ndarray (H, W, 3) - RGB image

    Returns:
        List of Detection objects
    """
    # Your detection model here
    boxes = [[100, 100, 200, 200], [250, 150, 400, 300]]
    labels = ["person", "car"]
    scores = [0.95, 0.87]

    detections = [
        Detection(box=box, label=label, score=score)
        for box, label, score in zip(boxes, labels, scores)
    ]

    return detections

# Use custom detection
gom = GraphOfMarks(custom_detector=my_detector)
result = gom.process_image("image.jpg")
```

### Custom Depth Estimation

```python
from gom import GraphOfMarks
import numpy as np

def my_depth_estimator(image, **kwargs):
    """
    Custom depth estimation function.

    Args:
        image: np.ndarray (H, W, 3) - RGB image

    Returns:
        np.ndarray (H, W) - Depth map [0, 1], 1=closest
    """
    h, w = image.shape[:2]

    # Your depth model here
    # Example: simple gradient
    depth_map = np.linspace(0, 1, h).reshape(-1, 1)
    depth_map = np.repeat(depth_map, w, axis=1)

    return depth_map.astype(np.float32)

# Use custom depth estimation
gom = GraphOfMarks(
    custom_depth_estimator=my_depth_estimator,
    use_depth=True
)
result = gom.process_image("image.jpg")
```

### Custom Relationship Extraction

```python
from gom import GraphOfMarks, Relationship

def my_relation_extractor(detections, image, **kwargs):
    """
    Custom relationship extraction function.

    Args:
        detections: List of Detection objects
        image: np.ndarray (H, W, 3) - RGB image

    Returns:
        List of Relationship objects
    """
    relationships = []

    # Your relationship logic here
    for i in range(len(detections) - 1):
        relationships.append(
            Relationship(
                source_id=i,
                target_id=i + 1,
                relation_type="next_to",
                confidence=0.9
            )
        )

    return relationships

# Use custom relationships
gom = GraphOfMarks(custom_relation_extractor=my_relation_extractor)
result = gom.process_image("image.jpg")
```

## Common Use Cases

### 1. Basic Image Processing

```python
from gom import GraphOfMarks

gom = GraphOfMarks()
result = gom.process_image("photo.jpg")

print(f"Detected {len(result['detections'])} objects")
print(f"Found {len(result['relations'])} relationships")
```

### 2. Fast Prototyping

```python
gom = GraphOfMarks(
    detectors=["yolov8"],
    sam_version="fast",
    use_depth=False
)
result = gom.process_image("photo.jpg")
```

### 3. High Quality Results

```python
gom = GraphOfMarks(
    detectors=["owlvit", "yolov8", "detectron2"],
    sam_version="sam2",
    use_depth=True
)
result = gom.process_image("photo.jpg")
```

### 4. Question-Aware Processing

```python
gom = GraphOfMarks()
result = gom.process_image(
    "photo.jpg",
    question="What objects are on the table?"
)
```

### 5. Batch Processing

```python
gom = GraphOfMarks()
images = ["img1.jpg", "img2.jpg", "img3.jpg"]
results = gom.process_batch(images)
```

### 6. Directory Processing

```python
gom = GraphOfMarks()
results = gom.process_directory(
    "images/",
    pattern="*.jpg",
    recursive=True
)
```

## Performance Tips

### For Speed

```python
gom = GraphOfMarks(
    detectors=["yolov8"],       # Single fast detector
    sam_version="fast",         # FastSAM
    use_depth=False,            # Skip depth
    use_clip_relations=False    # Skip CLIP relations
)
```

### For Quality

```python
gom = GraphOfMarks(
    detectors=["owlvit", "yolov8", "detectron2"],
    sam_version="sam2",
    use_depth=True,
    use_clip_relations=True
)
```

### For Memory

```python
# Process one at a time
for image_path in image_paths:
    result = gom.process_image(image_path)
    # Clear cache between images
    import gc, torch
    gc.collect()
    torch.cuda.empty_cache()
```

## Creating Your Own Examples

### Template Script

```python
from gom import GraphOfMarks
from pathlib import Path

def main():
    # Configure pipeline
    gom = GraphOfMarks(
        detectors=["yolov8"],
        output_folder="my_output"
    )

    # Process image
    image_path = "path/to/image.jpg"
    result = gom.process_image(image_path)

    # Use results
    print(f"Detected: {len(result['detections'])} objects")
    print(f"Relations: {len(result['relations'])} relationships")

    # Access scene graph
    scene_graph = result['scene_graph']
    print(f"Graph has {scene_graph.number_of_nodes()} nodes")

    # Export results
    import json
    with open("results.json", "w") as f:
        json.dump(result['scene_graph_json'], f, indent=2)

if __name__ == "__main__":
    main()
```

## Directory Structure

```
examples/
├── README.md              # This file
├── quickstart.py          # Quick start script
├── demo_notebook.ipynb    # Comprehensive Jupyter demo
└── demo_images/           # Place test images here (optional)
```

## Getting Help

- 📚 [Installation Guide](../INSTALLATION.md)
- 📖 [Package Usage Guide](../PACKAGE_USAGE.md)
- 📋 [Main README](../README.md)
- 🐛 [Report Issues](https://github.com/disi-unibo-nlp/graph-of-marks/issues)
- 💬 [Discussions](https://github.com/disi-unibo-nlp/graph-of-marks/discussions)

## Next Steps

1. **Run quick start**: `python quickstart.py`
2. **Explore notebook**: `jupyter notebook demo_notebook.ipynb`
3. **Try custom functions**: See templates above
4. **Read documentation**: [PACKAGE_USAGE.md](../PACKAGE_USAGE.md)
5. **Process your images**: Adapt examples to your use case

Happy scene understanding! 🎉
