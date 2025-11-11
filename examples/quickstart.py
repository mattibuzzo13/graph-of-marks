#!/usr/bin/env python
"""
Graph of Marks - Quick Start Example

This script demonstrates the basic usage of the Graph of Marks library.
Run this after installation to verify everything works.

Usage:
    python quickstart.py [path/to/image.jpg]

If no image is provided, a test image will be created.
"""

import sys
import numpy as np
from pathlib import Path
from PIL import Image

# Import Graph of Marks
try:
    from gom import GraphOfMarks
    print("✅ Successfully imported Graph of Marks")
except ImportError as e:
    print(f"❌ Failed to import Graph of Marks: {e}")
    print("\nPlease install the package first:")
    print("  pip install -e .")
    sys.exit(1)


def create_test_image(path: str = "test_image.jpg") -> str:
    """Create a simple test image with colored rectangles."""
    print(f"📸 Creating test image: {path}")

    # Create a white canvas
    img = np.ones((480, 640, 3), dtype=np.uint8) * 255

    # Draw some colored rectangles (simulating objects)
    # Red rectangle
    img[100:200, 100:250] = [255, 0, 0]
    # Green rectangle
    img[150:300, 300:450] = [0, 255, 0]
    # Blue rectangle
    img[280:400, 150:300] = [0, 0, 255]
    # Yellow rectangle
    img[200:280, 450:580] = [255, 255, 0]

    # Save image
    Image.fromarray(img).save(path)
    print(f"   Saved to: {path}")

    return path


def main():
    """Main function."""
    print("=" * 70)
    print("Graph of Marks - Quick Start")
    print("=" * 70)
    print()

    # Check if image path provided
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        if not Path(image_path).exists():
            print(f"❌ Image not found: {image_path}")
            print("Creating test image instead...")
            image_path = create_test_image()
    else:
        print("No image provided, creating test image...")
        image_path = create_test_image()

    print()

    # Example 1: Basic Usage
    print("=" * 70)
    print("Example 1: Basic Usage")
    print("=" * 70)
    print()

    print("Creating Graph of Marks pipeline with default settings...")
    gom = GraphOfMarks(
        detectors=["yolov8"],  # Use YOLOv8 for fast demo
        output_folder="quickstart_output"
    )
    print("✅ Pipeline created")
    print()

    print(f"Processing image: {image_path}")
    print("(This may take a few seconds on first run while models load...)")

    try:
        result = gom.process_image(image_path)

        print()
        print("✅ Processing complete!")
        print()
        print("📊 Results:")
        print(f"   - Detected {len(result['detections'])} objects")
        print(f"   - Found {len(result['relations'])} relationships")

        if 'output_path' in result:
            print(f"   - Visualization saved to: {result['output_path']}")

        print()

        if result['detections']:
            print("🔍 Detected Objects:")
            for i, det in enumerate(result['detections'][:5], 1):
                print(f"   {i}. {det['label']} (confidence: {det['score']:.2f})")
            if len(result['detections']) > 5:
                print(f"   ... and {len(result['detections']) - 5} more")

        print()

        if result['relations']:
            print("🔗 Relationships:")
            for i, rel in enumerate(result['relations'][:5], 1):
                src = result['detections'][rel['source_id']]['label']
                tgt = result['detections'][rel['target_id']]['label']
                rel_type = rel['relation_type']
                print(f"   {i}. {src} --{rel_type}--> {tgt}")
            if len(result['relations']) > 5:
                print(f"   ... and {len(result['relations']) - 5} more")

    except Exception as e:
        print(f"❌ Error during processing: {e}")
        print()
        print("This might be because:")
        print("  1. Models haven't been downloaded yet")
        print("  2. CUDA/GPU is not available (processing will be slow)")
        print("  3. Not enough memory")
        print()
        print("Try running: bash download_ckpt.sh")
        return

    print()
    print("=" * 70)
    print("Example 2: Configuration Options")
    print("=" * 70)
    print()

    print("You can customize the pipeline with different options:")
    print()
    print("# Fast configuration")
    print("gom = GraphOfMarks(")
    print("    detectors=['yolov8'],")
    print("    sam_version='fast',")
    print("    use_depth=False")
    print(")")
    print()
    print("# High-quality configuration")
    print("gom = GraphOfMarks(")
    print("    detectors=['owlvit', 'yolov8', 'detectron2'],")
    print("    sam_version='sam2',")
    print("    use_depth=True")
    print(")")
    print()

    print("=" * 70)
    print("Example 3: Question-Aware Processing")
    print("=" * 70)
    print()

    print("Process images with specific questions:")
    print()
    print("result = gom.process_image(")
    print("    'image.jpg',")
    print("    question='What objects are on the table?'")
    print(")")
    print()

    print("=" * 70)
    print("Next Steps")
    print("=" * 70)
    print()
    print("1. Run the demo notebook:")
    print("   jupyter notebook examples/demo_notebook.ipynb")
    print()
    print("2. Read the documentation:")
    print("   - INSTALLATION.md - Installation guide")
    print("   - PACKAGE_USAGE.md - Usage guide")
    print("   - README.md - Project overview")
    print()
    print("3. Try custom functions:")
    print("   See examples/demo_notebook.ipynb for custom segmentation,")
    print("   detection, and depth estimation examples")
    print()
    print("4. Process your own images:")
    print("   python quickstart.py path/to/your/image.jpg")
    print()
    print("=" * 70)
    print("✅ Quick start completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
