
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import torch
from datasets import load_dataset

# Add src to path
sys.path.append(str(Path.cwd() / "src"))

# Import Graph of Marks
from gom import GraphOfMarks
from gom.segmentation import Sam2Segmenter
from gom.vqa.models import OllamaWrapper

# Setup directories
OUTPUT_DIR = Path("demo_output")
OUTPUT_DIR.mkdir(exist_ok=True)

print(f"\n✅ Setup complete!")
print(f"Output directory: {OUTPUT_DIR.absolute()}")

# --- Part 1: Load Data ---
print("Loading COCO dataset...")
try:
    dataset = load_dataset("detection-datasets/coco", split="val", streaming=True)
    # Get first image
    sample = next(iter(dataset.take(1)))
    coco_image = sample['image']
    print("✅ Loaded COCO image")
except Exception as e:
    print(f"⚠️ Failed to load COCO: {e}")
    print("Using dummy image instead")
    coco_image = Image.new('RGB', (640, 480), color='white')

# Save original image for reference
coco_image.save(OUTPUT_DIR / "original.jpg")

# --- Part 2: Initialize Plugins ---

# SAM2
print("\nInitializing SAM2 Segmenter...")
try:
    # Using default config/checkpoint paths which might need adjustment
    # Assuming user has them in standard locations or we might need to mock/skip
    sam2_plugin = Sam2Segmenter(
        model_cfg="configs/sam2.1/sam2.1_hiera_l.yaml",
        checkpoint="./checkpoints/sam2.1_hiera_large.pt"
    )
    custom_segmenter_func = sam2_plugin.segment
    print("✅ SAM2 Plugin ready")
except Exception as e:
    print(f"⚠️ Failed to initialize SAM2: {e}")
    print("Ensure SAM2 is installed and checkpoints are available.")
    # Fallback to None to allow script to continue if possible (GoM might fail if segmenter is None but sam_version is not set)
    custom_segmenter_func = None

# Ollama
print("\nInitializing Ollama Wrapper...")
MODEL_NAME = "qwen3-vl:2b" # User requested qwen3-vl:2b
try:
    vqa_model = OllamaWrapper(model_name=MODEL_NAME, temperature=0.1)
    print(f"✅ Ollama wrapper initialized with {MODEL_NAME}")
except Exception as e:
    print(f"⚠️ Error initializing Ollama: {e}")
    vqa_model = None

# --- Part 3: GoM Processing ---
print("\nInitializing GoM with Custom Plugin...")
try:
    gom = GraphOfMarks(
        detectors=["yolov8"],  # Use YOLOv8 for detection
        custom_segmenter=custom_segmenter_func, # Inject SAM2 as a plugin
        output_folder=str(OUTPUT_DIR),
        label_mode="alphabetic" # Use A, B, C labels
    )

    # Process image
    print("Processing image...")
    result = gom.process_image(
        coco_image,
        save_visualization=True
    )

    print(f"Found {len(result['detections'])} objects")
    if result.get('output_path'):
        print(f"Saved to: {result['output_path']}")
except Exception as e:
    print(f"❌ GoM Processing failed: {e}")
    result = {}

# --- Part 4: VQA Experiments ---
if vqa_model and result:
    question = "Describe the spatial relationships between objects in this scene."
    print(f"\nQuestion: {question}")
    
    # 1. Baseline
    raw_path = OUTPUT_DIR / "exp_baseline.jpg"
    coco_image.save(raw_path)
    print("\n--- Baseline ---")
    print(vqa_model.generate(question, image_path=str(raw_path)))
    
    # 2. GoM
    if result.get('output_path'):
        print("\n--- Graph of Marks ---")
        scene_graph_str = result.get('scene_graph_json', '')
        prompt_with_context = f"Context: {scene_graph_str}\n\nQuestion: {question}"
        print(vqa_model.generate(prompt_with_context, image_path=result['output_path']))

print("\nDone!")
