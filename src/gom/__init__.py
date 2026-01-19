"""
Graph of Marks (GoM) - Visual Scene Understanding Pipeline

Extracts objects, masks, depth, and relationships from images.
Supports custom functions for detection, segmentation, and depth.

Quick Start:
    >>> from gom import Gom
    >>>
    >>> # Default models (YOLOv8 + SAM-HQ + Depth Anything V2)
    >>> gom = Gom()
    >>> result = gom.process("scene.jpg")
    >>>
    >>> # Custom detection function
    >>> def my_detector(image):
    ...     boxes, labels, scores = run_yolo(image)
    ...     return boxes, labels, scores
    >>>
    >>> gom = Gom(detect_fn=my_detector)
    >>> result = gom.process("scene.jpg")

GoM Visual Prompting Styles (AAAI 2026 Paper):
    The library supports different visual prompting configurations as described
    in the Graph-of-Mark paper. Use the `style` parameter to switch:

    >>> # SoM-style with numeric IDs (no relations)
    >>> gom = Gom(style="som_numeric")
    >>>
    >>> # Full GoM with textual IDs and labeled relations (best for VQA)
    >>> gom = Gom(style="gom_text_labeled")
    >>>
    >>> # Full GoM with numeric IDs and labeled relations (best for REC)
    >>> gom = Gom(style="gom_numeric_labeled")

    Available styles:
        - "som_text": Set-of-Mark with textual IDs (oven_1, chair_2)
        - "som_numeric": Set-of-Mark with numeric IDs (1, 2, 3)
        - "gom_text": GoM with textual IDs + relation arrows
        - "gom_numeric": GoM with numeric IDs + relation arrows
        - "gom_text_labeled": GoM with textual IDs + labeled relations
        - "gom_numeric_labeled": GoM with numeric IDs + labeled relations

    For Visual + Textual SG prompting (multimodal), access:
        result["scene_graph_text"]   # Triples format for LLM prompts
        result["scene_graph_prompt"] # Compact inline format

Custom Functions:
    detect_fn(image: Image) -> (boxes, labels, scores)
        boxes: List of [x1, y1, x2, y2]
        labels: List of class names
        scores: List of confidence values

    segment_fn(image: Image, boxes: List) -> List[np.ndarray]
        Returns binary masks (H, W) for each box

    depth_fn(image: Image) -> np.ndarray
        Returns normalized depth map (H, W) in [0, 1]

Exports:
    Gom, GraphOfMarks, create_pipeline
    GOM_STYLE_PRESETS, GomStyle
    ImageGraphPreprocessor
    Detection, Relationship, Box, MaskDict
    PreprocessorConfig, SegmenterConfig, RelationsConfig, VisualizerConfig

Version: 0.1.3
"""
from __future__ import annotations

__all__ = [
    # High-level API
    "Gom",
    "GraphOfMarks",
    "create_pipeline",
    # GoM style presets (AAAI 2026 paper configurations)
    "GOM_STYLE_PRESETS",
    "GomStyle",
    # Core pipeline (advanced)
    "ImageGraphPreprocessor",
    # Types
    "Detection",
    "Relationship",
    "Box",
    "MaskDict",
    # Configuration
    "PreprocessorConfig",
    "SegmenterConfig",
    "RelationsConfig",
    "VisualizerConfig",
    "default_config",
]

__version__ = "0.1.4"

# Core public types
from .types import Detection, Relationship, Box, MaskDict

# Configuration objects
from .config import (
    PreprocessorConfig,
    SegmenterConfig,
    RelationsConfig,
    VisualizerConfig,
    default_config,
)

# Pipeline (advanced users)
from .pipeline.preprocessor import ImageGraphPreprocessor

# High-level API
from .api import Gom, GraphOfMarks, create_pipeline, GOM_STYLE_PRESETS, GomStyle
