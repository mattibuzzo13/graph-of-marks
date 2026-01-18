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
    ImageGraphPreprocessor
    Detection, Relationship, Box, MaskDict
    PreprocessorConfig, SegmenterConfig, RelationsConfig, VisualizerConfig

Version: 0.1.2
"""
from __future__ import annotations

__all__ = [
    # High-level API
    "Gom",
    "GraphOfMarks",
    "create_pipeline",
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

__version__ = "0.1.3"

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
from .api import Gom, GraphOfMarks, create_pipeline
