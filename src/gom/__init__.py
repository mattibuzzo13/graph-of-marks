"""
Graph of Marks (GoM) - Visual Scene Understanding Pipeline

Extracts objects, masks, depth, and relationships from images.
Supports custom functions for detection, segmentation, and depth.

Key Design Principle:
    Models are loaded ONCE at initialization. Processing configuration
    (thresholds, limits, visualization options) can be changed per-call
    via ProcessingConfig, similar to how LLM inference uses SamplingParams.

Quick Start:
    >>> from gom import GoM, ProcessingConfig
    >>>
    >>> # Initialize once (loads models)
    >>> gom = GoM()
    >>>
    >>> # Process with default settings
    >>> result = gom.process("scene.jpg")
    >>>
    >>> # Process with different configurations (no model reload!)
    >>> result = gom.process("scene.jpg", config=ProcessingConfig(threshold=0.8))
    >>> result = gom.process("scene.jpg", config=ProcessingConfig(style="gom_numeric_labeled"))
    >>> result = gom.process("scene.jpg", config=ProcessingConfig(
    ...     max_detections=5,
    ...     max_relations_per_object=2,
    ...     display_relationships=False
    ... ))

GoM Visual Prompting Styles (AAAI 2026 Paper):
    Use the `style` parameter in ProcessingConfig to switch configurations:

    >>> config = ProcessingConfig(style="gom_text_labeled")  # Best for VQA
    >>> config = ProcessingConfig(style="gom_numeric_labeled")  # Best for REC

    Available styles:
        - "som_text": Set-of-Mark with textual IDs (oven_1, chair_2)
        - "som_numeric": Set-of-Mark with numeric IDs (1, 2, 3)
        - "gom_text": GoM with textual IDs + relation arrows
        - "gom_numeric": GoM with numeric IDs + relation arrows
        - "gom_text_labeled": GoM with textual IDs + labeled relations
        - "gom_numeric_labeled": GoM with numeric IDs + labeled relations

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
    GoM, ProcessingConfig, create_pipeline
    GOM_STYLE_PRESETS, GomStyle
    ImageGraphPreprocessor, PreprocessorConfig
    Detection, Relationship, Box, MaskDict
    SegmenterConfig, RelationsConfig, VisualizerConfig
"""
from __future__ import annotations

__all__ = [
    # High-level API
    "GoM",
    "ProcessingConfig",
    "create_pipeline",
    "run",
    # Backward compatibility aliases
    "Gom",
    "GraphOfMarks",
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

__version__ = "0.1.19"

# High-level API
from .api import GOM_STYLE_PRESETS, GoM, Gom, GomStyle, GraphOfMarks, ProcessingConfig, create_pipeline, run

# Configuration objects
from .config import (
    PreprocessorConfig,
    RelationsConfig,
    SegmenterConfig,
    VisualizerConfig,
    default_config,
)

# Pipeline (advanced users)
from .pipeline.preprocessor import ImageGraphPreprocessor

# Core public types
from .types import Box, Detection, MaskDict, Relationship
