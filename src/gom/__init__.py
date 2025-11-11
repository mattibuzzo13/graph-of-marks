# igp/__init__.py
"""
Graph of Marks (GoM) - Visual Scene Understanding Pipeline

This package provides a comprehensive toolkit for visual scene understanding through
multi-modal detection, segmentation, and relationship extraction. It combines multiple
state-of-the-art computer vision models to build rich scene graphs from images.

Key Components:
    - Detection: Multi-model object detection with ensemble fusion
    - Segmentation: Instance segmentation with SAM variants
    - Relationships: Spatial and semantic relationship extraction
    - Visualization: Publication-quality rendering with customizable styles
    - Graph: Scene graph construction and serialization
    - VQA: Visual Question Answering integration

Quick Start:
    >>> from gom import GraphOfMarks
    >>>
    >>> # Basic usage with default models
    >>> gom = GraphOfMarks()
    >>> result = gom.process_image("scene.jpg")
    >>> print(f"Detected {len(result['detections'])} objects")
    >>>
    >>> # With custom configuration
    >>> gom = GraphOfMarks(
    ...     detectors=["yolov8", "owlvit"],
    ...     sam_version="sam2",
    ...     use_depth=True
    ... )
    >>> result = gom.process_image("scene.jpg", question="What is on the table?")
    >>>
    >>> # With custom segmentation function
    >>> def my_segmenter(image, boxes):
    ...     # Your custom segmentation logic
    ...     return masks
    >>>
    >>> gom = GraphOfMarks(custom_segmenter=my_segmenter)
    >>> result = gom.process_image("scene.jpg")

Advanced Usage - Custom Functions:
    The library supports custom implementations for key components:

    1. Custom Segmentation:
        >>> def custom_segment(image, boxes, **kwargs):
        ...     # boxes: List of [x1, y1, x2, y2]
        ...     # Return: Dict with 'masks' key containing binary masks
        ...     masks = your_segmentation_model(image, boxes)
        ...     return {'masks': masks}

    2. Custom Detection:
        >>> def custom_detect(image, **kwargs):
        ...     # Return: List of Detection objects
        ...     boxes, labels, scores = your_detector(image)
        ...     return [Detection(box=b, label=l, score=s)
        ...             for b, l, s in zip(boxes, labels, scores)]

    3. Custom Depth Estimation:
        >>> def custom_depth(image, **kwargs):
        ...     # Return: numpy array of normalized depth values [0, 1]
        ...     depth_map = your_depth_model(image)
        ...     return depth_map

    4. Custom Relationship Extraction:
        >>> def custom_relations(detections, image, **kwargs):
        ...     # Return: List of Relationship objects
        ...     relations = your_relation_model(detections, image)
        ...     return relations

Exports:
    Main API:
        GraphOfMarks: High-level interface for scene understanding
        ImageGraphPreprocessor: Lower-level pipeline (advanced users)

    Types:
        Detection: Object detection dataclass
        Relationship: Relationship dataclass
        Box: Bounding box type alias
        MaskDict: Segmentation mask dictionary

    Configuration:
        PreprocessorConfig: Main pipeline configuration
        SegmenterConfig: Segmentation settings
        RelationsConfig: Relationship extraction settings
        VisualizerConfig: Visualization rendering settings
        default_config: Factory for default configuration

Version: 0.1.0
"""
from __future__ import annotations

__all__ = [
    # High-level API
    "GraphOfMarks",
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

__version__ = "0.1.0"

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
from .api import GraphOfMarks
