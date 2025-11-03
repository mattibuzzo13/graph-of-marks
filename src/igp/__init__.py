# igp/__init__.py
"""
Image Graph Preprocessor (IGP) - Main Package

This package provides a comprehensive toolkit for visual scene understanding through
multi-modal detection, segmentation, and relationship extraction. It combines multiple
state-of-the-art computer vision models to build rich scene graphs from images.

Key Components:
    - Detection: Multi-model object detection with ensemble fusion
    - Segmentation: Instance segmentation with SAM variants
    - Relationships: Spatial and semantic relationship extraction
    - Visualization: Publication-quality rendering with customizable styles
    - Graph: Scene graph construction and serialization

Typical Usage:
    >>> from igp import PreprocessorConfig
    >>> from igp.pipeline.preprocessor import ImageGraphPreprocessor
    >>> 
    >>> config = PreprocessorConfig(detectors=["yolov8", "owlvit"])
    >>> preprocessor = ImageGraphPreprocessor(config)
    >>> result = preprocessor.process_image("scene.jpg")
    >>> print(f"Detected {len(result['detections'])} objects")

Exports:
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
    # exported types
    "Detection",
    "Relationship",
    "Box",
    "MaskDict",
    # config re-exports
    "PreprocessorConfig",
    "SegmenterConfig",
    "RelationsConfig",
    "VisualizerConfig",
    "default_config",
]

__version__ = "0.1.0"  # package version

# Core public types
from .types import Detection, Relationship, Box, MaskDict

# Re-export configuration objects from submodules
from .config import (
    PreprocessorConfig,
    SegmenterConfig,
    RelationsConfig,
    VisualizerConfig,
    default_config,
)
