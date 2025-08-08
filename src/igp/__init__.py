# igp/__init__.py
from __future__ import annotations

__all__ = [
    # tipi
    "Detection",
    "Relationship",
    "Box",
    "MaskDict",
    # config re-export
    "PreprocessorConfig",
    "SegmenterConfig",
    "RelationsConfig",
    "VisualizerConfig",
    "default_config",
]

__version__ = "0.1.0"

# Tipi principali
from .types import Detection, Relationship, Box, MaskDict

# Re-export dei config dalle rispettive sottosezioni
from .config import (
    PreprocessorConfig,
    SegmenterConfig,
    RelationsConfig,
    VisualizerConfig,
    default_config,
)
