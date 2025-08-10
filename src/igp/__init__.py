# igp/__init__.py
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
