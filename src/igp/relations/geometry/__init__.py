# igp/relations/geometry/__init__.py
# Unified interface for geometry utilities
# Re-exports all functions to maintain backward compatibility

from __future__ import annotations

# Core utilities
from .core import (
    as_xyxy,
    area,
    center,
    center_distance,
    iou,
    iou_matrix,
    giou,
    diou,
    horizontal_overlap,
    vertical_overlap,
    edge_gap,
    overlap_ratio,
    is_inside,
    contains,
)

# Vectorized operations
from .vectorized import (
    centers_vectorized,
    areas_vectorized,
    pairwise_distances_vectorized,
    horizontal_overlaps_vectorized,
    vertical_overlaps_vectorized,
)

# Mask operations
from .masks import (
    mask_iou,
    depth_stats_from_map,
)

# Spatial predicates
from .predicates import (
    orientation_label,
    is_on_top_of,
    is_below_of,
    is_in_front_of,
    is_behind_of,
)

# Nearest relation builder
from .nearest import (
    build_precise_nearest_relation,
)

__all__ = [
    # Core
    "as_xyxy",
    "area",
    "center",
    "center_distance",
    "iou",
    "iou_matrix",
    "giou",
    "diou",
    "horizontal_overlap",
    "vertical_overlap",
    "edge_gap",
    "overlap_ratio",
    "is_inside",
    "contains",
    # Vectorized
    "centers_vectorized",
    "areas_vectorized",
    "pairwise_distances_vectorized",
    "horizontal_overlaps_vectorized",
    "vertical_overlaps_vectorized",
    # Masks
    "mask_iou",
    "depth_stats_from_map",
    # Predicates
    "orientation_label",
    "is_on_top_of",
    "is_below_of",
    "is_in_front_of",
    "is_behind_of",
    # Nearest
    "build_precise_nearest_relation",
]
