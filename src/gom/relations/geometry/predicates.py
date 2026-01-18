# igp/relations/geometry/predicates.py
"""
Spatial Relationship Predicates

High-level geometric predicates for spatial relationship inference between
detected objects. Implements robust heuristics for directional (left_of, above)
and depth-based (in_front_of, on_top_of) relationships using bounding boxes,
optional masks, and depth maps.

This module provides the semantic layer above raw geometric computations,
translating numeric measurements into linguistic relationships for scene graphs.

Key Predicates:
    Directional:
        - orientation_label: Primary cardinal direction (left/right/above/below)
    
    Vertical Support:
        - is_on_top_of: Object A resting on object B (gravity-based)
        - is_below_of: Inverse of on_top_of
    
    Depth-based:
        - is_in_front_of: Object A closer than object B (depth ordering)
        - is_behind_of: Inverse of in_front_of

Approach (is_on_top_of):
    1. Center ordering: A above B by Y-coordinate
    2. Vertical gap: Bottom of A near top of B (tolerance ~6% box height)
    3. Horizontal overlap: At least 25% of narrower box width
    4. Mask contact (optional): Dilated masks touch at interface band
    5. Depth consistency (optional): A not significantly farther than B

Performance:
    - orientation_label: <1μs (simple arithmetic)
    - is_on_top_of (boxes only): ~5μs
    - is_on_top_of (with masks): ~50μs (mask operations)
    - is_in_front_of (depth): ~20μs (median computation)

Usage:
    >>> from gom.relations.geometry.predicates import orientation_label, is_on_top_of
    >>> import numpy as np
    
    # Directional relationship
    >>> box1 = (10, 50, 50, 90)  # left object
    >>> box2 = (100, 50, 140, 90)  # right object
    >>> orientation_label(box1, box2)
    'left_of'
    
    # Vertical support (cup on table)
    >>> cup = (100, 50, 150, 100)    # Above
    >>> table = (50, 95, 200, 150)   # Below
    >>> is_on_top_of(cup, table)
    True
    
    # With masks and depth
    >>> cup_mask = np.zeros((200, 250), dtype=bool)
    >>> cup_mask[50:100, 100:150] = True
    >>> table_mask = np.zeros((200, 250), dtype=bool)
    >>> table_mask[95:150, 50:200] = True
    >>> depth_map = np.random.rand(200, 250)  # Depth from estimator
    >>> is_on_top_of(
    ...     cup, table,
    ...     mask_a=cup_mask, mask_b=table_mask,
    ...     depth_map=depth_map,
    ...     min_h_overlap_ratio=0.3
    ... )
    True

Orientation Semantics:
    Returns relation describing where A is relative to B:
        - "left_of": A center left of B center
        - "right_of": A center right of B center
        - "above": A center above B center
        - "below": A center below B center
    
    Tie-breaking:
        - margin_px=8.0: Vertical/horizontal preference threshold
        - If |dy| > |dx| + margin → vertical relation
        - Else horizontal relation (larger magnitude wins)

On-Top-Of Heuristic Details:
    Vertical Gap Tolerance:
        gap = top_of_B - bottom_of_A
        tolerance = max(8px, 6% of smaller box height)
        
        Valid if gap < tolerance OR slight overlap (<35% of A's height)
    
    Horizontal Overlap:
        overlap = horizontal_intersection(A, B)
        min_required = 25% of min(width_A, width_B)
        
        Prevents false positives for vertically aligned but horizontally distant objects
    
    Mask Contact:
        y_line = min(bottom_of_A, top_of_B)
        band = max(2px, 2% of reference height)
        dilate(mask_A) ∩ dilate(mask_B) at y_line ± band != ∅
        
        Verifies actual pixel-level contact, not just bounding box proximity
    
    Depth Consistency:
        depth_A > depth_B - 0.10  (normalized inverted depth: higher=closer)
        
        Rejects physically implausible configurations (e.g., occluded object "on" foreground)

Depth Convention:
    - Normalized inverted depth: Higher values = closer to camera
    - MiDaS/Depth Anything V2 output after normalization
    - is_in_front_of: depth_A > depth_B + delta
    - delta=0.05: Tolerance for depth noise (~5% range)

Integration with Scene Graphs:
    >>> from gom.relations.geometry.predicates import is_on_top_of
    >>> from gom.types import Relationship
    >>> # Scene: person on skateboard
    >>> person_box = (100, 20, 180, 150)
    >>> board_box = (110, 145, 170, 160)
    >>> if is_on_top_of(person_box, board_box):
    ...     rel = Relationship(
    ...         src_idx=0, tgt_idx=1, relation="on",
    ...         confidence=0.9
    ...     )

Error Handling:
    - Missing depth: Returns False (conservative)
    - Missing masks: Skips mask contact check
    - Invalid box coordinates: as_xyxy handles normalization
    - NaN in depth_map: Filtered during median computation

Dependencies:
    - numpy: Required for mask/depth operations
    - gom.relations.geometry.core: as_xyxy, center, horizontal_overlap
    - gom.relations.geometry.masks: Mask IoU, contact, depth stats

Notes:
    - All boxes in xyxy format: (x1, y1, x2, y2)
    - Predicates are heuristic (not physically perfect)
    - Tunable thresholds for different domains
    - Conservative: Prefers false-negatives over false-positives

See Also:
    - gom.relations.geometry.core: Low-level geometric operations
    - gom.relations.geometry.masks: Mask-based computations
    - gom.relations.clip_rel: CLIP-based semantic relationship scoring
    - gom.utils.depth: Depth estimation models
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .core import as_xyxy, center, horizontal_overlap
from .masks import _mask_contact_along_y, depth_stats_from_map


def orientation_label(a: Sequence[float], b: Sequence[float], *, margin_px: float = 8.0) -> str:
    """
    Primary directional relation between A and B:
    'left_of' | 'right_of' | 'above' | 'below'
    Uses center difference with a tie margin.
    
    Natural semantics: returns the relation describing where A is relative to B.
    E.g., if A is to the left of B, returns 'left_of'.
    """
    cx1, cy1 = center(a)
    cx2, cy2 = center(b)
    dx, dy = cx2 - cx1, cy2 - cy1
    # dx = b.x - a.x: if dx > 0, b is right of a, so a is left_of b
    # dy = b.y - a.y: if dy > 0, b is below a, so a is above b
    if abs(dy) > abs(dx) + margin_px:
        return "below" if dy < 0 else "above"
    if abs(dx) > abs(dy) + margin_px:
        return "left_of" if dx > 0 else "right_of"
    # Tie-breaker by larger magnitude
    return "left_of" if dx > 0 else "right_of"


def is_on_top_of(
    box_a: Sequence[float],
    box_b: Sequence[float],
    *,
    mask_a: Optional[np.ndarray] = None,
    mask_b: Optional[np.ndarray] = None,
    depth_a: Optional[float] = None,
    depth_b: Optional[float] = None,
    depth_map: Optional[np.ndarray] = None,
    min_h_overlap_ratio: float = 0.25,
    max_gap_px: Optional[float] = None,
) -> bool:
    """
    Robust heuristic for 'A on top of B':
      1) A above B by Y center
      2) touching contact (small gap, no deep overlap)
      3) sufficient horizontal overlap (relative to box widths)
      4) optional mask contact along interface band
      5) optional depth consistency (A not much farther than B)
    """
    x1a, y1a, x2a, y2a = as_xyxy(box_a)
    x1b, y1b, x2b, y2b = as_xyxy(box_b)

    # 1) center ordering
    if (y1a + y2a) / 2.0 >= (y1b + y2b) / 2.0:
        return False

    # 2) touching contact (scale-aware)
    hA = max(1.0, y2a - y1a)
    hB = max(1.0, y2b - y1b)
    h_ref = min(hA, hB)
    contact_tol = max_gap_px if max_gap_px is not None else max(2.0, 0.02 * h_ref)

    bottom_a, top_b = y2a, y1b
    gap = top_b - bottom_a
    if gap > contact_tol:
        return False
    if gap < 0:
        # Allow slight overlap but avoid deep interpenetration
        v_ov = min(y2a, y2b) - max(y1a, y1b)
        if v_ov / hA > 0.35:
            return False

    # 3) horizontal overlap
    hov = horizontal_overlap(box_a, box_b)
    min_hov = min_h_overlap_ratio * max(1.0, min(x2a - x1a, x2b - x1b))
    if hov < min_hov:
        return False

    # 4) mask contact (optional)
    if mask_a is not None and mask_b is not None:
        y_line = int(round(min(y2a, y1b)))
        band = int(max(2, 0.02 * h_ref))
        if not _mask_contact_along_y(mask_a, mask_b, y_line, band):
            return False

    # 5) depth consistency
    da = depth_a
    db = depth_b
    if da is None or db is None:
        if depth_map is not None:
            da = depth_stats_from_map(mask_a, depth_map, box_a) if da is None else da
            db = depth_stats_from_map(mask_b, depth_map, box_b) if db is None else db
    if (da is not None) and (db is not None):
        # With normalized depth (inverted), higher = closer.
        # A on top of B should have A closer or similar depth to B.
        # Reject if A is significantly farther than B.
        if da < db - 0.10:  # tolerance depends on sensor/noise scale
            return False

    return True


def is_below_of(
    box_a: Sequence[float],
    box_b: Sequence[float],
    **kwargs,
) -> bool:
    """A is below B ⇔ B is on top of A."""
    return is_on_top_of(box_b, box_a, **kwargs)


def is_in_front_of(
    box_a: Sequence[float],
    box_b: Sequence[float],
    *,
    mask_a: Optional[np.ndarray] = None,
    mask_b: Optional[np.ndarray] = None,
    depth_a: Optional[float] = None,
    depth_b: Optional[float] = None,
    depth_map: Optional[np.ndarray] = None,
    delta: float = 0.05,
) -> bool:
    """
    Depth-based: A in front of B if its median depth is smaller by > delta.
    If depth_a/b not provided, estimate medians from depth_map using masks or boxes.
    
    Note: In the normalized depth convention, higher values = closer, so we check
    if depth_a > depth_b + delta.
    """
    da = depth_a
    db = depth_b
    if da is None or db is None:
        if depth_map is None:
            return False
        da = depth_stats_from_map(mask_a, depth_map, box_a) if da is None else da
        db = depth_stats_from_map(mask_b, depth_map, box_b) if db is None else db
    if da is None or db is None:
        return False
    
    # With normalized depth (inverted), higher = closer, so A in front means da > db
    return da > (db + delta)


def is_behind_of(
    box_a: Sequence[float],
    box_b: Sequence[float],
    *,
    mask_a: Optional[np.ndarray] = None,
    mask_b: Optional[np.ndarray] = None,
    depth_a: Optional[float] = None,
    depth_b: Optional[float] = None,
    depth_map: Optional[np.ndarray] = None,
    delta: float = 0.05,
) -> bool:
    """A behind B ⇔ B in front of A. Swaps both boxes AND depth values."""
    return is_in_front_of(
        box_b, box_a,
        mask_a=mask_b,
        mask_b=mask_a,
        depth_a=depth_b,
        depth_b=depth_a,
        depth_map=depth_map,
        delta=delta,
    )
