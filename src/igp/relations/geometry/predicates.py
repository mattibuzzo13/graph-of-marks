# igp/relations/geometry/predicates.py
# Spatial predicates for relation inference: on_top_of, below, in_front_of, behind, orientation

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
      2) vertical gap small (allow slight overlap)
      3) sufficient horizontal overlap (relative to box widths)
      4) optional mask contact along interface band
      5) optional depth consistency (A not much farther than B)
    """
    x1a, y1a, x2a, y2a = as_xyxy(box_a)
    x1b, y1b, x2b, y2b = as_xyxy(box_b)

    # 1) center ordering
    if (y1a + y2a) / 2.0 >= (y1b + y2b) / 2.0:
        return False

    # 2) vertical gap tolerance (scale-aware)
    hA = max(1.0, y2a - y1a)
    hB = max(1.0, y2b - y1b)
    h_ref = min(hA, hB)
    tol_px = max_gap_px if max_gap_px is not None else max(8.0, 0.06 * h_ref)

    bottom_a, top_b = y2a, y1b
    gap = top_b - bottom_a
    if gap > tol_px:
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
