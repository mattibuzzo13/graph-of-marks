# igp/relations/geometry.py
# Geometric heuristics for object relations: IoU, distances, orientation,
# "on_top_of"/"below" checks with optional masks/depth, and a precise "nearest" relation.

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False
    cv2 = None

import numpy as np


def as_xyxy(b: Sequence[float]) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = b[:4]
    return float(x1), float(y1), float(x2), float(y2)


def iou(b1: Sequence[float], b2: Sequence[float]) -> float:
    x1, y1, x2, y2 = as_xyxy(b1)
    X1, Y1, X2, Y2 = as_xyxy(b2)
    ix1, iy1 = max(x1, X1), max(y1, Y1)
    ix2, iy2 = min(x2, X2), min(y2, Y2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    a1 = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
    a2 = max(0.0, (X2 - X1)) * max(0.0, (Y2 - Y1))
    union = a1 + a2 - inter
    return float(inter / union) if union > 0 else 0.0


def center_distance(b1: Sequence[float], b2: Sequence[float]) -> float:
    x1, y1, x2, y2 = as_xyxy(b1)
    X1, Y1, X2, Y2 = as_xyxy(b2)
    cx1, cy1 = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    cx2, cy2 = (X1 + X2) / 2.0, (Y1 + Y2) / 2.0
    return float(math.hypot(cx2 - cx1, cy2 - cy1))


def edge_gap(b1: Sequence[float], b2: Sequence[float]) -> float:
    x1, y1, x2, y2 = as_xyxy(b1)
    X1, Y1, X2, Y2 = as_xyxy(b2)
    gap_x = max(0.0, max(x1 - X2, X1 - x2))
    gap_y = max(0.0, max(y1 - Y2, Y1 - y2))
    return float(math.hypot(gap_x, gap_y))


def orientation_label(dx: float, dy: float, *, margin_px: float = 5.0) -> str:
    """
    Determine the dominant orientation ('left_of'/'right_of'/'above'/'below'),
    or 'near' if the offset is within the margin.
    """
    if abs(dx) >= abs(dy) and abs(dx) > margin_px:
        return "right_of" if dx > 0 else "left_of"
    if abs(dy) > margin_px:
        return "below" if dy > 0 else "above"
    return "near"


def is_on_top_of(
    box_a: Sequence[float],
    box_b: Sequence[float],
    *,
    mask_a: Optional[np.ndarray] = None,
    mask_b: Optional[np.ndarray] = None,
    depth_a: Optional[float] = None,
    depth_b: Optional[float] = None,
    on_top_gap_px: int = 8,
    on_top_horiz_overlap: float = 0.35,
) -> bool:
    """
    Robust heuristic for 'A on top of B' using:
      1) Y-center ordering,
      2) limited vertical gap,
      3) sufficient horizontal overlap,
      4) optional mask contact along the junction,
      5) optional consistency with depth estimates.
    """
    x1a, y1a, x2a, y2a = as_xyxy(box_a)
    x1b, y1b, x2b, y2b = as_xyxy(box_b)

    # 1) A must be above B (by Y-center)
    if (y1a + y2a) / 2.0 >= (y1b + y2b) / 2.0:
        return False

    # 2) Vertical gap (allow slight overlap, but not too much)
    bottom_a, top_b = y2a, y1b
    h_ref = min(y2a - y1a, y2b - y1b)
    tol_px = max(on_top_gap_px, int(0.06 * max(1.0, h_ref)))
    gap = top_b - bottom_a
    if gap > tol_px:
        return False
    if gap < 0:
        vert_overlap = min(y2a, y2b) - max(y1a, y1b)
        if vert_overlap / max(1.0, (y2a - y1a)) > 0.35:
            return False

    # 3) Horizontal overlap
    overlap_x = max(0.0, min(x2a, x2b) - max(x1a, x1b))
    ratio_x = overlap_x / max(1e-6, min((x2a - x1a), (x2b - x1b)))
    if ratio_x < on_top_horiz_overlap:
        return False

    # 4) Mask contact (optional)
    if mask_a is not None and mask_b is not None and _HAS_CV2:
        band = max(2, int(0.02 * max(1.0, h_ref)))
        H = mask_a.shape[0]
        ya0 = np.clip(int(bottom_a - band), 0, H - 1)
        ya1 = np.clip(int(bottom_a + band), 0, H - 1)
        yb0 = np.clip(int(top_b - band), 0, H - 1)
        yb1 = np.clip(int(top_b + band), 0, H - 1)

        band_a = mask_a[ya0:ya1, :]
        band_b = mask_b[yb0:yb1, :]
        min_h = min(band_a.shape[0], band_b.shape[0])
        if min_h > 0:
            band_a = band_a[:min_h, :]
            band_b = band_b[:min_h, :]
            contact = np.logical_and(band_a, band_b).any()
            if not contact:
                # Light dilation for tolerance
                k = np.ones((3, 3), np.uint8)
                da = cv2.dilate(band_a.astype(np.uint8), k)
                db = cv2.dilate(band_b.astype(np.uint8), k)
                if not (da & db).any():
                    return False

    # 5) Depth consistency (optional)
    if depth_a is not None and depth_b is not None:
        if depth_a > depth_b + 0.05:
            return False

    return True


def is_below_of(
    box_a: Sequence[float],
    box_b: Sequence[float],
    **kwargs,
) -> bool:
    """A is below B ⇔ B is on top of A."""
    return is_on_top_of(box_b, box_a, **kwargs)


def build_precise_nearest_relation(
    i: int,
    j: int,
    boxes: Sequence[Sequence[float]],
    *,
    margin_px: int = 20,
) -> dict:
    """
    Build a precise 'nearest' relation between i and j by combining
    proximity levels (touching/very_close/close/near) and orientation.
    """
    b1, b2 = boxes[i], boxes[j]
    x1, y1, x2, y2 = as_xyxy(b1)
    X1, Y1, X2, Y2 = as_xyxy(b2)
    cx1, cy1 = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    cx2, cy2 = (X1 + X2) / 2.0, (Y1 + Y2) / 2.0

    dist_px = float(math.hypot(cx2 - cx1, cy2 - cy1))
    orient = orientation_label(cx2 - cx1, cy2 - cy1, margin_px=margin_px)
    iou_val = iou(b1, b2)
    gap = edge_gap(b1, b2)
    
    # Reference size to normalize distances
    avg_size = (x2 - x1 + y2 - y1 + X2 - X1 + Y2 - Y1) / 4.0
    
    # More conservative, size-aware thresholds
    if iou_val > 0.15 or gap <= 2:  # higher IoU, tighter gap
        prox = "touching"
    elif gap <= max(3, avg_size * 0.02) and dist_px / avg_size < 0.08:  # very close
        prox = "very_close"
    elif gap <= max(8, avg_size * 0.06) and dist_px / avg_size < 0.15:  # close
        prox = "close"
    else:
        prox = "near"

    if prox == "near":
        relation = orient if orient != "near" else "near"
    else:
        relation = f"{prox}_{orient}" if orient != "near" else prox

    return {
        "src_idx": i,
        "tgt_idx": j,
        "relation": relation,
        "distance": dist_px,
    }
