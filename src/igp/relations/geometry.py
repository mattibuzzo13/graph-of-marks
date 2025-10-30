# igp/relations/geometry.py
# Geometric utilities for relation inference:
# - box metrics: IoU/GIoU/DIoU, centers, overlaps, containment
# - mask IoU/contact and depth stats
# - robust predicates: on_top_of/below, orientation, in_front_of/behind
# - precise nearest relation combining distance, gap, IoU, orientation

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import math
import numpy as np

try:
    import cv2
    _HAS_CV2 = True
except Exception:
    cv2 = None  # type: ignore
    _HAS_CV2 = False


# ------------------------------ box basics ------------------------------

def as_xyxy(b: Sequence[float]) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = b[:4]
    return float(x1), float(y1), float(x2), float(y2)


def area(b: Sequence[float]) -> float:
    x1, y1, x2, y2 = as_xyxy(b)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def center(b: Sequence[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = as_xyxy(b)
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def center_distance(b1: Sequence[float], b2: Sequence[float]) -> float:
    cx1, cy1 = center(b1)
    cx2, cy2 = center(b2)
    return float(math.hypot(cx2 - cx1, cy2 - cy1))


def iou(b1: Sequence[float], b2: Sequence[float]) -> float:
    x1, y1, x2, y2 = as_xyxy(b1)
    X1, Y1, X2, Y2 = as_xyxy(b2)
    ix1, iy1 = max(x1, X1), max(y1, Y1)
    ix2, iy2 = min(x2, X2), min(y2, Y2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    a1 = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a2 = max(0.0, X2 - X1) * max(0.0, Y2 - Y1)
    union = a1 + a2 - inter
    return float(inter / union) if union > 0 else 0.0


def iou_matrix(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """
    Vectorized IoU between boxes1 (N,4) and boxes2 (M,4) in xyxy.
    """
    a = np.asarray(boxes1, dtype=np.float32).reshape(-1, 4)
    b = np.asarray(boxes2, dtype=np.float32).reshape(-1, 4)
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)

    ax1, ay1, ax2, ay2 = a.T
    bx1, by1, bx2, by2 = b.T

    inter_x1 = np.maximum(ax1[:, None], bx1[None, :])
    inter_y1 = np.maximum(ay1[:, None], by1[None, :])
    inter_x2 = np.minimum(ax2[:, None], bx2[None, :])
    inter_y2 = np.minimum(ay2[:, None], by2[None, :])

    inter_w = np.clip(inter_x2 - inter_x1, 0, None)
    inter_h = np.clip(inter_y2 - inter_y1, 0, None)
    inter = inter_w * inter_h

    area_a = np.clip(ax2 - ax1, 0, None) * np.clip(ay2 - ay1, 0, None)
    area_b = np.clip(bx2 - bx1, 0, None) * np.clip(by2 - by1, 0, None)

    union = area_a[:, None] + area_b[None, :] - inter
    return inter / (union + 1e-7)


def _enclosing_box(b1: Sequence[float], b2: Sequence[float]) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = as_xyxy(b1)
    X1, Y1, X2, Y2 = as_xyxy(b2)
    return min(x1, X1), min(y1, Y1), max(x2, X2), max(y2, Y2)


def giou(b1: Sequence[float], b2: Sequence[float]) -> float:
    """
    Generalized IoU (Rezatofighi et al., 2019) for more stable overlap scoring.
    """
    i = iou(b1, b2)
    cx1, cy1, cx2, cy2 = _enclosing_box(b1, b2)
    c_area = max(0.0, cx2 - cx1) * max(0.0, cy2 - cy1)
    if c_area <= 0:
        return i
    a1 = area(b1)
    a2 = area(b2)
    inter = i * (a1 + a2 - i * (a1 + a2))
    union = a1 + a2 - inter
    return float(i - (c_area - union) / max(c_area, 1e-7))


def diou(b1: Sequence[float], b2: Sequence[float]) -> float:
    """
    Distance-IoU (Zheng et al., 2020): IoU penalized by center distance.
    """
    i = iou(b1, b2)
    cx1, cy1 = center(b1)
    cx2, cy2 = center(b2)
    xC1, yC1, xC2, yC2 = _enclosing_box(b1, b2)
    c_diag2 = (xC2 - xC1) ** 2 + (yC2 - yC1) ** 2
    if c_diag2 <= 0:
        return i
    d2 = (cx2 - cx1) ** 2 + (cy2 - cy1) ** 2
    return float(i - d2 / c_diag2)


# --------------------------- overlaps and gaps ---------------------------

def horizontal_overlap(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, _, ax2, _ = as_xyxy(a)
    bx1, _, bx2, _ = as_xyxy(b)
    left = max(ax1, bx1)
    right = min(ax2, bx2)
    return max(0.0, right - left)


def vertical_overlap(a: Sequence[float], b: Sequence[float]) -> float:
    _, ay1, _, ay2 = as_xyxy(a)
    _, by1, _, by2 = as_xyxy(b)
    top = max(ay1, by1)
    bottom = min(ay2, by2)
    return max(0.0, bottom - top)


def edge_gap(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = as_xyxy(a)
    bx1, by1, bx2, by2 = as_xyxy(b)
    gap_x = max(0.0, max(ax1 - bx2, bx1 - ax2))
    gap_y = max(0.0, max(ay1 - by2, by1 - ay2))
    return float(math.hypot(gap_x, gap_y))


def overlap_ratio(a: Sequence[float], b: Sequence[float]) -> float:
    """Intersection-over-smaller-area, helpful to detect containment."""
    ax = area(a)
    bx = area(b)
    if ax <= 0 or bx <= 0:
        return 0.0
    x1, y1, x2, y2 = as_xyxy(a)
    X1, Y1, X2, Y2 = as_xyxy(b)
    ix1, iy1 = max(x1, X1), max(y1, Y1)
    ix2, iy2 = min(x2, X2), min(y2, Y2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    return float(inter / min(ax, bx))


def is_inside(a: Sequence[float], b: Sequence[float], *, tol: float = 1.0) -> bool:
    """True if box a is fully inside box b (with tolerance in pixels)."""
    ax1, ay1, ax2, ay2 = as_xyxy(a)
    bx1, by1, bx2, by2 = as_xyxy(b)
    return (ax1 >= bx1 - tol) and (ay1 >= by1 - tol) and (ax2 <= bx2 + tol) and (ay2 <= by2 + tol)


def contains(a: Sequence[float], b: Sequence[float], *, tol: float = 1.0) -> bool:
    """True if box a fully contains box b (with tolerance)."""
    return is_inside(b, a, tol=tol)


# ---------------------------- masks and depth ----------------------------

def mask_iou(m1: np.ndarray, m2: np.ndarray) -> float:
    """IoU between binary masks (any truthy != 0 treated as True)."""
    if m1 is None or m2 is None:
        return 0.0
    a = m1.astype(bool)
    b = m2.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union > 0 else 0.0


def _dilate_bool(mask: np.ndarray, k: int = 1) -> np.ndarray:
    """Small, dependency-free dilation for boolean masks."""
    if k <= 0:
        return mask.astype(bool)
    m = mask.astype(bool)
    out = m.copy()
    # 8-neighborhood dilation repeated k times
    for _ in range(k):
        shifted = [
            np.pad(m[1:, :], ((0, 1), (0, 0)), constant_values=False),
            np.pad(m[:-1, :], ((1, 0), (0, 0)), constant_values=False),
            np.pad(m[:, 1:], ((0, 0), (0, 1)), constant_values=False),
            np.pad(m[:, :-1], ((0, 0), (1, 0)), constant_values=False),
            np.pad(m[1:, 1:], ((0, 1), (0, 1)), constant_values=False),
            np.pad(m[1:, :-1], ((0, 1), (1, 0)), constant_values=False),
            np.pad(m[:-1, 1:], ((1, 0), (0, 1)), constant_values=False),
            np.pad(m[:-1, :-1], ((1, 0), (1, 0)), constant_values=False),
        ]
        out = m | shifted[0] | shifted[1] | shifted[2] | shifted[3] | shifted[4] | shifted[5] | shifted[6] | shifted[7]
        m = out
    return out


def _mask_contact_along_y(mask_a: np.ndarray, mask_b: np.ndarray, y: int, band: int) -> bool:
    """
    Check contact between two masks along a horizontal band around y.
    """
    H = min(mask_a.shape[0], mask_b.shape[0])
    y0 = max(0, y - band)
    y1 = min(H, y + band + 1)
    if y0 >= y1:
        return False
    a = mask_a[y0:y1, :].astype(bool)
    b = mask_b[y0:y1, :].astype(bool)
    if _HAS_CV2:
        k = np.ones((3, 3), np.uint8)
        a = cv2.dilate(a.astype(np.uint8), k).astype(bool)
        b = cv2.dilate(b.astype(np.uint8), k).astype(bool)
    else:
        a = _dilate_bool(a, 1)
        b = _dilate_bool(b, 1)
    return bool(np.logical_and(a, b).any())


def depth_stats_from_map(mask: Optional[np.ndarray], depth_map: Optional[np.ndarray], box: Optional[Sequence[float]] = None) -> Optional[float]:
    """
    Return median depth within mask if available, else within box region.
    
    Note: The depth convention depends on the depth estimator. With MiDaS and our
    normalization (inverted), higher values = closer to camera.
    """
    if depth_map is None:
        return None
    dm = np.asarray(depth_map)
    if dm.ndim != 2:
        return None
    if mask is not None:
        vals = dm[mask.astype(bool)]
        vals = vals[np.isfinite(vals)]
        if vals.size > 0:
            return float(np.median(vals))
    if box is not None:
        x1, y1, x2, y2 = as_xyxy(box)
        x1i, y1i, x2i, y2i = max(0, int(x1)), max(0, int(y1)), min(dm.shape[1], int(x2)), min(dm.shape[0], int(y2))
        if x2i > x1i and y2i > y1i:
            vals = dm[y1i:y2i, x1i:x2i].ravel()
            vals = vals[np.isfinite(vals)]
            if vals.size > 0:
                return float(np.median(vals))
    return None


# ----------------------------- predicates ------------------------------

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


# ---------------------- precise nearest relation ----------------------

def build_precise_nearest_relation(
    i: int,
    j: int,
    boxes: Sequence[Sequence[float]],
    *,
    margin_px: int = 20,
) -> dict:
    """
    Build a nearest-neighbor relation with informative label:
    - proximity tiers (touching/very_close/close/near) using gap and size
    - direction via orientation_label
    """
    a = boxes[i]
    b = boxes[j]
    x1, y1, x2, y2 = as_xyxy(a)
    X1, Y1, X2, Y2 = as_xyxy(b)
    cx1, cy1 = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    cx2, cy2 = (X1 + X2) / 2.0, (Y1 + Y2) / 2.0

    dist_px = float(math.hypot(cx2 - cx1, cy2 - cy1))
    orient = orientation_label(a, b, margin_px=float(margin_px))
    iou_val = iou(a, b)
    gap = edge_gap(a, b)

    # size-aware thresholds
    avg_size = max(1.0, (x2 - x1 + y2 - y1 + X2 - X1 + Y2 - Y1) / 4.0)

    if iou_val > 0.15 or gap <= 2.0:
        prox = "touching"
    elif gap <= max(3.0, avg_size * 0.02) and dist_px / avg_size < 0.08:
        prox = "very_close"
    elif gap <= max(8.0, avg_size * 0.06) and dist_px / avg_size < 0.15:
        prox = "close"
    else:
        prox = "near"

    # CRITICAL: proximity relations (touching/very_close/close) MUST have a direction
    # Only "near" can exist without proximity prefix if it's a simple directional relation
    if prox in ("touching", "very_close", "close"):
        # Always combine proximity with orientation for these relations
        if orient:
            relation = f"{prox}_{orient}"
        else:
            # Fallback: if no clear orientation, default to right_of
            relation = f"{prox}_right_of"
    else:
        # For "near", use simple orientation if available, otherwise "near"
        relation = orient if orient else "near"

    return {
        "src_idx": int(i),
        "tgt_idx": int(j),
        "relation": relation,
        "distance": dist_px,
    }