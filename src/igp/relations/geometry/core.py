# igp/relations/geometry/core.py
# Core box utilities: format conversion, area, center, IoU, distance

from __future__ import annotations

from typing import Sequence, Tuple

import math
import numpy as np


# ------------------------ format conversion ------------------------

def as_xyxy(b: Sequence[float]) -> Tuple[float, float, float, float]:
    """Convert box to xyxy format."""
    x1, y1, x2, y2 = b[:4]
    return float(x1), float(y1), float(x2), float(y2)


# ------------------------ basic metrics ------------------------

def area(b: Sequence[float]) -> float:
    """Compute box area."""
    x1, y1, x2, y2 = as_xyxy(b)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def center(b: Sequence[float]) -> Tuple[float, float]:
    """Compute box center (cx, cy)."""
    x1, y1, x2, y2 = as_xyxy(b)
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def center_distance(b1: Sequence[float], b2: Sequence[float]) -> float:
    """Euclidean distance between box centers."""
    cx1, cy1 = center(b1)
    cx2, cy2 = center(b2)
    return float(math.hypot(cx2 - cx1, cy2 - cy1))


# ------------------------ IoU variants ------------------------

def iou(b1: Sequence[float], b2: Sequence[float]) -> float:
    """Standard Intersection over Union."""
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
    Vectorized IoU between boxes1 (N,4) and boxes2 (M,4) in xyxy format.
    Returns: (N, M) matrix of IoU values.
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
    """Compute smallest enclosing box around b1 and b2."""
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


# ------------------------ overlaps and gaps ------------------------

def horizontal_overlap(a: Sequence[float], b: Sequence[float]) -> float:
    """Horizontal overlap in pixels."""
    ax1, _, ax2, _ = as_xyxy(a)
    bx1, _, bx2, _ = as_xyxy(b)
    left = max(ax1, bx1)
    right = min(ax2, bx2)
    return max(0.0, right - left)


def vertical_overlap(a: Sequence[float], b: Sequence[float]) -> float:
    """Vertical overlap in pixels."""
    _, ay1, _, ay2 = as_xyxy(a)
    _, by1, _, by2 = as_xyxy(b)
    top = max(ay1, by1)
    bottom = min(ay2, by2)
    return max(0.0, bottom - top)


def edge_gap(a: Sequence[float], b: Sequence[float]) -> float:
    """Euclidean distance between closest edges of two boxes."""
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


# ------------------------ containment ------------------------

def is_inside(a: Sequence[float], b: Sequence[float], *, tol: float = 1.0) -> bool:
    """True if box a is fully inside box b (with tolerance in pixels)."""
    ax1, ay1, ax2, ay2 = as_xyxy(a)
    bx1, by1, bx2, by2 = as_xyxy(b)
    return (ax1 >= bx1 - tol) and (ay1 >= by1 - tol) and (ax2 <= bx2 + tol) and (ay2 <= by2 + tol)


def contains(a: Sequence[float], b: Sequence[float], *, tol: float = 1.0) -> bool:
    """True if box a fully contains box b (with tolerance)."""
    return is_inside(b, a, tol=tol)
