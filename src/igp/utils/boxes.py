# igp/utils/boxes.py
# Utility routines for axis-aligned bounding boxes in (x1, y1, x2, y2).
# - Scalar ops (area, iou, center, gap) and robust clamp/convert helpers.
# - Optional NumPy vectorized IoU matrix and NMS.

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

try:
    import numpy as np
    _HAS_NP = True
except Exception:
    _HAS_NP = False

Number = float
Box = Sequence[Number]  # [x1, y1, x2, y2]


def area(box: Box) -> float:
    x1, y1, x2, y2 = box[:4]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def intersect(box1: Box, box2: Box) -> float:
    x1, y1, x2, y2 = box1[:4]
    X1, Y1, X2, Y2 = box2[:4]
    ix1 = max(x1, X1)
    iy1 = max(y1, Y1)
    ix2 = min(x2, X2)
    iy2 = min(y2, Y2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    return iw * ih


def iou(box1: Box, box2: Box) -> float:
    inter = intersect(box1, box2)
    if inter == 0.0:
        return 0.0
    a1 = area(box1)
    a2 = area(box2)
    return inter / max(1e-9, (a1 + a2 - inter))


def center(box: Box) -> Tuple[float, float]:
    x1, y1, x2, y2 = box[:4]
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def center_distance(b1: Box, b2: Box) -> float:
    from math import hypot
    cx1, cy1 = center(b1)
    cx2, cy2 = center(b2)
    return hypot(cx2 - cx1, cy2 - cy1)


def edge_gap(b1: Box, b2: Box) -> float:
    from math import hypot
    gap_x = max(0.0, max(b1[0] - b2[2], b2[0] - b1[2]))
    gap_y = max(0.0, max(b1[1] - b2[3], b2[1] - b1[3]))
    return hypot(gap_x, gap_y)


def clamp_xyxy(box: Box, W: int, H: int) -> List[int]:
    # Clamp to valid pixel bounds and enforce at least 1 px size.
    W = max(1, int(W))
    H = max(1, int(H))
    x1, y1, x2, y2 = box[:4]
    x1 = int(min(max(round(x1), 0), W - 1))
    y1 = int(min(max(round(y1), 0), H - 1))
    x2 = int(min(max(round(x2), 0), W - 1))
    y2 = int(min(max(round(y2), 0), H - 1))
    if x2 <= x1:
        x2 = min(W - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(H - 1, y1 + 1)
    return [x1, y1, x2, y2]


def to_xywh(box: Box) -> List[float]:
    x1, y1, x2, y2 = box[:4]
    return [float(x1), float(y1), max(0.0, x2 - x1), max(0.0, y2 - y1)]


def from_xywh(box_xywh: Sequence[Number]) -> List[float]:
    x, y, w, h = box_xywh[:4]
    return [float(x), float(y), float(x + w), float(y + h)]


def union(b1: Box, b2: Box) -> List[float]:
    return [min(b1[0], b2[0]), min(b1[1], b2[1]), max(b1[2], b2[2]), max(b1[3], b2[3])]


# -------- Optional NumPy helpers --------

def iou_matrix(boxes1: "np.ndarray", boxes2: "np.ndarray") -> "np.ndarray":
    if not _HAS_NP:
        raise ImportError("NumPy non disponibile per iou_matrix")
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


def nms(boxes: "np.ndarray", scores: "np.ndarray", iou_thresh: float = 0.5) -> List[int]:
    """
    Basic NMS returning indices to keep (sorted by score desc).
    """
    if not _HAS_NP:
        raise ImportError("NumPy non disponibile per nms")
    b = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    s = np.asarray(scores, dtype=np.float32).reshape(-1)
    order = np.argsort(-s)
    keep: List[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        ious = iou_matrix(b[i:i+1], b[rest]).reshape(-1)
        rest = rest[ious <= float(iou_thresh)]
        order = rest
    return keep