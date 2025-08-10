# igp/utils/boxes.py
# Utility routines for axis-aligned bounding boxes in (x1, y1, x2, y2) format.

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

Number = float
Box = Sequence[Number]  # [x1, y1, x2, y2]


def area(box: Box) -> float:
    # Rectangle area; clamps negative width/height to 0.
    x1, y1, x2, y2 = box[:4]
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    return w * h


def intersect(box1: Box, box2: Box) -> float:
    # Intersection area between two boxes (0 if disjoint).
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
    # Intersection-over-Union with small-denominator protection.
    inter = intersect(box1, box2)
    if inter == 0.0:
        return 0.0
    a1 = area(box1)
    a2 = area(box2)
    return inter / max(1e-9, (a1 + a2 - inter))


def center(box: Box) -> Tuple[float, float]:
    # Geometric center of the box.
    x1, y1, x2, y2 = box[:4]
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def center_distance(b1: Box, b2: Box) -> float:
    # Euclidean distance between box centers.
    from math import hypot
    cx1, cy1 = center(b1)
    cx2, cy2 = center(b2)
    return hypot(cx2 - cx1, cy2 - cy1)


def edge_gap(b1: Box, b2: Box) -> float:
    # Minimum distance between box edges (0 if they overlap).
    from math import hypot
    gap_x = max(0.0, max(b1[0] - b2[2], b2[0] - b1[2]))
    gap_y = max(0.0, max(b1[1] - b2[3], b2[1] - b1[3]))
    return hypot(gap_x, gap_y)


def clamp_xyxy(box: Box, W: int, H: int) -> List[int]:
    # Clamp a box to valid pixel bounds [0..W-1]×[0..H-1] and enforce at least 1 px size.
    x1, y1, x2, y2 = box[:4]
    x1 = max(0, min(int(round(x1)), W - 2))
    y1 = max(0, min(int(round(y1)), H - 2))
    x2 = max(x1 + 1, min(int(round(x2)), W - 1))
    y2 = max(y1 + 1, min(int(round(y2)), H - 1))
    return [x1, y1, x2, y2]


def to_xywh(box: Box) -> List[float]:
    # Convert (x1, y1, x2, y2) → (x, y, w, h) with non-negative width/height.
    x1, y1, x2, y2 = box[:4]
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]


def from_xywh(box_xywh: Sequence[Number]) -> List[float]:
    # Convert (x, y, w, h) → (x1, y1, x2, y2).
    x, y, w, h = box_xywh[:4]
    return [x, y, x + w, y + h]


def union(b1: Box, b2: Box) -> List[float]:
    # Axis-aligned rectangle that tightly encloses both input boxes.
    x1 = min(b1[0], b2[0])
    y1 = min(b1[1], b2[1])
    x2 = max(b1[2], b2[2])
    y2 = max(b1[3], b2[3])
    return [x1, y1, x2, y2]
