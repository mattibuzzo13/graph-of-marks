# igp/relations/geometry/nearest.py
# Build precise nearest-neighbor relations with proximity tiers and direction

from __future__ import annotations

from typing import Sequence

import math

from .core import as_xyxy, iou, edge_gap
from .predicates import orientation_label


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
