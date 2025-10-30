# igp/relations/geometry/masks.py
# Mask operations: IoU, contact detection, depth statistics

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

try:
    import cv2
    _HAS_CV2 = True
except Exception:
    cv2 = None  # type: ignore
    _HAS_CV2 = False

from .core import as_xyxy


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


def depth_stats_from_map(
    mask: Optional[np.ndarray], 
    depth_map: Optional[np.ndarray], 
    box: Optional[Sequence[float]] = None
) -> Optional[float]:
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
