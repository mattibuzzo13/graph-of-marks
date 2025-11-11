"""
Utility helpers for detector output normalization and Detection construction.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from gom.types import Detection

logger = logging.getLogger(__name__)


def _normalize_box(box_like: Sequence[float]) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = box_like[:4]
    return float(x1), float(y1), float(x2), float(y2)


def _normalize_mask(mask: Optional[Any]) -> Optional[np.ndarray]:
    """Ensure mask is a boolean numpy array (H, W) or None."""
    if mask is None:
        return None
    try:
        arr = np.asarray(mask)
        if arr.dtype != np.bool_:
            arr = arr.astype(bool)
        # If mask has a channel dimension (1,H,W) or (N,H,W) take first
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim != 2:
            logger.debug("Mask has unexpected ndim=%s, attempting to reshape", arr.ndim)
            # Try to squeeze
            arr = np.squeeze(arr)
        if arr.ndim != 2:
            logger.debug("Could not coerce mask to 2D boolean array; returning None")
            return None
        return arr
    except Exception:
        logger.exception("Failed to normalize mask")
        return None


def make_detection(
    box: Sequence[float],
    label: str,
    score: float = 1.0,
    *,
    source: Optional[str] = None,
    mask: Optional[Any] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Detection:
    """Create a canonical `Detection` object with some normalization and safety.

    - box: xyxy numeric sequence
    - label: string label
    - score: coerced to float and clamped to [0,1]
    - mask: optional mask stored under extra['segmentation'] if present
    - extra: merged into final extra dict
    """
    b = _normalize_box(box)
    sc = float(score) if score is not None else 1.0
    # clamp
    if sc < 0.0:
        sc = 0.0
    if sc > 1.0:
        sc = 1.0

    final_extra: Dict[str, Any] = {} if extra is None else dict(extra)
    m = _normalize_mask(mask)
    if m is not None:
        final_extra.setdefault("segmentation", m)

    try:
        det = Detection(box=b, label=str(label), score=sc, source=source, extra=final_extra or None)
        # keep label lowercase by default for stable matching, but preserve original
        # stored label too (users can choose to read det.label as-is)
        try:
            det.label = det.label  # no-op placeholder; callers may choose to lower
        except Exception:
            pass
        return det
    except TypeError:
        # Defensive fallback in case Detection signature differs
        try:
            return Detection(box=b, label=str(label), score=sc, extra=final_extra or None)
        except TypeError:
            # Last resort: construct minimal Detection-like object (raise is alternative)
            return Detection(box=b, label=str(label), score=sc)
