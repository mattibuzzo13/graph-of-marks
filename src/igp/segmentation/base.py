# igp/segmentation/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple, Optional

import numpy as np


@dataclass
class SegmenterConfig:
    device: Optional[str] = None
    close_holes: bool = False
    hole_kernel: int = 7
    min_hole_area: int = 100


class Segmenter(ABC):
    """
    Base interface for SAM-like segmenters.
    """

    def __init__(self, config: SegmenterConfig | None = None) -> None:
        self.config = config or SegmenterConfig()

    # --------- Required API ---------
    @abstractmethod
    def segment(self, image_pil, boxes: Sequence[Sequence[float]]) -> List[Dict[str, Any]]:
        """
        Compute a mask for each input box.

        Args:
            image_pil: input PIL image.
            boxes: list of [x1, y1, x2, y2] boxes in pixel coordinates.

        Returns:
            A list of dicts with keys:
              - 'segmentation': np.ndarray(bool, H, W)
              - 'bbox': [x, y, w, h] (xywh)
              - 'predicted_iou': float
        """
    
    # --------- Common utilities ---------
    @staticmethod
    def clamp_box_xyxy(box: Sequence[float], W: int, H: int) -> List[int]:
        x1, y1, x2, y2 = box[:4]
        x1 = max(0, min(int(round(x1)), W - 2))
        y1 = max(0, min(int(round(y1)), H - 2))
        x2 = max(x1 + 1, min(int(round(x2)), W - 1))
        y2 = max(y1 + 1, min(int(round(y2)), H - 1))
        return [x1, y1, x2, y2]

    @staticmethod
    def bbox_from_mask(mask: np.ndarray) -> List[int]:
        """
        Convert a boolean mask into an xywh bbox (in pixels). If empty, return [0, 0, 0, 0].
        """
        ys, xs = np.where(mask)
        if len(xs) == 0 or len(ys) == 0:
            return [0, 0, 0, 0]
        x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        return [x1, y1, x2 - x1 + 1, y2 - y1 + 1]

    def close_mask_holes(self, mask: np.ndarray) -> np.ndarray:
        """
        Close internal holes and thin gaps using:
          1) morphological closing, and
          2) filling background components smaller than `min_hole_area`.

        If OpenCV is not available, return the mask unchanged.
        """
        if not self.config.close_holes:
            return mask

        try:
            import cv2  # type: ignore
        except Exception:
            return mask

        m = (mask.astype(np.uint8) * 255) if mask.dtype != np.uint8 else mask.copy()
        if m.max() == 1:
            m *= 255

        k = max(1, int(self.config.hole_kernel))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)

        inv = 255 - m
        h, w = m.shape[:2]
        flood = inv.copy()
        ff_mask = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(flood, ff_mask, (0, 0), 0)
        holes = cv2.bitwise_and(inv, flood)

        # remove only small holes
        num, lab = cv2.connectedComponents(holes, connectivity=4)
        out = m.copy()
        for lab_id in range(1, num):
            area = int((lab == lab_id).sum())
            if area < int(self.config.min_hole_area):
                out[lab == lab_id] = 255

        return (out > 0).astype(bool)
