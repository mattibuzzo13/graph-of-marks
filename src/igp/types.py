# igp/types.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, TypedDict

import numpy as np

# Bounding box in XYXY pixel coordinates: (x1, y1, x2, y2)
Box = Tuple[float, float, float, float]


@dataclass
class Detection:
    """
    Canonical detection record:
      - box: bounding box in XYXY pixel coords (x1, y1, x2, y2)
      - label: class name as string (e.g., "person")
      - score: confidence in [0, 1]
      - source: optional detector name (e.g., "yolov8", "owlvit")
      - extra: optional payload (e.g., a binary mask from Detectron2)
    """
    box: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    label: str
    score: float = 1.0
    source: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None


class MaskDict(TypedDict, total=False):
    """
    SAM-style segmentation bundle:
      - segmentation: boolean mask array of shape (H, W)
      - bbox: object box in XYWH pixel format [x, y, w, h]
      - predicted_iou: model's IoU estimate for the mask
    """
    segmentation: np.ndarray
    bbox: List[int]
    predicted_iou: float


@dataclass
class Relationship:
    """
    Directed relation between objects i → j:
      - relation: canonical predicate (e.g., "left_of", "on_top_of", "near", "touching")
      - distance: geometric priority metric (pixels or normalized)
      - relation_raw: optional unnormalized/verb phrase (e.g., from CLIP text)
      - clip_sim: optional similarity score when CLIP is used
    """
    src_idx: int
    tgt_idx: int
    relation: str
    distance: float = float("inf")
    relation_raw: Optional[str] = None
    clip_sim: Optional[float] = None
