# igp/types.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, TypedDict

import numpy as np

# Box in formato XYXY (pixel)
Box = Tuple[float, float, float, float]


@dataclass
class Detection:
    """
    Detection normalizzata:
      - box: XYXY in pixel
      - score: confidenza [0..1]
      - label: stringa
      - extra: dati opzionali (es. mask binaria di Detectron2)
    """
    box: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    label: str
    score: float = 1.0
    source: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None


class MaskDict(TypedDict, total=False):
    """
    Maschera SAM-like:
      - segmentation: np.ndarray(bool, H, W)
      - bbox: [x, y, w, h] (xywh, pixel)
      - predicted_iou: float
    """
    segmentation: np.ndarray
    bbox: List[int]
    predicted_iou: float


@dataclass
class Relationship:
    """
    Relazione direzionata fra oggetti i→j.
      - relation: nome canonico (e.g., 'left_of', 'on_top_of', 'near', 'touching')
      - distance: metrica di priorità (pixel o normalizzata)
      - relation_raw / clip_sim: opzionali per relazioni basate su CLIP
    """
    src_idx: int
    tgt_idx: int
    relation: str
    distance: float = float("inf")
    relation_raw: Optional[str] = None
    clip_sim: Optional[float] = None
