# igp/fusion/wbf.py
# Weighted Boxes Fusion (WBF) helper with optional ensemble-boxes backend.
# - Aggregates detections from multiple detectors
# - Uses ensemble_boxes.weighted_boxes_fusion when available
# - Falls back to per-class NMS if WBF implementation is not installed
# - Expects Detection objects with .box (xyxy), .label, .score, optional .source

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from igp.types import Detection

try:
    # Optional dependency: pip install ensemble-boxes
    from ensemble_boxes import weighted_boxes_fusion as _wbf_impl  # type: ignore
    _HAVE_WBF = True
except Exception:
    _HAVE_WBF = False

# Fallback to NMS if ensemble-boxes is not available
try:
    from .nms import nms as _fallback_nms
except Exception:
    _fallback_nms = None  # handled at runtime


# ---------------------------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------------------------
def compute_iou_vectorized(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """
    Vectorized IoU computation.
    boxes1: (N,4) [x1,y1,x2,y2], boxes2: (M,4)
    Returns: (N,M) IoU matrix.
    """
    boxes1 = np.asarray(boxes1, dtype=np.float32)
    boxes2 = np.asarray(boxes2, dtype=np.float32)
    if boxes1.size == 0 or boxes2.size == 0:
        return np.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=np.float32)

    x1_max = np.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
    y1_max = np.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
    x2_min = np.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
    y2_min = np.minimum(boxes1[:, None, 3], boxes2[None, :, 3])

    inter_w = np.maximum(0.0, x2_min - x1_max)
    inter_h = np.maximum(0.0, y2_min - y1_max)
    inter_area = inter_w * inter_h

    area1 = np.maximum(0.0, boxes1[:, 2] - boxes1[:, 0]) * np.maximum(0.0, boxes1[:, 3] - boxes1[:, 1])
    area2 = np.maximum(0.0, boxes2[:, 2] - boxes2[:, 0]) * np.maximum(0.0, boxes2[:, 3] - boxes2[:, 1])

    union_area = area1[:, None] + area2[None, :] - inter_area
    return inter_area / np.maximum(union_area, 1e-6)


def fuse_detections_wbf(
    detections: List[Detection],
    image_size: Tuple[int, int],
    *,
    iou_thr: float = 0.55,
    skip_box_thr: float = 0.0,
    weights_by_source: Optional[Dict[str, float]] = None,
    default_weight: float = 1.0,
    sort_desc: bool = True,
) -> List[Detection]:
    """
    Perform Weighted Boxes Fusion over detections from multiple detectors.

    Args:
        detections: list of Detection (xyxy boxes in pixels, label str, score float).
        image_size: (width, height) in pixels.
        iou_thr: IoU threshold for grouping boxes.
        skip_box_thr: drop boxes with score < skip_box_thr before fusion.
        weights_by_source: per-source weights (e.g., {"owlvit": 2.0, "yolov8": 1.5}).
        default_weight: fallback weight for unknown sources.
        sort_desc: sort output by descending score.

    Returns:
        List[Detection] fused (boxes in pixel coordinates).
    """
    if not detections:
        return []

    W, H = image_size
    if W <= 0 or H <= 0:
        raise ValueError("image_size must be (width>0, height>0)")

    # Group by source
    by_src: Dict[str, List[Detection]] = defaultdict(list)
    for d in detections:
        by_src[_get_source(d)].append(d)

    # Build label vocabulary
    labels_sorted = sorted({_get_label(d) for d in detections})
    label2id = {lab: i for i, lab in enumerate(labels_sorted)}
    id2label = {i: lab for lab, i in label2id.items()}

    # Prepare inputs for ensemble-boxes: per-model lists
    list_boxes: List[List[List[float]]] = []
    list_scores: List[List[float]] = []
    list_labels: List[List[int]] = []
    weights: List[float] = []

    # Sensible defaults for per-source weights consistent with the pipeline
    default_weights_map = {"owlvit": 2.0, "yolov8": 1.5, "yolo": 1.5, "detectron2": 1.0}
    wmap = dict(default_weights_map)
    if weights_by_source:
        wmap.update(weights_by_source)

    for src, dets in by_src.items():
        boxes_norm: List[List[float]] = []
        scores_: List[float] = []
        labels_id: List[int] = []

        for d in dets:
            score = float(getattr(d, "score", 1.0))
            if score < skip_box_thr:
                continue
            x1, y1, x2, y2 = _as_xyxy(d.box)
            # Normalize to [0, 1]
            boxes_norm.append([x1 / W, y1 / H, x2 / W, y2 / H])
            scores_.append(score)
            labels_id.append(label2id[_get_label(d)])

        list_boxes.append(boxes_norm)
        list_scores.append(scores_)
        list_labels.append(labels_id)
        weights.append(float(wmap.get(src, default_weight)))

    # If ensemble-boxes not installed, fallback to per-class NMS
    if not _HAVE_WBF:
        if _fallback_nms is None:
            raise RuntimeError("ensemble-boxes not available and fallback NMS not importable.")
        return _fallback_nms(detections, iou_thr=iou_thr, class_aware=True, sort_desc=sort_desc)

    # Apply WBF
    b_fused, s_fused, l_fused = _wbf_impl(
        list_boxes, list_scores, list_labels,
        weights=weights,
        iou_thr=float(iou_thr),
        skip_box_thr=float(skip_box_thr),
    )

    # Denormalize and build Detection objects
    out: List[Detection] = []
    for b, s, l in zip(b_fused, s_fused, l_fused):
        x1 = float(b[0] * W)
        y1 = float(b[1] * H)
        x2 = float(b[2] * W)
        y2 = float(b[3] * H)
        label = id2label[int(l)]
        out.append(_make_detection((x1, y1, x2, y2), label, float(s), source="fusion:wbf"))

    if sort_desc:
        out.sort(key=lambda d: float(getattr(d, "score", 0.0)), reverse=True)
    return out


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _get_source(d: Detection) -> str:
    # Compatibility: support 'from'/'from_' in addition to 'source'
    src = getattr(d, "source", None)
    if src is None:
        src = getattr(d, "from_", None) or getattr(d, "from", None)
    return str(src) if src is not None else "unknown"


def _get_label(d: Detection) -> str:
    return str(getattr(d, "label", ""))


def _as_xyxy(box_like: Sequence[float]) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = box_like[:4]
    return float(x1), float(y1), float(x2), float(y2)


def _make_detection(
    box_xyxy: Sequence[float],
    label: str,
    score: float,
    *,
    source: str = "fusion:wbf",
) -> Detection:
    x1, y1, x2, y2 = _as_xyxy(box_xyxy)
    try:
        return Detection(box=(x1, y1, x2, y2), label=label, score=float(score), source=source)
    except TypeError:
        try:
            return Detection(box=(x1, y1, x2, y2), label=label, score=float(score))
        except TypeError:
            return Detection(box=(x1, y1, x2, y2), label=label)


__all__ = ["fuse_detections_wbf"]