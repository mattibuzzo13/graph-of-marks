# igp/fusion/wbf.py
# Weighted Boxes Fusion (WBF) helper with optional ensemble-boxes backend.
# - Aggregates detections from multiple detectors
# - Uses ensemble_boxes.weighted_boxes_fusion when available
# - Falls back to per-class NMS if WBF implementation is not installed
# - Expects Detection objects with .box (xyxy), .label, .score, optional .source

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from gom.types import Detection

logger = logging.getLogger(__name__)

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
    weights_by_source: Optional[Dict[str, object]] = None,
    default_weight: float = 1.0,
    sort_desc: bool = True,
    fallback_to_original: bool = False,
    mask_fusion: str = "weighted",
    mask_threshold: float = 0.5,
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

    # Group by source and build a flat list of original items for later mask fusion
    by_src: Dict[str, List[Detection]] = defaultdict(list)
    orig_boxes: List[Tuple[float, float, float, float]] = []
    orig_scores: List[float] = []
    orig_labels: List[str] = []
    orig_masks: List[Optional[np.ndarray]] = []
    orig_sources: List[str] = []
    for d in detections:
        src = _get_source(d)
        by_src[src].append(d)
        b = _as_xyxy(d.box)
        orig_boxes.append(b)
        orig_scores.append(float(getattr(d, "score", 1.0)))
        orig_labels.append(_get_label(d))
        # masks may be in extra['segmentation'] or d.extra
        m = None
        if getattr(d, "extra", None) and isinstance(d.extra, dict):
            m = d.extra.get("segmentation")
        orig_masks.append(m)
        orig_sources.append(src)

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

        # Skip sources that produced no boxes after thresholding. Ensemble-boxes
        # expects per-model lists to be non-empty; including empty lists may
        # raise or produce undefined behavior. Also avoid adding a weight for
        # an empty source.
        if not boxes_norm:
            logger.debug("WBF: skipping source %s because it produced no boxes after threshold", src)
            continue

        list_boxes.append(boxes_norm)
        list_scores.append(scores_)
        list_labels.append(labels_id)
        weights.append(float(wmap.get(src, default_weight)))

    # If ensemble-boxes not installed, fallback to per-class NMS
    fused_boxes: List[Tuple[float, float, float, float]] = []
    fused_scores: List[float] = []
    fused_labels: List[str] = []
    out: List[Detection] = []

    if not _HAVE_WBF:
        if _fallback_nms is None:
            raise RuntimeError("ensemble-boxes not available and fallback NMS not importable.")
        logger.debug("WBF: ensemble-boxes not available, using fallback NMS")
        kept = _fallback_nms(detections, iou_thr=iou_thr, class_aware=True, sort_desc=sort_desc)
        for d in kept:
            b = _as_xyxy(d.box)
            fused_boxes.append(b)
            fused_scores.append(float(getattr(d, "score", 1.0)))
            fused_labels.append(_get_label(d))
            out.append(_make_detection(b, _get_label(d), float(getattr(d, "score", 1.0)), source="fusion:nms"))
    else:
        # If after filtering there are no per-source boxes to fuse, either return
        # the original detections (fallback) or an empty list depending on
        # `fallback_to_original`.
        if not list_boxes:
            if fallback_to_original:
                logger.debug("WBF: no boxes after filtering; returning original detections as fallback")
                return detections
            logger.debug("WBF: no boxes after filtering; returning empty list")
            return []

        # Sanitization: ensemble-boxes expects boxes normalized in [0,1].
        # Clip coordinates to [0,1] and ensure x1 < x2, y1 < y2. Remove any
        # invalid boxes (zero-area) to avoid warnings and unpredictable
        # behaviour from the backend.
        def _sanitize_boxes_per_model(boxes_model: List[List[float]]) -> List[List[float]]:
            sanitized: List[List[float]] = []
            for box in boxes_model:
                if len(box) < 4:
                    continue
                x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
                # clip
                x1 = max(0.0, min(1.0, x1))
                y1 = max(0.0, min(1.0, y1))
                x2 = max(0.0, min(1.0, x2))
                y2 = max(0.0, min(1.0, y2))
                # enforce ordering
                if x2 <= x1 or y2 <= y1:
                    # try to repair small numerical issues by nudging
                    if x2 <= x1:
                        x2 = min(1.0, x1 + 1e-3)
                    if y2 <= y1:
                        y2 = min(1.0, y1 + 1e-3)
                # discard zero-area boxes
                if (x2 - x1) <= 1e-6 or (y2 - y1) <= 1e-6:
                    continue
                sanitized.append([x1, y1, x2, y2])
            return sanitized

        # Apply sanitization per-model to avoid ensemble-boxes warnings
        list_boxes_sanitized: List[List[List[float]]] = [_sanitize_boxes_per_model(bm) for bm in list_boxes]

        # If any model became empty after sanitization, ensemble-boxes may still
        # work but empty lists can cause issues; we keep empty lists (the
        # implementation handles them) but log if we removed many boxes.
        try:
            b_fused, s_fused, l_fused = _wbf_impl(
                list_boxes_sanitized, list_scores, list_labels,
                weights=weights,
                iou_thr=float(iou_thr),
                skip_box_thr=float(skip_box_thr),
            )
        except Exception:
            # In case ensemble-boxes fails (unexpected), log and fallback to
            # per-class NMS via the existing path below.
            logger.exception("ensemble_boxes WBF failed; falling back to NMS")
            if _fallback_nms is None:
                raise
            kept = _fallback_nms(detections, iou_thr=iou_thr, class_aware=True, sort_desc=sort_desc)
            for d in kept:
                b = _as_xyxy(d.box)
                fused_boxes.append(b)
                fused_scores.append(float(getattr(d, "score", 1.0)))
                fused_labels.append(_get_label(d))
                out.append(_make_detection(b, _get_label(d), float(getattr(d, "score", 1.0)), source="fusion:nms"))
            # proceed to mask fusion below
            if sort_desc:
                out.sort(key=lambda d: float(getattr(d, "score", 0.0)), reverse=True)
            return out

        # Denormalize and build Detection objects
        for b, s, l in zip(b_fused, s_fused, l_fused):
            x1 = float(b[0] * W)
            y1 = float(b[1] * H)
            x2 = float(b[2] * W)
            y2 = float(b[3] * H)
            label = id2label[int(l)]
            fused_boxes.append((x1, y1, x2, y2))
            fused_scores.append(float(s))
            fused_labels.append(label)
            out.append(_make_detection((x1, y1, x2, y2), label, float(s), source="fusion:wbf"))

    # Now perform mask fusion: for each fused box, find contributing original
    # detections (same label, IoU > iou_thr) and combine their masks using
    # weights (score * source_weight * per-class weight if provided).
    def _get_source_weight(src: str, lab: str) -> float:
        if not weights_by_source:
            return float(default_weight)
        w = weights_by_source.get(src, None)
        if w is None:
            return float(default_weight)
        # w may be a dict (per-class) or a float
        if isinstance(w, dict):
            return float(w.get(lab, default_weight))
        try:
            return float(w)
        except Exception:
            return float(default_weight)

    if orig_masks and any(m is not None for m in orig_masks) and mask_fusion != "none":
        # Precompute orig boxes/numpy arrays
        import numpy as _np

        orig_boxes_np = _np.asarray(orig_boxes, dtype=_np.float32) if orig_boxes else _np.zeros((0, 4), dtype=_np.float32)

        def _iou_np(a, b):
            # a: (4,), b: (N,4)
            ax1, ay1, ax2, ay2 = a
            bx1, by1, bx2, by2 = b.T
            inter_x1 = _np.maximum(ax1, bx1)
            inter_y1 = _np.maximum(ay1, by1)
            inter_x2 = _np.minimum(ax2, bx2)
            inter_y2 = _np.minimum(ay2, by2)
            inter_w = _np.clip(inter_x2 - inter_x1, 0, None)
            inter_h = _np.clip(inter_y2 - inter_y1, 0, None)
            inter = inter_w * inter_h
            area_a = max((ax2 - ax1) * (ay2 - ay1), 1e-7)
            area_b = _np.clip(bx2 - bx1, 0, None) * _np.clip(by2 - by1, 0, None)
            union = area_a + area_b - inter
            return inter / (union + 1e-7)

        fused_out: List[Detection] = []
        for fb, fscore, flab, det in zip(fused_boxes, fused_scores, fused_labels, out):
            # find originals with same label
            candidates_idx = [i for i, lab in enumerate(orig_labels) if lab == flab]
            if not candidates_idx:
                fused_out.append(det)
                continue
            cand_boxes = orig_boxes_np[candidates_idx]
            ious = _iou_np(_np.asarray(fb, dtype=_np.float32), cand_boxes)
            # select contributors with IoU > iou_thr
            contrib_idx = [candidates_idx[i] for i, val in enumerate(ious) if val > iou_thr]
            masks_to_fuse = []
            weights_list = []
            for idx in contrib_idx:
                m = orig_masks[idx]
                if m is None:
                    continue
                # ensure boolean numpy mask and resize to image shape if needed
                try:
                    mm = _np.asarray(m).astype(bool)
                except Exception:
                    continue
                # resize mask to image HxW if shape differs
                try:
                    H_img, W_img = H, W
                    if mm.shape != (H_img, W_img):
                        from PIL import Image as _PILImage
                        mm_img = _PILImage.fromarray(mm.astype(_np.uint8) * 255)
                        mm_img = mm_img.resize((W_img, H_img), resample=_PILImage.NEAREST)
                        mm = _np.asarray(mm_img).astype(bool)
                except Exception:
                    # if resize fails, skip this mask
                    continue

                masks_to_fuse.append(mm.astype(_np.float32))
                src = orig_sources[idx]
                w_src = _get_source_weight(src, flab)
                sc = float(orig_scores[idx])
                weights_list.append(w_src * sc)

            if masks_to_fuse and weights_list:
                # ensure all masks same shape
                shapes = {m.shape for m in masks_to_fuse}
                if len(shapes) == 1:
                    stacked = _np.stack(masks_to_fuse, axis=0)
                    w_arr = _np.asarray(weights_list, dtype=_np.float32)
                    if mask_fusion == "union":
                        fused_bool = _np.any(stacked >= mask_threshold, axis=0)
                    elif mask_fusion == "majority":
                        # majority by weighted votes
                        w_arr = w_arr.reshape((-1, 1, 1))
                        votes = _np.sum(stacked * w_arr, axis=0)
                        fused_bool = votes >= (mask_threshold * _np.sum(w_arr))
                    else:
                        # default: weighted average (compat)
                        w_arr = w_arr.reshape((-1, 1, 1))
                        fused_mask = (_np.sum(stacked * w_arr, axis=0) / (_np.sum(w_arr) + 1e-7))
                        fused_bool = fused_mask >= mask_threshold

                    # attach to detection.extra
                    try:
                        if det.extra is None:
                            det.extra = {"segmentation": fused_bool}
                        elif isinstance(det.extra, dict):
                            det.extra["segmentation"] = fused_bool
                        else:
                            det.extra = {"segmentation": fused_bool}
                    except Exception:
                        logger.exception("Failed to attach fused mask to detection")
            fused_out.append(det)
        # sort if requested
        if sort_desc:
            fused_out.sort(key=lambda d: float(getattr(d, "score", 0.0)), reverse=True)
        return fused_out

    # If no masks to fuse, return out as-is (sorted if requested)
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