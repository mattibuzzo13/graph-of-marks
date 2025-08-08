# igp/fusion/wbf.py
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

from igp.types import Detection

try:
    # pip install ensemble-boxes
    from ensemble_boxes import weighted_boxes_fusion as _wbf_impl  # type: ignore
    _HAVE_WBF = True
except Exception:
    _HAVE_WBF = False

# fallback a NMS se ensemble-boxes non è disponibile
try:
    from .nms import nms as _fallback_nms
except Exception:
    _fallback_nms = None  # verrà gestito runtime


# ---------------------------------------------------------------------------
# API PUBBLICA
# ---------------------------------------------------------------------------

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
    Esegue Weighted Boxes Fusion su una lista di Detection provenienti da più detector.

    Args:
        detections: lista di Detection (box xyxy in pixel, label str, score float).
        image_size: (width, height) dell'immagine.
        iou_thr: soglia IoU per la fusione.
        skip_box_thr: ignora box con score < skip_box_thr prima della fusione.
        weights_by_source: pesi per sorgente (es. {"owlvit": 2.0, "yolov8": 1.5}).
        default_weight: peso di default.
        sort_desc: ordina il risultato per score decrescente.

    Returns:
        Lista di Detection fuse (box in pixel).
    """
    if not detections:
        return []

    W, H = image_size
    if W <= 0 or H <= 0:
        raise ValueError("image_size non valido: atteso (width>0, height>0)")

    # Raggruppa per 'source'
    by_src: Dict[str, List[Detection]] = defaultdict(list)
    for d in detections:
        by_src[_get_source(d)].append(d)

    # Vocabolario label <-> id
    labels_sorted = sorted({_get_label(d) for d in detections})
    label2id = {lab: i for i, lab in enumerate(labels_sorted)}
    id2label = {i: lab for lab, i in label2id.items()}

    # Prepara inputs per ensemble-boxes: liste per-modello
    list_boxes: List[List[List[float]]] = []
    list_scores: List[List[float]] = []
    list_labels: List[List[int]] = []
    weights: List[float] = []

    # Pesi di default coerenti con la pipeline (OWL > YOLO > Detectron2)
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
            # normalizza in [0,1]
            boxes_norm.append([x1 / W, y1 / H, x2 / W, y2 / H])
            scores_.append(score)
            labels_id.append(label2id[_get_label(d)])

        list_boxes.append(boxes_norm)
        list_scores.append(scores_)
        list_labels.append(labels_id)
        weights.append(float(wmap.get(src, default_weight)))

    # Se ensemble-boxes non è disponibile, fallback a NMS per classe
    if not _HAVE_WBF:
        if _fallback_nms is None:
            raise RuntimeError(
                "ensemble-boxes non disponibile e fallback NMS non importabile."
            )
        return _fallback_nms(detections, iou_thr=iou_thr, class_aware=True, sort_desc=sort_desc)

    # Applica WBF
    b_fused, s_fused, l_fused = _wbf_impl(
        list_boxes, list_scores, list_labels,
        weights=weights,
        iou_thr=float(iou_thr),
        skip_box_thr=float(skip_box_thr),
    )

    # Denormalizza e re-istanza Detection
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
    # compat: supporta 'from'/'from_' oltre a 'source'
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
