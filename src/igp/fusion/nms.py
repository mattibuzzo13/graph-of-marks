# igp/fusion/nms.py
from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

from igp.types import Detection


# ---------------------------------------------------------------------------
# API PUBBLICA
# ---------------------------------------------------------------------------

def nms(
    detections: List[Detection],
    *,
    iou_thr: float = 0.55,
    class_aware: bool = True,
    sort_desc: bool = True,
) -> List[Detection]:
    """
    Non-Maximum Suppression su una lista di Detection.

    Args:
        detections: lista di Detection (box xyxy, label, score).
        iou_thr: soglia IoU per soppressione.
        class_aware: se True applica NMS per classe; altrimenti globale.
        sort_desc: ordina il risultato per score decrescente.

    Returns:
        Lista filtrata di Detection.
    """
    if not detections:
        return []

    groups: Dict[str, List[Detection]] = defaultdict(list)
    if class_aware:
        for d in detections:
            groups[_get_label(d)].append(d)
    else:
        groups["__all__"] = list(detections)

    kept: List[Detection] = []
    for _, dets in groups.items():
        # ordina per score
        dets_sorted = sorted(dets, key=lambda d: float(_get_score(d)), reverse=True)
        suppress = [False] * len(dets_sorted)

        for i in range(len(dets_sorted)):
            if suppress[i]:
                continue
            kept.append(dets_sorted[i])
            box_i = _as_xyxy(dets_sorted[i].box)
            for j in range(i + 1, len(dets_sorted)):
                if suppress[j]:
                    continue
                if iou(box_i, _as_xyxy(dets_sorted[j].box)) >= iou_thr:
                    suppress[j] = True

    if sort_desc:
        kept.sort(key=lambda d: float(_get_score(d)), reverse=True)
    return kept


def soft_nms(
    detections: List[Detection],
    *,
    iou_thr: float = 0.55,
    sigma: float = 0.5,
    score_thresh: float = 1e-3,
    class_aware: bool = True,
    method: str = "linear",  # "linear" | "gaussian"
    sort_desc: bool = True,
) -> List[Detection]:
    """
    Soft-NMS (linear o gaussian) su Detection.

    Note:
        - È una versione semplice in-place su una copia; non richiede NumPy.
        - 'method' seleziona il decadimento dello score.
    """
    if not detections:
        return []

    def _decay_fn(iou_val: float) -> float:
        if method == "gaussian":
            # gaussian weighting
            from math import exp
            return exp(-(iou_val ** 2) / sigma)
        # linear weighting
        return max(0.0, 1.0 - iou_val) if iou_val > iou_thr else 1.0

    groups: Dict[str, List[Detection]] = defaultdict(list)
    if class_aware:
        for d in detections:
            groups[_get_label(d)].append(_clone_det(d))
    else:
        groups["__all__"] = [_clone_det(d) for d in detections]

    kept: List[Detection] = []
    for _, dets in groups.items():
        # lavora su una lista mutabile di copie
        work = dets[:]
        while work:
            # prendi il migliore
            work.sort(key=lambda d: float(_get_score(d)), reverse=True)
            best = work[0]
            kept.append(best)
            box_best = _as_xyxy(best.box)

            # applica decadimento agli altri
            rest: List[Detection] = []
            for d in work[1:]:
                ov = iou(box_best, _as_xyxy(d.box))
                new_score = float(_get_score(d)) * _decay_fn(ov)
                if new_score >= score_thresh:
                    _set_score(d, new_score)
                    rest.append(d)
            work = rest

    if sort_desc:
        kept.sort(key=lambda d: float(_get_score(d)), reverse=True)
    return kept


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def iou(b1: Sequence[float], b2: Sequence[float]) -> float:
    x1, y1, x2, y2 = _as_xyxy(b1)
    X1, Y1, X2, Y2 = _as_xyxy(b2)
    ix1 = max(x1, X1)
    iy1 = max(y1, Y1)
    ix2 = min(x2, X2)
    iy2 = min(y2, Y2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    a1 = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
    a2 = max(0.0, (X2 - X1)) * max(0.0, (Y2 - Y1))
    union = a1 + a2 - inter
    return float(inter / union) if union > 0.0 else 0.0


def _as_xyxy(box_like: Sequence[float]) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = box_like[:4]
    return float(x1), float(y1), float(x2), float(y2)


def _get_label(d: Detection) -> str:
    return str(getattr(d, "label", ""))


def _get_score(d: Detection) -> float:
    return float(getattr(d, "score", 0.0))


def _set_score(d: Detection, new_score: float) -> None:
    try:
        d.score = float(new_score)  # type: ignore[attr-defined]
    except Exception:
        # se Detection è immutabile, si potrebbe creare una nuova istanza,
        # ma qui assumiamo che sia mutabile come nel resto del progetto.
        pass

def labelwise_nms(
    boxes: List[List[float]], 
    labels: List[str], 
    scores: List[float], 
    iou_threshold: float = 0.5
) -> List[int]:
    """
    Applica NMS per ogni classe separatamente e restituisce gli indici da mantenere.
    
    Args:
        boxes: Lista di bounding boxes in formato [x1, y1, x2, y2]
        labels: Lista delle etichette corrispondenti
        scores: Lista dei punteggi di confidenza
        iou_threshold: Soglia IoU per la soppressione
        
    Returns:
        Lista di indici delle detection da mantenere
    """
    if not boxes or len(boxes) != len(labels) or len(boxes) != len(scores):
        return []
    
    # Raggruppa per etichetta
    label_groups: Dict[str, List[int]] = defaultdict(list)
    for i, label in enumerate(labels):
        label_groups[label].append(i)
    
    keep_indices = []
    
    # Applica NMS per ogni gruppo di etichette
    for label, indices in label_groups.items():
        if not indices:
            continue
            
        # Ordina per score decrescente
        indices_sorted = sorted(indices, key=lambda i: scores[i], reverse=True)
        
        suppressed = set()
        
        for i in indices_sorted:
            if i in suppressed:
                continue
                
            keep_indices.append(i)
            box_i = boxes[i]
            
            # Sopprimi le detection con IoU alto
            for j in indices_sorted:
                if j == i or j in suppressed:
                    continue
                    
                if iou(box_i, boxes[j]) >= iou_threshold:
                    suppressed.add(j)
    
    # Ordina gli indici risultanti per score decrescente
    keep_indices.sort(key=lambda i: scores[i], reverse=True)
    return keep_indices

def _clone_det(d: Detection) -> Detection:
    # copia shallow con campi canonici noti; mantiene 'source' se presente
    x1, y1, x2, y2 = _as_xyxy(d.box)
    label = _get_label(d)
    score = _get_score(d)
    source = getattr(d, "source", None)
    try:
        return Detection(box=(x1, y1, x2, y2), label=label, score=score, source=source)
    except TypeError:
        try:
            return Detection(box=(x1, y1, x2, y2), label=label, score=score)
        except TypeError:
            return Detection(box=(x1, y1, x2, y2), label=label)


__all__ = ["nms", "soft_nms", "iou", "labelwise_nms"]
