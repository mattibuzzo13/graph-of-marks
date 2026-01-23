# igp/fusion/nms.py
# Efficient NMS utilities with NumPy backend and optional Torch/Torchvision acceleration.
# Provides:
# - nms: flexible API (array boxes -> indices OR list[Detection] -> filtered list[Detection])
# - soft_nms: simple NumPy soft-NMS implementation (returns indices)
# - iou: vectorized IoU matrix
# - labelwise_nms: per-class NMS (returns indices)

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import torch
    from torchvision.ops import nms as tv_nms  # type: ignore
    _HAS_TORCHVISION = True
except Exception:
    torch = None  # type: ignore
    tv_nms = None  # type: ignore
    _HAS_TORCHVISION = False

try:
    from gom.types import Detection
except Exception:
    Detection = None  # type: ignore


ArrayLike = Union[np.ndarray, Sequence[float]]


def _ensure_np1d(x, dtype=None) -> np.ndarray:
    arr = np.asarray(x, dtype=dtype)
    return arr.reshape(-1)


def _ensure_np2d(x, dtype=None) -> np.ndarray:
    arr = np.asarray(x, dtype=dtype)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 4)
    return arr


def iou(boxes1: ArrayLike, boxes2: ArrayLike) -> np.ndarray:
    """
    Vectorized IoU between boxes1 (N,4) and boxes2 (M,4).
    Boxes in xyxy format.
    Returns (N, M) IoU matrix.
    """
    a = _ensure_np2d(boxes1, np.float32)
    b = _ensure_np2d(boxes2, np.float32)
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)

    ax1, ay1, ax2, ay2 = a.T
    bx1, by1, bx2, by2 = b.T

    inter_x1 = np.maximum(ax1[:, None], bx1[None, :])
    inter_y1 = np.maximum(ay1[:, None], by1[None, :])
    inter_x2 = np.minimum(ax2[:, None], bx2[None, :])
    inter_y2 = np.minimum(ay2[:, None], by2[None, :])

    inter_w = np.clip(inter_x2 - inter_x1, 0, None)
    inter_h = np.clip(inter_y2 - inter_y1, 0, None)
    inter = inter_w * inter_h

    area_a = np.clip(ax2 - ax1, 0, None) * np.clip(ay2 - ay1, 0, None)
    area_b = np.clip(bx2 - bx1, 0, None) * np.clip(by2 - by1, 0, None)

    union = area_a[:, None] + area_b[None, :] - inter
    return inter / (union + 1e-7)


def nms_numpy(boxes: ArrayLike, scores: ArrayLike, iou_thr: float = 0.5, topk: Optional[int] = None) -> np.ndarray:
    """
    Pure NumPy NMS. Returns indices of kept boxes in original order.
    """
    boxes = _ensure_np2d(boxes, np.float32)
    scores = _ensure_np1d(scores, np.float32)

    N = boxes.shape[0]
    if N == 0:
        return np.empty((0,), dtype=np.int64)

    order = scores.argsort()[::-1]
    if topk is not None:
        order = order[: int(topk)]

    x1 = boxes[order, 0]
    y1 = boxes[order, 1]
    x2 = boxes[order, 2]
    y2 = boxes[order, 3]
    areas = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break

        xx1 = np.maximum(x1[0], x1[1:])
        yy1 = np.maximum(y1[0], y1[1:])
        xx2 = np.minimum(x2[0], x2[1:])
        yy2 = np.minimum(y2[0], y2[1:])

        w = np.clip(xx2 - xx1, 0, None)
        h = np.clip(yy2 - yy1, 0, None)
        inter = w * h
        iou_vals = inter / (areas[0] + areas[1:] - inter + 1e-7)
        keep_mask = np.where(iou_vals <= iou_thr)[0] + 1
        order = order[keep_mask]
        x1, y1, x2, y2 = x1[keep_mask], y1[keep_mask], x2[keep_mask], y2[keep_mask]
        areas = areas[keep_mask]

    return np.asarray(keep, dtype=np.int64)


def nms_torch(boxes: ArrayLike, scores: ArrayLike, iou_thr: float = 0.5, topk: Optional[int] = None, device: Optional[str] = None) -> np.ndarray:
    """
    Torch/Torchvision NMS wrapper. Returns kept indices relative to original input.
    """
    if not _HAS_TORCHVISION:
        return nms_numpy(boxes, scores, iou_thr=iou_thr, topk=topk)

    boxes = _ensure_np2d(boxes, np.float32)
    scores = _ensure_np1d(scores, np.float32)
    dev = device or ("cuda" if (torch is not None and torch.cuda.is_available()) else "cpu")  # type: ignore

    t_boxes = torch.as_tensor(boxes, device=dev)  # type: ignore
    t_scores = torch.as_tensor(scores, device=dev)  # type: ignore

    if topk is not None and boxes.shape[0] > topk:
        order = torch.argsort(t_scores, descending=True)[: int(topk)]  # type: ignore
        t_boxes_sub = t_boxes[order]
        t_scores_sub = t_scores[order]
        kept_rel = tv_nms(t_boxes_sub, t_scores_sub, float(iou_thr))  # type: ignore
        kept = order[kept_rel]
    else:
        kept = tv_nms(t_boxes, t_scores, float(iou_thr))  # type: ignore

    return kept.detach().cpu().numpy().astype(np.int64)


# ---------- robust helpers for labels/scores coercion ----------

def _as_float_array(arr) -> Optional[np.ndarray]:
    """
    Try to convert to float32 array; return None if not possible.
    """
    try:
        a = np.asarray(arr)
        if a.dtype.kind in ("f", "i", "u"):
            return a.astype(np.float32, copy=False)
        return a.astype(np.float32)
    except Exception:
        return None


def _labels_to_ids(arr) -> np.ndarray:
    """
    Convert labels array (str/int/float) to stable int32 ids.
    - int/uint -> cast to int32
    - float -> if all near-integers, cast; else map unique values to ids
    - str/object -> map sorted unique strings to ids
    """
    a = np.asarray(arr)
    kind = a.dtype.kind
    if kind in ("i", "u"):
        return a.astype(np.int32, copy=False)
    if kind == "f":
        rounded = np.rint(a)
        if np.allclose(a, rounded):
            return rounded.astype(np.int32, copy=False)
        uniq_vals = [float(v) for v in np.unique(a)]
        mapping = {v: i for i, v in enumerate(uniq_vals)}
        return np.asarray([mapping[float(v)] for v in a], dtype=np.int32)
    # Strings / objects
    as_str = a.astype(str)
    uniq = sorted(set(as_str.tolist()))
    mapping = {s: i for i, s in enumerate(uniq)}
    return np.asarray([mapping[s] for s in as_str], dtype=np.int32)


def _coerce_labels_scores(
    a2: ArrayLike,
    a3: ArrayLike,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Infer which is labels vs scores without assuming numeric dtypes for labels.
    Priority:
      - If a2 looks like labels (str/object/int) and a3 can be float -> (labels=a2, scores=a3)
      - Else if a3 looks like labels and a2 can be float -> (labels=a3, scores=a2)
      - Else fall back to favor the one that can convert to float as scores.
    Returns (labels_np:int32, scores_np:float32).
    """
    a2_np = np.asarray(a2)
    a3_np = np.asarray(a3)
    s2 = _as_float_array(a2_np)
    s3 = _as_float_array(a3_np)

    a2_label_like = a2_np.dtype.kind in ("i", "u", "O", "U", "S")
    a3_label_like = a3_np.dtype.kind in ("i", "u", "O", "U", "S")

    if a2_label_like and s3 is not None:
        return _labels_to_ids(a2_np), s3
    if a3_label_like and s2 is not None:
        return _labels_to_ids(a3_np), s2

    # If both numeric, prefer integer-like as labels and float as scores
    if s2 is not None and a3_np.dtype.kind in ("i", "u") and s3 is None:
        return _labels_to_ids(a3_np), s2
    if s3 is not None and a2_np.dtype.kind in ("i", "u") and s2 is None:
        return _labels_to_ids(a2_np), s3

    # Fallbacks: choose the one convertible to float as scores
    if s3 is not None:
        return _labels_to_ids(a2_np), s3
    if s2 is not None:
        return _labels_to_ids(a3_np), s2

    raise ValueError("Cannot infer labels and scores: provide numeric scores and label-like array.")


# ---------- APIs ----------

def labelwise_nms(
    boxes: ArrayLike,
    a2: ArrayLike,
    a3: Optional[ArrayLike] = None,
    iou_thr: float = 0.5,
    iou_threshold: Optional[float] = None,
    topk_per_class: Optional[int] = None,
    backend: str = "auto",
    device: Optional[str] = None,
) -> np.ndarray:
    """
    Run per-class NMS and return kept indices (relative to input arrays).

    Backward-compat:
    - Supports both orderings:
        (boxes, scores, labels)
        (boxes, labels, scores)
    - Accepts `iou_threshold` as alias for `iou_thr`.
    """
    if iou_threshold is not None:
        iou_thr = float(iou_threshold)

    boxes_np = _ensure_np2d(boxes, np.float32)
    if a3 is None:
        raise ValueError("labelwise_nms requires both labels and scores arrays.")
    labels_np, scores_np = _coerce_labels_scores(a2, a3)

    if boxes_np.shape[0] == 0:
        return np.empty((0,), dtype=np.int64)

    kept: List[int] = []
    for cls in np.unique(labels_np):
        mask = (labels_np == cls)
        idxs = np.nonzero(mask)[0]
        if idxs.size == 0:
            continue

        sub_boxes = boxes_np[idxs]
        sub_scores = scores_np[idxs]

        if backend == "torch" or (backend == "auto" and _HAS_TORCHVISION):
            kept_rel = nms_torch(sub_boxes, sub_scores, iou_thr=iou_thr, topk=topk_per_class, device=device)
        else:
            kept_rel = nms_numpy(sub_boxes, sub_scores, iou_thr=iou_thr, topk=topk_per_class)

        kept.extend(idxs[kept_rel].tolist())

    kept = np.asarray(kept, dtype=np.int64)
    order = scores_np[kept].argsort()[::-1]
    return kept[order]


def soft_nms(
    boxes: ArrayLike,
    scores: ArrayLike,
    iou_thr: float = 0.5,
    sigma: float = 0.5,
    method: str = "linear",
    score_thr: float = 0.001,
) -> np.ndarray:
    """
    Simple NumPy Soft-NMS implementation.
    Returns indices of boxes with final score >= score_thr, sorted by final score desc.
    method: "linear" | "gaussian" | "original" (original uses IoU suppression)
    """
    boxes = _ensure_np2d(boxes, np.float32).copy()
    scores = _ensure_np1d(scores, np.float32).copy()
    N = boxes.shape[0]
    if N == 0:
        return np.empty((0,), dtype=np.int64)

    idxs = np.arange(N)
    keep = []

    while idxs.size > 0:
        max_idx = np.argmax(scores[idxs])
        current = idxs[max_idx]
        keep.append(current)
        if idxs.size == 1:
            break
        others = np.delete(idxs, max_idx)

        # compute IoU between current and others
        iou_vals = iou(boxes[[current]], boxes[others])[0]

        if method == "linear":
            decay = np.where(iou_vals > iou_thr, 1 - iou_vals, 1.0)
        elif method == "gaussian":
            decay = np.exp(- (iou_vals ** 2) / sigma)
        else:  # original
            decay = np.where(iou_vals > iou_thr, 0.0, 1.0)

        scores[others] = scores[others] * decay
        idxs = np.array([i for i in idxs if scores[i] >= score_thr])

    # sort keep by final score desc
    keep_scores = scores[keep]
    order = np.argsort(keep_scores)[::-1]
    return np.asarray(keep, dtype=np.int64)[order]


def _detection_to_arrays(dets: List["Detection"]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Helper to convert list[Detection] -> (boxes, scores, labels_idx, labels_str)
    labels_idx will be numeric ids assigned deterministically from sorted unique labels.
    """
    boxes = []
    scores = []
    labels_str = []
    for d in dets:
        b = getattr(d, "box", (0, 0, 0, 0))
        boxes.append([float(b[0]), float(b[1]), float(b[2]), float(b[3])])
        scores.append(float(getattr(d, "score", 1.0)))
        labels_str.append(str(getattr(d, "label", "")))
    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    uniq = sorted(set(labels_str))
    lab2id = {l: i for i, l in enumerate(uniq)}
    labels_idx = np.asarray([lab2id[l] for l in labels_str], dtype=np.int32)
    return boxes, scores, labels_idx, uniq


def _make_detection_from_original(d: "Detection"):
    return d


def nms(
    arg: Union[List["Detection"], ArrayLike],
    *,
    scores: Optional[ArrayLike] = None,
    labels: Optional[ArrayLike] = None,
    iou_thr: float = 0.5,
    class_aware: bool = False,
    sort_desc: bool = True,
    topk: Optional[int] = None,
    topk_per_class: Optional[int] = None,
    backend: str = "auto",
    device: Optional[str] = None,
) -> Union[np.ndarray, List["Detection"]]:
    """
    Flexible NMS API:
    - If `arg` is an array of boxes -> requires scores, returns indices kept.
    - If `arg` is list[Detection] -> returns filtered list[Detection] (kept).
      Use class_aware=True to run per-class NMS for Detection list.
    """
    # list[Detection] case
    if isinstance(arg, list) and (len(arg) == 0 or Detection is None or hasattr(arg[0], "box")):
        dets: List["Detection"] = arg  # type: ignore
        if not dets:
            return []
        boxes_np, scores_np, labels_idx, uniq_labels = _detection_to_arrays(dets)

        if class_aware:
            kept_idxs = labelwise_nms(
                boxes_np,
                labels_idx,  # support (boxes, labels, scores)
                scores_np,
                iou_thr=iou_thr,
                topk_per_class=topk_per_class,
                backend=backend,
                device=device,
            )
        else:
            if backend == "torch" or (backend == "auto" and _HAS_TORCHVISION):
                kept_idxs = nms_torch(boxes_np, scores_np, iou_thr=iou_thr, topk=topk, device=device)
            else:
                kept_idxs = nms_numpy(boxes_np, scores_np, iou_thr=iou_thr, topk=topk)

        kept_list = [dets[int(i)] for i in kept_idxs.tolist()]
        if sort_desc:
            kept_list.sort(key=lambda d: float(getattr(d, "score", 0.0)), reverse=True)
        return kept_list

    # array boxes case
    boxes_np = _ensure_np2d(arg, np.float32)  # type: ignore
    if scores is None and labels is None:
        raise ValueError("scores/labels must be provided when first argument is array of boxes")
    if class_aware:
        if scores is None or labels is None:
            raise ValueError("labels and scores are required when class_aware=True for array input")
        # support both (boxes, scores, labels) and (boxes, labels, scores)
        labels_np, scores_np = _coerce_labels_scores(labels, scores)
        return labelwise_nms(
            boxes_np,
            labels_np,
            scores_np,
            iou_thr=iou_thr,
            topk_per_class=topk_per_class,
            backend=backend,
            device=device,
        )
    else:
        if scores is None:
            raise ValueError("scores must be provided for non class-aware array NMS")
        scores_np = _ensure_np1d(scores, np.float32)
        if backend == "torch" or (backend == "auto" and _HAS_TORCHVISION):
            return nms_torch(boxes_np, scores_np, iou_thr=iou_thr, topk=topk, device=device)
        return nms_numpy(boxes_np, scores_np, iou_thr=iou_thr, topk=topk)


# ------------------------------------------------------------------------------
# SOTA NMS Methods (2024)
# ------------------------------------------------------------------------------

def soft_nms_gaussian(
    boxes: ArrayLike,
    scores: ArrayLike,
    iou_threshold: float = 0.5,
    sigma: float = 0.5,
    score_threshold: float = 0.001,
) -> np.ndarray:
    """
    Soft-NMS with Gaussian decay (ICCV 2017).
    
    Instead of removing overlapping boxes, decays their scores based on IoU.
    Better preserves high-confidence overlapping detections.
    
    Paper: Improving Object Detection With One Line of Code
    https://arxiv.org/abs/1704.04503
    
    Args:
        boxes: (N, 4) boxes in xyxy format
        scores: (N,) confidence scores
        iou_threshold: IoU threshold for suppression
        sigma: Gaussian decay parameter (default: 0.5)
        score_threshold: Minimum score to keep (default: 0.001)
    
    Returns:
        Indices of boxes to keep (sorted by score descending)
    
    Examples:
        >>> indices = soft_nms_gaussian(boxes, scores, iou_threshold=0.5)
        >>> kept_boxes = boxes[indices]
    """
    boxes = _ensure_np2d(boxes, np.float32)
    scores = _ensure_np1d(scores, np.float32).copy()  # Will modify in-place
    N = len(boxes)
    
    if N == 0:
        return np.array([], dtype=np.int64)
    
    # Get sorted indices (descending score)
    order = np.argsort(-scores)
    keep = []
    
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        
        if len(order) == 1:
            break
        
        # Compute IoU with remaining boxes
        ious = iou(boxes[i:i+1], boxes[order[1:]])[0]
        
        # Gaussian decay: score *= exp(-(iou^2) / sigma)
        decay = np.exp(-(ious ** 2) / sigma)
        scores[order[1:]] *= decay
        
        # Remove boxes below threshold and re-sort
        valid_mask = scores[order[1:]] > score_threshold
        order = order[1:][valid_mask]
        order = order[np.argsort(-scores[order])]
    
    return np.array(keep, dtype=np.int64)


def diou_nms(
    boxes: ArrayLike,
    scores: ArrayLike,
    iou_threshold: float = 0.5,
) -> np.ndarray:
    """
    DIoU-NMS: Distance-IoU aware NMS (AAAI 2020).
    
    Considers both IoU and center point distance.
    Better for crowded scenes with overlapping objects.
    
    Paper: Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression
    https://arxiv.org/abs/1911.08287
    
    DIoU = IoU - (distance^2) / (diagonal^2)
    
    Args:
        boxes: (N, 4) boxes in xyxy format
        scores: (N,) confidence scores
        iou_threshold: DIoU threshold for suppression
    
    Returns:
        Indices of boxes to keep (sorted by score descending)
    """
    boxes = _ensure_np2d(boxes, np.float32)
    scores = _ensure_np1d(scores, np.float32)
    N = len(boxes)
    
    if N == 0:
        return np.array([], dtype=np.int64)
    
    # Compute box centers and enclosing box diagonals
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    centers_x = (x1 + x2) / 2
    centers_y = (y1 + y2) / 2
    
    # Get sorted indices
    order = np.argsort(-scores)
    keep = []
    
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        
        if len(order) == 1:
            break
        
        # Compute IoU
        ious = iou(boxes[i:i+1], boxes[order[1:]])[0]
        
        # Compute center distances
        dx = centers_x[i] - centers_x[order[1:]]
        dy = centers_y[i] - centers_y[order[1:]]
        center_dist_sq = dx ** 2 + dy ** 2
        
        # Compute enclosing box diagonal
        enclose_x1 = np.minimum(x1[i], x1[order[1:]])
        enclose_y1 = np.minimum(y1[i], y1[order[1:]])
        enclose_x2 = np.maximum(x2[i], x2[order[1:]])
        enclose_y2 = np.maximum(y2[i], y2[order[1:]])
        enclose_diag_sq = (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2
        
        # DIoU = IoU - (center_dist^2) / (diagonal^2)
        dious = ious - (center_dist_sq / (enclose_diag_sq + 1e-7))
        
        # Keep boxes with DIoU < threshold
        mask = dious < iou_threshold
        order = order[1:][mask]
    
    return np.array(keep, dtype=np.int64)


def matrix_nms(
    boxes: ArrayLike,
    scores: ArrayLike,
    iou_threshold: float = 0.5,
    sigma: float = 2.0,
) -> np.ndarray:
    """
    Matrix-NMS: Parallel, efficient NMS (ECCV 2020).
    
    Computes all pairwise IoUs in parallel, then decays scores
    based on IoU with higher-scoring boxes.
    
    Paper: SOLOv2: Dynamic and Fast Instance Segmentation
    https://arxiv.org/abs/2003.10152
    
    Args:
        boxes: (N, 4) boxes in xyxy format
        scores: (N,) confidence scores
        iou_threshold: IoU threshold
        sigma: Decay parameter (default: 2.0)
    
    Returns:
        Indices of boxes to keep
    """
    boxes = _ensure_np2d(boxes, np.float32)
    scores = _ensure_np1d(scores, np.float32).copy()
    N = len(boxes)
    
    if N == 0:
        return np.array([], dtype=np.int64)
    
    # Compute full IoU matrix (N x N)
    iou_matrix = iou(boxes, boxes)
    
    # Sort by score descending
    order = np.argsort(-scores)
    
    # Decay scores based on IoU with higher-scoring boxes
    for i in range(N):
        idx = order[i]
        # Get IoUs with all higher-scoring boxes
        higher_ious = iou_matrix[idx, order[:i]]
        
        if len(higher_ious) > 0:
            # Decay based on max IoU with higher-scoring box
            max_iou = higher_ious.max()
            decay = np.exp(-(max_iou ** 2) / sigma)
            scores[idx] *= decay
    
    # Keep boxes above threshold
    keep_mask = scores > iou_threshold * scores.max()
    keep_indices = np.where(keep_mask)[0]
    
    # Sort by updated scores
    keep_indices = keep_indices[np.argsort(-scores[keep_indices])]
    
    return keep_indices


def adaptive_nms(
    boxes: ArrayLike,
    scores: ArrayLike,
    iou_threshold: float = 0.5,
    density_aware: bool = True,
) -> np.ndarray:
    """
    Adaptive NMS: Adjusts threshold based on local density (CVPR 2019).
    
    In crowded regions, uses lower IoU threshold.
    In sparse regions, uses higher IoU threshold.
    
    Paper: Adaptive NMS: Refining Pedestrian Detection in a Crowd
    https://arxiv.org/abs/1904.03629
    
    Args:
        boxes: (N, 4) boxes in xyxy format
        scores: (N,) confidence scores
        iou_threshold: Base IoU threshold
        density_aware: Enable density-based threshold adaptation
    
    Returns:
        Indices of boxes to keep
    """
    boxes = _ensure_np2d(boxes, np.float32)
    scores = _ensure_np1d(scores, np.float32)
    N = len(boxes)
    
    if N == 0:
        return np.array([], dtype=np.int64)
    
    # Compute IoU matrix
    iou_matrix = iou(boxes, boxes)
    
    # Estimate local density for each box
    if density_aware:
        # Count neighbors within IoU threshold
        neighbors = (iou_matrix > 0.3).sum(axis=1) - 1  # Exclude self
        # Normalize density to [0, 1]
        max_neighbors = neighbors.max()
        density = neighbors / (max_neighbors + 1e-7) if max_neighbors > 0 else np.zeros_like(neighbors)
        # Adaptive threshold: lower in dense regions
        adaptive_thresholds = iou_threshold * (1.0 - 0.3 * density)
    else:
        adaptive_thresholds = np.full(N, iou_threshold)
    
    # Standard NMS with adaptive thresholds
    order = np.argsort(-scores)
    keep = []
    
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        
        if len(order) == 1:
            break
        
        # Use adaptive threshold for current box
        threshold = adaptive_thresholds[i]
        ious = iou_matrix[i, order[1:]]
        mask = ious < threshold
        order = order[1:][mask]
    
    return np.array(keep, dtype=np.int64)


__all__ = [
    "nms", 
    "soft_nms", 
    "iou", 
    "labelwise_nms",
    # SOTA methods
    "soft_nms_gaussian",
    "diou_nms",
    "matrix_nms",
    "adaptive_nms",
]