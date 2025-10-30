# igp/relations/geometry/vectorized.py
# Vectorized batch operations for efficient processing of multiple boxes

from __future__ import annotations

from typing import Optional

import numpy as np


def centers_vectorized(boxes: np.ndarray) -> np.ndarray:
    """
    Compute centers for multiple boxes at once.
    
    Args:
        boxes: array of shape (N, 4) in xyxy format
    
    Returns:
        centers: array of shape (N, 2) with (cx, cy) for each box
    """
    boxes = np.asarray(boxes, dtype=np.float32)
    if boxes.ndim == 1:
        boxes = boxes.reshape(1, -1)
    
    cx = (boxes[:, 0] + boxes[:, 2]) / 2.0
    cy = (boxes[:, 1] + boxes[:, 3]) / 2.0
    return np.stack([cx, cy], axis=1)


def areas_vectorized(boxes: np.ndarray) -> np.ndarray:
    """
    Compute areas for multiple boxes at once.
    
    Args:
        boxes: array of shape (N, 4) in xyxy format
    
    Returns:
        areas: array of shape (N,) with area for each box
    """
    boxes = np.asarray(boxes, dtype=np.float32)
    if boxes.ndim == 1:
        boxes = boxes.reshape(1, -1)
    
    w = np.maximum(0, boxes[:, 2] - boxes[:, 0])
    h = np.maximum(0, boxes[:, 3] - boxes[:, 1])
    return w * h


def pairwise_distances_vectorized(centers1: np.ndarray, centers2: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute pairwise distances between centers.
    
    Args:
        centers1: array of shape (N, 2)
        centers2: array of shape (M, 2), if None uses centers1
    
    Returns:
        distances: array of shape (N, M) with Euclidean distances
    """
    centers1 = np.asarray(centers1, dtype=np.float32)
    if centers2 is None:
        centers2 = centers1
    else:
        centers2 = np.asarray(centers2, dtype=np.float32)
    
    # Broadcasting: (N, 1, 2) - (1, M, 2) = (N, M, 2)
    diff = centers1[:, np.newaxis, :] - centers2[np.newaxis, :, :]
    distances = np.sqrt(np.sum(diff**2, axis=2))
    return distances


def horizontal_overlaps_vectorized(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """
    Compute horizontal overlap for all pairs.
    
    Args:
        boxes1: array of shape (N, 4)
        boxes2: array of shape (M, 4)
    
    Returns:
        overlaps: array of shape (N, M)
    """
    boxes1 = np.asarray(boxes1, dtype=np.float32).reshape(-1, 4)
    boxes2 = np.asarray(boxes2, dtype=np.float32).reshape(-1, 4)
    
    # Broadcast to (N, M)
    x1_1 = boxes1[:, 0:1]  # (N, 1)
    x2_1 = boxes1[:, 2:3]  # (N, 1)
    x1_2 = boxes2[:, 0:1].T  # (1, M)
    x2_2 = boxes2[:, 2:3].T  # (1, M)
    
    overlap_x1 = np.maximum(x1_1, x1_2)
    overlap_x2 = np.minimum(x2_1, x2_2)
    overlaps = np.maximum(0, overlap_x2 - overlap_x1)
    
    return overlaps


def vertical_overlaps_vectorized(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """
    Compute vertical overlap for all pairs.
    
    Args:
        boxes1: array of shape (N, 4)
        boxes2: array of shape (M, 4)
    
    Returns:
        overlaps: array of shape (N, M)
    """
    boxes1 = np.asarray(boxes1, dtype=np.float32).reshape(-1, 4)
    boxes2 = np.asarray(boxes2, dtype=np.float32).reshape(-1, 4)
    
    # Broadcast to (N, M)
    y1_1 = boxes1[:, 1:2]  # (N, 1)
    y2_1 = boxes1[:, 3:4]  # (N, 1)
    y1_2 = boxes2[:, 1:2].T  # (1, M)
    y2_2 = boxes2[:, 3:4].T  # (1, M)
    
    overlap_y1 = np.maximum(y1_1, y1_2)
    overlap_y2 = np.minimum(y2_1, y2_2)
    overlaps = np.maximum(0, overlap_y2 - overlap_y1)
    
    return overlaps
