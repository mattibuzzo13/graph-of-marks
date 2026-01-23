# igp/types.py
"""
Core Type Definitions for Image Graph Preprocessor

This module defines the fundamental data structures used throughout the IGP pipeline
for representing detection results, segmentation masks, and relationships between objects.

Types:
    Box: Type alias for bounding box coordinates (x1, y1, x2, y2)
    Detection: Dataclass for object detection results
    MaskDict: TypedDict for SAM-style segmentation masks
    Relationship: Dataclass for spatial/semantic relationships

Design Philosophy:
    - Immutable by default (dataclasses with frozen=False for flexibility)
    - Type-safe with comprehensive type hints
    - Compatible with numpy, PIL, and matplotlib
    - Minimal dependencies (only numpy for mask arrays)

Usage:
    >>> from gom.types import Detection, Relationship, Box
    >>> det = Detection(
    ...     box=(100, 200, 300, 400),
    ...     label="person",
    ...     score=0.95,
    ...     source="yolov8"
    ... )
    >>> rel = Relationship(
    ...     src_idx=0, tgt_idx=1,
    ...     relation="left_of",
    ...     distance=150.0
    ... )
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, TypedDict

import numpy as np

# Bounding box in XYXY pixel coordinates: (x1, y1, x2, y2)
# where (x1, y1) is top-left corner and (x2, y2) is bottom-right corner
Box = Tuple[float, float, float, float]


@dataclass
class Detection:
    """
    Canonical detection record from object detectors.
    
    Represents a single detected object with bounding box, class label, and
    confidence score. Optionally includes source detector name and additional
    metadata (e.g., segmentation masks, features).
    
    Attributes:
        box: Bounding box in XYXY pixel coordinates (x1, y1, x2, y2)
            where (x1, y1) is top-left and (x2, y2) is bottom-right.
            Coordinates are absolute pixels, not normalized.
        label: Human-readable class name (e.g., "person", "car", "table").
               Should be lowercase for consistency.
        score: Detection confidence score in range [0.0, 1.0].
              Higher values indicate more confident detections.
              Default: 1.0 (for ground-truth annotations)
        source: Optional identifier for the detector that produced this detection
               (e.g., "yolov8", "owlvit", "detectron2", "grounding_dino").
               Useful for multi-detector fusion and provenance tracking.
        extra: Optional dictionary for additional detector-specific data:
               - Segmentation masks (from Detectron2, SAM)
               - Feature vectors (from CLIP)
               - Keypoints (from pose estimators)
               - Any other metadata
    
    Examples:
        >>> # Simple detection from YOLO
        >>> det = Detection(
        ...     box=(100.5, 200.3, 350.2, 450.7),
        ...     label="person",
        ...     score=0.95,
        ...     source="yolov8"
        ... )
        
        >>> # Detection with segmentation mask
        >>> det_with_mask = Detection(
        ...     box=(50, 50, 200, 200),
        ...     label="cat",
        ...     score=0.88,
        ...     source="detectron2",
        ...     extra={"mask": binary_mask_array}
        ... )
    
    Notes:
        - Box coordinates should be valid (x2 > x1, y2 > y1)
        - Labels are case-sensitive but lowercase is recommended
        - Score typically from softmax/sigmoid output of detector
        - Extra field is intentionally flexible for extensibility
    """
    box: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    label: str
    score: float = 1.0
    source: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None


class MaskDict(TypedDict, total=False):
    """
    SAM-style segmentation mask bundle.
    
    This TypedDict follows the Segment Anything Model (SAM) output format,
    containing a binary segmentation mask along with associated metadata.
    Used for instance segmentation results from SAM, SAM-HQ, FastSAM, etc.
    
    Attributes:
        segmentation: Binary mask array of shape (H, W) with dtype bool or uint8.
                     True/1 indicates foreground pixels belonging to the object,
                     False/0 indicates background.
        bbox: Bounding box in XYWH format [x, y, width, height] in pixels,
             where (x, y) is the top-left corner. Note: This differs from
             Detection.box which uses XYXY format.
        predicted_iou: Model's self-estimated Intersection over Union score
                      for mask quality. Range [0.0, 1.0], higher is better.
                      Useful for filtering low-quality masks.
    
    Notes:
        - total=False means all fields are optional (TypedDict with partial keys)
        - segmentation is the only strictly required field in practice
        - bbox can be recomputed from segmentation mask if missing
        - predicted_iou may not be available from all segmentation models
    
    Examples:
        >>> mask_dict: MaskDict = {
        ...     "segmentation": np.array([[0, 1, 1], [0, 1, 1]], dtype=bool),
        ...     "bbox": [100, 200, 50, 80],
        ...     "predicted_iou": 0.92
        ... }
        
        >>> # Minimal mask (just segmentation)
        >>> minimal_mask: MaskDict = {
        ...     "segmentation": binary_mask
        ... }
    
    SAM Output Format:
        This matches the dictionary structure returned by:
        - facebook/segment-anything (SAM)
        - SysCV/sam-hq (SAM-HQ)
        - CASIA-IVA-Lab/FastSAM
        Ensures compatibility with SAM ecosystem.
    """
    segmentation: np.ndarray
    bbox: List[int]
    predicted_iou: float


@dataclass
class Relationship:
    """
    Directed spatial or semantic relationship between two objects.
    
    Represents a relationship predicate connecting two detected objects in a scene.
    Relationships are directed: src_idx → tgt_idx with a specific relation type.
    Used for building scene graphs and structured scene representations.
    
    Attributes:
        src_idx: Index of the source (subject) object in the detections list.
                Example: For "person left_of car", person is the source.
        tgt_idx: Index of the target (object) object in the detections list.
                Example: For "person left_of car", car is the target.
        relation: Canonical relationship predicate (normalized form).
                 Common spatial relations: "left_of", "right_of", "above", "below",
                     "on_top_of", "under", "in_front_of", "behind"
                 Proximity relations: "near", "far", "touching", "adjacent"
                 Semantic relations: "holding", "wearing", "riding", "sitting_on"
                 Should be lowercase with underscores for consistency.
        distance: Geometric distance metric in pixels (or normalized).
                 Used for prioritizing relationships and filtering distant pairs.
                 Lower values indicate closer/stronger relationships.
                 Default: float("inf") for unknown distance.
        relation_raw: Optional unnormalized relationship text.
                     Stores original form before canonicalization (e.g., from CLIP).
                     Example: "is positioned to the left of" → relation="left_of"
        clip_sim: Optional CLIP similarity score when relation extracted via CLIP.
                 Range [0.0, 1.0], higher indicates more confident relationship.
                 Useful for filtering weak semantic relationships.
    
    Examples:
        >>> # Spatial relationship with distance
        >>> rel = Relationship(
        ...     src_idx=0,  # person
        ...     tgt_idx=1,  # car
        ...     relation="left_of",
        ...     distance=150.5
        ... )
        
        >>> # Semantic relationship from CLIP
        >>> clip_rel = Relationship(
        ...     src_idx=2,  # person
        ...     tgt_idx=3,  # bicycle
        ...     relation="riding",
        ...     relation_raw="riding on",
        ...     clip_sim=0.87,
        ...     distance=25.0
        ... )
        
        >>> # Simple proximity relation
        >>> near_rel = Relationship(
        ...     src_idx=4,
        ...     tgt_idx=5,
        ...     relation="near",
        ...     distance=50.0
        ... )
    
    Notes:
        - Relationships are directional (src → tgt)
        - distance=inf indicates uncalculated or irrelevant distance
        - Multiple relationships can exist between same object pair
        - Indices reference the detections list from preprocessing
    
    Scene Graph Representation:
        Relationships form edges in a directed scene graph:
        - Nodes: Detected objects (via src_idx, tgt_idx)
        - Edges: Relationships with labels (relation attribute)
        - Edge attributes: distance, clip_sim for filtering/ranking
    """
    src_idx: int
    tgt_idx: int
    relation: str
    distance: float = float("inf")
    relation_raw: Optional[str] = None
    clip_sim: Optional[float] = None
