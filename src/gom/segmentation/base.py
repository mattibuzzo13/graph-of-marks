# igp/segmentation/base.py
"""
Segmentation Model Abstract Base Class

Defines the interface for SAM-like segmentation models that generate masks
from bounding box prompts. Provides common utilities for mask post-processing,
hole filling, and component filtering.

Key Features:
    - Abstract interface for SAM variants (SAM1, SAM2, SAM-HQ, FastSAM)
    - Mask post-processing: hole filling, component removal
    - Multiple backend support (OpenCV, SciPy, NumPy fallbacks)
    - XYWH bbox extraction from masks
    - Box clamping to image bounds

Supported Segmenters:
    - SAM1Segmenter: Original Segment Anything (sam1.py)
    - SAM2Segmenter: Segment Anything v2 (sam2.py)
    - SAMHQSegmenter: High-quality SAM variant (samhq.py)
    - FastSAMSegmenter: Real-time SAM (fastsam.py)

Architecture:
    SegmenterConfig: Configuration for post-processing
        - device: Compute device
        - close_holes: Enable morphological hole filling
        - hole_kernel: Kernel size for closing operation
        - min_hole_area: Minimum hole size to fill (pixels)
        - remove_small_components: Filter small mask components
        - min_component_area: Minimum component size to keep
    
    Segmenter (ABC):
        - segment(image, boxes) → List[MaskDict] [abstract]
        - clamp_box_xyxy(box, W, H) → [x1,y1,x2,y2] [static]
        - bbox_from_mask(mask) → [x,y,w,h] [static]
        - close_mask_holes(mask) → refined_mask
        - remove_small_components_from_mask(mask) → filtered_mask

Output Format (MaskDict):
    {
        'segmentation': np.ndarray(bool, H, W),  # Binary mask
        'bbox': [x, y, w, h],                     # XYWH format
        'predicted_iou': float,                   # Quality score
        'area': int,                              # Mask area in pixels
        'stability_score': float                  # Optional confidence
    }

Usage:
    >>> from gom.segmentation import SAM1Segmenter, SegmenterConfig
    >>> 
    >>> # Configure post-processing
    >>> config = SegmenterConfig(
    ...     device="cuda",
    ...     close_holes=True,
    ...     min_hole_area=100,
    ...     remove_small_components=True,
    ...     min_component_area=500
    ... )
    >>> 
    >>> # Initialize segmenter
    >>> segmenter = SAM1Segmenter(config, checkpoint_path="sam_vit_h.pth")
    >>> 
    >>> # Generate masks from boxes
    >>> boxes = [[100, 100, 300, 300], [400, 200, 600, 500]]
    >>> masks = segmenter.segment(image, boxes)
    >>> for mask_dict in masks:
    ...     mask = mask_dict['segmentation']
    ...     print(f"Area: {mask.sum()} pixels")

Hole Filling Algorithm:
    1. Morphological closing (dilate → erode)
    2. Flood fill from background
    3. Identify holes (interior background regions)
    4. Filter by area: fill holes < min_hole_area
    5. Fallback chain: OpenCV → SciPy → NumPy

Backend Priority:
    - OpenCV (cv2): Fastest, most robust
    - SciPy (ndimage): Good fallback
    - NumPy: Minimal naive implementation

Notes:
    - All boxes must be in XYXY pixel coordinates
    - Output masks are in SAM format (XYWH bbox)
    - Post-processing is optional but recommended
    - Device auto-selection: CUDA if available

See Also:
    - gom.segmentation.sam1: SAM v1 implementation
    - gom.segmentation.sam2: SAM v2 implementation
    - gom.segmentation.refinement: Advanced post-processing
    - gom.types.MaskDict: Mask dictionary type definition
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


@dataclass
class SegmenterConfig:
    """
    Configuration for segmentation models and post-processing.
    
    Attributes:
        device: Compute device ("cuda", "cpu", "mps", None=auto)
        close_holes: Enable morphological hole filling (bool, default False)
        hole_kernel: Kernel size for closing operation (int, default 7)
        min_hole_area: Minimum hole size to fill in pixels (int, default 100)
        remove_small_components: Filter out small mask regions (bool, default False)
        min_component_area: Minimum component size in pixels (int, default 0)
    
    Examples:
        >>> # Aggressive cleaning
        >>> config = SegmenterConfig(
        ...     close_holes=True,
        ...     min_hole_area=50,
        ...     remove_small_components=True,
        ...     min_component_area=1000
        ... )
        
        >>> # Minimal processing
        >>> config = SegmenterConfig(device="cuda")
    
    Notes:
        - close_holes: Fills interior holes via morphological operations
        - min_hole_area: Holes smaller than this are filled
        - remove_small_components: Removes disconnected mask regions
        - Larger hole_kernel = more aggressive smoothing
    """
    device: Optional[str] = None
    close_holes: bool = False
    hole_kernel: int = 7
    min_hole_area: int = 100
    remove_small_components: bool = False
    min_component_area: int = 0


class Segmenter(ABC):
    """
    Abstract base class for SAM-like segmentation models.
    
    Subclasses must implement:
        - segment(image_pil, boxes) → List[MaskDict]
    
    Provides utilities:
        - clamp_box_xyxy: Box clamping to image bounds
        - bbox_from_mask: XYWH extraction from binary mask
        - close_mask_holes: Morphological hole filling
        - remove_small_components_from_mask: Component filtering
    
    Attributes:
        config: SegmenterConfig instance
    """

    def __init__(self, config: SegmenterConfig | None = None) -> None:
        """
        Initialize segmenter with configuration.
        
        Args:
            config: SegmenterConfig instance (None creates default)
        """
        self.config = config or SegmenterConfig()

    # --------- Required API ---------
    @abstractmethod
    def segment(self, image_pil, boxes: Sequence[Sequence[float]]) -> List[Dict[str, Any]]:
        """
        Generate segmentation masks from bounding box prompts.

        Args:
            image_pil: Input PIL image
            boxes: List of boxes in XYXY format [[x1, y1, x2, y2], ...]
        
        Returns:
            List of mask dictionaries with keys:
                - 'segmentation': np.ndarray(bool, H, W) - Binary mask
                - 'bbox': [x, y, w, h] - Bounding box in XYWH format
                - 'predicted_iou': float - Quality score [0, 1]
                - 'area': int - Mask area in pixels (optional)
                - 'stability_score': float - Confidence score (optional)
        
        Notes:
            - Input boxes must be in XYXY pixel coordinates
            - Output bbox is in XYWH (SAM format)
            - Empty boxes may return empty masks
            - Post-processing applied based on config
        
        Example:
            >>> masks = segmenter.segment(image, [[100, 100, 300, 300]])
            >>> mask_array = masks[0]['segmentation']
            >>> bbox_xywh = masks[0]['bbox']
        """
        raise NotImplementedError

    # --------- Common utilities ---------
    @staticmethod
    def clamp_box_xyxy(box: Sequence[float], W: int, H: int) -> List[int]:
        """
        Clamp XYXY box to image dimensions ensuring valid box.
        
        Args:
            box: [x1, y1, x2, y2] in pixel coordinates
            W: Image width
            H: Image height
        
        Returns:
            Clamped box [x1, y1, x2, y2] with guarantees:
                - 0 <= x1 < x2 <= W-1
                - 0 <= y1 < y2 <= H-1
                - x2 > x1 and y2 > y1 (non-degenerate)
        
        Example:
            >>> box = [-10, 50, 1920, 1100]
            >>> clamped = Segmenter.clamp_box_xyxy(box, W=1920, H=1080)
            >>> clamped
            [0, 50, 1919, 1079]
        
        Notes:
            - Handles out-of-bounds coordinates
            - Ensures minimum box size (1x1 pixel)
            - Safe for very small images (W=1, H=1)
        """
        W = max(1, int(W))
        H = max(1, int(H))
        x1, y1, x2, y2 = box[:4]
        x1 = int(np.clip(round(x1), 0, W - 1))
        y1 = int(np.clip(round(y1), 0, H - 1))
        x2 = int(np.clip(round(x2), 0, W - 1))
        y2 = int(np.clip(round(y2), 0, H - 1))
        if x2 <= x1:
            x2 = min(W - 1, x1 + 1)
        if y2 <= y1:
            y2 = min(H - 1, y1 + 1)
        return [x1, y1, x2, y2]

    @staticmethod
    def bbox_from_mask(mask: np.ndarray) -> List[int]:
        """
        Extract XYWH bounding box from binary mask.
        
        Args:
            mask: Binary mask (H, W) or (1, H, W) numpy array
        
        Returns:
            XYWH box [x, y, width, height] in pixels
            Returns [0, 0, 0, 0] for empty masks
        
        Example:
            >>> mask = np.zeros((100, 100), dtype=bool)
            >>> mask[20:40, 30:60] = True
            >>> bbox = Segmenter.bbox_from_mask(mask)
            >>> bbox
            [30, 20, 30, 20]  # x=30, y=20, w=30, h=20
        
        Notes:
            - Squeezes extra dimensions automatically
            - Inclusive: width = x_max - x_min + 1
            - Returns zeros for completely empty masks
        """
        m = np.asarray(mask)
        if m.ndim > 2:
            m = m.squeeze()
        ys, xs = np.where(m)
        if len(xs) == 0 or len(ys) == 0:
            return [0, 0, 0, 0]
        x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        return [x1, y1, x2 - x1 + 1, y2 - y1 + 1]

    def close_mask_holes(self, mask: np.ndarray) -> np.ndarray:
        """
        Fill interior holes in binary mask using morphological operations.
        
        Algorithm:
            1. Morphological closing (dilation → erosion)
            2. Flood fill from border to identify background
            3. Detect holes (interior background regions)
            4. Fill holes smaller than min_hole_area
        
        Args:
            mask: Binary mask (H, W) numpy array
        
        Returns:
            Hole-filled mask (bool array, same shape)
        
        Backend Priority:
            1. OpenCV (cv2): Fastest, most robust
            2. SciPy (ndimage): Good fallback
            3. NumPy: Minimal naive closing
        
        Example:
            >>> # Mask with interior hole
            >>> mask = create_donut_mask(100, 100)
            >>> filled = segmenter.close_mask_holes(mask)
            >>> assert filled.sum() > mask.sum()  # Hole filled
        
        Configuration:
            - config.close_holes: Must be True to apply
            - config.hole_kernel: Closing kernel size (default 7)
            - config.min_hole_area: Minimum hole size to fill (default 100px)
        
        Notes:
            - Returns input unchanged if config.close_holes=False
            - Small holes (<min_hole_area) are filled
            - Large holes (>min_hole_area) are preserved
            - Robust to degenerate cases (empty masks, single pixels)
        """
        if not self.config.close_holes:
            return mask.astype(bool)

        m_bool = mask.astype(bool)

        # OpenCV path (faster/more robust)
        try:
            import cv2  # type: ignore

            k = max(1, int(self.config.hole_kernel))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

            m_u8 = (m_bool.astype(np.uint8) * 255)
            m_u8 = cv2.morphologyEx(m_u8, cv2.MORPH_CLOSE, kernel)

            # Find holes: floodfill on inverted background
            inv = 255 - m_u8
            h, w = inv.shape[:2]
            flood = inv.copy()
            ff_mask = np.zeros((h + 2, w + 2), np.uint8)
            cv2.floodFill(flood, ff_mask, (0, 0), 0)
            holes = cv2.bitwise_and(inv, flood)

            # Remove only small holes via connectedComponentsWithStats
            num, labels, stats, _ = cv2.connectedComponentsWithStats((holes > 0).astype(np.uint8), connectivity=4)
            out = m_u8.copy()
            min_area = int(self.config.min_hole_area)
            for lab_id in range(1, num):
                area = int(stats[lab_id, cv2.CC_STAT_AREA])
                if area < min_area:
                    out[labels == lab_id] = 255

            return (out > 0).astype(bool)

        except Exception:
            pass  # fallback SciPy/NumPy

        # SciPy fallback: closing + fill_holes + filter by area
        try:
            from scipy.ndimage import binary_closing, binary_fill_holes, label  # type: ignore

            k = max(1, int(self.config.hole_kernel))
            structure = np.ones((k, k), dtype=bool)
            closed = binary_closing(m_bool, structure=structure)

            # Find holes (background components inside the object)
            filled = binary_fill_holes(closed)
            holes = np.logical_and(filled, np.logical_not(closed))

            if self.config.min_hole_area > 0:
                lab, num = label(holes)
                out = closed.copy()
                min_area = int(self.config.min_hole_area)
                for lab_id in range(1, num + 1):
                    area = int((lab == lab_id).sum())
                    if area < min_area:
                        out[lab == lab_id] = True
                return out.astype(bool)
            else:
                return filled.astype(bool)

        except Exception:
            # Last fallback: lightweight NumPy closing (naive dilation/erosion)
            return self._binary_closing_numpy(m_bool, radius=max(1, int(self.config.hole_kernel // 2)))

    @staticmethod
    def _binary_closing_numpy(mask: np.ndarray, radius: int = 3) -> np.ndarray:
        """
        Lightweight closing without dependencies: repeats 3x3 dilation and erosion ~radius times.
        Useful as fallback if OpenCV/SciPy are not available.
        """
        if radius <= 0:
            return mask.astype(bool)
        m = mask.astype(bool)
        for _ in range(radius):
            m = Segmenter._dilate_3x3_bool(m)
        for _ in range(radius):
            m = Segmenter._erode_3x3_bool(m)
        return m

    @staticmethod
    def _dilate_3x3_bool(m: np.ndarray) -> np.ndarray:
        s = [
            np.pad(m[1:, :], ((0, 1), (0, 0)), constant_values=False),
            np.pad(m[:-1, :], ((1, 0), (0, 0)), constant_values=False),
            np.pad(m[:, 1:], ((0, 0), (0, 1)), constant_values=False),
            np.pad(m[:, :-1], ((0, 0), (1, 0)), constant_values=False),
            np.pad(m[1:, 1:], ((0, 1), (0, 1)), constant_values=False),
            np.pad(m[1:, :-1], ((0, 1), (1, 0)), constant_values=False),
            np.pad(m[:-1, 1:], ((1, 0), (0, 1)), constant_values=False),
            np.pad(m[:-1, :-1], ((1, 0), (1, 0)), constant_values=False),
        ]
        return m | s[0] | s[1] | s[2] | s[3] | s[4] | s[5] | s[6] | s[7]

    @staticmethod
    def _erode_3x3_bool(m: np.ndarray) -> np.ndarray:
        s = [
            np.pad(m[1:, :], ((0, 1), (0, 0)), constant_values=True),
            np.pad(m[:-1, :], ((1, 0), (0, 0)), constant_values=True),
            np.pad(m[:, 1:], ((0, 0), (0, 1)), constant_values=True),
            np.pad(m[:, :-1], ((0, 0), (1, 0)), constant_values=True),
            np.pad(m[1:, 1:], ((0, 1), (0, 1)), constant_values=True),
            np.pad(m[1:, :-1], ((0, 1), (1, 0)), constant_values=True),
            np.pad(m[:-1, 1:], ((1, 0), (0, 1)), constant_values=True),
            np.pad(m[:-1, :-1], ((1, 0), (1, 0)), constant_values=True),
        ]
        return m & s[0] & s[1] & s[2] & s[3] & s[4] & s[5] & s[6] & s[7]

    def remove_small_components(self, mask: np.ndarray, min_area: int) -> np.ndarray:
        """
        Rimuove componenti con area < min_area. Usa OpenCV se presente, altrimenti SciPy; fallback NumPy.
        """
        if min_area <= 0:
            return mask.astype(bool)

        m = mask.astype(bool)

        # OpenCV
        try:
            import cv2  # type: ignore
            num, labels, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=4)
            keep = np.zeros_like(m, dtype=bool)
            for lab_id in range(1, num):
                area = int(stats[lab_id, cv2.CC_STAT_AREA])
                if area >= min_area:
                    keep |= (labels == lab_id)
            return keep
        except Exception:
            pass

        # SciPy
        try:
            from scipy.ndimage import label  # type: ignore
            lab, num = label(m)
            keep = np.zeros_like(m, dtype=bool)
            for lab_id in range(1, num + 1):
                area = int((lab == lab_id).sum())
                if area >= min_area:
                    keep |= (lab == lab_id)
            return keep
        except Exception:
            # Fallback: nessuna rimozione possibile
            return m

    def postprocess_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Applica chiusura fori e rimozione componenti piccole secondo config.
        """
        m = mask.astype(bool)
        m = self.close_mask_holes(m)
        if self.config.remove_small_components and self.config.min_component_area > 0:
            m = self.remove_small_components(m, self.config.min_component_area)
        return m