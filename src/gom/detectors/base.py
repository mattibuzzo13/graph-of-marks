# igp/detectors/base.py
"""
Object Detector Abstract Base Class

Defines the interface for all object detection models in the IGP pipeline.
Provides common functionality for score thresholding, batch processing,
RGB normalization, and resource management.

Key Features:
    - Abstract interface for pluggable detectors
    - Automatic RGB conversion for any PIL image mode
    - Score threshold filtering
    - Optional batch processing support
    - Context manager protocol for resource cleanup
    - Thread-pool parallelization fallback

Supported Detectors:
    - YOLOv8: Real-time detection (see yolov8.py)
    - OWL-ViT: Open-vocabulary detection (see owlvit.py)
    - Detectron2: Mask R-CNN variants (see detectron2.py)
    - GroundingDINO: Text-grounded detection (see grounding_dino.py)

Interface Contract:
    Required:
        - detect(image) -> List[Detection]: Single-image inference
    
    Optional:
        - detect_batch(images) -> List[List[Detection]]: Batch inference
        - warmup(example_image, use_half): Memory allocation warm-up
        - close(): Resource cleanup
        - supports_batch property: Batch capability flag

Architecture:
    Detector (ABC)
    ├── detect(image) [abstract]
    ├── detect_batch(images) [threadpool default]
    ├── run(image) [public API: RGB + detect + threshold]
    ├── warmup() [optional hook]
    ├── close() [optional hook]
    └── _ensure_rgb(image), _apply_score_threshold(dets) [internal]

Usage:
    >>> from gom.detectors import YOLODetector
    >>> 
    >>> # Context manager usage (auto cleanup)
    >>> with YOLODetector("yolov8x", device="cuda") as detector:
    ...     detections = detector.run(image, score_threshold=0.5)
    ...     for det in detections:
    ...         print(f"{det.label}: {det.score:.2f}")
    
    >>> # Manual lifecycle
    >>> detector = YOLODetector("yolov8x")
    >>> detector.warmup()
    >>> detections = detector.detect(image)
    >>> detector.close()

Subclassing:
    >>> class CustomDetector(Detector):
    ...     def __init__(self, model_path, **kwargs):
    ...         super().__init__("custom", **kwargs)
    ...         self.model = load_model(model_path, device=self.device)
    ...     
    ...     def detect(self, image: Image.Image) -> List[Detection]:
    ...         rgb = self._ensure_rgb(image)
    ...         results = self.model(rgb)
    ...         return [Detection(box=box, label=lbl, score=s) 
    ...                 for box, lbl, s in results]
    ...     
    ...     def close(self):
    ...         del self.model
    ...         torch.cuda.empty_cache()

Notes:
    - All coordinates must be absolute pixels in XYXY format
    - Labels should be lowercase for consistency
    - Score thresholding happens in run(), not detect()
    - Default device selection: CUDA if available, else CPU

See Also:
    - gom.types.Detection: Detection data structure
    - gom.detectors.manager: Multi-detector orchestration
"""
from __future__ import annotations

import concurrent.futures
import logging
import os
from abc import ABC, abstractmethod
from typing import List, Optional, Sequence

from PIL import Image

from gom.types import Detection


class Detector(ABC):
    """
    Abstract base class for all object detectors.

    Subclass requirements:
    - implement `detect(image)`, returning List[Detection] with bbox coordinates
      in pixels (x1, y1, x2, y2) and a score in [0, 1];
    - normalize labels consistently (recommended: lowercase);
    - optional: override `detect_batch`, `warmup`, and `close`.

    This class provides:
    - image normalization to RGB,
    - configurable score-threshold filtering,
    - context manager support (`with ...`),
    - `run()` convenience method: RGB → detect → threshold filter.
    
    Attributes:
        name: Human-readable detector identifier (e.g., "yolov8", "owlvit")
        device: Compute device ("cuda", "cpu", "mps")
        score_threshold: Minimum detection confidence (optional, applied in run())
    """

    #: Human-readable detector name (e.g., "yolov8", "owlvit", "detectron2")
    name: str

    def __init__(
        self,
        name: str,
        *,
        device: Optional[str] = None,
        score_threshold: Optional[float] = None,
    ) -> None:
        """
        Initialize detector with device and score threshold.
        
        Args:
            name: Detector identifier string
            device: Target compute device (None = auto-detect CUDA/CPU)
            score_threshold: Minimum score for detections (None = no filtering)
        
        Notes:
            - device=None uses CUDA if available, otherwise CPU
            - score_threshold is applied in run(), not detect()
        """
        self.name = name
        
        # Handle device=None with a sensible fallback (CUDA if available, else CPU).
        if device is None:
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"
        else:
            self.device = device
            
        self.score_threshold = score_threshold

    # -------------------- lifecycle hooks --------------------

    def warmup(self) -> None:
        """
        Optional initialization hook for memory allocation and model loading.

        Subclasses may override to:
        - Allocate GPU memory
        - Load model weights
        - Run dummy inference to stabilize memory allocation
        
        Signature Extensions:
            Some detectors accept:
            - example_image: Optional[PIL.Image] for size-based allocation
            - use_half: Optional[bool] for FP16 optimization
        
        Default:
            No-op (does nothing)
        
        Example:
            >>> detector = YOLODetector("yolov8x")
            >>> detector.warmup()  # Loads model to GPU
            >>> # Now ready for inference
        """
        return None

    def close(self) -> None:
        """
        Optional cleanup hook to release resources.
        
        Use for:
            - Releasing GPU memory
            - Closing file handles
            - Cleaning up temporary files
        
        Default:
            No-op (does nothing)
        
        Example:
            >>> detector.close()
            >>> torch.cuda.empty_cache()  # Common cleanup pattern
        """
        return None

    # -------------------- capabilities -----------------------

    @property
    def supports_batch(self) -> bool:
        """
        Whether detector implements efficient batch processing.
        
        Returns:
            True if detect_batch() is optimized, False if it uses fallback
        
        Notes:
            - Default implementation returns False
            - Subclasses with native batching should override to return True
            - Affects pipeline batch strategy selection
        """
        return False

    # -------------------- required API -----------------------

    @abstractmethod
    def detect(self, image: Image.Image) -> List[Detection]:
        """
        Run detection on a single PIL image.

        Args:
            image: Input PIL Image (any mode, will be converted to RGB)
        
        Returns:
            List of Detection objects with:
                - box: [x1, y1, x2, y2] in absolute pixel coordinates
                - label: Object class label (lowercase recommended)
                - score: Confidence in [0.0, 1.0]
                - source: Detector name (optional)
        
        Notes:
            - Accepts any PIL image mode; use _ensure_rgb() before inference
            - Returns unfiltered detections; thresholding happens in run()
            - Coordinates must be absolute pixels (not normalized)
        
        Example:
            >>> def detect(self, image: Image.Image) -> List[Detection]:
            ...     rgb = self._ensure_rgb(image)
            ...     results = self.model(rgb)
            ...     return [Detection(box=[x1,y1,x2,y2], label=lbl, score=s)
            ...             for x1,y1,x2,y2, lbl, s in results]
        """
        raise NotImplementedError

    # -------------------- optional/batch API -----------------

    def detect_batch(self, images: Sequence[Image.Image]) -> List[List[Detection]]:
        """
        Batch inference on multiple images.
        
        Default Implementation:
            Parallelizes detect() calls across thread pool. Effective for
            native/C-bound detectors where GIL is released during inference.
        
        Args:
            images: Sequence of PIL images to process
        
        Returns:
            List of detection lists, one per input image (order preserved)
        
        Performance:
            - Default: ThreadPoolExecutor with min(len(images), CPU_COUNT) workers
            - Native batching: Override for GPU batch parallelism
        
        Subclass Override:
            >>> def detect_batch(self, images):
            ...     # True batch processing
            ...     tensor_batch = torch.stack([preprocess(img) for img in images])
            ...     results = self.model(tensor_batch)
            ...     return [parse_result(r) for r in results]
        
        Notes:
            - Empty input returns empty list
            - Exceptions fallback to empty detection list for that image
            - Order is preserved despite as_completed() usage
        
        Example:
            >>> images = [Image.open(f"img{i}.jpg") for i in range(10)]
            >>> batch_results = detector.detect_batch(images)
            >>> print(len(batch_results[0]))  # Detections from first image
        """
        if not images:
            return []

        max_workers = min(len(images), (os.cpu_count() or 4))
        results: List[List[Detection]] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(self.detect, img) for img in images]
            for f in concurrent.futures.as_completed(futures):
                try:
                    results.append(f.result())
                except Exception:
                    # preserve ordering: fall back to synchronous detect for this index
                    logging.exception("detect_batch: worker failed; falling back to sync detect")
                    results.append([])

        # as_completed loses original order; rebuild results in original order
        ordered_results = [None] * len(images)
        for i, fut in enumerate(futures):
            try:
                ordered_results[i] = fut.result()
            except Exception:
                ordered_results[i] = []
        return ordered_results

    # -------------------- generic helpers --------------------

    def _ensure_rgb(self, image: Image.Image) -> Image.Image:
        """
        Convert image to RGB mode if needed.
        
        Args:
            image: Input PIL image (any mode)
        
        Returns:
            RGB PIL image (same image if already RGB)
        
        Notes:
            - Handles RGBA, L, P, CMYK, etc.
            - No-op if already RGB (returns same object)
            - Required before most tensor conversions
        """
        if image.mode != "RGB":
            return image.convert("RGB")
        return image

    def _apply_score_threshold(self, detections: List[Detection]) -> List[Detection]:
        """
        Filter detections by score threshold.
        
        Args:
            detections: List of Detection objects
        
        Returns:
            Filtered list keeping only detections with score >= threshold
        
        Behavior:
            - If self.score_threshold is None: returns all detections
            - Detections without score attribute: always kept
            - Otherwise: filters by score >= self.score_threshold
        
        Example:
            >>> detector.score_threshold = 0.5
            >>> dets = [Detection(score=0.3), Detection(score=0.7)]
            >>> filtered = detector._apply_score_threshold(dets)
            >>> len(filtered)  # 1 (only 0.7 passes)
        """
        th = self.score_threshold
        if th is None:
            return detections
        return [d for d in detections if getattr(d, "score", None) is None or d.score >= th]

    def run(self, image: Image.Image) -> List[Detection]:
        """
        High-level detection API with RGB normalization and threshold filtering.
        
        Pipeline:
            1. Convert image to RGB (_ensure_rgb)
            2. Run detection (detect)
            3. Apply score threshold (_apply_score_threshold)
        
        Args:
            image: Input PIL image (any mode)
        
        Returns:
            Filtered list of Detection objects
        
        Notes:
            - Preferred public API over detect()
            - Handles all image mode conversions
            - Applies score_threshold automatically
        
        Example:
            >>> detector = YOLODetector("yolov8x", score_threshold=0.5)
            >>> detections = detector.run(image)  # Auto-filtered
            >>> all(d.score >= 0.5 for d in detections)  # True
        """
        img = self._ensure_rgb(image)
        dets = self.detect(img)
        return self._apply_score_threshold(dets)

    # -------------------- context manager --------------------

    def __enter__(self) -> "Detector":
        """
        Context manager entry: calls warmup().
        
        Returns:
            self for use in with statement
        
        Example:
            >>> with YOLODetector("yolov8x") as detector:
            ...     detections = detector.run(image)
        """
        self.warmup()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """
        Context manager exit: calls close() for cleanup.
        
        Args:
            exc_type, exc, tb: Exception info (standard context manager signature)
        
        Notes:
            - Always calls close(), even on exception
            - Does not suppress exceptions (returns None)
        """
        self.close()

    # -------------------- utility -----------------------------

    def __repr__(self) -> str:  # pragma: no cover
        """String representation for debugging."""
        return (
            f"{self.__class__.__name__}("
            f"name={self.name!r}, device={self.device!r}, "
            f"score_threshold={self.score_threshold!r})"
        )
