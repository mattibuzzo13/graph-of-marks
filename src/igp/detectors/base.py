# igp/detectors/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Sequence, Optional
from PIL import Image

from igp.types import Detection


class Detector(ABC):
    """
    Abstract base class for all detectors used by ImageGraphPreprocessor.

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
        """Optional hook, e.g., to allocate/load models into memory."""
        return None

    def close(self) -> None:
        """Optional hook to release resources (GPU memory, file handles, etc.)."""
        return None

    # -------------------- capabilities -----------------------

    @property
    def supports_batch(self) -> bool:
        """Whether the detector supports efficient batched inference."""
        return False

    # -------------------- required API -----------------------

    @abstractmethod
    def detect(self, image: Image.Image) -> List[Detection]:
        """
        Run detection on a single PIL image.

        Notes:
        - Accepts any input mode; call `_ensure_rgb` before tensor use.
        - Returns a list of Detection in absolute pixel coordinates.
        - Thresholding is not required here; `_apply_score_threshold` will handle it.
        """
        raise NotImplementedError

    # -------------------- optional/batch API -----------------

    def detect_batch(self, images: Sequence[Image.Image]) -> List[List[Detection]]:
        """
        Default implementation: calls `detect` for each image.
        Subclasses with true batching should override this.
        """
        return [self.detect(img) for img in images]

    # -------------------- generic helpers --------------------

    def _ensure_rgb(self, image: Image.Image) -> Image.Image:
        """Convert image to 'RGB' mode if needed."""
        if image.mode != "RGB":
            return image.convert("RGB")
        return image

    def _apply_score_threshold(self, detections: List[Detection]) -> List[Detection]:
        """
        Apply score-threshold filtering if `self.score_threshold` is set.
        Detections without a `score` attribute are always kept.
        """
        th = self.score_threshold
        if th is None:
            return detections
        return [d for d in detections if getattr(d, "score", None) is None or d.score >= th]

    def run(self, image: Image.Image) -> List[Detection]:
        """
        Convenience: normalize to RGB, run `detect`, then apply threshold filtering.
        """
        img = self._ensure_rgb(image)
        dets = self.detect(img)
        return self._apply_score_threshold(dets)

    # -------------------- context manager --------------------

    def __enter__(self) -> "Detector":
        self.warmup()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # -------------------- utility -----------------------------

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"{self.__class__.__name__}("
            f"name={self.name!r}, device={self.device!r}, "
            f"score_threshold={self.score_threshold!r})"
        )
