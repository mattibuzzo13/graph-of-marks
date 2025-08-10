# igp/detectors/detectron2.py
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
from PIL import Image

from igp.types import Detection
from igp.detectors.base import Detector

# Keep Detectron2 imports local to this module to avoid overhead elsewhere.
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog


class Detectron2Detector(Detector):
    """
    Detectron2 wrapper for object detection / instance segmentation.

    - Defaults to 'COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml'
      with model zoo weights.
    - Returns Detection with: box (x1, y1, x2, y2), label (str), score (float),
      and, if available, mask (bool np.ndarray of shape [H, W]).
    """

    def __init__(
        self,
        *,
        name: str = "detectron2",
        model_config: str = "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml",
        weights: Optional[str] = None,
        device: Optional[str] = None,
        score_threshold: Optional[float] = 0.5,
        return_masks: bool = True,
        class_names: Optional[Sequence[str]] = None,
    ) -> None:
        """
        Args:
            name: human-readable detector name (default: 'detectron2').
            model_config: model zoo path or local .yaml file.
            weights: URL or local path to weights; if None, use model zoo weights.
            device: 'cuda' or 'cpu' (if None, falls back to base class device).
            score_threshold: runtime confidence threshold (also set in cfg).
            return_masks: if True, include predicted masks when available.
            class_names: optional override for class names.
        """
        super().__init__(name=name, device=device, score_threshold=score_threshold)
        self.model_config = model_config
        self.weights = weights
        self.return_masks = return_masks

        # Build config and predictor.
        self.cfg = self._build_cfg(
            model_config=model_config,
            weights=weights,
            device=device,
            score_threshold=score_threshold,
        )
        self.predictor = DefaultPredictor(self.cfg)

        # Class names (use provided list or try to fetch from MetadataCatalog).
        if class_names is not None:
            self.classes = list(class_names)
        else:
            # Most model zoo configs have DATASETS.TRAIN set.
            try:
                train_ds = self.cfg.DATASETS.TRAIN[0]
                self.classes = list(MetadataCatalog.get(train_ds).thing_classes)
            except Exception:
                # Conservative fallback: index by class id.
                self.classes = []

    # --------------- lifecycle ----------------

    def close(self) -> None:
        """Release predictor resources (notably GPU memory)."""
        try:
            # Explicitly drop main fields.
            del self.predictor
        except Exception:
            pass
        # Clear CUDA cache if available.
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    # --------------- core API -----------------

    def detect(self, image: Image.Image) -> List[Detection]:
        img_np = np.array(image)  # H, W, 3 (RGB)

        outputs = self.predictor(img_np)
        instances = outputs["instances"].to("cpu")

        # Base predictions.
        boxes = instances.pred_boxes.tensor.numpy().tolist() if instances.has("pred_boxes") else []
        scores = instances.scores.numpy().tolist() if instances.has("scores") else []
        classes = instances.pred_classes.numpy().tolist() if instances.has("pred_classes") else []

        # Optional masks.
        masks_np: Optional[np.ndarray] = None
        if self.return_masks and instances.has("pred_masks"):
            # (N, H, W) bool
            masks_np = instances.pred_masks.numpy().astype(bool)

        detections: List[Detection] = []
        num = min(len(boxes), len(scores), len(classes))
        for i in range(num):
            box = boxes[i]
            score = float(scores[i])
            cls_id = int(classes[i])

            # Map class id to label if available.
            if self.classes and 0 <= cls_id < len(self.classes):
                label = str(self.classes[cls_id])
            else:
                label = str(cls_id)

            mask = masks_np[i] if (masks_np is not None and i < masks_np.shape[0]) else None

            det = self._make_detection(box=box, label=label, score=score, mask=mask)
            detections.append(det)

        # Return raw detections; any global score threshold may be applied by the caller.
        return detections

    def detect_batch(self, images: Sequence[Image.Image]) -> List[List[Detection]]:
        # Simple fallback: run single-image inference.
        # For speed, a true batched path would be preferable, but DefaultPredictor is per-image.
        return [self.detect(img) for img in images]

    # --------------- helpers ------------------

    def _build_cfg(
        self,
        *,
        model_config: str,
        weights: Optional[str],
        device: Optional[str],
        score_threshold: Optional[float],
    ):
        cfg = get_cfg()
        # Load config from model zoo (or local file path).
        try:
            cfg.merge_from_file(model_zoo.get_config_file(model_config))
            cfg.MODEL.WEIGHTS = (
                weights if weights is not None else model_zoo.get_checkpoint_url(model_config)
            )
        except Exception:
            # If not in the model zoo, interpret as a local config file.
            cfg.merge_from_file(model_config)
            if weights is not None:
                cfg.MODEL.WEIGHTS = weights

        if score_threshold is not None:
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = float(score_threshold)

        if device is not None:
            cfg.MODEL.DEVICE = device  # "cuda" | "cpu"

        return cfg

    def _make_detection(
        self,
        *,
        box: Sequence[float],
        label: str,
        score: float,
        mask: Optional[np.ndarray],
    ) -> Detection:
        """
        Create a Detection while being robust to the actual dataclass signature
        of `igp.types.Detection`.
        """
        # Box as (x1, y1, x2, y2) floats.
        b = tuple(float(x) for x in box[:4])

        # Try the richest set of fields first; gracefully degrade on TypeError.
        try:
            # If Detection supports all fields.
            return Detection(box=b, label=label, score=score, mask=mask, source=self.name)
        except TypeError:
            try:
                # If it does not support `source`.
                return Detection(box=b, label=label, score=score, mask=mask)
            except TypeError:
                try:
                    # If it does not support `mask`.
                    return Detection(box=b, label=label, score=score)
                except TypeError:
                    # Minimal fallback — check the concrete Detection definition.
                    from igp.types import Detection as DetectionType
                    # Use only required fields.
                    return DetectionType(box=b, label=label, score=score)


__all__ = ["Detectron2Detector"]
