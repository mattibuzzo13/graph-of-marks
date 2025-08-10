# igp/detectors/yolov8.py
# Minimal wrapper around Ultralytics YOLOv8.
# Provides single-image detection, optional horizontal-flip TTA,
# and robust `Detection` creation without changing core logic.

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import torch
from PIL import Image

from igp.detectors.base import Detector
from igp.types import Detection

try:
    from ultralytics import YOLO
except Exception as e:  # pragma: no cover
    # Keep import-time error message concise; installation is required by the user.
    raise ImportError(
        "Ultralytics YOLOv8 non è installato. Installa con: pip install ultralytics"
    ) from e


class YOLOv8Detector(Detector):
    """
    Minimal wrapper for Ultralytics YOLOv8.

    - Returns List[Detection] with boxes (x1, y1, x2, y2) as floats, label as str, and score as float.
    - Optional horizontal-flip TTA with bbox remapping back to the original image frame.
    """

    def __init__(
        self,
        *,
        model_path: str = "yolov8x.pt",
        device: Optional[str] = None,
        score_threshold: float = 0.5,
        imgsz: int = 640,
        tta_hflip: bool = False,
    ) -> None:
        # Base initialization (sets name, device selection, and global score threshold).
        super().__init__(
            name="yolov8",
            device=device,
            score_threshold=score_threshold
        )
        
        self.model = YOLO(model_path)
        # `self.device` and `self.score_threshold` already come from the base class.
        self.imgsz = int(imgsz)
        self.tta_hflip = bool(tta_hflip)

        # Move the model to the selected device (handle different Ultralytics internals).
        try:
            self.model.to(self.device)
        except Exception:
            try:
                self.model.model.to(self.device)
            except Exception:
                pass

    def detect(self, image: Image.Image) -> List[Detection]:
        """Single-image detection; global score filtering is handled by `run()`."""
        dets = self._detect_once(image)

        if self.tta_hflip:
            flipped = image.transpose(Image.FLIP_LEFT_RIGHT)
            dets_flip = self._detect_once(flipped)
            W = image.size[0]
            remapped: List[Detection] = []
            for d in dets_flip:
                x1, y1, x2, y2 = self._as_xyxy(d.box)
                new_box = (W - x2, y1, W - x1, y2)
                remapped.append(self._rebox(d, new_box))
            dets.extend(remapped)

        # Return raw detections; `run()` will apply any global threshold.
        return dets

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------
    @torch.inference_mode()
    def _detect_once(self, image: Image.Image) -> List[Detection]:
        image_np = np.array(image)
        results_list = self.model.predict(
            image_np,
            conf=self.score_threshold,  # model-level threshold (kept; caller may apply another)
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )

        if not results_list:
            return []

        results = results_list[0]
        # Retrieve class-name mapping (varies across Ultralytics versions).
        names = getattr(results, "names", None)
        if names is None:
            names = getattr(self.model, "names", None)
        if names is None:
            # Last-resort fallback.
            names = getattr(getattr(self.model, "model", object()), "names", {})  # type: ignore[attr-defined]

        detections: List[Detection] = []
        try:
            xyxy = results.boxes.xyxy
            conf = results.boxes.conf
            cls_ = results.boxes.cls
        except Exception:
            # Unexpected structure; return empty for safety.
            return []

        for box_t, conf_t, cls_t in zip(xyxy, conf, cls_):
            try:
                score = float(conf_t.item())
                x1, y1, x2, y2 = [float(v) for v in box_t.tolist()[:4]]
                cls_idx = int(cls_t.item())
                label = str(names.get(cls_idx, cls_idx)) if isinstance(names, dict) else str(cls_idx)
                detections.append(self._make_detection((x1, y1, x2, y2), label, score))
            except Exception:
                # Skip malformed entries but continue processing others.
                continue

        return detections

    @staticmethod
    def _as_xyxy(box_like: Sequence[float]) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = box_like[:4]
        return float(x1), float(y1), float(x2), float(y2)

    def _make_detection(self, box_xyxy: Sequence[float], label: str, score: float) -> Detection:
        b = self._as_xyxy(box_xyxy)
        # Construct Detection robustly across possible dataclass signatures.
        try:
            return Detection(box=b, label=label, score=score, source="yolov8")
        except TypeError:
            try:
                return Detection(box=b, label=label, score=score)
            except TypeError:
                return Detection(box=b, label=label)

    def _rebox(self, det: Detection, new_box_xyxy: Sequence[float]) -> Detection:
        b = self._as_xyxy(new_box_xyxy)
        # Return a new Detection with the updated box; handle immutable dataclasses.
        try:
            return Detection(
                box=b,
                label=getattr(det, "label", ""),
                score=getattr(det, "score", 1.0),
                source=getattr(det, "source", "yolov8"),
            )
        except TypeError:
            try:
                return Detection(
                    box=b,
                    label=getattr(det, "label", ""),
                    score=getattr(det, "score", 1.0),
                )
            except TypeError:
                return Detection(box=b, label=getattr(det, "label", ""))


__all__ = ["YOLOv8Detector"]
