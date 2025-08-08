# igp/detectors/yolov8.py
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
    raise ImportError(
        "Ultralytics YOLOv8 non è installato. Installa con: pip install ultralytics"
    ) from e


class YOLOv8Detector(Detector):
    """
    Wrapper minimale per Ultralytics YOLOv8.

    - Ritorna List[Detection] con box in formato (x1, y1, x2, y2) float, label str, score float.
    - Opzionale TTA con flip orizzontale e rimappatura bbox.
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
        # ✅ CHIAMATA ALLA BASE
        super().__init__(
            name="yolov8",
            device=device,
            score_threshold=score_threshold
        )
        
        self.model = YOLO(model_path)
        # ✅ RIMUOVI: self.device e self.score_threshold sono già dalla base
        self.imgsz = int(imgsz)
        self.tta_hflip = bool(tta_hflip)

        # Porta il modello sul device
        try:
            self.model.to(self.device)
        except Exception:
            try:
                self.model.model.to(self.device)
            except Exception:
                pass

    def detect(self, image: Image.Image) -> List[Detection]:
        """✅ Rimuovi filtro difensivo finale - gestito da run()"""
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

        # ✅ RIMUOVI il filtro difensivo - gestito da run()
        return dets

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------
    @torch.inference_mode()
    def _detect_once(self, image: Image.Image) -> List[Detection]:
        image_np = np.array(image)
        results_list = self.model.predict(
            image_np,
            conf=self.score_threshold,  
            imgsz=self.imgsz,
            device=self.device,  
            verbose=False,
        )

        if not results_list:
            return []

        results = results_list[0]
        # names: dict[int,str] (dipende dalla versione, gestiamo fallback)
        names = getattr(results, "names", None)
        if names is None:
            names = getattr(self.model, "names", None)
        if names is None:
            # ulteriore fallback
            names = getattr(getattr(self.model, "model", object()), "names", {})  # type: ignore[attr-defined]

        detections: List[Detection] = []
        try:
            xyxy = results.boxes.xyxy
            conf = results.boxes.conf
            cls_ = results.boxes.cls
        except Exception:
            # struttura inattesa
            return []

        for box_t, conf_t, cls_t in zip(xyxy, conf, cls_):
            try:
                score = float(conf_t.item())
                x1, y1, x2, y2 = [float(v) for v in box_t.tolist()[:4]]
                cls_idx = int(cls_t.item())
                label = str(names.get(cls_idx, cls_idx)) if isinstance(names, dict) else str(cls_idx)
                detections.append(self._make_detection((x1, y1, x2, y2), label, score))
            except Exception:
                continue

        return detections

    @staticmethod
    def _as_xyxy(box_like: Sequence[float]) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = box_like[:4]
        return float(x1), float(y1), float(x2), float(y2)

    def _make_detection(self, box_xyxy: Sequence[float], label: str, score: float) -> Detection:
        b = self._as_xyxy(box_xyxy)
        # Crea Detection in modo robusto rispetto alla firma reale della dataclass
        try:
            return Detection(box=b, label=label, score=score, source="yolov8")
        except TypeError:
            try:
                return Detection(box=b, label=label, score=score)
            except TypeError:
                return Detection(box=b, label=label)

    def _rebox(self, det: Detection, new_box_xyxy: Sequence[float]) -> Detection:
        b = self._as_xyxy(new_box_xyxy)
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
