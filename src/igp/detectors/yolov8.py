# igp/detectors/yolov8.py
# Minimal wrapper around Ultralytics YOLOv8, ottimizzato per efficienza.
# - Batch inference reale
# - Fuse Conv+BN
# - FP16 su GPU (opzionale/auto)
# - max_det configurabile
# - TTA orizzontale opzionale
# - Creazione Detection robusta

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
    raise ImportError("Ultralytics YOLOv8 non è installato. Installa con: pip install ultralytics") from e


class YOLOv8Detector(Detector):
    """
    Wrapper minimale per YOLOv8 (Ultralytics) con ottimizzazioni runtime.

    Restituisce: List[Detection] con box (x1,y1,x2,y2) float, label str, score float.
    """

    def __init__(
        self,
        *,
        model_path: str = "yolov8x.pt",
        device: Optional[str] = None,
        score_threshold: float = 0.5,
        imgsz: int = 640,
        tta_hflip: bool = False,
        max_det: int = 300,
        batch_size: int = 16,
        use_half: Optional[bool] = None,  # None = auto (True se cuda disponibile)
    ) -> None:
        super().__init__(name="yolov8", device=device, score_threshold=score_threshold)

        self.model = YOLO(model_path)
        self.imgsz = int(imgsz)
        self.tta_hflip = bool(tta_hflip)
        self.max_det = int(max_det)
        self.batch_size = int(batch_size)

        # Sposta su device selezionato
        try:
            self.model.to(self.device)
        except Exception:
            try:
                self.model.model.to(self.device)  # vecchie versioni
            except Exception:
                pass

        # Fuse Conv+BN se disponibile
        try:
            self.model.fuse()
        except Exception:
            pass

        # FP16 su GPU (auto o forzato via argomento)
        if use_half is None:
            self.use_half = (str(self.device).startswith("cuda") and torch.cuda.is_available())
        else:
            self.use_half = bool(use_half) and (str(self.device).startswith("cuda") and torch.cuda.is_available())
        if self.use_half:
            try:
                self.model.model.half()
            except Exception:
                self.use_half = False  # fallback se non supportato

        # Eval mode
        try:
            self.model.model.eval()
        except Exception:
            pass

        # Cache dei nomi classi (best-effort)
        self.names = self._resolve_names()

    # ------------------ API ------------------

    def detect(self, image: Image.Image) -> List[Detection]:
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

        return dets

    @property
    def supports_batch(self) -> bool:
        return True

    def detect_batch(self, images: Sequence[Image.Image]) -> List[List[Detection]]:
        if not images:
            return []
        if self.tta_hflip:
            return self._detect_batch_with_tta(images)
        return self._detect_batch_once(images)

    def close(self) -> None:
        try:
            del self.model
        except Exception:
            pass
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    # --------------- Implementazione ----------------

    @torch.inference_mode()
    def _detect_batch_once(self, images: Sequence[Image.Image]) -> List[List[Detection]]:
        images_np = [np.array(self._ensure_rgb(img)) for img in images]

        results_list = self._predict(images_np)
        if not results_list:
            return [[] for _ in images]

        return [self._parse_single_result(res) for res in results_list]

    def _detect_batch_with_tta(self, images: Sequence[Image.Image]) -> List[List[Detection]]:
        dets_original = self._detect_batch_once(images)
        flipped_images = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in images]
        dets_flipped = self._detect_batch_once(flipped_images)

        all_detections: List[List[Detection]] = []
        for img, dets_o, dets_f in zip(images, dets_original, dets_flipped):
            W = img.size[0]
            remapped: List[Detection] = []
            for d in dets_f:
                x1, y1, x2, y2 = self._as_xyxy(d.box)
                new_box = (W - x2, y1, W - x1, y2)
                remapped.append(self._rebox(d, new_box))
            all_detections.append(dets_o + remapped)
        return all_detections

    @torch.inference_mode()
    def _detect_once(self, image: Image.Image) -> List[Detection]:
        image_np = np.array(self._ensure_rgb(image))
        results_list = self._predict(image_np)
        if not results_list:
            return []
        return self._parse_single_result(results_list[0])

    # --------------- Helpers ----------------

    def _predict(self, inputs):
        """
        Wrapper robusto a differenze di versione di Ultralytics.
        """
        kwargs = dict(
            conf=self.score_threshold,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
            max_det=self.max_det,
        )
        # Alcune versioni accettano 'half' e 'batch'
        try:
            return self.model.predict(inputs, half=self.use_half, batch=self.batch_size, **kwargs)
        except TypeError:
            # Fallback senza argomenti extra
            return self.model.predict(inputs, **kwargs)

    def _parse_single_result(self, results) -> List[Detection]:
        detections: List[Detection] = []
        try:
            boxes = results.boxes
            xyxy = boxes.xyxy
            conf = boxes.conf
            cls_ = boxes.cls
        except Exception:
            return detections

        names = self._get_names(results)

        for box_t, conf_t, cls_t in zip(xyxy, conf, cls_):
            try:
                score = float(conf_t.item())
                x1, y1, x2, y2 = [float(v) for v in box_t.tolist()[:4]]
                cls_idx = int(cls_t.item())
                if isinstance(names, dict):
                    label = str(names.get(cls_idx, cls_idx))
                elif isinstance(names, (list, tuple)) and 0 <= cls_idx < len(names):
                    label = str(names[cls_idx])
                else:
                    label = str(cls_idx)
                detections.append(self._make_detection((x1, y1, x2, y2), label, score))
            except Exception:
                continue
        return detections

    def _get_names(self, results):
        # Priorità: results.names -> model.names -> model.model.names
        names = getattr(results, "names", None)
        if names is None:
            names = getattr(self.model, "names", None)
        if names is None:
            names = getattr(getattr(self.model, "model", object()), "names", None)
        return names if names is not None else self.names

    def _resolve_names(self):
        names = getattr(self.model, "names", None)
        if names is None:
            names = getattr(getattr(self.model, "model", object()), "names", None)
        return names if names is not None else {}

    @staticmethod
    def _ensure_rgb(img: Image.Image) -> Image.Image:
        if isinstance(img, Image.Image) and img.mode != "RGB":
            return img.convert("RGB")
        return img

    @staticmethod
    def _as_xyxy(box_like: Sequence[float]) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = box_like[:4]
        return float(x1), float(y1), float(x2), float(y2)

    def _make_detection(self, box_xyxy: Sequence[float], label: str, score: float) -> Detection:
        b = self._as_xyxy(box_xyxy)
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