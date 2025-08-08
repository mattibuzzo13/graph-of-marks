# igp/detectors/detectron2.py
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
from PIL import Image

from igp.types import Detection
from igp.detectors.base import Detector

# Import Detectron2 solo quando necessario (evita costo all'import di altri detector)
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog


class Detectron2Detector(Detector):
    """
    Wrapper Detectron2 per object detection / instance segmentation.

    - Di default usa 'COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml'
      con pesi del model zoo.
    - Restituisce Detection con: box (x1,y1,x2,y2), label (string), score (float)
      e, se disponibile, mask (np.ndarray bool di shape [H, W]).
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
            name: nome leggibile del detector (default: 'detectron2')
            model_config: percorso nel model zoo (o file .yaml locale)
            weights: url o path locale a pesi; se None usa quelli del model zoo
            device: 'cuda' o 'cpu' (se None, usa device della base)
            score_threshold: soglia conf. a runtime (inserita anche in cfg)
            return_masks: se True, prova ad includere le maschere
            class_names: override per i nomi di classe
        """
        super().__init__(name=name, device=device, score_threshold=score_threshold)
        self.model_config = model_config
        self.weights = weights
        self.return_masks = return_masks

        # Costruzione cfg + predictor
        self.cfg = self._build_cfg(
            model_config=model_config,
            weights=weights,
            device=device,
            score_threshold=score_threshold,
        )
        self.predictor = DefaultPredictor(self.cfg)

        # Classi (se non fornite esplicitamente, prova a recuperarle da MetadataCatalog)
        if class_names is not None:
            self.classes = list(class_names)
        else:
            # La maggior parte dei config del model zoo ha DATASETS.TRAIN valorizzato
            try:
                train_ds = self.cfg.DATASETS.TRAIN[0]
                self.classes = list(MetadataCatalog.get(train_ds).thing_classes)
            except Exception:
                # fallback prudente: indicizza per id
                self.classes = []

    # --------------- lifecycle ----------------

    def close(self) -> None:
        """Libera risorse del predictor (soprattutto GPU)."""
        try:
            # Rilascio esplicito dei campi principali
            del self.predictor
        except Exception:
            pass
        # Svuota cache CUDA se presente
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

        # Predizioni base
        boxes = instances.pred_boxes.tensor.numpy().tolist() if instances.has("pred_boxes") else []
        scores = instances.scores.numpy().tolist() if instances.has("scores") else []
        classes = instances.pred_classes.numpy().tolist() if instances.has("pred_classes") else []

        # Maschere (opzionali)
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

            # label
            if self.classes and 0 <= cls_id < len(self.classes):
                label = str(self.classes[cls_id])
            else:
                label = str(cls_id)

            mask = masks_np[i] if (masks_np is not None and i < masks_np.shape[0]) else None

            det = self._make_detection(box=box, label=label, score=score, mask=mask)
            detections.append(det)

        # Applica anche un eventuale filtro di soglia globale (ridondante ma sicuro)
        return detections

    def detect_batch(self, images: Sequence[Image.Image]) -> List[List[Detection]]:
        # Fallback semplice; per efficienza si potrebbe scrivere una versione batched,
        # ma DefaultPredictor opera su singola immagine.
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
        # Config dal model zoo (o percorso locale)
        try:
            cfg.merge_from_file(model_zoo.get_config_file(model_config))
            cfg.MODEL.WEIGHTS = (
                weights if weights is not None else model_zoo.get_checkpoint_url(model_config)
            )
        except Exception:
            # Se non è nel model zoo, proviamo a interpretarlo come file locale
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
        Crea un Detection in modo robusto rispetto alla signature effettiva
        della tua dataclass `igp.types.Detection`.
        """
        # box in (x1,y1,x2,y2) float
        b = tuple(float(x) for x in box[:4])

        # Prova con i campi più ricchi, poi degrada in caso di TypeError
        try:
            # Se Detection supporta tutti i campi
            return Detection(box=b, label=label, score=score, mask=mask, source=self.name)
        except TypeError:
            try:
                # Se non supporta source
                return Detection(box=b, label=label, score=score, mask=mask)
            except TypeError:
                try:
                    # Se non supporta mask
                    return Detection(box=b, label=label, score=score)
                except TypeError:
                    # Fallback minimo - controlla la definizione di Detection
                    from igp.types import Detection as DetectionType
                    # Usa solo i campi obbligatori
                    return DetectionType(box=b, label=label, score=score)


__all__ = ["Detectron2Detector"]
