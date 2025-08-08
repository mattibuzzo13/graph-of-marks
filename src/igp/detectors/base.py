# igp/detectors/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Sequence, Optional
from PIL import Image

from igp.types import Detection


class Detector(ABC):
    """
    Base astratta per tutti i detector usati da ImageGraphPreprocessor.

    Requisiti per le sottoclassi:
    - implementare `detect(image)` restituendo List[Detection] con coordinate
      bbox in pixel (x1, y1, x2, y2) e score in [0,1];
    - gestire la conversione del label in forma coerente (consigliato: lowercase);
    - opzionale: override di `detect_batch`, `warmup`, `close`.

    Questa classe fornisce:
    - normalizzazione dell'immagine a RGB,
    - filtro per soglia di score (se configurato),
    - context manager (with ...),
    - metodo `run()` come scorciatoia: RGB + detect + filtro soglia.
    """

    #: Nome leggibile del detector (es. "yolov8", "owlvit", "detectron2")
    name: str

    def __init__(
        self,
        name: str,
        *,
        device: Optional[str] = None,
        score_threshold: Optional[float] = None,
    ) -> None:
        self.name = name
        
        # ✅ GESTISCI device None con fallback intelligente
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
        """Hook opzionale, es. allocazione/modello in memoria."""
        return None

    def close(self) -> None:
        """Hook opzionale per liberare risorse (GPU, file handle, ecc.)."""
        return None

    # -------------------- capacità ----------------------------

    @property
    def supports_batch(self) -> bool:
        """Indica se il detector supporta un'inferenza batched efficiente."""
        return False

    # -------------------- API obbligatoria --------------------

    @abstractmethod
    def detect(self, image: Image.Image) -> List[Detection]:
        """
        Esegue la detection su una singola immagine PIL.

        Note:
        - Accetta qualsiasi modalità input; consigliato chiamare `_ensure_rgb` prima di usare il tensore.
        - Restituisce una lista di Detection in coordinate assolute (pixel).
        - Non è necessario applicare qui la soglia: verrà gestita da `_apply_score_threshold`.
        """
        raise NotImplementedError

    # -------------------- API opzionale/batch -----------------

    def detect_batch(self, images: Sequence[Image.Image]) -> List[List[Detection]]:
        """
        Implementazione di default: effettua `detect` immagine per immagine.
        Le sottoclassi che supportano batching dovrebbero fare override.
        """
        return [self.detect(img) for img in images]

    # -------------------- helper generici ---------------------

    def _ensure_rgb(self, image: Image.Image) -> Image.Image:
        """Converte l'immagine in modalità 'RGB' se necessario."""
        if image.mode != "RGB":
            return image.convert("RGB")
        return image

    def _apply_score_threshold(self, detections: List[Detection]) -> List[Detection]:
        """
        Applica il filtro di soglia su score se `self.score_threshold` è impostata.
        Se un Detection non ha l'attributo `score`, viene mantenuto in ogni caso.
        """
        th = self.score_threshold
        if th is None:
            return detections
        return [d for d in detections if getattr(d, "score", None) is None or d.score >= th]

    def run(self, image: Image.Image) -> List[Detection]:
        """
        Scorciatoia: normalizza immagine a RGB, chiama `detect` e filtra per soglia.
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
