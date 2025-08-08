# igp/utils/depth.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple, Optional

try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

import numpy as np
from PIL import Image


@dataclass
class DepthConfig:
    model_name: str = "DPT_Large"   # MiDaS variant
    device: Optional[str] = None


class DepthEstimator:
    """
    Wrapper per MiDaS via torch.hub.
    Fornisce profondità relativa normalizzata in [0,1] (valore alto = più vicino).
    """
    def __init__(self, config: DepthConfig | None = None) -> None:
        self.config = config or DepthConfig()
        self._ok = _HAS_TORCH
        if not self._ok:
            self.model = None
            self.transform = None
            return

        device = self.config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        # Caricamento MiDaS + transforms
        self.model = torch.hub.load("intel-isl/MiDaS", self.config.model_name).to(device).eval()
        transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        # Usa la trasformazione corretta per il modello
        if self.config.model_name.lower().startswith("dpt"):
            self.transform = transforms.dpt_transform
        else:
            self.transform = transforms.small_transform
        self.device = device

    def available(self) -> bool:
        return self._ok and (self.model is not None)

    @torch.inference_mode()
    def relative_depth_at(self, image: Image.Image, centers: Sequence[Tuple[float, float]]) -> List[float]:
        """
        Ritorna valori normalizzati in [0,1] (alto = vicino) per i centroidi
        specificati in coordinate dell'immagine originale.
        """
        if not self.available() or not centers:
            return [0.5] * len(centers)

        # PIL -> numpy BGR
        import cv2  # lazy import; MiDaS dipende da OpenCV solo per conversione qui
        img_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # trasformazione + forward
        im_t = self.transform(img_np).to(self.device)
        depth = self.model(im_t).squeeze().detach().cpu().numpy()  # (H_d, W_d)

        Hd, Wd = depth.shape[:2]
        W0, H0 = image.size  # PIL: (W, H)

        vals: List[float] = []
        for (cx, cy) in centers:
            x_d = int(np.clip(cx / max(1.0, W0) * Wd, 0, Wd - 1))
            y_d = int(np.clip(cy / max(1.0, H0) * Hd, 0, Hd - 1))
            vals.append(float(depth[y_d, x_d]))

        arr = np.asarray(vals, dtype=np.float32)
        if arr.size == 0:
            return []
        rng = float(np.ptp(arr))
        if rng < 1e-6:
            return [0.5] * len(vals)

        # Nota: MiDaS restituisce valori “maggiore = più lontano”
        # Invertiamo poi normalizziamo, così 1.0 = più vicino
        arr = (arr - arr.min()) / rng
        arr = 1.0 - arr
        return arr.tolist()
