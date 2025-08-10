# igp/utils/depth.py
# Lightweight MiDaS wrapper for relative depth.
# - Loads a MiDaS model via torch.hub and the corresponding transforms.
# - Returns normalized relative depth in [0, 1] where higher = closer.
# - If PyTorch is unavailable, methods degrade gracefully.

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
    MiDaS wrapper via torch.hub.
    Provides normalized relative depth in [0, 1] (larger value = closer).
    """
    def __init__(self, config: DepthConfig | None = None) -> None:
        self.config = config or DepthConfig()
        self._ok = _HAS_TORCH
        if not self._ok:
            self.model = None
            self.transform = None
            return

        device = self.config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        # Load MiDaS model and its transforms
        self.model = torch.hub.load("intel-isl/MiDaS", self.config.model_name).to(device).eval()
        transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        # Pick the correct transform for the chosen model
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
        Return normalized values in [0, 1] (higher = closer) for the given
        centroid coordinates, expressed in the original image space.
        """
        if not self.available() or not centers:
            return [0.5] * len(centers)

        # PIL → NumPy BGR (MiDaS reference)
        import cv2  # lazy import; OpenCV is used only for this conversion
        img_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Transform + forward pass
        im_t = self.transform(img_np).to(self.device)
        depth = self.model(im_t).squeeze().detach().cpu().numpy()  # (H_d, W_d)

        Hd, Wd = depth.shape[:2]
        W0, H0 = image.size  # PIL gives (W, H)

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

        # Note: MiDaS predicts larger = farther; invert and normalize so 1.0 = closer
        arr = (arr - arr.min()) / rng
        arr = 1.0 - arr
        return arr.tolist()
