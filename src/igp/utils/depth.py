# igp/utils/depth.py
# Lightweight MiDaS wrapper for relative depth.
# - Loads MiDaS via torch.hub and its transforms (lazy, CPU/GPU aware).
# - Provides full normalized depth map [0,1] (higher = closer) and sampling utils.
# - Graceful degradation when torch is missing.

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False
    torch = None  # type: ignore


@dataclass
class DepthConfig:
    model_name: str = "DPT_Large"   # MiDaS variant (e.g., "DPT_Large", "DPT_Hybrid")
    device: Optional[str] = None
    fp16_on_cuda: bool = True


class DepthEstimator:
    """
    MiDaS wrapper via torch.hub.
    Provides normalized relative depth in [0, 1] (larger value = closer).
    """
    def __init__(self, config: DepthConfig | None = None) -> None:
        self.config = config or DepthConfig()
        self._ok = bool(_HAS_TORCH)
        self.model = None
        self.transform = None
        self.device = "cpu"
        self._amp_enabled = False
        self._amp_dtype = None

        if not self._ok:
            return

        self.device = self.config.device or ("cuda" if torch.cuda.is_available() else "cpu")  # type: ignore[attr-defined]
        # Load MiDaS model and its transforms (weights are cached by torch.hub)
        self.model = torch.hub.load("intel-isl/MiDaS", self.config.model_name).to(self.device).eval()  # type: ignore[attr-defined]
        transforms = torch.hub.load("intel-isl/MiDaS", "transforms")  # type: ignore[attr-defined]
        if self.config.model_name.lower().startswith("dpt"):
            self.transform = transforms.dpt_transform
        else:
            self.transform = transforms.small_transform

        self._amp_enabled = (self.device == "cuda" and bool(self.config.fp16_on_cuda))
        self._amp_dtype = torch.float16 if self._amp_enabled else torch.float32  # type: ignore[attr-defined]

    def available(self) -> bool:
        return self._ok and (self.model is not None) and (self.transform is not None)

    @torch.inference_mode()  # type: ignore[misc]
    def infer_map(self, image: Image.Image) -> Optional[np.ndarray]:
        """
        Return normalized depth map in [0, 1] where higher = closer.
        Returns None if MiDaS is unavailable.
        """
        if not self.available():
            return None

        # Prefer BGR np array (as in MiDaS reference); fallback to RGB if cv2 missing
        try:
            import cv2  # type: ignore
            img_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        except Exception:
            img_np = np.array(image)

        device_type = "cuda" if self._amp_enabled else "cpu"
        with torch.autocast(device_type=device_type, dtype=self._amp_dtype, enabled=self._amp_enabled), torch.inference_mode():  # type: ignore[attr-defined]
            inp = self.transform(img_np).to(self.device)  # type: ignore[operator]
            pred = self.model(inp).squeeze().detach().cpu().numpy()  # type: ignore[operator]

        # Normalize and invert: MiDaS larger = farther → invert so 1.0 = closer
        pred = np.asarray(pred, dtype=np.float32)
        if not np.isfinite(pred).any():
            return None
        # robust min-max
        finite = pred[np.isfinite(pred)]
        pmin, pmax = np.percentile(finite, [2.0, 98.0])
        rng = max(1e-6, float(pmax - pmin))
        norm = np.clip((pred - pmin) / rng, 0.0, 1.0)
        return 1.0 - norm

    def relative_depth_at(self, image: Image.Image, centers: Sequence[Tuple[float, float]]) -> List[float]:
        """
        Sample normalized values in [0, 1] (higher = closer) at given centers.
        """
        if not centers:
            return []
        dm = self.infer_map(image)
        if dm is None:
            return [0.5] * len(centers)
        H, W = dm.shape[:2]
        vals: List[float] = []
        for (cx, cy) in centers:
            x = int(np.clip(round(cx), 0, W - 1))
            y = int(np.clip(round(cy), 0, H - 1))
            vals.append(float(dm[y, x]))
        return vals

    def median_in_mask(self, image: Image.Image, mask: np.ndarray) -> Optional[float]:
        """
        Median normalized depth inside a boolean mask. Returns None if unavailable.
        """
        dm = self.infer_map(image)
        if dm is None:
            return None
        m = mask.astype(bool)
        if dm.shape != m.shape:
            # naive resize via nearest if shapes differ
            try:
                import cv2  # type: ignore
                m = cv2.resize(m.astype(np.uint8), (dm.shape[1], dm.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
            except Exception:
                # fallback: crop/pad center
                H, W = dm.shape[:2]
                mh, mw = m.shape[:2]
                y0 = max(0, (mh - H) // 2)
                x0 = max(0, (mw - W) // 2)
                m = m[y0:y0 + H, x0:x0 + W]
                m = np.pad(m, ((0, max(0, H - m.shape[0])), (0, max(0, W - m.shape[1]))), constant_values=False)
                m = m[:H, :W]
        vals = dm[m]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return None
        return float(np.median(vals))