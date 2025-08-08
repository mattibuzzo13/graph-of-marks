# igp/utils/caption.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

try:
    import torch
    from transformers import BlipProcessor, BlipForConditionalGeneration
    _HAS_TRANSFORMERS = True
except Exception:
    _HAS_TRANSFORMERS = False

from PIL import Image


@dataclass
class CaptionerConfig:
    model_id: str = "Salesforce/blip-image-captioning-base"
    device: Optional[str] = None
    max_new_tokens: int = 20


class Captioner:
    """
    Wrapper leggero per BLIP (image captioning).
    Se transformers/torch non sono disponibili, ritorna stringhe vuote.
    """
    def __init__(self, config: CaptionerConfig | None = None) -> None:
        self.config = config or CaptionerConfig()
        self._ok = _HAS_TRANSFORMERS
        if not self._ok:
            self.processor = None
            self.model = None
            return

        device = self.config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = BlipProcessor.from_pretrained(self.config.model_id)
        self.model = BlipForConditionalGeneration.from_pretrained(self.config.model_id).to(device).eval()
        self.device = device

    @torch.inference_mode()
    def caption(self, image: Image.Image) -> str:
        if not self._ok:
            return ""
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        out = self.model.generate(**inputs, max_new_tokens=int(self.config.max_new_tokens))
        return self.processor.decode(out[0], skip_special_tokens=True)
