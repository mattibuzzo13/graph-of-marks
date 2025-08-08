# igp/utils/clip_utils.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

try:
    import torch
    from transformers import CLIPProcessor, CLIPModel
    _HAS_CLIP = True
except Exception:
    _HAS_CLIP = False

from PIL import Image


@dataclass
class CLIPConfig:
    model_id: str = "openai/clip-vit-large-patch14"
    device: Optional[str] = None


class CLIPWrapper:
    """
    Utilità per embedding immagine/testo e similarità con CLIP.
    """
    def __init__(self, config: CLIPConfig | None = None) -> None:
        self.config = config or CLIPConfig()
        self._ok = _HAS_CLIP
        if not self._ok:
            self.processor = None
            self.model = None
            return

        device = self.config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = CLIPProcessor.from_pretrained(self.config.model_id)
        self.model = CLIPModel.from_pretrained(self.config.model_id).to(device).eval()
        self.device = device

    def available(self) -> bool:
        return self._ok

    @torch.inference_mode() 
    def image_features(self, images: Sequence[Image.Image]) -> "torch.Tensor | None":
        if not self._ok:
            return None
        inputs = self.processor(images=list(images), return_tensors="pt").to(self.device)
        feats = self.model.get_image_features(**inputs)
        return torch.nn.functional.normalize(feats, dim=-1)

    @torch.inference_mode() 
    def text_features(self, texts: Sequence[str]) -> "torch.Tensor | None":
        if not self._ok:
            return None
        inputs = self.processor(text=list(texts), return_tensors="pt", padding=True, truncation=True).to(self.device)
        feats = self.model.get_text_features(**inputs)
        return torch.nn.functional.normalize(feats, dim=-1)

    @staticmethod
    def cosine_sim(a: "torch.Tensor", b: "torch.Tensor") -> "torch.Tensor":
        return a @ b.T

    @torch.inference_mode() 
    def best_labels_by_text(self, query: str, labels: Sequence[str], threshold: float = 0.25) -> List[Tuple[str, float]]:
        """
        Trova le label più affini alla query testuale.
        """
        if not self._ok or not labels:
            return []
        qf = self.text_features([query])
        lf = self.text_features(list(labels))
        sims = self.cosine_sim(qf, lf).squeeze(0)  # [len(labels)]
        out = [(labels[i], float(sims[i])) for i in range(len(labels)) if float(sims[i]) >= float(threshold)]
        out.sort(key=lambda x: x[1], reverse=True)
        return out

    @torch.inference_mode()
    def best_relation(self, crop: Image.Image, subj: str, obj: str, templates: Sequence[str]) -> Tuple[str, float]:
        """
        Sceglie la relazione {tmpl.format(subj=..., obj=...)} con sim. massima.
        """
        if not self._ok or not templates:
            return ("", 0.0)
        imf = self.image_features([crop])
        texts = [t.format(subj=subj, obj=obj) for t in templates]
        tf = self.text_features(texts)
        sims = self.cosine_sim(imf, tf).squeeze(0)  # [len(templates)]
        best = int(sims.argmax())
        return texts[best], float(sims[best])
