# igp/utils/clip_utils.py
# Thin CLIP wrapper with robust APIs:
# - Lazy load HF CLIP (processor+model), device/precision handling.
# - Exposes encode_image/encode_text and get_* aliases, cosine similarity.
# - Direct similarity(image, prompts) utility for convenience.
# - Graceful degradation when dependencies are missing.

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple, Union

try:
    import torch
    from transformers import CLIPProcessor, CLIPModel  # type: ignore
    _HAS_CLIP = True
except Exception:
    _HAS_CLIP = False
    torch = None  # type: ignore

from PIL import Image


@dataclass
class CLIPConfig:
    model_id: str = "openai/clip-vit-large-patch14"
    device: Optional[str] = None
    fp16_on_cuda: bool = True


class CLIPWrapper:
    """
    Utility wrapper for CLIP image/text embeddings and similarities.
    Provides multiple method names for compatibility with callers.
    """
    def __init__(self, config: CLIPConfig | None = None) -> None:
        self.config = config or CLIPConfig()
        self._ok = bool(_HAS_CLIP)
        self.processor = None
        self.model = None
        self.device = "cpu"
        self._amp_enabled = False
        self._amp_dtype = None

        if not self._ok:
            return

        self.device = self.config.device or ("cuda" if torch.cuda.is_available() else "cpu")  # type: ignore[attr-defined]
        self.processor = CLIPProcessor.from_pretrained(self.config.model_id)  # type: ignore[operator]
        self.model = CLIPModel.from_pretrained(self.config.model_id).to(self.device).eval()  # type: ignore[operator]
        self._amp_enabled = (self.device == "cuda" and bool(self.config.fp16_on_cuda))
        self._amp_dtype = torch.float16 if self._amp_enabled else torch.float32  # type: ignore[attr-defined]

    def available(self) -> bool:
        return self._ok and (self.processor is not None) and (self.model is not None)

    # ----- embeddings -----

    def encode_image(self, images: Union[Image.Image, Sequence[Image.Image]]):
        """Return L2-normalized image features as torch.Tensor [N, D]."""
        if not self.available():
            return None
        imgs = [images] if isinstance(images, Image.Image) else list(images)
        with torch.inference_mode(), torch.autocast(  # type: ignore[attr-defined]
            device_type=("cuda" if self._amp_enabled else "cpu"),
            dtype=self._amp_dtype,
            enabled=self._amp_enabled,
        ):
            inputs = self.processor(images=imgs, return_tensors="pt").to(self.device)
            feats = self.model.get_image_features(**inputs)  # type: ignore[operator]
            return torch.nn.functional.normalize(feats, dim=-1)  # type: ignore[attr-defined]

    def encode_text(self, texts: Union[str, Sequence[str]]):
        """Return L2-normalized text features as torch.Tensor [N, D]."""
        if not self.available():
            return None
        tx = [texts] if isinstance(texts, str) else list(texts)
        with torch.inference_mode(), torch.autocast(  # type: ignore[attr-defined]
            device_type=("cuda" if self._amp_enabled else "cpu"),
            dtype=self._amp_dtype,
            enabled=self._amp_enabled,
        ):
            inputs = self.processor(text=tx, return_tensors="pt", padding=True, truncation=True).to(self.device)
            feats = self.model.get_text_features(**inputs)  # type: ignore[operator]
            return torch.nn.functional.normalize(feats, dim=-1)  # type: ignore[attr-defined]

    # Aliases for compatibility
    get_image_features = encode_image
    get_text_features = encode_text

    # ----- similarities -----

    @staticmethod
    def cosine_sim(a, b):
        """Cosine similarity given already normalized embeddings."""
        if a is None or b is None:
            return None
        return a @ b.T

    def similarities(
        self,
        images: Sequence[Image.Image],
        texts: Sequence[str],
    ) -> Optional["torch.Tensor"]:
        """Return cosine similarity matrix [len(images), len(texts)]."""
        if not self.available() or not images or not texts:
            return None
        imf = self.encode_image(images)
        tf = self.encode_text(texts)
        return self.cosine_sim(imf, tf)

    def similarity(self, image: Image.Image, texts: Sequence[str]):
        """
        Convenience single-image similarity against multiple prompts.
        Returns a 1D tensor [len(texts)] on CPU when available, else [].
        """
        sims = self.similarities([image], list(texts))
        if sims is None:
            return []
        return sims.squeeze(0).detach().cpu()

    # ----- higher-level helpers -----

    def best_labels_by_text(self, query: str, labels: Sequence[str], threshold: float = 0.25) -> List[Tuple[str, float]]:
        """
        Rank labels by similarity to a text query and return those >= threshold.
        """
        if not self.available() or not labels:
            return []
        qf = self.encode_text([query])
        lf = self.encode_text(list(labels))
        sims = self.cosine_sim(qf, lf).squeeze(0)  # type: ignore[union-attr]
        out = [(labels[i], float(sims[i])) for i in range(len(labels)) if float(sims[i]) >= float(threshold)]
        out.sort(key=lambda x: x[1], reverse=True)
        return out

    def best_relation(self, crop: Image.Image, subj: str, obj: str, templates: Sequence[str]) -> Tuple[str, float]:
        """
        Choose the relation template with maximum CLIP similarity for a crop.
        Templates may contain '{subj}'/'{obj}' placeholders.
        """
        if not self.available() or not templates:
            return ("", 0.0)
        prompts = [t.format(subj=subj, obj=obj) for t in templates]
        sims = self.similarity(crop, prompts)
        if sims is None or len(prompts) == 0:
            return ("", 0.0)
        best = int(sims.argmax().item())
        return prompts[best], float(sims[best].item())