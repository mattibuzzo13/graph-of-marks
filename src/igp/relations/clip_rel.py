# igp/relations/clip_rel.py
# CLIP-based relation scoring between two detected objects.
# Uses an image crop covering both boxes and compares against a small set of
# relation text prompts; returns the best-matching relation and score.

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import torch
from PIL import Image


# Default natural-language relation templates queried with CLIP.
_DEFAULT_TEMPLATES = [
    "on top of",
    "under",
    "inside",
    "holding",
    "riding",
    "touching",
    "next to",
    "in front of",
    "behind",
]

# Canonicalization map (spaces → underscores; aligned with the pipeline labels).
_CANON = {
    "on top of": "on_top_of",
    "under": "below",
    "inside": "inside",
    "holding": "holding",
    "riding": "riding",
    "touching": "touching",
    "next to": "next_to",
    "in front of": "in_front_of",
    "behind": "behind",
}


def canonicalize_relation(text: str) -> str:
    """Convert a free-text relation into the canonical label used by the pipeline."""
    key = " ".join(text.strip().lower().split())
    return _CANON.get(key, key.replace(" ", "_"))


def union_crop(
    image_pil: Image.Image,
    box_a: Sequence[float],
    box_b: Sequence[float],
) -> Image.Image:
    """
    Crop the minimal axis-aligned rectangle that contains both input boxes.
    Input and output coordinates are in pixel (xyxy).
    """
    W, H = image_pil.size
    x1 = int(max(0, min(box_a[0], box_b[0])))
    y1 = int(max(0, min(box_a[1], box_b[1])))
    x2 = int(min(W - 1, max(box_a[2], box_b[2])))
    y2 = int(min(H - 1, max(box_a[3], box_b[3])))
    if x2 <= x1 or y2 <= y1:
        return image_pil.crop((0, 0, 1, 1)).convert("RGB")
    return image_pil.crop((x1, y1, x2, y2)).convert("RGB")


class ClipRelScorer:
    """
    Score the most plausible relation between (subject, object) using CLIP on a
    crop that includes both objects.

    Dependencies (passed in at init):
      - clip_processor: transformers.CLIPProcessor
      - clip_model: transformers.CLIPModel
      - device: "cpu" or "cuda"
    """

    def __init__(
        self,
        clip_processor,
        clip_model,
        device: Optional[str] = None,
        default_templates: Optional[Iterable[str]] = None,
    ) -> None:
        self.processor = clip_processor
        self.model = clip_model.eval()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.templates = list(default_templates) if default_templates else list(_DEFAULT_TEMPLATES)

    @torch.inference_mode()
    def best_relation(
        self,
        image_pil: Image.Image,
        box_i: Sequence[float],
        box_j: Sequence[float],
        label_i: str,
        label_j: str,
        *,
        templates: Optional[Iterable[str]] = None,
    ) -> Tuple[str, str, float]:
        """
        Return (canonical_relation, raw_relation_text, score) for the pair (i, j).

        The score is the cosine similarity between the crop image embedding and
        the best-matching relation text embedding among the provided templates.
        """
        tmpl = list(templates) if templates else self.templates

        crop = union_crop(image_pil, box_i, box_j)

        # 1) Image features
        im_inputs = self.processor(images=crop, return_tensors="pt").to(self.device)
        im_feat = self.model.get_image_features(**im_inputs)
        im_feat = im_feat / im_feat.norm(dim=-1, keepdim=True)

        # 2) Text features — plain templates (no placeholders)
        texts = [t for t in tmpl]
        txt_inputs = self.processor(text=texts, return_tensors="pt", padding=True).to(self.device)
        txt_feat = self.model.get_text_features(**txt_inputs)
        txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)

        # 3) Similarity and argmax over templates
        sims = torch.matmul(im_feat, txt_feat.T).squeeze(0)  # [len(templates)]
        best = int(sims.argmax())
        best_sim = float(sims[best])
        rel_raw = texts[best]
        rel_canon = canonicalize_relation(rel_raw)
        return rel_canon, rel_raw, best_sim
