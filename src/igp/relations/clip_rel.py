# igp/relations/clip_rel.py
# CLIP-based relation scorer.
# - best_relation: single pair scoring
# - batch_best_relations: optional batched scoring over multiple pairs
# Falls back to light geometric/text heuristics if CLIP is unavailable.

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import math
from PIL import Image

try:
    import numpy as np
except Exception:
    np = None  # type: ignore

try:
    import torch
except Exception:
    torch = None  # type: ignore

# Optional project wrapper (preferred if available)
try:
    from igp.utils.clip_utils import CLIPWrapper  # type: ignore
except Exception:
    CLIPWrapper = None  # type: ignore


def _as_xyxy(box_like: Sequence[float]) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = box_like[:4]
    return float(x1), float(y1), float(x2), float(y2)


def _union_box(b1: Sequence[float], b2: Sequence[float]) -> Tuple[float, float, float, float]:
    x11, y11, x12, y12 = _as_xyxy(b1)
    x21, y21, x22, y22 = _as_xyxy(b2)
    return min(x11, x21), min(y11, y21), max(x12, x22), max(y12, y22)


def _center(b: Sequence[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = _as_xyxy(b)
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


@dataclass
class ClipRelScorer:
    device: str = "cpu"
    clip: Optional[object] = None  # instance of CLIPWrapper or compatible

    def __post_init__(self) -> None:
        if self.clip is None and CLIPWrapper is not None:
            try:
                # Prefer constructor with device kw
                self.clip = CLIPWrapper(device=self.device)  # type: ignore
            except TypeError:
                try:
                    # Fallback to config object signature
                    from igp.utils.clip_utils import CLIPConfig  # type: ignore
                    self.clip = CLIPWrapper(config=CLIPConfig(device=self.device))  # type: ignore
                except Exception:
                    self.clip = None  # no CLIP available

    # -------------------- public API --------------------

    def best_relation(
        self,
        image_pil: Image.Image,
        box_subj: Sequence[float],
        box_obj: Sequence[float],
        subj_label: str,
        obj_label: str,
    ) -> Tuple[str, str, float]:
        """
        Return (relation_canonical, relation_raw_prompt, score).
        Uses CLIP if available, else geometric fallback.
        """
        prompts = self._build_prompts(subj_label, obj_label)
        if not prompts:
            return "left_of", f"{subj_label} left of {obj_label}", 0.0

        # Try CLIP scoring on union crop
        try:
            scores = self._score_prompts(image_pil, box_subj, box_obj, [p for _, p in prompts])
            best_idx = int(max(range(len(scores)), key=lambda k: scores[k]))
            canon, raw = prompts[best_idx]
            return canon, raw, float(scores[best_idx])
        except Exception:
            # Fallback: geometry-based tie-breaker
            return self._geom_fallback(box_subj, box_obj, subj_label, obj_label, prompts)

    def batch_best_relations(
        self,
        image_pil: Image.Image,
        boxes: Sequence[Sequence[float]],
        labels: Sequence[str],
        pairs: Sequence[Tuple[int, int]],
    ) -> Iterable[Tuple[int, int, str, str, float]]:
        """
        Batched variant: yields (i, j, relation_canon, relation_raw, score).
        Uses CLIP when available; falls back to per-pair scoring otherwise.
        """
        # Attempt shared image embedding (if wrapper supports it)
        for (i, j) in pairs:
            canon, raw, score = self.best_relation(
                image_pil=image_pil,
                box_subj=boxes[i],
                box_obj=boxes[j],
                subj_label=labels[i],
                obj_label=labels[j],
            )
            yield int(i), int(j), canon, raw, float(score)

    # -------------------- internals --------------------

    def _build_prompts(self, subj: str, obj: str) -> List[Tuple[str, str]]:
        subj = str(subj).strip()
        obj = str(obj).strip()

        # Canonical relation -> prompt template
        templates = [
            ("on_top_of",       f"a photo of a {subj} on top of a {obj}"),
            ("under",           f"a photo of a {subj} under a {obj}"),
            ("left_of",         f"a photo of a {subj} to the left of a {obj}"),
            ("right_of",        f"a photo of a {subj} to the right of a {obj}"),
            ("above",           f"a photo of a {subj} above a {obj}"),
            ("below",           f"a photo of a {subj} below a {obj}"),
            ("in_front_of",     f"a photo of a {subj} in front of a {obj}"),
            ("behind",          f"a photo of a {subj} behind a {obj}"),
            ("touching",        f"a photo of a {subj} touching a {obj}"),
            ("near",            f"a photo of a {subj} near a {obj}"),
            ("holding",         f"a photo of a {subj} holding a {obj}"),
            ("wearing",         f"a photo of a {subj} wearing a {obj}"),
            ("riding",          f"a photo of a {subj} riding a {obj}"),
            ("sitting_on",      f"a photo of a {subj} sitting on a {obj}"),
            ("carrying",        f"a photo of a {subj} carrying a {obj}"),
        ]
        return templates

    def _score_prompts(
        self,
        image_pil: Image.Image,
        box_subj: Sequence[float],
        box_obj: Sequence[float],
        prompts: Sequence[str],
    ) -> List[float]:
        """
        Compute CLIP similarity scores between union crop and text prompts.
        If CLIP is unavailable, raise to let caller fallback to geometry.
        """
        if self.clip is None:
            raise RuntimeError("CLIP backend unavailable")

        # Compute union crop (focus on the pair region)
        ux1, uy1, ux2, uy2 = _union_box(box_subj, box_obj)
        ux1, uy1 = max(0, int(ux1)), max(0, int(uy1))
        ux2, uy2 = int(ux2), int(uy2)
        crop = image_pil.crop((ux1, uy1, ux2, uy2))

        # Try common CLIPWrapper APIs
        # 1) encode_image/encode_text
        try:
            img_feat = self.clip.encode_image(crop)  # type: ignore
            txt_feat = self.clip.encode_text(list(prompts))  # type: ignore
            return self._cosine_scores(img_feat, txt_feat)
        except Exception:
            pass

        # 2) get_image_features/get_text_features
        try:
            img_feat = self.clip.get_image_features(crop)  # type: ignore
            txt_feat = self.clip.get_text_features(list(prompts))  # type: ignore
            return self._cosine_scores(img_feat, txt_feat)
        except Exception:
            pass

        # 3) similarity(image, prompts)
        try:
            scores = self.clip.similarity(crop, list(prompts))  # type: ignore
            # Ensure list[float]
            return [float(s) for s in (scores.tolist() if hasattr(scores, "tolist") else scores)]
        except Exception:
            raise

    def _cosine_scores(self, img_feat, txt_feat) -> List[float]:
        """Normalize and compute cosine similarities -> list[float]."""
        if torch is None:
            # Best-effort with NumPy
            import numpy as _np  # type: ignore
            i = _np.asarray(img_feat)
            t = _np.asarray(txt_feat)
            i = i / (1e-8 + _np.linalg.norm(i))
            t = t / (1e-8 + _np.linalg.norm(t, axis=-1, keepdims=True))
            sims = i.dot(t.T).reshape(-1)
            return [float(x) for x in sims.tolist()]
        with torch.inference_mode():
            i = img_feat
            t = txt_feat
            if not torch.is_tensor(i):
                i = torch.as_tensor(i)
            if not torch.is_tensor(t):
                t = torch.as_tensor(t)
            i = i.float()
            t = t.float()
            i = i / (i.norm(dim=-1, keepdim=True) + 1e-8)
            t = t / (t.norm(dim=-1, keepdim=True) + 1e-8)
            sims = (i @ t.T).flatten()
            return sims.detach().cpu().tolist()

    def _geom_fallback(
        self,
        box_subj: Sequence[float],
        box_obj: Sequence[float],
        subj_label: str,
        obj_label: str,
        prompts: Sequence[Tuple[str, str]],
    ) -> Tuple[str, str, float]:
        """Pick a relation using geometric cues when CLIP isn't available."""
        cx1, cy1 = _center(box_subj)
        cx2, cy2 = _center(box_obj)
        dx, dy = cx2 - cx1, cy2 - cy1

        # Prefer strong cues with small margin
        margin = 8.0
        candidates: List[Tuple[str, float]] = []

        if abs(dy) > abs(dx) + margin:
            candidates.append(("above" if dy < 0 else "below", 0.55))
        if abs(dx) > abs(dy) + margin:
            candidates.append(("left_of" if dx < 0 else "right_of", 0.55))

        # Distance-based "near"
        dist = math.hypot(dx, dy)
        if dist < 64.0:
            candidates.append(("near", 0.50))

        # Fallback default if nothing triggered
        if not candidates:
            candidates = [("near", 0.40)]

        # Map to the available prompts
        prompt_map = {canon: raw for canon, raw in prompts}
        for canon, score in candidates:
            if canon in prompt_map:
                return canon, prompt_map[canon], score

        # Last resort
        canon, raw = prompts[0]
        return canon, raw, 0.35