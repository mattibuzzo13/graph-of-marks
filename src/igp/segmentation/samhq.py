# igp/segmentation/samhq.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image

from .base import Segmenter, SegmenterConfig


_SAM_HQ_URLS = {
    "vit_b": "https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_b.pth",
    "vit_l": "https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_l.pth",
    "vit_h": "https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_h.pth",
}


class SamHQSegmenter(Segmenter):
    """
    SAM-HQ (SysCV). Richiede `segment_anything_hq`.
    Usa box-prompt con eventuale fallback a point-prompt centrale.
    """

    def __init__(
        self,
        model_type: str = "vit_h",
        checkpoint: Optional[str] = None,
        *,
        config: Optional[SegmenterConfig] = None,
        auto_download: bool = True,
    ) -> None:
        super().__init__(config)
        self.device = self.config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type

        try:
            from segment_anything_hq import sam_model_registry, SamPredictor  # type: ignore
        except Exception as e:
            raise ImportError(
                "segment_anything_hq non installato. Installazione: "
                "pip install git+https://github.com/SysCV/sam-hq.git"
            ) from e

        ckpt_path = self._resolve_checkpoint(checkpoint, model_type, auto_download)
        self._sam_predictor_cls = SamPredictor
        self._sam_model = sam_model_registry[model_type](checkpoint=str(ckpt_path)).to(self.device).eval()
        self._predictor = SamPredictor(self._sam_model)

    @torch.inference_mode()
    def segment(self, image_pil: Image.Image, boxes: Sequence[Sequence[float]]) -> List[Dict[str, Any]]:
        image_np = np.array(image_pil)
        H, W = image_np.shape[:2]
        out: List[Dict[str, Any]] = []

        self._predictor.set_image(image_np)
        try:
            for box in boxes:
                x1, y1, x2, y2 = self.clamp_box_xyxy(box, W, H)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                mask = None
                score = 0.0

                # 1) box prompt
                try:
                    box_arr = np.array([[x1, y1, x2, y2]], dtype=float)
                    masks_box, scores_box, _ = self._predictor.predict(
                        box=box_arr, multimask_output=False
                    )
                    mask = masks_box[0]
                    score = float(scores_box[0]) if scores_box is not None else 1.0
                except Exception:
                    mask = None

                # 2) fallback a punto centrale se necessario
                if mask is None or mask.sum() < 50:
                    try:
                        masks_pt, scores_pt, _ = self._predictor.predict(
                            point_coords=np.array([[cx, cy]]),
                            point_labels=np.array([1]),
                            multimask_output=False,
                        )
                        mask = masks_pt[0]
                        score = float(scores_pt[0]) if scores_pt is not None else score
                    except Exception:
                        mask = np.zeros((H, W), dtype=bool)
                        score = 0.0

                mask = mask.astype(bool)
                if self.config.close_holes:
                    mask = self.close_mask_holes(mask)

                out.append(
                    {
                        "segmentation": mask,
                        "bbox": self.bbox_from_mask(mask),
                        "predicted_iou": float(score),
                    }
                )
            return out
        finally:
            self._predictor.reset_image()
            if hasattr(self._predictor, "features"):
                delattr(self._predictor, "features")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ----------------- internals -----------------

    def _resolve_checkpoint(self, checkpoint: Optional[str], model_type: str, auto_download: bool) -> Path:
        if checkpoint:
            p = Path(checkpoint)
            if not p.exists():
                raise FileNotFoundError(f"Checkpoint SAM-HQ non trovato: {checkpoint}")
            return p

        fname = f"sam_hq_{model_type}.pth"
        p = Path("./checkpoints") / fname
        p.parent.mkdir(parents=True, exist_ok=True)

        if not p.exists():
            if not auto_download:
                raise FileNotFoundError(f"Checkpoint SAM-HQ assente: {p}")
            from torch.hub import download_url_to_file
            url = _SAM_HQ_URLS.get(model_type, _SAM_HQ_URLS["vit_h"])
            download_url_to_file(url, str(p), progress=True)
        return p
