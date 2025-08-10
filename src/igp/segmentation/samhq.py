# igp/segmentation/samhq.py
# High-level: SAM-HQ segmenter wrapper.
# - Loads a SAM-HQ model (SysCV) and exposes a uniform .segment() API.
# - Primary prompt is the bounding box; if mask is too small, falls back to a
#   single positive point at the box center.
# - Optionally closes small mask holes if configured.
# - Cleans up GPU memory after prediction.

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image

from .base import Segmenter, SegmenterConfig


# Official SAM-HQ checkpoint URLs (used when auto_download=True and no local ckpt is provided).
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
        # Device selection (explicit in config wins; otherwise CUDA if available).
        super().__init__(config)
        self.device = self.config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type

        # Import the SAM-HQ variant; provide an actionable error if missing.
        try:
            from segment_anything_hq import sam_model_registry, SamPredictor  # type: ignore
        except Exception as e:
            raise ImportError(
                "segment_anything_hq non installato. Installazione: "
                "pip install git+https://github.com/SysCV/sam-hq.git"
            ) from e

        # Resolve or download the checkpoint; construct predictor on selected device.
        ckpt_path = self._resolve_checkpoint(checkpoint, model_type, auto_download)
        self._sam_predictor_cls = SamPredictor
        self._sam_model = sam_model_registry[model_type](checkpoint=str(ckpt_path)).to(self.device).eval()
        self._predictor = SamPredictor(self._sam_model)

    @torch.inference_mode()
    def segment(self, image_pil: Image.Image, boxes: Sequence[Sequence[float]]) -> List[Dict[str, Any]]:
        # Convert once to NumPy; set the image for the predictor session.
        image_np = np.array(image_pil)
        H, W = image_np.shape[:2]
        out: List[Dict[str, Any]] = []

        self._predictor.set_image(image_np)
        try:
            for box in boxes:
                # Clamp to valid image bounds to avoid degenerate prompts.
                x1, y1, x2, y2 = self.clamp_box_xyxy(box, W, H)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                mask = None
                score = 0.0

                # 1) Try box-prompt inference first (single best mask).
                try:
                    box_arr = np.array([[x1, y1, x2, y2]], dtype=float)
                    masks_box, scores_box, _ = self._predictor.predict(
                        box=box_arr, multimask_output=False
                    )
                    mask = masks_box[0]
                    score = float(scores_box[0]) if scores_box is not None else 1.0
                except Exception:
                    mask = None

                # 2) If box fails or returns an unreasonably small mask, try a center point.
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
                        # Final defensive fallback: empty mask and zero score.
                        mask = np.zeros((H, W), dtype=bool)
                        score = 0.0

                # Optional hole closing (morphology + small-hole fill) via base utilities.
                mask = mask.astype(bool)
                if self.config.close_holes:
                    mask = self.close_mask_holes(mask)

                # Package result: segmentation mask, bbox derived from mask, and predicted IOU score.
                out.append(
                    {
                        "segmentation": mask,
                        "bbox": self.bbox_from_mask(mask),
                        "predicted_iou": float(score),
                    }
                )
            return out
        finally:
            # Reset predictor state and free GPU cache between images.
            self._predictor.reset_image()
            if hasattr(self._predictor, "features"):
                delattr(self._predictor, "features")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ----------------- internals -----------------

    def _resolve_checkpoint(self, checkpoint: Optional[str], model_type: str, auto_download: bool) -> Path:
        # Use provided path if present; otherwise resolve a default filename under ./checkpoints.
        if checkpoint:
            p = Path(checkpoint)
            if not p.exists():
                raise FileNotFoundError(f"Checkpoint SAM-HQ non trovato: {checkpoint}")
            return p

        fname = f"sam_hq_{model_type}.pth"
        p = Path("./checkpoints") / fname
        p.parent.mkdir(parents=True, exist_ok=True)

        # If missing and auto-download allowed, fetch from the official URL map.
        if not p.exists():
            if not auto_download:
                raise FileNotFoundError(f"Checkpoint SAM-HQ assente: {p}")
            from torch.hub import download_url_to_file
            url = _SAM_HQ_URLS.get(model_type, _SAM_HQ_URLS["vit_h"])
            download_url_to_file(url, str(p), progress=True)
        return p
