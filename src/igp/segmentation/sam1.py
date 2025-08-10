# igp/segmentation/sam1.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence, Optional

import numpy as np
import torch
from PIL import Image

from .base import Segmenter, SegmenterConfig


# Official SAM v1 checkpoint URLs (Meta)
_SAM_URLS = {
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
}


class Sam1Segmenter(Segmenter):
    """
    Segment Anything v1 (Meta). Requires the `segment_anything` package.
    Uses a bounding-box prompt with a robust fallback to a single center point.
    """

    def __init__(
        self,
        checkpoint: Optional[str] = None,
        model_type: str = "vit_h",
        *,
        points_per_side: int = 32,                  # kept for API compatibility; unused here
        pred_iou_thresh: float = 0.8,               # not used directly
        stability_score_thresh: float = 0.85,       # not used directly
        min_mask_region_area: int = 300,            # not used directly
        config: Optional[SegmenterConfig] = None,
        auto_download: bool = True,
    ) -> None:
        super().__init__(config)
        self.device = self.config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type

        try:
            from segment_anything import sam_model_registry, SamPredictor  # type: ignore
        except Exception as e:
            raise ImportError(
                "segment_anything is not installed. Install from: "
                "https://github.com/facebookresearch/segment-anything"
            ) from e

        ckpt_path = self._resolve_checkpoint(checkpoint, model_type, auto_download)
        self._sam_predictor_cls = SamPredictor
        self._sam_model = sam_model_registry[model_type](checkpoint=str(ckpt_path)).to(self.device).eval()
        self._predictor = SamPredictor(self._sam_model)

    # ----------------- public API -----------------

    @torch.inference_mode()
    def segment(self, image_pil: Image.Image, boxes: Sequence[Sequence[float]]) -> List[Dict[str, Any]]:
        image_np = np.array(image_pil)
        H, W = image_np.shape[:2]

        out: List[Dict[str, Any]] = []
        self._predictor.set_image(image_np)

        try:
            for box in boxes:
                x1, y1, x2, y2 = self.clamp_box_xyxy(box, W, H)

                # Try progressively shrunken boxes for robustness; stop at first valid mask
                mask_ok = None
                score_ok = 0.0
                for shrink in (0, 2, 4, 8, 12, 16):
                    xs = max(0, x1 + shrink)
                    ys = max(0, y1 + shrink)
                    xe = max(xs + 1, x2 - shrink)
                    ye = max(ys + 1, y2 - shrink)
                    if xe <= xs or ye <= ys:
                        continue

                    box_arr = np.asarray([[xs, ys, xe, ye]], dtype=float)
                    masks_box, scores_box, _ = self._predictor.predict(
                        box=box_arr, multimask_output=True
                    )
                    best = int(scores_box.argmax())
                    mask = masks_box[best]
                    score = float(scores_box[best])

                    if mask.sum() < 50:
                        # Fallback: single positive point at the (clamped) box center
                        cx, cy = (xs + xe) // 2, (ys + ye) // 2
                        masks_pt, scores_pt, _ = self._predictor.predict(
                            point_coords=np.array([[cx, cy]]),
                            point_labels=np.array([1]),
                            multimask_output=False,
                        )
                        mask = masks_pt[0]
                        score = float(scores_pt[0])

                    mask = mask.astype(bool)
                    if self.config.close_holes:
                        mask = self.close_mask_holes(mask)

                    mask_ok, score_ok = mask, score
                    break  # valid mask obtained; proceed to next box

                if mask_ok is None:
                    # Final fallback: empty mask
                    mask_ok = np.zeros((H, W), dtype=bool)
                    score_ok = 0.0

                bbox_xywh = self.bbox_from_mask(mask_ok)
                out.append(
                    {
                        "segmentation": mask_ok,
                        "bbox": bbox_xywh,
                        "predicted_iou": float(score_ok),
                    }
                )
            return out
        finally:
            # GPU memory cleanup between calls
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
                raise FileNotFoundError(f"SAM-1 checkpoint not found: {checkpoint}")
            return p

        # Default filename per model type
        fname = {
            "vit_h": "sam_vit_h_4b8939.pth",
            "vit_l": "sam_vit_l_0b3195.pth",
            "vit_b": "sam_vit_b_01ec64.pth",
        }.get(model_type, "sam_vit_h_4b8939.pth")
        p = Path("./checkpoints") / fname
        p.parent.mkdir(parents=True, exist_ok=True)

        if not p.exists():
            if not auto_download:
                raise FileNotFoundError(f"Missing SAM-1 checkpoint: {p}")
            # Download from official URL if absent
            url = _SAM_URLS.get(model_type, _SAM_URLS["vit_h"])
            from torch.hub import download_url_to_file
            download_url_to_file(url, str(p))
        return p
