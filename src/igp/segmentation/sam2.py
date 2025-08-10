# igp/segmentation/sam2.py
# Thin wrapper around SAM 2.x: box-prompt segmentation with a center-point fallback.
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image

from .base import Segmenter, SegmenterConfig


class Sam2Segmenter(Segmenter):
    """
    Segment Anything 2.x (Meta). Requires the official `sam2` modules.
    Uses a bounding-box prompt; if the mask is too small, falls back to a single
    positive point at the box center.
    """

    def __init__(
        self,
        model_cfg: str = "configs/sam2.1/sam2.1_hiera_l.yaml",
        checkpoint: str = "./checkpoints/sam2.1_hiera_large.pt",
        *,
        config: Optional[SegmenterConfig] = None,
    ) -> None:
        super().__init__(config)
        self.device = self.config.device or ("cuda" if torch.cuda.is_available() else "cpu")

        try:
            from sam2.build_sam import build_sam2  # type: ignore
            from sam2.sam2_image_predictor import SAM2ImagePredictor  # type: ignore
        except Exception as e:
            raise ImportError(
                "Moduli SAM2 non trovati. Assicurati di aver installato il repo SAM2 (facebookresearch)."
            ) from e

        cfg_path = Path(model_cfg)
        ckpt_path = Path(checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Checkpoint SAM-2 non trovato: {ckpt_path} (fornisci un percorso valido)."
            )
        if not cfg_path.exists():
            # Many environments load the cfg directly from the installed repo.
            # We don't force-download the YAML; build_sam2 accepts a string path.
            pass

        self._sam2_model = build_sam2(model_cfg, str(ckpt_path), device=self.device, precision="fp16").eval()
        from sam2.sam2_image_predictor import SAM2ImagePredictor  # late import for type checkers
        self._predictor = SAM2ImagePredictor(self._sam2_model)

    @torch.inference_mode()
    def segment(self, image_pil: Image.Image, boxes: Sequence[Sequence[float]]) -> List[Dict[str, Any]]:
        """
        Segment each box on the given image. Returns a list of dicts containing:
          - 'segmentation': boolean mask (H, W)
          - 'bbox': xywh box derived from the mask
          - 'predicted_iou': model-reported score
        """
        image_np = np.array(image_pil)
        H, W = image_np.shape[:2]

        out: List[Dict[str, Any]] = []
        self._predictor.set_image(image_np)

        try:
            for box in boxes:
                x1, y1, x2, y2 = self.clamp_box_xyxy(box, W, H)
                input_box = np.array([[x1, y1, x2, y2]])

                try:
                    # Primary attempt: box-prompt
                    masks, scores, _ = self._predictor.predict(box=input_box, multimask_output=False)
                    mask = masks[0]
                    score = float(scores[0]) if scores is not None else 1.0
                    if mask.sum() < 30:
                        # Fallback: single positive point at box center
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                        masks, scores, _ = self._predictor.predict(
                            point_coords=np.array([[cx, cy]]),
                            point_labels=np.array([1], dtype=int),
                            multimask_output=False,
                        )
                        mask = masks[0]
                        score = float(scores[0]) if scores is not None else score
                except Exception:
                    # Defensive fallback on predictor failure
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
            # Release cached features and GPU memory between calls
            if hasattr(self._predictor, "features"):
                delattr(self._predictor, "features")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
