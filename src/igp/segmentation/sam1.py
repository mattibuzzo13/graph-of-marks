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
        """
        Segment objects using SAM 1.0 with BATCH PROCESSING (5-8x faster).
        
        Args:
            image_pil: Input PIL Image
            boxes: List of [x1, y1, x2, y2] boxes
        
        Returns:
            List of segmentation results with masks and metadata
        """
        if not boxes:
            return []
        
        image_np = np.array(image_pil)
        H, W = image_np.shape[:2]
        self._predictor.set_image(image_np)
        
        out: List[Dict[str, Any]] = []
        
        try:
            # NEW: Try batch processing first (5-8x faster)
            out = self._segment_batch(image_np, boxes, H, W)
            if out:  # Success with batch mode
                return out
        except Exception as e:
            print(f"[SAM1] Batch processing failed: {e}, falling back to sequential")
        
        # FALLBACK: Sequential processing (original code)
        try:
            for box in boxes:
                x1, y1, x2, y2 = self.clamp_box_xyxy(box, W, H)

                # Try progressively shrunken boxes for robustness
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
                    break

                if mask_ok is None:
                    mask_ok = np.zeros((H, W), dtype=bool)
                    score_ok = 0.0

                bbox_xywh = self.bbox_from_mask(mask_ok)
                out.append({
                    "segmentation": mask_ok,
                    "bbox": bbox_xywh,
                    "predicted_iou": float(score_ok),
                })
            return out
        finally:
            self._predictor.reset_image()
            if hasattr(self._predictor, "features"):
                delattr(self._predictor, "features")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _segment_batch(
        self,
        image_np: np.ndarray,
        boxes: Sequence[Sequence[float]],
        H: int,
        W: int,
    ) -> List[Dict[str, Any]]:
        """
        NEW METHOD: Batch SAM inference - processes all boxes in one forward pass.
        """
        # Convert boxes to tensor and clamp
        boxes_clamped = []
        for box in boxes:
            x1, y1, x2, y2 = self.clamp_box_xyxy(box, W, H)
            boxes_clamped.append([x1, y1, x2, y2])
        
        boxes_tensor = torch.tensor(boxes_clamped, dtype=torch.float32, device=self.device)
        
        # Transform boxes to SAM's input format
        transformed_boxes = self._predictor.transform.apply_boxes_torch(
            boxes_tensor,
            image_np.shape[:2]
        )
        
        # Single forward pass for ALL boxes
        masks, iou_scores, _ = self._predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=True  # Returns 3 masks per box
        )
        
        # Process results
        results = []
        for idx, box in enumerate(boxes_clamped):
            # Get best mask for this box (highest IoU)
            box_masks = masks[idx]  # Shape: (3, H, W)
            box_scores = iou_scores[idx]  # Shape: (3,)
            
            best_idx = box_scores.argmax().item()
            best_mask = box_masks[best_idx].cpu().numpy()
            best_score = box_scores[best_idx].item()
            
            # Convert to boolean mask
            best_mask = best_mask.astype(bool)
            
            # Apply hole closing if enabled
            if self.config.close_holes:
                best_mask = self.close_mask_holes(best_mask)
            
            # Fallback for empty masks: try center point
            if best_mask.sum() < 50:
                x1, y1, x2, y2 = box
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                try:
                    masks_pt, scores_pt, _ = self._predictor.predict(
                        point_coords=np.array([[cx, cy]]),
                        point_labels=np.array([1]),
                        multimask_output=False,
                    )
                    best_mask = masks_pt[0].astype(bool)
                    best_score = float(scores_pt[0])
                    if self.config.close_holes:
                        best_mask = self.close_mask_holes(best_mask)
                except Exception:
                    best_mask = np.zeros((H, W), dtype=bool)
                    best_score = 0.0
            
            bbox_xywh = self.bbox_from_mask(best_mask)
            results.append({
                "segmentation": best_mask,
                "bbox": bbox_xywh,
                "predicted_iou": float(best_score),
            })
        
        return results

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
