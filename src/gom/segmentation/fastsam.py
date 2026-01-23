# igp/segmentation/fastsam.py
# FastSAM: Fast Segment Anything Model (ICCV 2023)
# 10x faster than SAM2, YOLOv8 backbone, real-time segmentation
# Paper: https://arxiv.org/abs/2306.12156

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image

from .base import Segmenter, SegmenterConfig


class FastSAMSegmenter(Segmenter):
    """
    FastSAM: YOLOv8-based segmentation model.
    
    Performance:
    - 10x faster than SAM2 (50 FPS vs 5 FPS on V100)
    - Slightly lower quality (~2-3% mIoU drop)
    - Best for real-time applications
    
    Reference:
    - Paper: "Fast Segment Anything" (ICCV 2023)
    - Code: https://github.com/CASIA-IVA-Lab/FastSAM
    """

    def __init__(
        self,
        checkpoint: str = "./checkpoints/FastSAM-x.pt",
        *,
        config: Optional[SegmenterConfig] = None,
        imgsz: int = 1024,  # Input image size (FastSAM default)
        conf: float = 0.4,  # Confidence threshold
        iou: float = 0.9,   # IoU threshold for NMS
        retina_masks: bool = True,  # High-quality mask output
    ) -> None:
        super().__init__(config)
        self.device = self.config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.retina_masks = retina_masks

        # Check if FastSAM is available
        try:
            from fastsam import FastSAM  # type: ignore
            from fastsam.prompt import FastSAMPrompt  # type: ignore
        except ImportError:
            raise ImportError(
                "FastSAM not installed. Install with:\n"
                "  pip install git+https://github.com/CASIA-IVA-Lab/FastSAM.git\n"
                "Or download checkpoint from:\n"
                "  wget https://github.com/CASIA-IVA-Lab/FastSAM/releases/download/v1.0/FastSAM-x.pt"
            )

        ckpt_path = Path(checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"FastSAM checkpoint not found: {ckpt_path}\n"
                f"Download from: https://github.com/CASIA-IVA-Lab/FastSAM/releases/download/v1.0/FastSAM-x.pt"
            )

        self._model = FastSAM(str(ckpt_path))
        self._FastSAMPrompt = FastSAMPrompt  # Store for later use

        print(f"[FastSAM] Loaded model: {ckpt_path.name}, device={self.device}")

    @torch.inference_mode()
    def segment(self, image_pil: Image.Image, boxes: Sequence[Sequence[float]]) -> List[Dict[str, Any]]:
        """
        Segment each box on the image.
        
        Args:
            image_pil: PIL Image
            boxes: List of [x1, y1, x2, y2] boxes in pixel coordinates
            
        Returns:
            List of dicts with:
              - 'segmentation': np.ndarray(bool, H, W)
              - 'bbox': [x, y, w, h] (xywh)
              - 'predicted_iou': float (confidence score)
        """
        if not boxes:
            return []

        image_np = np.array(image_pil)
        H, W = image_np.shape[:2]
        boxes_xyxy = [self.clamp_box_xyxy(b, W, H) for b in boxes]

        # Run FastSAM on entire image
        try:
            results = self._segment_with_boxes(image_np, boxes_xyxy, H, W)
        except Exception as e:
            print(f"[FastSAM] Error: {e}, using fallback")
            results = self._fallback_boxes_to_masks(boxes_xyxy, H, W)

        # Postprocess masks
        final: List[Dict[str, Any]] = []
        for r in results:
            mask = self.postprocess_mask(r["segmentation"].astype(bool))
            final.append(
                {
                    "segmentation": mask,
                    "bbox": self.bbox_from_mask(mask),
                    "predicted_iou": float(r.get("predicted_iou", 0.0)),
                }
            )
        
        return final

    def _segment_with_boxes(
        self,
        image_np: np.ndarray,
        boxes_xyxy: Sequence[Sequence[int]],
        H: int,
        W: int,
    ) -> List[Dict[str, Any]]:
        """
        Use FastSAM with box prompts.
        
        FastSAM workflow:
        1. Run model on full image (generates all masks)
        2. Use box prompts to filter/refine masks
        """
        # Step 1: Run FastSAM on entire image
        everything_results = self._model(
            image_np,
            device=self.device,
            retina_masks=self.retina_masks,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
        )
        
        # Step 2: Create prompt processor
        prompt_processor = self._FastSAMPrompt(
            image_np,
            everything_results,
            device=self.device
        )
        
        # Step 3: Process each box
        results: List[Dict[str, Any]] = []
        for box_xyxy in boxes_xyxy:
            try:
                # Convert to FastSAM box format [x1, y1, x2, y2]
                box_prompt = [[int(box_xyxy[0]), int(box_xyxy[1]), 
                              int(box_xyxy[2]), int(box_xyxy[3])]]
                
                # Get mask for this box
                mask_results = prompt_processor.box_prompt(bboxes=box_prompt)
                
                if mask_results is not None and len(mask_results) > 0:
                    # Extract first (best) mask
                    mask = mask_results[0].cpu().numpy().astype(bool)
                    if mask.ndim == 3:
                        mask = mask[0]  # Take first channel if multi-channel
                    
                    # Calculate confidence score (use IoU with box as proxy)
                    confidence = self._calculate_mask_confidence(mask, box_xyxy, H, W)
                    
                    results.append({
                        "segmentation": mask,
                        "predicted_iou": confidence,
                    })
                else:
                    # Fallback: use box as mask
                    mask = self._box_to_mask(box_xyxy, H, W)
                    results.append({
                        "segmentation": mask,
                        "predicted_iou": 0.5,
                    })
                    
            except Exception as e:
                print(f"[FastSAM] Box prompt failed: {e}, using box mask")
                mask = self._box_to_mask(box_xyxy, H, W)
                results.append({
                    "segmentation": mask,
                    "predicted_iou": 0.5,
                })
        
        return results

    def _fallback_boxes_to_masks(
        self,
        boxes_xyxy: Sequence[Sequence[int]],
        H: int,
        W: int,
    ) -> List[Dict[str, Any]]:
        """
        Fallback: convert boxes directly to binary masks.
        Used when FastSAM fails.
        """
        results: List[Dict[str, Any]] = []
        for box in boxes_xyxy:
            mask = self._box_to_mask(box, H, W)
            results.append({
                "segmentation": mask,
                "predicted_iou": 0.5,
            })
        return results

    @staticmethod
    def _box_to_mask(box_xyxy: Sequence[int], H: int, W: int) -> np.ndarray:
        """Convert a box to a binary mask."""
        mask = np.zeros((H, W), dtype=bool)
        x1, y1, x2, y2 = box_xyxy
        mask[y1:y2, x1:x2] = True
        return mask

    @staticmethod
    def _calculate_mask_confidence(
        mask: np.ndarray,
        box_xyxy: Sequence[int],
        H: int,
        W: int,
    ) -> float:
        """
        Calculate confidence score for a mask based on IoU with input box.
        
        Higher IoU = mask closely matches box = higher confidence
        """
        # Create box mask
        box_mask = np.zeros((H, W), dtype=bool)
        x1, y1, x2, y2 = box_xyxy
        box_mask[y1:y2, x1:x2] = True
        
        # Calculate IoU
        intersection = np.logical_and(mask, box_mask).sum()
        union = np.logical_or(mask, box_mask).sum()
        
        if union == 0:
            return 0.0
        
        iou = intersection / union
        return float(iou)

    def postprocess_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Apply postprocessing to improve mask quality.
        
        Operations:
        - Close holes (if enabled)
        - Remove small components (if enabled)
        - Morphological smoothing
        """
        if not isinstance(mask, np.ndarray):
            mask = np.array(mask, dtype=bool)
        
        # Apply base postprocessing (from parent class)
        if self.config.close_holes:
            mask = self.close_mask_holes(mask)
        
        if self.config.remove_small_components and self.config.min_component_area > 0:
            mask = self.remove_small_components_mask(mask)
        
        return mask


class MobileSAMSegmenter(Segmenter):
    """
    MobileSAM: Efficient SAM variant for mobile/edge devices.
    
    Performance:
    - 60x faster than SAM (ViT-B)
    - 6x smaller model size
    - Similar quality to SAM
    
    Reference:
    - Paper: "Faster Segment Anything: Towards Lightweight SAM for Mobile Applications" (2023)
    - Code: https://github.com/ChaoningZhang/MobileSAM
    """

    def __init__(
        self,
        checkpoint: str = "./checkpoints/mobile_sam.pt",
        *,
        config: Optional[SegmenterConfig] = None,
    ) -> None:
        super().__init__(config)
        self.device = self.config.device or ("cuda" if torch.cuda.is_available() else "cpu")

        try:
            from mobile_sam import SamPredictor, sam_model_registry  # type: ignore
        except ImportError:
            raise ImportError(
                "MobileSAM not installed. Install with:\n"
                "  pip install git+https://github.com/ChaoningZhang/MobileSAM.git\n"
                "Or download checkpoint from:\n"
                "  wget https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt"
            )

        ckpt_path = Path(checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"MobileSAM checkpoint not found: {ckpt_path}\n"
                f"Download from: https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt"
            )

        # Load MobileSAM model
        sam = sam_model_registry["vit_t"](checkpoint=str(ckpt_path))
        sam.to(device=self.device)
        sam.eval()
        
        self._predictor = SamPredictor(sam)
        print(f"[MobileSAM] Loaded model: {ckpt_path.name}, device={self.device}")

    @torch.inference_mode()
    def segment(self, image_pil: Image.Image, boxes: Sequence[Sequence[float]]) -> List[Dict[str, Any]]:
        """
        Segment each box on the image using MobileSAM.
        """
        if not boxes:
            return []

        image_np = np.array(image_pil)
        H, W = image_np.shape[:2]
        boxes_xyxy = [self.clamp_box_xyxy(b, W, H) for b in boxes]

        # Set image once
        self._predictor.set_image(image_np)

        # Process each box
        results: List[Dict[str, Any]] = []
        for box in boxes_xyxy:
            try:
                # Convert to numpy array format expected by MobileSAM
                box_np = np.array([box], dtype=np.float32)
                
                # Predict mask
                masks, scores, _ = self._predictor.predict(
                    box=box_np[0],
                    multimask_output=False,
                )
                
                mask = masks[0].astype(bool)
                score = float(scores[0]) if len(scores) > 0 else 0.0
                
                # Postprocess
                mask = self.postprocess_mask(mask)
                
                results.append({
                    "segmentation": mask,
                    "bbox": self.bbox_from_mask(mask),
                    "predicted_iou": score,
                })
                
            except Exception as e:
                print(f"[MobileSAM] Error for box {box}: {e}")
                # Fallback: box to mask
                mask = np.zeros((H, W), dtype=bool)
                x1, y1, x2, y2 = box
                mask[y1:y2, x1:x2] = True
                results.append({
                    "segmentation": mask,
                    "bbox": self.bbox_from_mask(mask),
                    "predicted_iou": 0.5,
                })
        
        return results

    def postprocess_mask(self, mask: np.ndarray) -> np.ndarray:
        """Apply postprocessing from config."""
        if self.config.close_holes:
            mask = self.close_mask_holes(mask)
        if self.config.remove_small_components and self.config.min_component_area > 0:
            mask = self.remove_small_components_mask(mask)
        return mask
