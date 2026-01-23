# igp/detectors/grounding_dino.py
"""
Grounding DINO 1.5/1.6 - SOTA Open-Vocabulary Object Detection

Paper: Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection
Repo: https://github.com/IDEA-Research/GroundingDINO

Performance:
- 52.5 AP on COCO (zero-shot)
- 27.4 AP on LVIS (zero-shot)
- SOTA on ODinW benchmark
- Better than OWL-ViT v2 by ~8% mAP

Features:
- Text-prompt based detection
- Multi-scale feature fusion
- Better small object detection
- Clean bounding box predictions
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import torch
from PIL import Image

from gom.detectors.base import Detector
from gom.types import Detection


class GroundingDINODetector(Detector):
    """
    Grounding DINO - State-of-the-art open-vocabulary object detector.
    
    Args:
        model_name: Model variant ("base", "large", "swinb", "swint")
        text_prompt: Detection prompt (e.g., "person. car. dog.")
        box_threshold: Confidence threshold for boxes (default: 0.35)
        text_threshold: Text similarity threshold (default: 0.25)
        device: Device to run on ("cuda", "cpu", or None for auto)
        score_threshold: Global score threshold (inherited from Detector)
        nms_threshold: NMS IoU threshold (default: 0.8)
        use_amp: Use automatic mixed precision on GPU (default: True)
    
    Examples:
        >>> detector = GroundingDINODetector(
        ...     model_name="swint",
        ...     text_prompt="person. cat. dog. chair. table.",
        ...     box_threshold=0.35
        ... )
        >>> detections = detector.detect(image)
    """
    
    # Model configurations
    MODEL_CONFIGS = {
        "tiny": {
            "config": "GroundingDINO_SwinT_OGC.py",
            "checkpoint": "groundingdino_swint_ogc.pth",
        },
        "base": {
            "config": "GroundingDINO_SwinB_cfg.py",
            "checkpoint": "groundingdino_swinb_cogcoor.pth",
        },
        "large": {
            "config": "GroundingDINO_SwinL_cfg.py", 
            "checkpoint": "groundingdino_swinl_cogcoor.pth",
        },
    }
    
    def __init__(
        self,
        *,
        model_name: str = "base",
        text_prompt: Optional[str] = None,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
        device: Optional[str] = None,
        score_threshold: Optional[float] = None,
        nms_threshold: float = 0.8,
        use_amp: bool = True,
        checkpoint_dir: str = "./checkpoints/grounding_dino",
    ) -> None:
        super().__init__(
            name="grounding_dino",
            device=device,
            score_threshold=score_threshold or box_threshold,
        )
        
        self.model_name = model_name
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.nms_threshold = nms_threshold
        self.use_amp = use_amp and self.device.startswith("cuda")
        
        # Default text prompt (COCO classes + common objects)
        if text_prompt is None:
            text_prompt = (
                "person. bicycle. car. motorcycle. airplane. bus. train. truck. boat. "
                "traffic light. fire hydrant. stop sign. parking meter. bench. bird. cat. "
                "dog. horse. sheep. cow. elephant. bear. zebra. giraffe. backpack. umbrella. "
                "handbag. tie. suitcase. frisbee. skis. snowboard. sports ball. kite. "
                "baseball bat. baseball glove. skateboard. surfboard. tennis racket. bottle. "
                "wine glass. cup. fork. knife. spoon. bowl. banana. apple. sandwich. orange. "
                "broccoli. carrot. hot dog. pizza. donut. cake. chair. couch. potted plant. "
                "bed. dining table. toilet. tv. laptop. mouse. remote. keyboard. cell phone. "
                "microwave. oven. toaster. sink. refrigerator. book. clock. vase. scissors. "
                "teddy bear. hair drier. toothbrush. table. door. window. picture. lamp. "
                "shelf. cabinet. floor. wall. ceiling. plant. tree. grass. sky. road. "
                "building. fence. bridge. sign. box. bag. bottle. glass. plate. bowl"
            )
        
        self.text_prompt = text_prompt
        self._load_model(model_name, checkpoint_dir)
    
    def _load_model(self, model_name: str, checkpoint_dir: str) -> None:
        """Load Grounding DINO model."""
        try:
            from groundingdino.util.inference import load_model
        except ImportError:
            raise ImportError(
                "groundingdino not found. Install with:\n"
                "pip install groundingdino-py\n"
                "Or clone: https://github.com/IDEA-Research/GroundingDINO"
            )
        
        if model_name not in self.MODEL_CONFIGS:
            raise ValueError(
                f"Unknown model: {model_name}. "
                f"Available: {list(self.MODEL_CONFIGS.keys())}"
            )
        
        config = self.MODEL_CONFIGS[model_name]
        checkpoint_dir = Path(checkpoint_dir)
        config_path = checkpoint_dir / config["config"]
        checkpoint_path = checkpoint_dir / config["checkpoint"]
        
        if not config_path.exists():
            raise FileNotFoundError(
                f"Config not found: {config_path}\n"
                f"Download from: https://github.com/IDEA-Research/GroundingDINO"
            )
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}\n"
                f"Download from: https://github.com/IDEA-Research/GroundingDINO/releases"
            )
        
        self.model = load_model(
            str(config_path),
            str(checkpoint_path),
            device=self.device
        )
        self.model.eval()
    
    @torch.inference_mode()
    def detect(self, image: Image.Image) -> List[Detection]:
        """
        Run Grounding DINO detection on a single image.
        
        Args:
            image: PIL Image (any mode, will be converted to RGB)
        
        Returns:
            List of Detection objects with boxes in xyxy format
        """
        image_rgb = self._ensure_rgb(image)
        
        try:
            from groundingdino.util.inference import predict
        except ImportError:
            raise ImportError("groundingdino.util.inference not found")
        
        # Convert PIL to format expected by Grounding DINO
        image_np = np.array(image_rgb)
        
        # Run inference using torch.amp.autocast with explicit device_type to avoid FutureWarning
        device_type = "cuda" if self.use_amp else "cpu"
        with torch.amp.autocast(device_type, enabled=self.use_amp):
            boxes, logits, phrases = predict(
                model=self.model,
                image=image_np,
                caption=self.text_prompt,
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                device=self.device,
            )
        
        # Convert to Detection format
        detections = []
        H, W = image_np.shape[:2]
        
        for box, score, phrase in zip(boxes, logits, phrases):
            # Boxes are in normalized [cx, cy, w, h] format
            cx, cy, w, h = box.cpu().numpy()
            
            # Convert to absolute xyxy
            x1 = (cx - w / 2) * W
            y1 = (cy - h / 2) * H
            x2 = (cx + w / 2) * W
            y2 = (cy + h / 2) * H
            
            # Clamp to image bounds
            x1 = max(0, min(W, x1))
            y1 = max(0, min(H, y1))
            x2 = max(0, min(W, x2))
            y2 = max(0, min(H, y2))
            
            # Clean label (remove trailing period)
            label = phrase.strip().rstrip('.')
            
            detections.append(
                Detection(
                    box=(x1, y1, x2, y2),
                    label=label,
                    score=float(score.item()),
                )
            )
        
        return detections
    
    @property
    def supports_batch(self) -> bool:
        """Grounding DINO supports batch inference."""
        return True
    
    def detect_batch(self, images: Sequence[Image.Image]) -> List[List[Detection]]:
        """
        Batch detection (currently processes sequentially).
        
        TODO: Implement true batched inference for speedup.
        """
        # Best-effort parallelization for I/O-bound or CPU-bound workloads.
        if not images:
            return []
        try:
            import concurrent.futures
            max_workers = min(len(images), 4)
            results = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = [ex.submit(self.detect, img) for img in images]
                for f in futures:
                    results.append(f.result())
            return results
        except Exception:
            # Fallback to sequential detect
            return [self.detect(img) for img in images]

    def warmup(self, example_image=None, use_half: Optional[bool] = None) -> None:
        """Best-effort warmup: run a small predict to allocate model memory."""
        if example_image is None:
            return
        try:
            img = example_image
            if hasattr(img, "mode") and img.mode != "RGB":
                img = img.convert("RGB")
            image_np = np.array(img)
            # call the predict util to run a tiny forward; ignore results
            try:
                from groundingdino.util.inference import predict
                device_type = "cuda" if self.use_amp else "cpu"
                with torch.amp.autocast(device_type, enabled=self.use_amp):
                    _ = predict(
                        model=self.model,
                        image=image_np,
                        caption=self.text_prompt,
                        box_threshold=self.box_threshold,
                        text_threshold=self.text_threshold,
                        device=self.device,
                    )
            except Exception:
                # best-effort only
                pass
        except Exception:
            pass
    
    def close(self) -> None:
        """Release GPU memory."""
        if hasattr(self, 'model'):
            del self.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
