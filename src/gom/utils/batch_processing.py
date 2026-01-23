# igp/utils/batch_processing.py
# Batch Processing for Detectors and Segmenters
# Process multiple images in parallel for GPU efficiency

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
from PIL import Image


@dataclass
class BatchConfig:
    """Configuration for batch processing."""
    
    batch_size: int = 4  # Number of images per batch
    num_workers: int = 2  # DataLoader workers
    pin_memory: bool = True  # Pin memory for faster GPU transfer
    prefetch_factor: int = 2  # Prefetch batches
    drop_last: bool = False  # Drop incomplete last batch
    
    # Image preprocessing
    resize_mode: str = "pad"  # "pad" | "resize" | "none"
    max_size: int = 1024  # Max dimension
    pad_value: int = 0  # Padding value


class BatchProcessor:
    """
    Batch processor for efficient multi-image inference.
    
    Benefits:
    - 2-4x speedup on GPU (batch_size=4-8)
    - Better GPU utilization
    - Parallel data loading
    
    Usage:
        >>> processor = BatchProcessor(batch_size=4)
        >>> results = processor.process_batch(images, detector.detect_batch)
    """
    
    def __init__(self, config: Optional[BatchConfig] = None):
        self.config = config or BatchConfig()
    
    def process_batch(
        self,
        images: List[Image.Image],
        process_fn: Callable[[torch.Tensor], List[Dict]],
        *,
        device: str = "cuda",
    ) -> List[Dict[str, Any]]:
        """
        Process images in batches.
        
        Args:
            images: List of PIL Images
            process_fn: Function that takes batched tensor and returns list of results
            device: "cuda" or "cpu"
            
        Returns:
            List of results (one per image)
        """
        if len(images) == 0:
            return []
        
        all_results = []
        
        # Process in batches
        for i in range(0, len(images), self.config.batch_size):
            batch_images = images[i:i + self.config.batch_size]
            
            # Prepare batch tensor
            batch_tensor, metadata = self._prepare_batch(batch_images, device)
            
            # Run inference
            with torch.no_grad():
                batch_results = process_fn(batch_tensor)
            
            # Post-process results (unpad, rescale)
            batch_results = self._postprocess_batch(
                batch_results, metadata, batch_images
            )
            
            all_results.extend(batch_results)
        
        return all_results
    
    def _prepare_batch(
        self,
        images: List[Image.Image],
        device: str,
    ) -> tuple[torch.Tensor, List[Dict]]:
        """
        Prepare batch tensor from list of images.
        
        Returns:
            (batch_tensor, metadata_list)
        """
        metadata = []
        tensors = []
        
        # Find max dimensions in batch
        max_h = max(img.height for img in images)
        max_w = max(img.width for img in images)
        
        # Round up to multiple of 32 (common for detection models)
        max_h = ((max_h + 31) // 32) * 32
        max_w = ((max_w + 31) // 32) * 32
        
        # Limit max size
        if max_h > self.config.max_size or max_w > self.config.max_size:
            scale = self.config.max_size / max(max_h, max_w)
            max_h = int(max_h * scale)
            max_w = int(max_w * scale)
        
        for img in images:
            orig_w, orig_h = img.size
            
            if self.config.resize_mode == "pad":
                # Pad to max size
                tensor, pad_info = self._pad_image(img, max_w, max_h)
                metadata.append({
                    "orig_size": (orig_w, orig_h),
                    "padded_size": (max_w, max_h),
                    "pad_left": pad_info[0],
                    "pad_top": pad_info[1],
                    "scale": 1.0,
                })
            
            elif self.config.resize_mode == "resize":
                # Resize to max size
                resized = img.resize((max_w, max_h), Image.BILINEAR)
                tensor = self._img_to_tensor(resized)
                metadata.append({
                    "orig_size": (orig_w, orig_h),
                    "resized_size": (max_w, max_h),
                    "scale_x": max_w / orig_w,
                    "scale_y": max_h / orig_h,
                })
            
            else:  # "none"
                tensor = self._img_to_tensor(img)
                metadata.append({
                    "orig_size": (orig_w, orig_h),
                })
            
            tensors.append(tensor)
        
        # Stack into batch
        batch_tensor = torch.stack(tensors).to(device)
        
        return batch_tensor, metadata
    
    def _pad_image(
        self,
        img: Image.Image,
        target_w: int,
        target_h: int,
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        """
        Pad image to target size (center padding).
        
        Returns:
            (padded_tensor, (pad_left, pad_top))
        """
        orig_w, orig_h = img.size
        
        # Calculate padding
        pad_left = (target_w - orig_w) // 2
        pad_top = (target_h - orig_h) // 2
        pad_right = target_w - orig_w - pad_left
        pad_bottom = target_h - orig_h - pad_top
        
        # Convert to numpy
        img_array = np.array(img)
        
        # Pad
        if img_array.ndim == 2:  # Grayscale
            img_array = img_array[:, :, np.newaxis]
        
        padded = np.pad(
            img_array,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode='constant',
            constant_values=self.config.pad_value,
        )
        
        # To tensor
        tensor = torch.from_numpy(padded).permute(2, 0, 1).float() / 255.0
        
        return tensor, (pad_left, pad_top)
    
    def _img_to_tensor(self, img: Image.Image) -> torch.Tensor:
        """Convert PIL Image to tensor [C, H, W]."""
        img_array = np.array(img)
        
        if img_array.ndim == 2:
            img_array = img_array[:, :, np.newaxis]
        
        tensor = torch.from_numpy(img_array).permute(2, 0, 1).float() / 255.0
        
        return tensor
    
    def _postprocess_batch(
        self,
        results: List[Dict],
        metadata: List[Dict],
        original_images: List[Image.Image],
    ) -> List[Dict]:
        """
        Post-process batch results (unpad, rescale boxes).
        
        Args:
            results: Raw results from model
            metadata: Batch metadata (padding, scaling info)
            original_images: Original images (for size reference)
            
        Returns:
            Post-processed results in original coordinates
        """
        processed = []
        
        for result, meta, orig_img in zip(results, metadata, original_images):
            # Adjust boxes based on padding/resizing
            if "boxes" in result and len(result["boxes"]) > 0:
                boxes = np.array(result["boxes"])
                
                if self.config.resize_mode == "pad":
                    # Remove padding offset
                    pad_left = meta["pad_left"]
                    pad_top = meta["pad_top"]
                    boxes[:, [0, 2]] -= pad_left
                    boxes[:, [1, 3]] -= pad_top
                
                elif self.config.resize_mode == "resize":
                    # Rescale to original size
                    scale_x = meta["scale_x"]
                    scale_y = meta["scale_y"]
                    boxes[:, [0, 2]] /= scale_x
                    boxes[:, [1, 3]] /= scale_y
                
                # Clamp to image bounds
                orig_w, orig_h = meta["orig_size"]
                boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w)
                boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h)
                
                result["boxes"] = boxes.tolist()
            
            processed.append(result)
        
        return processed


class BatchDetectorWrapper:
    """
    Wrapper to add batch processing to any detector.
    
    Usage:
        >>> detector = GroundingDINODetector()
        >>> batch_detector = BatchDetectorWrapper(detector, batch_size=4)
        >>> results = batch_detector.detect_multiple(images)
    """
    
    def __init__(
        self,
        detector: Any,
        batch_size: int = 4,
        device: Optional[str] = None,
    ):
        self.detector = detector
        self.batch_processor = BatchProcessor(BatchConfig(batch_size=batch_size))
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    def detect_multiple(
        self,
        images: List[Image.Image],
    ) -> List[Dict[str, Any]]:
        """
        Detect objects in multiple images (batched).
        
        Args:
            images: List of PIL Images
            
        Returns:
            List of detection results
        """
        # Check if detector has native batch support
        if hasattr(self.detector, 'detect_batch'):
            return self.batch_processor.process_batch(
                images,
                self.detector.detect_batch,
                device=self.device,
            )
        else:
            # Fallback: call detect() for each image (no batching)
            print("[Warning] Detector doesn't support batching, falling back to sequential")
            return [self.detector.detect(img) for img in images]
    
    def detect(self, image: Image.Image) -> Dict[str, Any]:
        """Single image detection (for compatibility)."""
        return self.detect_multiple([image])[0]


def create_batch_detector(detector: Any, batch_size: int = 4):
    """
    Convenience function to create batch detector.
    
    Args:
        detector: Detector instance
        batch_size: Batch size
        
    Returns:
        Batch-enabled detector
    """
    return BatchDetectorWrapper(detector, batch_size=batch_size)


def estimate_optimal_batch_size(
    model: torch.nn.Module,
    input_shape: tuple = (3, 640, 640),
    max_memory_gb: float = 10.0,
) -> int:
    """
    Estimate optimal batch size based on GPU memory.
    
    Args:
        model: PyTorch model
        input_shape: Input shape (C, H, W)
        max_memory_gb: Max GPU memory to use
        
    Returns:
        Recommended batch size
    """
    if not torch.cuda.is_available():
        return 1
    
    device = next(model.parameters()).device
    
    # Test batch sizes
    batch_sizes = [1, 2, 4, 8, 16, 32]
    max_batch_size = 1
    
    for bs in batch_sizes:
        try:
            # Create dummy batch
            dummy_input = torch.randn(bs, *input_shape, device=device)
            
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            # Forward pass
            with torch.no_grad():
                _ = model(dummy_input)
            
            # Check memory usage
            memory_used = torch.cuda.max_memory_allocated() / 1e9
            
            if memory_used < max_memory_gb:
                max_batch_size = bs
            else:
                break
            
            del dummy_input
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                break
            raise
    
    print(f"[BatchSize] Optimal batch size: {max_batch_size}")
    return max_batch_size
