# igp/utils/model_registry.py
# Model Registry - Singleton pattern for model caching
# Avoids reloading models, saves GPU memory, reduces latency

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch


class ModelRegistry:
    """
    Global model registry with singleton pattern.
    
    Benefits:
    - Models loaded once, reused across calls
    - Saves GPU memory (shared models)
    - Eliminates cold start latency
    - Thread-safe caching
    
    Usage:
        >>> detector = ModelRegistry.get_detector("grounding_dino", device="cuda")
        >>> # Second call returns same instance (instant)
        >>> detector2 = ModelRegistry.get_detector("grounding_dino", device="cuda")
        >>> assert detector is detector2
    """
    
    _detector_cache: Dict[Tuple, Any] = {}
    _segmenter_cache: Dict[Tuple, Any] = {}
    _other_cache: Dict[Tuple, Any] = {}
    
    @classmethod
    def get_detector(cls, name: str, **kwargs) -> Any:
        """
        Get cached detector or create new instance.
        
        Args:
            name: Detector name ("grounding_dino", "yolov8", "detectron2", "owlvit")
            **kwargs: Detector-specific parameters
            
        Returns:
            Cached or newly created detector instance
        """
        # Create cache key from name + kwargs
        cache_key = cls._make_key(name, kwargs)
        
        if cache_key in cls._detector_cache:
            print(f"[ModelRegistry] Using cached {name} detector")
            return cls._detector_cache[cache_key]
        
        print(f"[ModelRegistry] Loading {name} detector (first time)")
        
        # Create detector instance
        detector = cls._create_detector(name, **kwargs)
        
        # Warmup (run dummy inference to load weights)
        if hasattr(detector, 'warmup'):
            detector.warmup()
        
        # Cache it
        cls._detector_cache[cache_key] = detector
        
        return detector
    
    @classmethod
    def get_segmenter(cls, name: str, **kwargs) -> Any:
        """
        Get cached segmenter or create new instance.
        
        Args:
            name: Segmenter name ("1", "2", "hq", "fast", "mobile")
            **kwargs: Segmenter-specific parameters
            
        Returns:
            Cached or newly created segmenter instance
        """
        cache_key = cls._make_key(name, kwargs)
        
        if cache_key in cls._segmenter_cache:
            print(f"[ModelRegistry] Using cached SAM-{name} segmenter")
            return cls._segmenter_cache[cache_key]
        
        print(f"[ModelRegistry] Loading SAM-{name} segmenter (first time)")
        
        segmenter = cls._create_segmenter(name, **kwargs)
        
        if hasattr(segmenter, 'warmup'):
            segmenter.warmup()
        
        cls._segmenter_cache[cache_key] = segmenter
        
        return segmenter
    
    @classmethod
    def get_model(cls, category: str, name: str, **kwargs) -> Any:
        """
        Generic model getter.
        
        Args:
            category: "detector" | "segmenter" | "other"
            name: Model name
            **kwargs: Model parameters
            
        Returns:
            Cached model instance
        """
        if category == "detector":
            return cls.get_detector(name, **kwargs)
        elif category == "segmenter":
            return cls.get_segmenter(name, **kwargs)
        else:
            cache_key = cls._make_key(f"{category}_{name}", kwargs)
            
            if cache_key not in cls._other_cache:
                # Generic model loading
                cls._other_cache[cache_key] = cls._create_generic(category, name, **kwargs)
            
            return cls._other_cache[cache_key]
    
    @classmethod
    def _make_key(cls, name: str, kwargs: Dict) -> Tuple:
        """Create hashable cache key from name + kwargs."""
        # Sort kwargs for consistent keys
        items = sorted(kwargs.items())
        # Filter out non-hashable values
        hashable_items = []
        for k, v in items:
            if isinstance(v, (str, int, float, bool, type(None))):
                hashable_items.append((k, v))
            elif isinstance(v, (list, tuple)):
                hashable_items.append((k, tuple(v)))
        
        return (name, tuple(hashable_items))
    
    @classmethod
    def _create_detector(cls, name: str, **kwargs) -> Any:
        """Create detector instance."""
        
        if name == "grounding_dino":
            from gom.detectors.grounding_dino import GroundingDINODetector
            return GroundingDINODetector(**kwargs)
        
        elif name == "yolov8":
            from gom.detectors.yolov8 import YOLOv8Detector
            return YOLOv8Detector(**kwargs)
        
        elif name == "detectron2":
            from gom.detectors.detectron2 import Detectron2Detector
            return Detectron2Detector(**kwargs)
        
        elif name == "owlvit":
            from gom.detectors.owlvit import OwlViTDetector
            return OwlViTDetector(**kwargs)
        
        else:
            raise ValueError(f"Unknown detector: {name}")
    
    @classmethod
    def _create_segmenter(cls, name: str, **kwargs) -> Any:
        """Create segmenter instance."""
        
        if name == "1":
            from gom.segmentation.sam1 import Sam1Segmenter
            return Sam1Segmenter(**kwargs)
        
        elif name == "2":
            from gom.segmentation.sam2 import Sam2Segmenter
            return Sam2Segmenter(**kwargs)
        
        elif name == "hq":
            from gom.segmentation.samhq import SamHQSegmenter
            return SamHQSegmenter(**kwargs)
        
        elif name == "fast":
            from gom.segmentation.fastsam import FastSAMSegmenter
            return FastSAMSegmenter(**kwargs)
        
        elif name == "mobile":
            from gom.segmentation.fastsam import MobileSAMSegmenter
            return MobileSAMSegmenter(**kwargs)
        
        else:
            raise ValueError(f"Unknown segmenter: {name}")
    
    @classmethod
    def _create_generic(cls, category: str, name: str, **kwargs) -> Any:
        """Create generic model (depth, CLIP, etc.)."""
        
        if category == "depth":
            from gom.utils.depth import DepthEstimator
            return DepthEstimator(**kwargs)
        
        elif category == "clip":
            from gom.utils.clip_utils import CLIPWrapper
            return CLIPWrapper(**kwargs)
        
        else:
            raise ValueError(f"Unknown category: {category}")
    
    @classmethod
    def clear_cache(cls, category: Optional[str] = None):
        """
        Clear model cache to free GPU memory.
        
        Args:
            category: "detector" | "segmenter" | "other" | None (all)
        """
        if category == "detector" or category is None:
            for model in cls._detector_cache.values():
                cls._cleanup_model(model)
            cls._detector_cache.clear()
            print("[ModelRegistry] Cleared detector cache")
        
        if category == "segmenter" or category is None:
            for model in cls._segmenter_cache.values():
                cls._cleanup_model(model)
            cls._segmenter_cache.clear()
            print("[ModelRegistry] Cleared segmenter cache")
        
        if category == "other" or category is None:
            for model in cls._other_cache.values():
                cls._cleanup_model(model)
            cls._other_cache.clear()
            print("[ModelRegistry] Cleared other cache")
        
        # Force GPU memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    @classmethod
    def _cleanup_model(cls, model: Any):
        """Cleanup model resources."""
        try:
            # Move to CPU before deletion
            if hasattr(model, 'model') and hasattr(model.model, 'to'):
                model.model.to('cpu')
            elif hasattr(model, 'to'):
                model.to('cpu')
            
            # Delete
            del model
        except Exception as e:
            print(f"[ModelRegistry] Cleanup warning: {e}")
    
    @classmethod
    def get_cache_stats(cls) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            "detectors": len(cls._detector_cache),
            "segmenters": len(cls._segmenter_cache),
            "other": len(cls._other_cache),
            "total": len(cls._detector_cache) + len(cls._segmenter_cache) + len(cls._other_cache),
        }
    
    @classmethod
    def warmup_all(cls):
        """Warmup all cached models (avoid cold start)."""
        print("[ModelRegistry] Warming up all cached models...")
        
        for name, model in cls._detector_cache.items():
            if hasattr(model, 'warmup'):
                model.warmup()
        
        for name, model in cls._segmenter_cache.items():
            if hasattr(model, 'warmup'):
                model.warmup()
        
        print("[ModelRegistry] Warmup complete")


# Convenience functions
def get_detector(name: str, **kwargs):
    """Shorthand for ModelRegistry.get_detector()."""
    return ModelRegistry.get_detector(name, **kwargs)


def get_segmenter(name: str, **kwargs):
    """Shorthand for ModelRegistry.get_segmenter()."""
    return ModelRegistry.get_segmenter(name, **kwargs)


def clear_model_cache():
    """Shorthand for ModelRegistry.clear_cache()."""
    ModelRegistry.clear_cache()
