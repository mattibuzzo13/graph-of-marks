# igp/detectors/__init__.py
"""
Object Detection Models Package

Provides unified interface to multiple state-of-the-art object detection models
through the Detector abstract base class. Supports both closed-set and open-
vocabulary detection.

Available Detectors:
    - YOLOv8Detector: Real-time detection (YOLOv8n/s/m/l/x)
    - OwlViTDetector: Open-vocabulary detection (OWL-ViT)
    - Detectron2Detector: Mask R-CNN and variants (Detectron2)
    - GroundingDINODetector: Text-grounded detection (GroundingDINO)

Components:
    - Detector: Abstract base class defining detector interface
    - DetectorManager: Multi-detector orchestration and fusion

Key Features:
    - Unified API across all detector types
    - Automatic device management (CUDA/CPU)
    - Score threshold filtering
    - Batch processing support
    - Context manager protocol
    - Label normalization

Usage:
    >>> from gom.detectors import YOLOv8Detector, OwlViTDetector
    >>> 
    >>> # Real-time detection
    >>> yolo = YOLOv8Detector("yolov8x", device="cuda", score_threshold=0.5)
    >>> detections = yolo.run(image)
    >>> 
    >>> # Open-vocabulary detection
    >>> owl = OwlViTDetector(device="cuda")
    >>> detections = owl.run(image, text_queries=["person", "dog", "car"])
    >>> 
    >>> # Multi-detector fusion
    >>> from gom.detectors import DetectorManager
    >>> manager = DetectorManager(detectors=[yolo, owl])
    >>> fused = manager.detect_and_fuse(image)

Detector Comparison:
    
    YOLOv8:
        - Speed: ⭐⭐⭐⭐⭐ (fastest, 100+ FPS on GPU)
        - Accuracy: ⭐⭐⭐⭐ (good on COCO classes)
        - Vocabulary: ❌ Closed-set (80 COCO classes)
        - Use case: Real-time, known objects
    
    OwlViT:
        - Speed: ⭐⭐ (slower, ~10 FPS)
        - Accuracy: ⭐⭐⭐ (good zero-shot)
        - Vocabulary: ✅ Open (arbitrary text queries)
        - Use case: Novel objects, VQA
    
    Detectron2:
        - Speed: ⭐⭐⭐ (moderate, ~20 FPS)
        - Accuracy: ⭐⭐⭐⭐⭐ (state-of-art on COCO)
        - Vocabulary: ❌ Closed-set (configurable)
        - Use case: High accuracy, instance segmentation
    
    GroundingDINO:
        - Speed: ⭐⭐ (slower, transformer-based)
        - Accuracy: ⭐⭐⭐⭐⭐ (best open-vocabulary)
        - Vocabulary: ✅ Open (text grounding)
        - Use case: Complex queries, referring expressions

See Also:
    - gom.detectors.base: Detector abstract interface
    - gom.detectors.manager: Multi-detector orchestration
    - gom.fusion: Detection fusion strategies
"""
from .base import Detector
from .yolov8 import YOLOv8Detector
from .owlvit import OwlViTDetector
from .detectron2 import Detectron2Detector
from .grounding_dino import GroundingDINODetector
from .manager import DetectorManager

__all__ = [
    "Detector",
    "YOLOv8Detector",
    "OwlViTDetector",
    "Detectron2Detector",
    "GroundingDINODetector",
    "DetectorManager",
]
