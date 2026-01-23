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
        - Speed: Fastest (100+ FPS on GPU)
        - Accuracy: Good (COCO classes)
        - Vocabulary: Closed-set (80 COCO classes)
        - Use case: Real-time, known objects
    
    OwlViT:
        - Speed: Slower (~10 FPS)
        - Accuracy: Good (zero-shot)
        - Vocabulary: Open (arbitrary text queries)
        - Use case: Novel objects, VQA
    
    Detectron2:
        - Speed: Moderate (~20 FPS)
        - Accuracy: State-of-art (COCO)
        - Vocabulary: Closed-set (configurable)
        - Use case: High accuracy, instance segmentation
    
    GroundingDINO:
        - Speed: Slower (transformer-based)
        - Accuracy: Best (open-vocabulary)
        - Vocabulary: Open (text grounding)
        - Use case: Complex queries, referring expressions

See Also:
    - gom.detectors.base: Detector abstract interface
    - gom.detectors.manager: Multi-detector orchestration
    - gom.fusion: Detection fusion strategies
"""

from .base import Detector
from .manager import DetectorManager

# Try to import optional detectors - they may require additional dependencies
_AVAILABLE_DETECTORS = []

try:
    from .yolov8 import YOLOv8Detector
    _AVAILABLE_DETECTORS.append("YOLOv8Detector")
except ImportError:
    YOLOv8Detector = None  # type: ignore

try:
    from .owlvit import OwlViTDetector
    _AVAILABLE_DETECTORS.append("OwlViTDetector")
except ImportError:
    OwlViTDetector = None  # type: ignore

try:
    from .detectron2 import Detectron2Detector
    _AVAILABLE_DETECTORS.append("Detectron2Detector")
except ImportError:
    Detectron2Detector = None  # type: ignore

try:
    from .grounding_dino import GroundingDINODetector
    _AVAILABLE_DETECTORS.append("GroundingDINODetector")
except ImportError:
    GroundingDINODetector = None  # type: ignore

__all__ = [
    "Detector",
    "DetectorManager",
] + _AVAILABLE_DETECTORS
