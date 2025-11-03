"""Package exports for igp.detectors

Esporta i detector concreti e il DetectorManager per uso a livello di pipeline.
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
