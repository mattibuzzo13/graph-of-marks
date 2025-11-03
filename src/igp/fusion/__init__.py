# igp/fusion/__init__.py
"""
🚀 SOTA Detection Fusion Module (2024)

State-of-the-art fusion methods for multi-detector object detection:

**Classic Methods:**
- WBF (Weighted Boxes Fusion): ensemble-boxes based fusion
- NMS (Non-Maximum Suppression): Standard greedy NMS
- Soft-NMS: Score decay instead of removal

**SOTA Methods (2019-2024):**
- DIoU-NMS: Distance-IoU aware (AAAI 2020)
- Matrix-NMS: Parallel computation (ECCV 2020)
- Adaptive-NMS: Density-aware thresholds (CVPR 2019)
- Confluence: IoU + objectness fusion (CVPR 2021)

**Optimizations:**
- Vectorized NumPy operations
- Optional GPU acceleration (PyTorch/Torchvision)
- Mask fusion support
- Per-class/cross-class variants
"""

from __future__ import annotations

from .nms import (
    nms,
    soft_nms,
    iou,
    labelwise_nms,
    # SOTA methods
    soft_nms_gaussian,
    diou_nms,
    matrix_nms,
    adaptive_nms,
)

from .wbf import (
    fuse_detections_wbf,
    compute_iou_vectorized,
)

try:
    from .confluence import confluence_fusion
    _HAS_CONFLUENCE = True
except ImportError:
    confluence_fusion = None  # type: ignore
    _HAS_CONFLUENCE = False

__all__ = [
    # Classic
    "nms",
    "soft_nms",
    "iou",
    "labelwise_nms",
    "fuse_detections_wbf",
    "compute_iou_vectorized",
    # SOTA
    "soft_nms_gaussian",
    "diou_nms",
    "matrix_nms",
    "adaptive_nms",
    "confluence_fusion",
]


# Convenience function to get best available fusion method
def get_fusion_method(name: str = "auto"):
    """
    Get fusion method by name.
    
    Args:
        name: "auto", "wbf", "nms", "soft_nms", "diou", "matrix", "adaptive", "confluence"
    
    Returns:
        Callable fusion function
    
    Examples:
        >>> fusion_fn = get_fusion_method("wbf")
        >>> fused = fusion_fn(detections, image_size=(800, 600))
    """
    if name == "auto":
        # Prefer WBF if ensemble-boxes is available, else DIoU-NMS
        return fuse_detections_wbf
    
    methods = {
        "wbf": fuse_detections_wbf,
        "nms": nms,
        "soft_nms": soft_nms,
        "soft_nms_gaussian": soft_nms_gaussian,
        "diou": diou_nms,
        "diou_nms": diou_nms,
        "matrix": matrix_nms,
        "matrix_nms": matrix_nms,
        "adaptive": adaptive_nms,
        "adaptive_nms": adaptive_nms,
    }
    
    if name == "confluence" and _HAS_CONFLUENCE:
        methods["confluence"] = confluence_fusion
    
    if name not in methods:
        raise ValueError(
            f"Unknown fusion method: {name}. "
            f"Available: {', '.join(methods.keys())}"
        )
    
    return methods[name]
