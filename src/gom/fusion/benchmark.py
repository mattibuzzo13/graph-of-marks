# igp/fusion/benchmark.py
"""
Fusion Methods Benchmark & Comparison

Utility to compare different fusion methods on your data:
- Speed comparison
- Quality metrics (if ground truth available)
- Visualization of differences

Usage:
    >>> from gom.fusion.benchmark import compare_fusion_methods
    >>> results = compare_fusion_methods(detections, image_size=(800, 600))
    >>> print(results)
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    from gom.types import Detection
except ImportError:
    Detection = None  # type: ignore


def compare_fusion_methods(
    detections: List["Detection"],
    image_size: Tuple[int, int],
    *,
    methods: Optional[List[str]] = None,
    iou_threshold: float = 0.5,
    runs: int = 3,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Benchmark different fusion methods.
    
    Args:
        detections: List of Detection objects to fuse
        image_size: (width, height)
        methods: List of method names to test (default: all available)
        iou_threshold: IoU threshold for methods
        runs: Number of timing runs per method
        verbose: Print results
    
    Returns:
        Dict with results per method:
        {
            "wbf": {
                "time_ms": 12.3,
                "num_output": 45,
                "avg_score": 0.87,
            },
            ...
        }
    
    Examples:
        >>> results = compare_fusion_methods(dets, (800, 600))
        >>> fastest = min(results.items(), key=lambda x: x[1]["time_ms"])
        >>> print(f"Fastest: {fastest[0]} ({fastest[1]['time_ms']:.1f}ms)")
    """
    if methods is None:
        methods = ["wbf", "nms", "soft_nms", "diou_nms", "matrix_nms", "adaptive_nms"]
    
    from . import get_fusion_method
    from .nms import nms as nms_fn
    
    results = {}
    
    for method_name in methods:
        try:
            if verbose:
                print(f"\nTesting {method_name}...")
            
            # Get method
            if method_name in ["nms", "soft_nms", "diou_nms", "matrix_nms", "adaptive_nms"]:
                # These methods work on Detection lists directly
                method_fn = get_fusion_method(method_name)
            else:
                # WBF and confluence need special handling
                method_fn = get_fusion_method(method_name)
            
            # Warm-up run
            if method_name == "wbf":
                _ = method_fn(detections, image_size, iou_thr=iou_threshold)
            elif method_name in ["confluence"]:
                _ = method_fn(detections, image_size, confluence_threshold=iou_threshold)
            elif method_name in ["nms", "soft_nms"]:
                _ = nms_fn(detections, iou_thr=iou_threshold, class_aware=True)
            else:
                # DIoU, Matrix, Adaptive work on arrays
                import numpy as np
                boxes = np.array([list(d.box) for d in detections])
                scores = np.array([d.score for d in detections])
                _ = method_fn(boxes, scores, iou_threshold=iou_threshold)
            
            # Timed runs
            times = []
            for _ in range(runs):
                start = time.perf_counter()
                
                if method_name == "wbf":
                    output = method_fn(detections, image_size, iou_thr=iou_threshold)
                elif method_name == "confluence":
                    output = method_fn(detections, image_size, confluence_threshold=iou_threshold)
                elif method_name in ["nms", "soft_nms"]:
                    output = nms_fn(detections, iou_thr=iou_threshold, class_aware=True)
                else:
                    import numpy as np
                    boxes = np.array([list(d.box) for d in detections])
                    scores = np.array([d.score for d in detections])
                    labels = np.array([hash(d.label) % 1000 for d in detections])
                    kept_indices = method_fn(boxes, scores, iou_threshold=iou_threshold)
                    output = [detections[i] for i in kept_indices]
                
                elapsed = (time.perf_counter() - start) * 1000  # ms
                times.append(elapsed)
            
            avg_time = sum(times) / len(times)
            
            # Compute statistics
            num_output = len(output)
            avg_score = sum(d.score for d in output) / max(len(output), 1)
            
            results[method_name] = {
                "time_ms": avg_time,
                "time_std_ms": max(times) - min(times),
                "num_output": num_output,
                "avg_score": avg_score,
                "reduction": 1.0 - (num_output / max(len(detections), 1)),
            }
            
            if verbose:
                print(f"  [OK] {method_name}: {avg_time:.1f}ms, "
                      f"{num_output} outputs ({len(detections)-num_output} removed), "
                      f"avg_score={avg_score:.3f}")
        
        except Exception as e:
            if verbose:
                print(f"  [FAILED] {method_name}: ({e})")
            results[method_name] = {"error": str(e)}
    
    if verbose:
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        
        valid_results = {k: v for k, v in results.items() if "error" not in v}
        if valid_results:
            fastest = min(valid_results.items(), key=lambda x: x[1]["time_ms"])
            most_outputs = max(valid_results.items(), key=lambda x: x[1]["num_output"])
            fewest_outputs = min(valid_results.items(), key=lambda x: x[1]["num_output"])
            
            print(f"Fastest: {fastest[0]} ({fastest[1]['time_ms']:.1f}ms)")
            print(f"Most outputs: {most_outputs[0]} ({most_outputs[1]['num_output']} dets)")
            print(f"Most aggressive: {fewest_outputs[0]} ({fewest_outputs[1]['num_output']} dets)")
        print("="*60)
    
    return results


def visualize_fusion_comparison(
    detections: List["Detection"],
    image_size: Tuple[int, int],
    methods: Optional[List[str]] = None,
    output_path: Optional[str] = None,
):
    """
    Visualize outputs of different fusion methods side-by-side.
    
    Requires matplotlib.
    
    Args:
        detections: Input detections
        image_size: (width, height)
        methods: Methods to compare
        output_path: If provided, save figure to this path
    """
    try:
        import matplotlib.patches as patches
        import matplotlib.pyplot as plt
    except ImportError:
        logger.error("matplotlib not installed, cannot visualize")
        return
    
    if methods is None:
        methods = ["wbf", "nms", "diou_nms"]
    
    from . import get_fusion_method
    from .nms import nms as nms_fn
    
    n_methods = len(methods)
    fig, axes = plt.subplots(1, n_methods, figsize=(5*n_methods, 5))
    
    if n_methods == 1:
        axes = [axes]
    
    W, H = image_size
    
    for ax, method_name in zip(axes, methods):
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)  # Invert y-axis
        ax.set_aspect('equal')
        ax.set_title(f'{method_name}\n({len(detections)} -> ?)')
        
        try:
            # Run fusion
            if method_name == "wbf":
                method_fn = get_fusion_method(method_name)
                output = method_fn(detections, image_size, iou_thr=0.5)
            elif method_name in ["nms", "soft_nms"]:
                output = nms_fn(detections, iou_thr=0.5, class_aware=True)
            else:
                import numpy as np
                method_fn = get_fusion_method(method_name)
                boxes = np.array([list(d.box) for d in detections])
                scores = np.array([d.score for d in detections])
                kept_indices = method_fn(boxes, scores, iou_threshold=0.5)
                output = [detections[i] for i in kept_indices]
            
            # Update title with output count
            ax.set_title(f'{method_name}\n({len(detections)} -> {len(output)})')
            
            # Draw boxes
            for det in output:
                x1, y1, x2, y2 = det.box
                rect = patches.Rectangle(
                    (x1, y1), x2-x1, y2-y1,
                    linewidth=2,
                    edgecolor='red',
                    facecolor='none',
                    alpha=0.7
                )
                ax.add_patch(rect)
                
                # Add score text
                ax.text(
                    x1, y1-5,
                    f'{det.score:.2f}',
                    fontsize=8,
                    color='red',
                    weight='bold'
                )
        
        except Exception as e:
            ax.text(W/2, H/2, f'ERROR:\n{e}', ha='center', va='center', color='red')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {output_path}")
    else:
        plt.show()


__all__ = ["compare_fusion_methods", "visualize_fusion_comparison"]
