# igp/viz/rendering_opt.py
# 🚀 Rendering optimizations for matplotlib visualizations

from __future__ import annotations

from typing import List, Tuple, Optional
import numpy as np


class VectorizedMaskRenderer:
    """
    Vectorized operations for rendering multiple masks efficiently.
    Instead of drawing masks one-by-one, batch operations for better performance.
    """
    
    @staticmethod
    def blend_multiple_masks(
        image: np.ndarray,
        masks: List[np.ndarray],
        colors: List[Tuple[float, float, float]],
        alpha: float = 0.6
    ) -> np.ndarray:
        """
        Blend multiple masks onto image using vectorized numpy operations.
        
        Args:
            image: Base image (H, W, 3)
            masks: List of binary masks (H, W)
            colors: List of RGB colors (0-1 range)
            alpha: Transparency factor
            
        Returns:
            Blended image
        """
        H, W = image.shape[:2]
        result = image.copy().astype(np.float32) / 255.0
        
        # Pre-allocate color layer
        color_layer = np.zeros((H, W, 3), dtype=np.float32)
        mask_total = np.zeros((H, W), dtype=np.float32)
        
        # Vectorized blending: accumulate all masks at once
        for mask, color in zip(masks, colors):
            if mask is None:
                continue
            
            mask_bool = mask.astype(bool)
            color_arr = np.array(color, dtype=np.float32)
            
            # Add weighted contribution
            color_layer[mask_bool] += color_arr * alpha
            mask_total[mask_bool] += alpha
        
        # Normalize where masks overlap
        overlap = mask_total > 0
        if overlap.any():
            color_layer[overlap] /= mask_total[overlap, None]
        
        # Blend with original image
        result = (1 - alpha) * result + color_layer
        result = np.clip(result * 255, 0, 255).astype(np.uint8)
        
        return result


class BatchTextRenderer:
    """
    Batch text rendering to reduce matplotlib overhead.
    Pre-compute text positions and sizes, then render all at once.
    """
    
    def __init__(self):
        self.text_items = []
        
    def add_text(
        self,
        x: float,
        y: float,
        text: str,
        fontsize: int,
        color: str,
        bbox_params: Optional[dict] = None,
        ha: str = "center",
        va: str = "center",
        zorder: int = 5
    ):
        """Queue text for batched rendering."""
        self.text_items.append({
            "x": x,
            "y": y,
            "text": text,
            "fontsize": fontsize,
            "color": color,
            "bbox": bbox_params,
            "ha": ha,
            "va": va,
            "zorder": zorder,
        })
    
    def render_all(self, ax):
        """Render all queued text items at once."""
        # Sort by zorder for consistent rendering
        self.text_items.sort(key=lambda item: item.get("zorder", 5))
        
        for item in self.text_items:
            ax.text(
                item["x"],
                item["y"],
                item["text"],
                ha=item["ha"],
                va=item["va"],
                fontsize=item["fontsize"],
                color=item["color"],
                bbox=item["bbox"],
                zorder=item["zorder"],
            )
        
        self.text_items.clear()


class GeometricOptimizer:
    """
    Vectorized geometric operations for boxes and masks.
    """
    
    @staticmethod
    def compute_centers_vectorized(boxes: np.ndarray) -> np.ndarray:
        """
        Compute centers of all boxes at once.
        
        Args:
            boxes: Array of shape (N, 4) with [x1, y1, x2, y2]
            
        Returns:
            Array of shape (N, 2) with [cx, cy]
        """
        return (boxes[:, :2] + boxes[:, 2:]) / 2.0
    
    @staticmethod
    def compute_areas_vectorized(boxes: np.ndarray) -> np.ndarray:
        """
        Compute areas of all boxes at once.
        
        Args:
            boxes: Array of shape (N, 4) with [x1, y1, x2, y2]
            
        Returns:
            Array of shape (N,) with areas
        """
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        return widths * heights
    
    @staticmethod
    def compute_mask_areas_vectorized(masks: List[np.ndarray]) -> np.ndarray:
        """
        Compute areas of all masks at once.
        
        Args:
            masks: List of binary masks
            
        Returns:
            Array of areas
        """
        return np.array([mask.sum() if mask is not None else 0 for mask in masks])
    
    @staticmethod
    def clamp_boxes_vectorized(
        boxes: np.ndarray,
        width: int,
        height: int
    ) -> np.ndarray:
        """
        Clamp all boxes to image bounds at once.
        
        Args:
            boxes: Array of shape (N, 4) with [x1, y1, x2, y2]
            width: Image width
            height: Image height
            
        Returns:
            Clamped boxes
        """
        boxes = boxes.copy()
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, width - 1)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, height - 1)
        
        # Ensure x2 > x1 and y2 > y1
        boxes[:, 2] = np.maximum(boxes[:, 2], boxes[:, 0] + 1)
        boxes[:, 3] = np.maximum(boxes[:, 3], boxes[:, 1] + 1)
        
        return boxes


class ArrowOptimizer:
    """
    Optimize arrow rendering by pre-computing paths and reducing draw calls.
    """
    
    @staticmethod
    def compute_arrow_paths_batch(
        centers: np.ndarray,
        relations: List[Tuple[int, int]],
        curvature: float = 0.3
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Pre-compute all arrow paths using vectorized operations.
        
        Args:
            centers: Array of shape (N, 2) with [cx, cy]
            relations: List of (src_idx, tgt_idx) pairs
            curvature: Arrow curvature factor
            
        Returns:
            List of (path_x, path_y) arrays for each arrow
        """
        paths = []
        
        for src_idx, tgt_idx in relations:
            src = centers[src_idx]
            tgt = centers[tgt_idx]
            
            # Compute control point for Bezier curve
            mid = (src + tgt) / 2
            direction = tgt - src
            perpendicular = np.array([-direction[1], direction[0]])
            perpendicular = perpendicular / (np.linalg.norm(perpendicular) + 1e-8)
            
            control = mid + perpendicular * curvature * np.linalg.norm(direction)
            
            # Sample points along Bezier curve
            t = np.linspace(0, 1, 20)
            path_x = (1 - t)**2 * src[0] + 2 * (1 - t) * t * control[0] + t**2 * tgt[0]
            path_y = (1 - t)**2 * src[1] + 2 * (1 - t) * t * control[1] + t**2 * tgt[1]
            
            paths.append((path_x, path_y))
        
        return paths


# Performance monitoring decorator
def profile_rendering(func):
    """Decorator to profile rendering functions."""
    import time
    from functools import wraps
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"[PROFILE] {func.__name__} took {elapsed*1000:.2f} ms")
        return result
    
    return wrapper


# Example usage:
if __name__ == "__main__":
    # Test vectorized operations
    import time
    
    # Generate test data
    N = 100
    boxes = np.random.rand(N, 4) * 512
    boxes[:, 2:] += boxes[:, :2]  # Ensure x2 > x1, y2 > y1
    
    # Test centers computation
    start = time.perf_counter()
    centers_vec = GeometricOptimizer.compute_centers_vectorized(boxes)
    time_vec = time.perf_counter() - start
    
    start = time.perf_counter()
    centers_loop = np.array([((b[0]+b[2])/2, (b[1]+b[3])/2) for b in boxes])
    time_loop = time.perf_counter() - start
    
    print(f"Vectorized: {time_vec*1000:.3f} ms")
    print(f"Loop: {time_loop*1000:.3f} ms")
    print(f"Speedup: {time_loop/time_vec:.1f}x")
    print(f"Results match: {np.allclose(centers_vec, centers_loop)}")
