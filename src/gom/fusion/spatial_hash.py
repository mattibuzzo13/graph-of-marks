# igp/fusion/spatial_hash.py
# Spatial Hashing Grid for accelerating WBF fusion from O(N²) to O(N×k)
# Reduces IoU computations by ~90% through spatial indexing

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Set, Tuple

import numpy as np


class SpatialHashGrid:
    """
    Grid-based spatial index for fast neighbor queries.
    
    Divides image space into cells and indexes boxes by the cells they overlap.
    Enables O(1) neighbor queries instead of O(N) brute force search.
    
    Example:
        grid = SpatialHashGrid((1920, 1080), cell_size=100)
        for i, box in enumerate(boxes):
            grid.insert(box, i)
        
        neighbors = grid.query_neighbors(box)  # Returns indices of nearby boxes
    """
    
    def __init__(self, image_size: Tuple[int, int], cell_size: int = 100):
        """
        Args:
            image_size: (width, height) in pixels
            cell_size: Size of each grid cell in pixels (default 100px)
                       Smaller = more precise but more memory
                       Larger = less memory but more false positives
                       Optimal: ~average box size
        """
        self.W, self.H = image_size
        self.cell_size = cell_size
        
        # Grid dimensions
        self.grid_w = (self.W + cell_size - 1) // cell_size
        self.grid_h = (self.H + cell_size - 1) // cell_size
        
        # Storage: {(cell_x, cell_y): [box_indices]}
        self.cells: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        
    def _box_to_cells(self, box: np.ndarray) -> Tuple[int, int, int, int]:
        """Convert box coordinates to cell indices."""
        x1, y1, x2, y2 = box
        
        # Clamp to image bounds
        x1 = max(0, min(self.W - 1, x1))
        y1 = max(0, min(self.H - 1, y1))
        x2 = max(0, min(self.W - 1, x2))
        y2 = max(0, min(self.H - 1, y2))
        
        # Convert to cell coordinates
        c_x1 = int(x1 // self.cell_size)
        c_y1 = int(y1 // self.cell_size)
        c_x2 = int(x2 // self.cell_size)
        c_y2 = int(y2 // self.cell_size)
        
        # Ensure within grid bounds
        c_x1 = max(0, min(self.grid_w - 1, c_x1))
        c_y1 = max(0, min(self.grid_h - 1, c_y1))
        c_x2 = max(0, min(self.grid_w - 1, c_x2))
        c_y2 = max(0, min(self.grid_h - 1, c_y2))
        
        return c_x1, c_y1, c_x2, c_y2
    
    def insert(self, box: np.ndarray, idx: int) -> None:
        """
        Insert a box into the grid.
        
        Args:
            box: [x1, y1, x2, y2] in pixels
            idx: Index/ID of this box for later retrieval
        """
        c_x1, c_y1, c_x2, c_y2 = self._box_to_cells(box)
        
        # Add to all cells the box overlaps
        for cy in range(c_y1, c_y2 + 1):
            for cx in range(c_x1, c_x2 + 1):
                self.cells[(cx, cy)].append(idx)
    
    def query_neighbors(self, box: np.ndarray) -> Set[int]:
        """
        Find all boxes that spatially overlap with the query box's cells.
        
        Returns only candidates - still need IoU check, but reduces from
        O(N) to O(k) where k is typically 10-20 neighbors.
        
        Args:
            box: [x1, y1, x2, y2] in pixels
            
        Returns:
            Set of box indices that may overlap (includes box itself if inserted)
        """
        c_x1, c_y1, c_x2, c_y2 = self._box_to_cells(box)
        
        neighbors = set()
        for cy in range(c_y1, c_y2 + 1):
            for cx in range(c_x1, c_x2 + 1):
                neighbors.update(self.cells.get((cx, cy), []))
        
        return neighbors
    
    def clear(self) -> None:
        """Clear all indexed boxes."""
        self.cells.clear()
    
    def stats(self) -> Dict[str, float]:
        """
        Return grid statistics for tuning.
        
        Returns:
            Dict with:
                - num_cells_used: Number of non-empty cells
                - avg_boxes_per_cell: Average occupancy
                - max_boxes_per_cell: Worst case occupancy
                - load_factor: Fraction of cells occupied
        """
        if not self.cells:
            return {
                "num_cells_used": 0,
                "avg_boxes_per_cell": 0.0,
                "max_boxes_per_cell": 0,
                "load_factor": 0.0,
            }
        
        num_cells_used = len(self.cells)
        total_entries = sum(len(v) for v in self.cells.values())
        max_boxes = max(len(v) for v in self.cells.values())
        total_cells = self.grid_w * self.grid_h
        
        return {
            "num_cells_used": num_cells_used,
            "avg_boxes_per_cell": total_entries / num_cells_used,
            "max_boxes_per_cell": max_boxes,
            "load_factor": num_cells_used / max(total_cells, 1),
        }


def compute_iou_pairwise(boxes: np.ndarray, indices1: List[int], indices2: List[int]) -> np.ndarray:
    """
    Compute IoU only for specified pairs (optimized for spatial neighbors).
    
    Args:
        boxes: (N, 4) array of all boxes [x1, y1, x2, y2]
        indices1: List of box indices (query boxes)
        indices2: List of box indices (candidate neighbors)
        
    Returns:
        (len(indices1), len(indices2)) IoU matrix
    """
    if not indices1 or not indices2:
        return np.zeros((len(indices1), len(indices2)), dtype=np.float32)
    
    boxes1 = boxes[indices1]
    boxes2 = boxes[indices2]
    
    # Vectorized IoU computation
    x1_max = np.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
    y1_max = np.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
    x2_min = np.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
    y2_min = np.minimum(boxes1[:, None, 3], boxes2[None, :, 3])
    
    inter_w = np.maximum(0.0, x2_min - x1_max)
    inter_h = np.maximum(0.0, y2_min - y1_max)
    inter_area = inter_w * inter_h
    
    area1 = np.maximum(0.0, boxes1[:, 2] - boxes1[:, 0]) * np.maximum(0.0, boxes1[:, 3] - boxes1[:, 1])
    area2 = np.maximum(0.0, boxes2[:, 2] - boxes2[:, 0]) * np.maximum(0.0, boxes2[:, 3] - boxes2[:, 1])
    
    union_area = area1[:, None] + area2[None, :] - inter_area
    return inter_area / np.maximum(union_area, 1e-6)


__all__ = ["SpatialHashGrid", "compute_iou_pairwise"]
