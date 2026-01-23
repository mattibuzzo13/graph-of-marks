"""
Image Visualization Module

This module provides comprehensive visualization capabilities for object detection,
segmentation, and relationship extraction results. It supports:

- Bounding box and segmentation mask rendering
- Relationship arrows and labels
- Multiple output formats (PNG, JPG, SVG)
- Transparent background mode for publication-ready overlays
- Granular control over visualization elements
- Optimized rendering with vectorized operations

The main class is Visualizer, which takes a VisualizerConfig and provides
methods to draw annotated images with various combinations of visual elements.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import Bbox
from PIL import Image

# Optional dependencies for advanced features
try:
    import cv2  # type: ignore
except Exception:
    cv2 = None  # Morphological operations and contour detection unavailable

try:
    from adjustText import adjust_text  # type: ignore
except Exception:
    adjust_text = None  # Automatic label positioning unavailable

# Color utilities with fallback
try:
    from gom.utils.colors import color_for_label, text_color_for_bg  # type: ignore
except Exception:
    # Fallback to basic color cycling if advanced color utilities unavailable
    from gom.utils.colors import ColorCycler, text_color_for_bg  # type: ignore
    _color_cycler = ColorCycler()

    def color_for_label(
        label: str,
        idx: int = 0,
        sat_boost: float = 1.3,
        val_boost: float = 1.15,
        cache: Optional[dict] = None,
    ) -> str:
        """Fallback color assignment using basic color cycling."""
        return _color_cycler.color_for_label(label)

# Vectorized rendering optimizations (optional performance enhancement)
try:
    from gom.viz.rendering_opt import (
        BatchTextRenderer,
        VectorizedMaskRenderer,
    )
    RENDERING_OPT_AVAILABLE = True  # Optimized rendering paths available

except ImportError:
    RENDERING_OPT_AVAILABLE = False  # Fall back to standard matplotlib rendering


@dataclass
class VisualizerConfig:
    """
    Configuration for visualization rendering.
    
    This dataclass controls all aspects of visualization output including:
    - Which elements to display (labels, relationships, segmentation, etc.)
    - Visual styling (colors, fonts, line widths, transparency)
    - Performance optimizations (vectorized rendering, batched operations)
    - Output format options
    
    Attributes:
        Content Control:
            display_labels: Show object labels (default: True)
            display_relationships: Show relationship arrows between objects (default: True)
            display_relation_labels: Show text labels on relationship arrows (default: True)
            display_legend: Show color legend (default: True)
            show_segmentation: Show segmentation masks (default: True)
            fill_segmentation: Fill masks vs outline only (default: True)
            show_bboxes: Show bounding boxes (default: True)
            
        Performance Optimizations:
            use_vectorized_masks: Use optimized mask blending, 2-2.5x speedup (default: True)
            use_batch_text_renderer: Batch text rendering, 20-30% speedup (default: True)
            
        Typography and Styling:
            obj_fontsize_inside: Font size for labels inside boxes (default: 12)
            obj_fontsize_outside: Font size for labels outside boxes (default: 12)
            rel_fontsize: Font size for relationship labels (default: 10)
            legend_fontsize: Font size for legend text (default: 8)
            seg_fill_alpha: Transparency of segmentation fills 0.0-1.0 (default: 0.75)
            bbox_linewidth: Line width for bounding boxes in points (default: 2.0)
            rel_arrow_linewidth: Line width for relationship arrows (default: 2.5)
            rel_arrow_mutation_scale: Arrow head size scaling factor (default: 26.0)
            label_bbox_linewidth: Line width for object label borders (default: 3.0)
            relation_label_bbox_linewidth: Line width for relation label borders (default: 3.0)
            connector_linewidth: Line width for label-to-object connectors (default: 1.5)
            
        Relationship Processing:
            filter_redundant_relations: Remove duplicate relationships (default: True)
            cap_relations_per_object: Limit relationships per object (default: False)
            max_relations_per_object: Maximum relationships per object (default: 1)
            min_relations_per_object: Minimum relationships per object (default: 1)
            
        Label Configuration:
            label_mode: Label display mode - "original", "numeric", or "alphabetic" (default: "original")
            show_confidence: Append confidence scores to labels (default: False)
            
        Inside Label Placement:
            min_area_ratio_inside: Minimum object area ratio to allow inside labels (default: 0.006)
            inside_label_margin_px: Pixel margin around inside labels (default: 6)
            min_solidity_inside: Minimum mask solidity for inside placement (default: 0.45)
            measure_text_with_renderer: Use renderer for accurate text measurement (default: True)
            
        Overlap Resolution:
            resolve_overlaps: Enable automatic overlap resolution (default: True)
            adjust_text_profile: Overlap resolution aggressiveness - "dense" or default (default: "dense")
            micro_push_iters: Micro-adjustment iterations for fine-tuning (default: 100)
            
        Depth and 3D Rendering:
            use_depth_ordering: Sort objects by depth for proper occlusion (default: True)
            depth_key: Metadata key for depth information (default: "depth")
            
        Relation Label Positioning:
            relation_label_placement: Label position on arrow - "midpoint" (default: "midpoint")
            relation_label_offset_px: Pixel offset from arrow path (default: 10.0)
            relation_label_max_dist_px: Maximum label movement from arrow (default: 50.0)
            
        Color Enhancement:
            color_sat_boost: Saturation boost multiplier for colors (default: 1.30)
            color_val_boost: Value/brightness boost multiplier (default: 1.15)
            
        Spatial Relationship Rendering:
            on_top_gap_px: Vertical gap for "on_top_of" relationships (default: 8)
            on_top_horiz_overlap: Required horizontal overlap ratio (default: 0.35)

        Auto-Scaling:
            auto_scale_styles: Enable automatic font/arrow scaling for image size (default: True)
            style_ref_px: Reference image size for scaling calculations (default: 1000)
            style_ref_dpi: Reference DPI for scaling calculations (default: 100)
            style_scale_min: Minimum scale factor (default: 0.5)
            style_scale_max: Maximum scale factor (default: 2.0)
    """
    
    # Content control flags
    display_labels: bool = True
    display_relationships: bool = True
    display_relation_labels: bool = True
    display_legend: bool = False

    # Object rendering options
    show_segmentation: bool = True
    fill_segmentation: bool = True
    show_bboxes: bool = True

    # Performance optimization flags
    use_vectorized_masks: bool = True
    use_batch_text_renderer: bool = True

    # Typography and visual style
    font_family: str = "DejaVu Sans"  # Font family for all text elements
    obj_fontsize_inside: int = 9
    obj_fontsize_outside: int = 10
    rel_fontsize: int = 8
    legend_fontsize: int = 8
    seg_fill_alpha: float = 0.25
    bbox_linewidth: float = 2.0
    rel_arrow_linewidth: float = 2.0
    rel_arrow_mutation_scale: float = 26.0
    label_bbox_linewidth: float = 3.0
    relation_label_bbox_linewidth: float = 3.0
    connector_linewidth: float = 1.5

    # Relation post-processing
    filter_redundant_relations: bool = True
    cap_relations_per_object: bool = False
    max_relations_per_object: int = 1
    min_relations_per_object: int = 1

    # Label content/mode
    label_mode: str = "original"
    show_confidence: bool = False

    # Inside-placement
    min_area_ratio_inside: float = 0.006
    inside_label_margin_px: int = 6
    min_solidity_inside: float = 0.45
    measure_text_with_renderer: bool = True
    avoid_object_occlusion: bool = True
    allow_inside_large_objects: bool = True
    large_object_area_ratio: float = 0.04
    large_object_min_side_px: int = 120

    # Overlap resolution
    resolve_overlaps: bool = True
    adjust_text_profile: str = "dense"
    micro_push_iters: int = 100
    obj_label_max_dist_px: float = 60.0

    # Depth handling
    use_depth_ordering: bool = True
    depth_key: str = "depth"

    # Relation label placement
    relation_label_placement: str = "midpoint"
    relation_label_offset_px: float = 10.0
    relation_label_max_dist_px: float = 20.0

    # Global color tweaks
    color_sat_boost: float = 1.1
    color_val_boost: float = 1.1

    # Special knobs
    on_top_gap_px: int = 8
    on_top_horiz_overlap: float = 0.35

    # Auto-scaling for different image sizes/resolutions
    auto_scale_styles: bool = True
    style_ref_px: int = 1000  # Reference image size for scaling
    style_ref_dpi: int = 100  # Reference DPI
    style_scale_min: float = 0.5  # Minimum scale factor
    style_scale_max: float = 2.0  # Maximum scale factor


# ===============================================================
# VISUALIZER
# ===============================================================

class Visualizer:
    """
    Comprehensive visualization engine for object detection, segmentation, and relationships.
    
    This class provides a flexible rendering pipeline that combines:
    - Object detection bounding boxes with customizable styling
    - Instance segmentation masks with transparent overlays
    - Spatial and semantic relationships as directed arrows
    - Intelligent label placement (inside objects when possible, outside with connectors otherwise)
    - Optional legend generation showing all detected classes
    
    The visualizer supports both standard rendering with background images and transparent
    rendering for compositing. It includes performance optimizations like vectorized mask
    blending and batched text rendering.
    
    Typical Usage:
        >>> viz = Visualizer(VisualizerConfig(show_segmentation=True))
        >>> fig, ax = viz.draw(
        ...     image=img,
        ...     boxes=[[x1, y1, x2, y2], ...],
        ...     labels=["person", "car", ...],
        ...     scores=[0.95, 0.87, ...],
        ...     relationships=[{"subject": 0, "object": 1, "predicate": "next to"}],
        ...     masks=[mask_array_1, mask_array_2, ...],
        ... )
        >>> plt.show()
    
    Attributes:
        cfg: Configuration object controlling all visualization parameters
        SPATIAL_KEYS: Tuple of recognized spatial relationship predicates
    """

    SPATIAL_KEYS = (
        "left_of",
        "right_of",
        "above",
        "below",
        "on_top_of",
        "under",
        "in_front_of",
        "behind",
    )

    def __init__(self, config: Optional[VisualizerConfig] = None) -> None:
        """
        Initialize the visualizer with optional configuration.
        
        Args:
            config: VisualizerConfig instance with rendering parameters.
                   If None, uses default configuration.
        """
        self.cfg = config or VisualizerConfig()
        self._label2color_cache: Dict[str, str] = {}
        self._draw_background = True  # Track if we're drawing background

    # -----------------------------------------------------------
    # PUBLIC ENTRY POINT
    # -----------------------------------------------------------
    def draw(
        self,
        image: Image.Image,
        boxes: Sequence[Sequence[float]],
        labels: Sequence[str],
        scores: Sequence[float],
        relationships: Sequence[Dict[str, Any]],
        masks: Optional[Sequence[np.ndarray | Dict[str, Any]]] = None,
        save_path: Optional[str] = None,
        draw_background: bool = True,
        bg_color: Tuple[float, float, float, float] = (1, 1, 1, 0),
        dpi: int = 800,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Main entry point for rendering complete scene visualization.
        
        This method orchestrates the full rendering pipeline:
        1. Color assignment for each object class
        2. Canvas creation (with or without background image)
        3. Segmentation mask rendering (if enabled and available)
        4. Bounding box rendering (if enabled)
        5. Label placement with overlap resolution
        6. Relationship arrow rendering
        7. Legend generation
        8. Optional file saving
        
        Args:
            image: Input PIL Image to use as background (if draw_background=True)
            boxes: List of bounding boxes in [x1, y1, x2, y2] format (unnormalized pixel coordinates)
            labels: List of class labels for each detection
            scores: List of confidence scores for each detection (0.0-1.0)
            relationships: List of relationship dictionaries with keys:
                          - "subject": index of subject object
                          - "object": index of object object  
                          - "predicate": relationship type string
            masks: Optional list of segmentation masks. Each mask can be:
                  - numpy array (H, W) with binary mask
                  - dict with "segmentation" key containing the mask array
            save_path: Optional file path to save the rendered figure
            draw_background: If True, render image as background. If False, transparent background.
            bg_color: RGBA background color when draw_background=False (default: transparent white)
            dpi: Dots per inch for output resolution
        
        Returns:
            Tuple of (matplotlib Figure, matplotlib Axes) for further customization or display
        
        Notes:
            - Label placement uses intelligent inside/outside logic based on object area
            - Relationship arrows are rendered with optional labels at midpoints
            - Color assignment is consistent across all visualization elements
            - Performance optimizations activated via VisualizerConfig flags
        """
        # Store draw_background state for use in drawing methods
        self._draw_background = draw_background

        # Auto-scale fonts/arrows based on image size and resolution.
        restore_cfg = None
        if self.cfg.auto_scale_styles:
            W, H = image.size
            scale = max(W, H) / float(self.cfg.style_ref_px)
            scale *= float(dpi) / float(self.cfg.style_ref_dpi)
            scale = max(self.cfg.style_scale_min, min(self.cfg.style_scale_max, scale))
            restore_cfg = {
                "obj_fontsize_inside": self.cfg.obj_fontsize_inside,
                "obj_fontsize_outside": self.cfg.obj_fontsize_outside,
                "rel_fontsize": self.cfg.rel_fontsize,
                "legend_fontsize": self.cfg.legend_fontsize,
                "rel_arrow_linewidth": self.cfg.rel_arrow_linewidth,
                "rel_arrow_mutation_scale": self.cfg.rel_arrow_mutation_scale,
            }
            self.cfg.obj_fontsize_inside = max(8, int(round(self.cfg.obj_fontsize_inside * scale)))
            self.cfg.obj_fontsize_outside = max(8, int(round(self.cfg.obj_fontsize_outside * scale)))
            self.cfg.rel_fontsize = max(6, int(round(self.cfg.rel_fontsize * scale)))
            self.cfg.legend_fontsize = max(6, int(round(self.cfg.legend_fontsize * scale)))
            self.cfg.rel_arrow_linewidth = max(0.5, float(self.cfg.rel_arrow_linewidth * scale))
            self.cfg.rel_arrow_mutation_scale = max(8.0, float(self.cfg.rel_arrow_mutation_scale * scale))
        
        # Assign colors first (needed for all rendering paths)
        colors = self._assign_colors(labels)
        
        # When saving without background, use special rendering mode
        # that creates a clean canvas without the original image
        if not draw_background:
            # Check if we have something to draw (masks or relationships)
            has_masks = masks and self.cfg.show_segmentation
            has_relations = relationships and self.cfg.display_relationships
            
            if has_masks or has_relations:
                return self._draw_without_background(
                    image, boxes, labels, scores, relationships, masks, colors,
                    save_path, bg_color, dpi
                )
        
        fig, ax = self._create_canvas(image, draw_background, bg_color)

        # 1) preprocess relations
        relations = self._preprocess_relations(relationships, boxes)

        # 3) draw passes
        self._draw_objects(ax, boxes, masks, labels, scores, colors, image)
        avoid_objects = self._build_avoid_patches(ax, boxes) if self.cfg.avoid_object_occlusion else []
        
        # Draw object labels first and collect them
        obj_texts = self._draw_labels(ax, boxes, labels, scores, masks, colors, image, avoid_objects=avoid_objects)
        
        # Draw relationships and avoid object labels
        self._draw_relationships(ax, relations, boxes, colors, obj_texts, avoid_objects=avoid_objects)
        
        self._draw_legend(ax, labels, colors)

        # 4) finalize
        self._finalize_figure(fig, save_path, draw_background, bg_color, dpi)
        if restore_cfg is not None:
            for k, v in restore_cfg.items():
                setattr(self.cfg, k, v)
        return fig, ax

    @staticmethod
    def draw_detections_only(
        image: Image.Image,
        boxes: Sequence[Sequence[float]],
        labels: Sequence[str],
        scores: Sequence[float],
        save_path: str,
    ) -> None:
        """
        Render an image with only detection bounding boxes + labels (no masks, no relations).
        
        Args:
            image: Input PIL Image
            boxes: List of bounding boxes
            labels: List of labels
            scores: List of scores
            save_path: Path to save the output image
        """
        cfg = VisualizerConfig()
        cfg.display_labels = True
        cfg.display_relationships = False
        cfg.display_relation_labels = False
        cfg.show_segmentation = False
        cfg.show_bboxes = True
        cfg.display_legend = False

        viz = Visualizer(cfg)
        # Ensure parent directory exists
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        viz.draw(
            image=image,
            boxes=boxes,
            labels=labels,
            scores=scores,
            relationships=[],   # no relations
            masks=None,
            save_path=str(save_path),
            draw_background=True,
            bg_color=(1, 1, 1, 1),
        )

    @staticmethod
    def draw_segmentation_only(
        image: Image.Image,
        boxes: Sequence[Sequence[float]],
        labels: Sequence[str],
        scores: Sequence[float],
        masks: Sequence[np.ndarray],
        save_path: str,
    ) -> None:
        """
        Render segmentation masks (and optional boxes/labels) for a set of objects.
        
        Args:
            image: Input PIL Image
            boxes: List of bounding boxes
            labels: List of labels
            scores: List of scores
            masks: List of segmentation masks
            save_path: Path to save the output image
        """
        cfg = VisualizerConfig()
        cfg.display_labels = True
        cfg.display_relationships = False
        cfg.display_relation_labels = False
        cfg.show_segmentation = True
        cfg.fill_segmentation = True
        cfg.show_bboxes = True
        cfg.display_legend = False
        cfg.seg_fill_alpha = 0.25

        viz = Visualizer(cfg)
        # Ensure parent directory exists
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        viz.draw(
            image=image,
            boxes=boxes,
            labels=labels,
            scores=scores,
            relationships=[],
            masks=masks,
            save_path=str(save_path),
            draw_background=True,
            bg_color=(1, 1, 1, 1),
        )

    # -----------------------------------------------------------
    # CANVAS CREATION AND FINALIZATION
    # -----------------------------------------------------------
    def _create_canvas(
        self, image: Image.Image, draw_background: bool, bg_color: Tuple[float, float, float, float]
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create matplotlib figure and axes with appropriate background settings.
        
        Args:
            image: PIL Image used for dimensions and optional background
            draw_background: Whether to display the image as background
            bg_color: RGBA tuple for background color when image not shown
        
        Returns:
            Tuple of (Figure, Axes) ready for drawing
        
        Notes:
            - Figure size is set to match image dimensions at 100 DPI
            - Transparent backgrounds achieved by setting alpha=0 on figure and axes
            - Axes limits match image dimensions in pixel coordinates
        """
        W, H = image.size
        
        fig, ax = plt.subplots(figsize=(W / 100, H / 100))
        ax.axis("off")
        if draw_background:
            ax.imshow(image)
        else:
            ax.set_xlim(0, W)
            ax.set_ylim(H, 0)
            # Set transparent background for both figure and axes
            if len(bg_color) == 4 and bg_color[3] == 0:
                fig.patch.set_alpha(0)
                ax.patch.set_alpha(0)
                ax.set_facecolor('none')
            else:
                ax.set_facecolor(bg_color)
        return fig, ax

    def _finalize_figure(
        self, fig: plt.Figure, save_path: Optional[str], draw_background: bool,
        bg_color: Tuple[float, float, float, float], dpi: int
    ) -> None:
        """
        Apply final layout and optionally save figure to file.
        
        Args:
            fig: Matplotlib figure to finalize
            save_path: Optional path to save figure. If None, displays interactively.
            draw_background: Whether background was drawn (affects transparency)
            bg_color: Background color used (affects transparency detection)
            dpi: Dots per inch for output resolution
        
        Notes:
            - tight_layout() removes excess whitespace
            - Transparency enabled for PNG/SVG when bg_color alpha is 0
            - SVG gets additional 'facecolor=none' for proper transparency
            - Figure auto-closed after saving to free memory
        """
        fig.tight_layout()
        if save_path:
            # Determine if we need transparency
            is_transparent = not draw_background and (len(bg_color) == 4 and bg_color[3] == 0)
            
            # For SVG, also set facecolor to 'none' for true transparency
            kwargs = {
                'bbox_inches': 'tight',
                'transparent': is_transparent,
                'dpi': dpi,
            }
            
            if save_path.endswith('.svg') and is_transparent:
                kwargs['facecolor'] = 'none'
            
            fig.savefig(save_path, **kwargs)
            plt.close(fig)
        else:
            plt.show()

    # -----------------------------------------------------------
    # RELATIONSHIP PREPROCESSING AND COLOR ASSIGNMENT
    # -----------------------------------------------------------
    def _preprocess_relations(
        self, relationships: Sequence[Dict[str, Any]], boxes: Sequence[Sequence[float]]
    ) -> List[Dict[str, Any]]:
        """
        Apply filtering and capping to relationships before rendering.
        
        Args:
            relationships: Raw list of relationship dictionaries
            boxes: Bounding boxes used for spatial filtering
        
        Returns:
            Filtered and capped list of relationships to render
        
        Notes:
            - Redundant filtering removes duplicate/inverse spatial relationships
            - Capping limits visual clutter by restricting relationships per object
            - Both operations controlled by VisualizerConfig flags
        """
        rels = list(relationships)
        if self.cfg.filter_redundant_relations:
            rels = self._filter_redundant_relations(rels)
        if self.cfg.cap_relations_per_object:
            rels = self._cap_relations_per_object(rels, boxes)
        return rels

    def _assign_colors(self, labels: Sequence[str]) -> List[str]:
        """
        Assign consistent colors to object labels.
        
        Args:
            labels: List of object class labels
        
        Returns:
            List of hex color strings, one per label
        
        Notes:
            - Color assignment is deterministic based on label text
            - Indexed labels (e.g., "person_1", "person_2") share base color
            - Results cached for consistency across multiple draw calls
        """
        return [self._pick_color(lbl, i) for i, lbl in enumerate(labels)]

    def _pick_color(self, label: str, idx: int) -> str:
        """
        Select or generate a color for a specific label.
        
        Args:
            label: Object class label (may include numeric suffix)
            idx: Index of this object in the detection list
        
        Returns:
            Hex color string (e.g., "#FF5733")
        
        Notes:
            - Strips numeric suffixes to share colors across instances
            - Caches colors for consistency within and across frames
            - Falls back to index-based color generation if not cached
        """
        base = label.rsplit("_", 1)[0].lower()
        if base in self._label2color_cache:
            return self._label2color_cache[base]
        col = color_for_label(
            base,
            idx=idx,
            sat_boost=self.cfg.color_sat_boost,
            val_boost=self.cfg.color_val_boost,
            cache=self._label2color_cache,
        )
        self._label2color_cache[base] = col
        return col

    # ===========================================================
    # OBJECT RENDERING (BOUNDING BOXES AND SEGMENTATION MASKS)
    # ===========================================================
    def _draw_objects(
        self,
        ax: plt.Axes,
        boxes: Sequence[Sequence[float]],
        masks: Optional[Sequence[np.ndarray | Dict[str, Any]]],
        labels: Sequence[str],
        scores: Sequence[float],
        colors: Sequence[str],
        image: Image.Image,
    ) -> None:
        """
        Render bounding boxes and segmentation masks for all detected objects.
        
        This method implements two rendering modes:
        1. Vectorized rendering: Blends all masks in one operation (2-2.5x faster)
           then draws contours/boxes on top. Requires rendering_opt module.
        2. Standard rendering: Draws each mask individually with matplotlib patches.
        
        Objects are rendered in depth order (back to front) for proper occlusion.
        
        Args:
            ax: Matplotlib axes to draw on
            boxes: List of bounding boxes [x1, y1, x2, y2]
            masks: Optional list of segmentation masks
            labels: Object class labels (may contain depth suffixes)
            scores: Detection confidence scores
            colors: Pre-assigned colors for each object
            image: Original PIL image for background reference
        
        Notes:
            - Depth ordering extracted from label suffixes (e.g., "person_1" has depth 1)
            - Vectorized rendering provides significant performance improvement
            - Falls back to standard rendering if optimization unavailable
            - Segmentation fill transparency controlled by cfg.seg_fill_alpha
        """
        cfg = self.cfg
        if not boxes:
            return

        # Sort objects by depth for proper rendering order (back to front)
        ordered = []

        for i, box in enumerate(boxes):
            depth_idx = self._extract_depth_index(labels[i], i)
            ordered.append((i, box, depth_idx))
        ordered.sort(key=lambda x: x[2], reverse=True)

        # Vectorized rendering path: blend all masks at once for performance
        if cfg.show_segmentation and cfg.use_vectorized_masks and RENDERING_OPT_AVAILABLE and masks:
            # Collect masks and colors matching original order
            masks_list = []
            colors_list = []
            for (original_idx, box, depth) in ordered:
                m = self._get_mask_for_index(original_idx, masks)
                if m is not None and m.get("segmentation") is not None:
                    masks_list.append(m["segmentation"])
                    colors_list.append(colors[original_idx])
                else:
                    # Keep alignment with None for objects without mask
                    masks_list.append(None)
                    colors_list.append(colors[original_idx])

            # Convert PIL image to numpy background for blending
            try:
                bg_np = np.asarray(image)
            except Exception:
                bg_np = None

            blended = VectorizedMaskRenderer.blend_multiple_masks(
                masks=masks_list,
                colors=colors_list,
                background=bg_np,
                alpha=cfg.seg_fill_alpha,
            )
            ax.imshow(blended, extent=(0, blended.shape[1], blended.shape[0], 0), zorder=1)

            # Draw contours or bbox outlines for each object on top of blended masks
            for (original_idx, box, depth) in ordered:
                color = colors[original_idx]
                x1, y1, x2, y2 = map(int, box[:4])
                mask_info = self._get_mask_for_index(original_idx, masks)
                z_order = 2 + (len(boxes) - min(depth, len(boxes))) * 0.1
                if mask_info is not None and mask_info.get("segmentation") is not None:
                    # Draw contour stroke on top of blended image
                    self._draw_segmentation(ax, mask_info["segmentation"], color, cfg.bbox_linewidth, z_order)
                elif cfg.show_bboxes:
                    self._draw_bbox(ax, x1, y1, x2, y2, color, cfg.bbox_linewidth, z_order)
        else:
            # Standard rendering path: draw each mask individually
            for idx, box, depth in ordered:
                color = colors[idx]
                x1, y1, x2, y2 = map(int, box[:4])
                mask_info = self._get_mask_for_index(idx, masks)
                z_order = 1 + (len(boxes) - min(depth, len(boxes))) * 0.1

                if cfg.show_segmentation and mask_info is not None and mask_info.get("segmentation") is not None:
                    self._draw_segmentation(ax, mask_info["segmentation"], color, cfg.bbox_linewidth, z_order)
                elif cfg.show_bboxes:
                    self._draw_bbox(ax, x1, y1, x2, y2, color, cfg.bbox_linewidth, z_order)

    def _draw_bbox(
        self, ax: plt.Axes, x1: int, y1: int, x2: int, y2: int, color: str, linewidth: float, zorder: float = 2
    ) -> None:
        """
        Draw a bounding box rectangle.
        
        Args:
            ax: Matplotlib axes to draw on
            x1, y1: Top-left corner coordinates
            x2, y2: Bottom-right corner coordinates
            color: Hex color string for box edge
            linewidth: Width of box edge in points
            zorder: Rendering layer (higher values appear on top)
        """
        rect = patches.Rectangle(
            (x1, y1),
            max(1, x2 - x1),
            max(1, y2 - y1),
            linewidth=linewidth,
            edgecolor=color,
            facecolor="none",
            zorder=zorder,
        )
        ax.add_patch(rect)

    def _draw_segmentation(
        self, ax: plt.Axes, mask: np.ndarray, color: str, linewidth: float, zorder: float = 2
    ) -> None:
        """
        Draw segmentation mask with filled interior and opaque border.
        
        Uses OpenCV to extract contours for precise boundary rendering. Falls back
        to simple imshow if OpenCV unavailable.
        
        Args:
            ax: Matplotlib axes to draw on
            mask: Binary mask array (H, W) with 0/1 or 0/255 values
            color: Hex color string for fill and border
            linewidth: Width of contour border in points
            zorder: Rendering layer (higher values appear on top)
        
        Notes:
            - Fill transparency controlled by cfg.seg_fill_alpha
            - Border always opaque (alpha=1.0) for clarity
            - Handles both boolean and uint8 masks
            - Skips degenerate contours (< 3 points)
        """
        # Use seg_fill_alpha from config (respects user setting)
        alpha_to_use = self.cfg.seg_fill_alpha
        
        if cv2 is None:
            # Fallback rendering without contours
            ax.imshow(mask.astype(float), alpha=alpha_to_use, extent=(0, mask.shape[1], mask.shape[0], 0), 
                     cmap='Greys', vmin=0, vmax=1)
            return

        # Ensure mask is uint8 with 0-255 range
        mask_uint8 = (mask.astype(np.uint8) * 255) if mask.dtype != np.uint8 else mask.copy()
        if mask_uint8.max() == 1:
            mask_uint8 *= 255

        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return

        for cnt in contours:
            cnt = cnt.squeeze()
            if cnt.ndim != 2 or len(cnt) < 3:
                continue
            # Fill mask region with transparency
            ax.fill(cnt[:, 0], cnt[:, 1], color=color, alpha=alpha_to_use, zorder=zorder)
            # Draw opaque border for definition
            ax.plot(cnt[:, 0], cnt[:, 1], color=color, linewidth=linewidth, alpha=1.0, zorder=zorder + 0.1)

    # ===========================================================
    # RELATIONSHIP RENDERING
    # ===========================================================
    def _draw_relationships(
        self,
        ax: plt.Axes,
        relationships: Sequence[Dict[str, Any]],
        boxes: Sequence[Sequence[float]],
        colors: Sequence[str],
        object_texts: List[Any] = None,
        *,
        avoid_objects: Sequence[Any] = (),
    ) -> None:
        """
        Render relationship arrows and labels between objects.
        
        This method creates curved arrows connecting related objects and places
        labels at optimal positions along the arrows. It handles multiple relationships
        between the same pair by adjusting curvature and avoids overlapping with
        object labels.
        
        Args:
            ax: Matplotlib axes to draw on
            relationships: List of relationship dicts with keys:
                          - "src_idx": source object index
                          - "tgt_idx": target object index
                          - "relation": relationship type string
            boxes: Bounding boxes for computing object centers
            colors: Object colors (arrow uses source object's color)
            object_texts: Optional list of object label text objects to avoid
            avoid_objects: Optional patches (e.g., object boxes) to avoid
        
        Notes:
            - Arrows are curved to avoid overlapping parallel relationships
            - Multiple arrows between same pair get increasing curvature
            - Labels positioned at arc midpoint with offset
            - Arrow endpoints shrunk to avoid covering object centers
            - All arrows rendered with full opacity for visibility
        """
        cfg = self.cfg
        if not cfg.display_relationships or not relationships:
            return
        
        if object_texts is None:
            object_texts = []

        # Compute object centers for arrow endpoints
        centers = [
            ((float(b[0]) + float(b[2])) / 2.0, (float(b[1]) + float(b[3])) / 2.0)
            for b in boxes
        ]

        arrow_patches: List[patches.FancyArrowPatch] = []
        rel_texts: List[Any] = []
        rel_label_anchors: List[Tuple[float, float]] = []
        rel_label_dirs: List[Tuple[float, float]] = []

        # Track multiple arrows between same pair for curvature adjustment
        arrow_counts: Dict[Tuple[int, int], int] = {}

        for rel in relationships:
            src, tgt = int(rel["src_idx"]), int(rel["tgt_idx"])
            if not (0 <= src < len(centers) and 0 <= tgt < len(centers)):
                continue

            start, end = centers[src], centers[tgt]
            relation_name = str(rel.get("relation", "")).lower()
            color = colors[src]

            # Adjust curvature for multiple arrows between same objects
            arrow_counts[(src, tgt)] = arrow_counts.get((src, tgt), 0) + 1
            curvature = 0.2 + 0.1 * (arrow_counts[(src, tgt)] - 1)

            # Shrink arrow endpoints to avoid covering object centers
            p0, p1 = self._shrink_segment_px(start, end, 6, ax)
            arrow = patches.FancyArrowPatch(
                p0,
                p1,
                arrowstyle="->",
                color=color,
                alpha=1.0,
                linewidth=cfg.rel_arrow_linewidth,
                connectionstyle=f"arc3,rad={curvature}",
                mutation_scale=cfg.rel_arrow_mutation_scale,
                zorder=4,
            )
            ax.add_patch(arrow)
            arrow_patches.append(arrow)

            if cfg.display_relation_labels:
                readable = self._humanize_relation(relation_name)
                # Compute initial position at arc midpoint
                pos = self._get_optimal_relation_label_position(ax, arrow, readable)
                # Offset labels along the arrow normal to reduce collisions.
                count = arrow_counts[(src, tgt)] - 1
                if cfg.relation_label_offset_px and count > 0:
                    try:
                        to_px = ax.transData.transform
                        to_data = ax.transData.inverted().transform
                        p0_px = np.array(to_px(p0))
                        p1_px = np.array(to_px(p1))
                        v_px = p1_px - p0_px
                        v_norm = np.hypot(v_px[0], v_px[1])
                        if v_norm > 1e-6:
                            n_px = np.array([-v_px[1], v_px[0]]) / v_norm
                            sign = 1.0 if (count % 2 == 1) else -1.0
                            mag = float(cfg.relation_label_offset_px) * (1 + (count // 2))
                            pos_px = np.array(to_px(pos)) + n_px * mag * sign
                            pos = tuple(to_data(pos_px))
                    except Exception:
                        pass
                t = ax.text(
                    pos[0],
                    pos[1],
                    readable,
                    fontsize=cfg.rel_fontsize,
                    fontfamily=cfg.font_family,
                    ha="center",
                    va="center",
                    color="black",
                    bbox=dict(
                        boxstyle="round,pad=0.25",
                        facecolor="white",
                        alpha=0.95,
                        edgecolor=color,
                        linewidth=cfg.relation_label_bbox_linewidth,
                    ),
                    zorder=5,
                )
                try:
                    arrow_len_px = self._get_arrow_length_px(ax, arrow)
                    max_dist = min(
                        float(cfg.relation_label_max_dist_px),
                        max(12.0, arrow_len_px * 0.4),
                    )
                    t._gom_max_dist_px = max_dist
                except Exception:
                    pass
                rel_texts.append(t)
                rel_label_anchors.append(pos)
                rel_label_dirs.append((p1[0] - p0[0], p1[1] - p0[1]))

        # Resolve overlaps between relationship labels, avoiding object labels
        if cfg.resolve_overlaps and rel_texts:
            fig = ax.figure
            fig.canvas.draw()
            # Pass object_texts as fixed_texts to avoid overlapping them
            self._resolve_relation_vs_relation_overlaps(
                ax,
                rel_texts,
                arrow_patches,
                cfg.relation_label_max_dist_px,
                anchors=rel_label_anchors,
                arrow_dirs=rel_label_dirs,
                fixed_texts=object_texts,
                avoid_objects=avoid_objects,
            )
            # Second pass: adjustText + clamp to keep labels near arrows.
            self._resolve_overlaps(
                ax,
                movable_texts=rel_texts,
                movable_anchors=rel_label_anchors,
                fixed_texts=object_texts,
                arrows=arrow_patches,
            )
            if cfg.relation_label_max_dist_px and cfg.relation_label_max_dist_px > 0:
                to_px = ax.transData.transform
                to_data = ax.transData.inverted().transform
                default_max = float(cfg.relation_label_max_dist_px)
                for i, t in enumerate(rel_texts):
                    if i >= len(rel_label_anchors):
                        continue
                    max_dist = float(getattr(t, "_gom_max_dist_px", default_max))
                    anchor_px = np.array(to_px(rel_label_anchors[i]))
                    pos_px = np.array(to_px(t.get_position()))
                    delta = pos_px - anchor_px
                    dist = np.linalg.norm(delta)
                    if dist > max_dist and dist > 1e-6:
                        pos_px = anchor_px + (delta / dist) * max_dist
                        t.set_position(tuple(to_data(pos_px)))
            # If relation labels still overlap object labels, nudge outside labels away.
            movable_obj_texts = [t for t in object_texts if not getattr(t, "_gom_inside_label", False)]
            fixed_obj_texts = [t for t in object_texts if getattr(t, "_gom_inside_label", False)]
            if movable_obj_texts:
                anchors = [t.get_position() for t in movable_obj_texts]
                self._resolve_overlaps(
                    ax,
                    movable_texts=movable_obj_texts,
                    movable_anchors=anchors,
                    fixed_texts=list(rel_texts) + fixed_obj_texts,
                    arrows=arrow_patches,
                )
                if cfg.obj_label_max_dist_px and cfg.obj_label_max_dist_px > 0:
                    to_px = ax.transData.transform
                    to_data = ax.transData.inverted().transform
                    max_dist = float(cfg.obj_label_max_dist_px)
                    for i, t in enumerate(movable_obj_texts):
                        if i >= len(anchors):
                            continue
                        anchor_px = np.array(to_px(anchors[i]))
                        pos_px = np.array(to_px(t.get_position()))
                        delta = pos_px - anchor_px
                        dist = np.linalg.norm(delta)
                        if dist > max_dist and dist > 1e-6:
                            pos_px = anchor_px + (delta / dist) * max_dist
                            t.set_position(tuple(to_data(pos_px)))
                # Re-run relation resolution after object labels moved.
                self._resolve_relation_vs_relation_overlaps(
                    ax,
                    rel_texts,
                    arrow_patches,
                    cfg.relation_label_max_dist_px,
                    anchors=rel_label_anchors,
                    arrow_dirs=rel_label_dirs,
                    fixed_texts=object_texts,
                    avoid_objects=avoid_objects,
                )

    # ===========================================================
    # OBJECT LABEL PLACEMENT
    # ===========================================================
    def _draw_labels(
        self,
        ax: plt.Axes,
        boxes: Sequence[Sequence[float]],
        labels: Sequence[str],
        scores: Sequence[float],
        masks: Optional[Sequence[np.ndarray | Dict[str, Any]]],
        colors: Sequence[str],
        image: Image.Image,
        *,
        avoid_objects: Sequence[Any] = (),
    ) -> List[Any]:
        """
        Intelligently place object labels using multi-tier fallback strategy.
        
        Label placement follows this priority order:
        1. Inside object: Centered within the object if sufficient space and solidity
        2. On border: Just outside the top edge if inside placement fails
        3. With connector: Further outside with line connecting to object center
        
        The method can optionally use batch text rendering for performance when
        many labels are placed outside objects.
        
        Args:
            ax: Matplotlib axes to draw on
            boxes: List of bounding boxes [x1, y1, x2, y2]
            labels: Object class labels
            scores: Detection confidence scores
            masks: Optional segmentation masks for inside placement validation
            colors: Object colors for label styling
            image: Original image for dimension reference
        
        Returns:
            List of matplotlib Text objects for overlap resolution with relationships
        
        Notes:
            - Inside placement requires sufficient area (cfg.min_area_ratio_inside)
              and mask solidity (cfg.min_solidity_inside) if masks available
            - Border labels shifted slightly outward to avoid covering edges
            - Batch rendering provides ~20-30% speedup for many outside labels
            - Overlap resolution applied if cfg.resolve_overlaps enabled
        """
        cfg = self.cfg
        if not cfg.display_labels:
            return []

        W, H = image.size
        placed_texts: List[Any] = []
        inside_texts: List[Any] = []
        outside_texts: List[Any] = []
        outside_anchors: List[Tuple[float, float]] = []
        avoid_patches: List[Any] = list(avoid_objects)

        # Optionally collect outside labels for batch rendering
        batch_renderer = None
        batch_out_specs = []  # list of (border_x, border_y, max_dist_px)
        if cfg.use_batch_text_renderer and RENDERING_OPT_AVAILABLE:
            batch_renderer = BatchTextRenderer()

        fig = ax.figure
        renderer = None
        if cfg.resolve_overlaps:
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()

        placed_bbs: List[Any] = []
        avoid_bbs: List[Any] = []
        if renderer is not None and avoid_patches:
            avoid_bbs = [o.get_window_extent(renderer=renderer).expanded(1.02, 1.06) for o in avoid_patches]

        def _bbox_for_label(
            center_xy: Tuple[float, float],
            w_txt: float,
            h_txt: float,
            *,
            ha: str,
            va: str,
            padding: float,
        ):
            to_px = ax.transData.transform
            cx, cy = to_px(center_xy)
            w = w_txt + 2 * padding
            h = h_txt + 2 * padding
            if ha == "center":
                x0 = cx - w / 2.0
                x1 = cx + w / 2.0
            elif ha == "left":
                x0 = cx
                x1 = cx + w
            else:  # right
                x0 = cx - w
                x1 = cx
            if va == "center":
                y0 = cy - h / 2.0
                y1 = cy + h / 2.0
            elif va == "bottom":
                y0 = cy
                y1 = cy + h
            else:  # top
                y0 = cy - h
                y1 = cy
            return Bbox.from_extents(x0, y0, x1, y1)

        def _overlap_cost(bb) -> Tuple[int, float]:
            hits = 0
            area = 0.0
            for obb in avoid_bbs + placed_bbs:
                if bb.overlaps(obb):
                    hits += 1
                    ix0 = max(bb.x0, obb.x0)
                    iy0 = max(bb.y0, obb.y0)
                    ix1 = min(bb.x1, obb.x1)
                    iy1 = min(bb.y1, obb.y1)
                    area += max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
            return hits, area

        for i, box in enumerate(boxes):
            color = colors[i]
            x1, y1, x2, y2 = map(int, box[:4])
            w, h = max(1, x2 - x1), max(1, y2 - y1)
            label_text = self._format_label_text(labels[i], scores[i], obj_index=i)
            mask_info = self._get_mask_for_index(i, masks)
            small_obj_side_px = max(30, int(cfg.obj_fontsize_inside * 2.5))
            if not avoid_objects and min(w, h) <= small_obj_side_px:
                rect = patches.Rectangle((x1, y1), w, h, linewidth=0, edgecolor="none", facecolor="none", alpha=0.0)
                ax.add_patch(rect)
                avoid_patches.append(rect)

            # Try inside placement first (centered within object)
            allow_inside = self._can_draw_label_inside(
                image, box, mask_info, label_text, ax if cfg.measure_text_with_renderer else None
            )
            if cfg.avoid_object_occlusion:
                allow_inside = (
                    allow_inside
                    and cfg.allow_inside_large_objects
                    and self._is_large_object(image, box)
                )
            if allow_inside:
                txt_col = text_color_for_bg(color)
                t = ax.text(
                    (x1 + x2) / 2,
                    (y1 + y2) / 2,
                    label_text,
                    ha="center",
                    va="center",
                    fontsize=cfg.obj_fontsize_inside,
                    fontfamily=cfg.font_family,
                    color=txt_col,
                    bbox=dict(
                            facecolor=color,
                            alpha=0.95,
                            edgecolor=color,
                            linewidth=cfg.label_bbox_linewidth,
                            boxstyle="round,pad=0.25",
                    ),
                    zorder=7,
                )
                t._gom_inside_label = True
                placed_texts.append(t)
                inside_texts.append(t)
                continue

            # Fallback: place on border (top edge, shifted outward)
            border_x = (x1 + x2) / 2
            w_txt, h_txt = self._estimate_text_px(
                ax if cfg.measure_text_with_renderer else None,
                label_text,
                cfg.obj_fontsize_outside,
            )
            label_padding = 8
            label_h = h_txt + 2 * label_padding
            gap_px = max(3, int(label_h * 0.15))
            can_above = (y1 - (label_h + gap_px)) >= 0
            can_below = (y2 + (label_h + gap_px)) <= H
            cand = []
            if can_above:
                dx_data, dy_data = self._pixels_to_data(ax, 0, -gap_px)
                pos = (border_x + dx_data, y1 + dy_data)
                bb = _bbox_for_label(pos, w_txt, h_txt, ha="center", va="bottom", padding=8)
                cand.append((pos, "bottom", bb, "above"))
            if can_below:
                dx_data, dy_data = self._pixels_to_data(ax, 0, gap_px)
                pos = (border_x + dx_data, y2 + dy_data)
                bb = _bbox_for_label(pos, w_txt, h_txt, ha="center", va="top", padding=8)
                cand.append((pos, "top", bb, "below"))

            if cand:
                scored = []
                for pos, va, bb, side in cand:
                    hits, area = _overlap_cost(bb)
                    scored.append((hits, area, 0 if side == "below" else 1, pos, va, bb))
                scored.sort(key=lambda x: (x[0], x[1], x[2]))
                _, _, _, border_pos, va, bb = scored[0]
            else:
                # Fallback to original logic if no space available
                if can_above:
                    dx_data, dy_data = self._pixels_to_data(ax, 0, -gap_px)
                    border_pos = (border_x + dx_data, y1 + dy_data)
                    va = "bottom"
                    bb = _bbox_for_label(border_pos, w_txt, h_txt, ha="center", va=va, padding=8)
                else:
                    dx_data, dy_data = self._pixels_to_data(ax, 0, gap_px)
                    border_pos = (border_x + dx_data, y2 + dy_data)
                    va = "top"
                    bb = _bbox_for_label(border_pos, w_txt, h_txt, ha="center", va=va, padding=8)

            font_col = text_color_for_bg(color)
            if batch_renderer is not None:
                # Defer rendering; store spec for connector annotation later
                try:
                    base_max = float(cfg.obj_label_max_dist_px)
                    small_scale = min(1.0, max(0.25, min(w, h) / 120.0))
                    max_dist_px = base_max * small_scale
                except Exception:
                    max_dist_px = None
                batch_renderer.add_text(
                    border_pos[0],
                    border_pos[1],
                    label_text,
                    fontsize=cfg.obj_fontsize_outside,
                    color=font_col,
                    bbox_params=dict(facecolor=color, alpha=0.95, edgecolor=color, linewidth=cfg.label_bbox_linewidth, boxstyle="round,pad=0.25"),
                    fontfamily=cfg.font_family,
                    ha="center",
                    va=va,
                    zorder=7,
                )
                batch_out_specs.append((border_x, border_pos[1], max_dist_px))
            else:
                t = ax.text(
                    border_pos[0],
                    border_pos[1],
                    label_text,
                    fontsize=cfg.obj_fontsize_outside,
                    fontfamily=cfg.font_family,
                    color=font_col,
                    ha="center",
                    va=va,
                    bbox=dict(
                        facecolor=color,
                        alpha=0.95,
                        edgecolor=color,
                        linewidth=cfg.label_bbox_linewidth,
                        boxstyle="round,pad=0.25",
                    ),
                    zorder=7,
                )
                t._gom_inside_label = False
                try:
                    base_max = float(cfg.obj_label_max_dist_px)
                    small_scale = min(1.0, max(0.25, min(w, h) / 120.0))
                    t._gom_max_dist_px = base_max * small_scale
                except Exception:
                    pass
                placed_texts.append(t)
                outside_texts.append(t)
                outside_anchors.append((border_x, border_y))
                if bb is not None:
                    placed_bbs.append(bb)

                # connector dall’anchor (bordo) alla label
                ax.annotate(
                    "",
                    xy=(border_x, border_y),
                    xytext=t.get_position(),
                    arrowprops=dict(
                        arrowstyle="-",
                        color="gray",
                        alpha=0.45,
                        shrinkA=4,
                        shrinkB=4,
                        linewidth=cfg.connector_linewidth,
                        linestyle="-",
                    ),
                    zorder=6,
                )

        # If we deferred outside labels to batch rendering, render them now and
        # create connectors/anchors for overlap resolution.
        if batch_renderer is not None:
            created = batch_renderer.render_all(ax)
            # Created texts align with batch_out_specs order
            for t, (bx, by, max_dist_px) in zip(created, batch_out_specs):
                t._gom_inside_label = False
                if max_dist_px is not None:
                    t._gom_max_dist_px = max_dist_px
                placed_texts.append(t)
                outside_texts.append(t)
                outside_anchors.append((bx, by))
                if renderer is not None:
                    placed_bbs.append(t.get_window_extent(renderer=renderer).expanded(1.02, 1.06))
                # Draw connector line from anchor to label
                ax.annotate("", xy=(bx, by), xytext=t.get_position(), arrowprops=dict(arrowstyle="-", color="gray", alpha=0.45, shrinkA=4, shrinkB=4, linewidth=1, linestyle="-"), zorder=6)

        # Resolve overlaps between object labels
        if outside_texts and cfg.resolve_overlaps:
            fig = ax.figure
            fig.canvas.draw()
            self._resolve_object_overlaps_only(
                ax,
                outside_texts,
                outside_anchors,
                fixed_texts=inside_texts,
                avoid_objects=avoid_patches,
            )
        
        return placed_texts


    # ===========================================================
    # LEGEND GENERATION
    # ===========================================================
    def _draw_legend(self, ax: plt.Axes, labels: Sequence[str], colors: Sequence[str]) -> None:
        """
        Create a compact legend showing unique object classes.
        
        Args:
            ax: Matplotlib axes to draw legend on
            labels: All object labels (may contain duplicates with suffixes)
            colors: Corresponding colors for each label
        
        Notes:
            - Strips numeric suffixes to show only base classes
            - Limited to 10 entries to avoid clutter
            - Positioned in upper right corner
            - Uses small font size (cfg.legend_fontsize)
        """
        cfg = self.cfg
        if not cfg.display_legend or not labels:
            return
        # Extract unique base class names (without numeric suffixes)
        uniq_base = sorted({lab.rsplit("_", 1)[0] for lab in labels})
        handles = [patches.Patch(color=self._pick_color(lb, 0), label=lb) for lb in uniq_base[:10]]
        if handles:
            ax.legend(
                handles=handles,
                prop={"family": cfg.font_family, "size": cfg.legend_fontsize},
                loc="upper right",
            )

    # ===========================================================
    # NO-BACKGROUND RENDERING MODE
    # ===========================================================
    def _draw_without_background(
        self,
        image: Image.Image,
        boxes: Sequence[Sequence[float]],
        labels: Sequence[str],
        scores: Sequence[float],
        relationships: Sequence[Dict[str, Any]],
        masks: Optional[Sequence[np.ndarray | Dict[str, Any]]],
        colors: Sequence[str],
        save_path: Optional[str],
        bg_color: Tuple[float, float, float, float],
        dpi: int,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Render visualization without background image for transparent output.
        
        This special rendering mode creates a clean canvas showing only:
        - Segmentation masks (if available and enabled)
        - Relationship arrows (if available and enabled)
        
        Perfect for creating overlay graphics that can be composited later.
        
        Args:
            image: Original image (used for dimensions only)
            boxes: Bounding boxes for relationship rendering
            labels: Object labels for relationship rendering
            scores: Detection scores (unused in this mode)
            relationships: Relationships to render
            masks: Segmentation masks to render
            colors: Object colors
            save_path: Where to save the output
            bg_color: Background color (usually transparent)
            dpi: Output resolution
        
        Returns:
            Tuple of (Figure, Axes) for the transparent rendering
        
        Notes:
            - Figure and axes alpha set to 0 for true transparency
            - Only masks and relationships rendered (no boxes or labels)
            - Ideal for creating compositable overlay layers
        """
        W, H = image.size
        
        # Create figure with transparent background
        fig, ax = plt.subplots(figsize=(W / 100, H / 100))
        ax.axis("off")
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        
        # Set transparent background
        if len(bg_color) == 4 and bg_color[3] == 0:
            fig.patch.set_alpha(0)
            ax.patch.set_alpha(0)
            ax.set_facecolor('none')
        else:
            # White or custom color background
            ax.set_facecolor(bg_color[:3] if len(bg_color) >= 3 else (1, 1, 1))
        
        avoid_objects = self._build_avoid_patches(ax, boxes) if self.cfg.avoid_object_occlusion else []

        # Draw segmentation masks if enabled
        if self.cfg.show_segmentation and masks:
            self._draw_masks_on_clean_canvas(ax, masks, colors, W, H)
        
        # Draw relationships if enabled
        if self.cfg.display_relationships and relationships:
            relations = self._preprocess_relations(relationships, boxes)
            # Draw relationship arrows and labels
            self._draw_relationships(ax, relations, boxes, colors, [], avoid_objects=avoid_objects)
        
        # Draw labels if enabled (optional, for context)
        if self.cfg.display_labels:
            obj_texts = self._draw_labels(ax, boxes, labels, scores, masks, colors, image, avoid_objects=avoid_objects)
        
        # Save
        fig.tight_layout()
        if save_path:
            is_transparent = len(bg_color) == 4 and bg_color[3] == 0
            kwargs = {
                'bbox_inches': 'tight',
                'transparent': is_transparent,
                'dpi': dpi,
                'pad_inches': 0,
            }
            
            if save_path.endswith('.svg') and is_transparent:
                kwargs['facecolor'] = 'none'
            
            fig.savefig(save_path, **kwargs)
            plt.close(fig)
        else:
            plt.show()
            
        return fig, ax
    
    def _draw_masks_on_clean_canvas(
        self,
        ax: plt.Axes,
        masks: Sequence[np.ndarray | Dict[str, Any]],
        colors: Sequence[str],
        W: int,
        H: int,
    ) -> None:
        """
        Render segmentation masks on a clean canvas without background.
        
        Args:
            ax: Matplotlib axes to draw on
            masks: Sequence of mask data (arrays or dicts with 'segmentation' key)
            colors: Corresponding colors for each mask
            W: Canvas width in pixels
            H: Canvas height in pixels
        
        Notes:
            - Uses standard _draw_segmentation method with opaque rendering
            - Skips None or empty masks
            - Ideal for creating transparent overlays
        """
        # Create numpy array for compositing
        if len(colors) == 0:
            return
            
        # Use the standard segmentation drawing but with opaque alpha
        for i, (mask_data, color) in enumerate(zip(masks, colors)):
            if mask_data is None:
                continue
                
            mask_info = self._get_mask_for_index(i, masks)
            if not mask_info or "segmentation" not in mask_info:
                continue
                
            mask = mask_info["segmentation"]
            if mask is None or mask.size == 0:
                continue
            
            # Draw with opaque color (alpha=1.0)
            self._draw_segmentation(ax, mask, color, self.cfg.bbox_linewidth, zorder=2)
    
    def _hex_to_rgb(self, hex_color: str) -> Tuple[int, int, int]:
        """
        Convert hexadecimal color string to RGB tuple.
        
        Args:
            hex_color: Hex color string (with or without '#' prefix)
        
        Returns:
            RGB tuple with values in 0-255 range
        
        Example:
            >>> viz._hex_to_rgb("#FF5733")
            (255, 87, 51)
        """
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    # ===========================================================
    # HELPER FUNCTIONS (MASK ACCESS, DEPTH EXTRACTION, LABEL FORMATTING)
    # ===========================================================
    def _get_mask_for_index(
        self, i: int, masks: Optional[Sequence[np.ndarray | Dict[str, Any]]]
    ) -> Optional[Dict[str, Any]]:
        """
        Safely retrieve mask data for a specific object index.
        
        Handles multiple mask formats:
        - Direct numpy arrays
        - Dictionaries with 'segmentation' key
        - None values
        
        Args:
            i: Object index
            masks: Sequence of mask data in various formats
        
        Returns:
            Dictionary with 'segmentation' key, or None if unavailable
        """
        if masks is None or i >= len(masks) or masks[i] is None:
            return None
        m = masks[i]
        if isinstance(m, dict):
            return m
        if isinstance(m, np.ndarray):
            return {"segmentation": m}
        return None

    def _extract_depth_index(self, label: str, fallback_index: int, metadata: Optional[Dict[str, Any]] = None) -> int:
        """
        Extract depth/layer index from label or metadata for rendering order.
        
        Depth determines z-order during rendering (higher depth = rendered first = appears behind).
        
        Priority order:
        1. Explicit depth value from metadata dict
        2. Numeric suffix in label (e.g., "person_3" has depth 3)
        3. Fallback to provided index
        
        Args:
            label: Object label string (may contain numeric suffix)
            fallback_index: Default depth if no other source available
            metadata: Optional dict that may contain depth information
        
        Returns:
            Integer depth index for z-ordering
        
        Example:
            >>> viz._extract_depth_index("car_2", 0)
            2
            >>> viz._extract_depth_index("person", 5)
            5
        """
        if metadata and self.cfg.depth_key in metadata:
            try:
                return int(metadata[self.cfg.depth_key])
            except (ValueError, TypeError):
                pass
        import re
        match = re.search(r"_(\d+)$", label)
        if match:
            return int(match.group(1))
        return fallback_index

    def _format_label_text(self, label: str, score: float, obj_index: int = 0) -> str:
        """
        Format label text according to configured display mode.
        
        Supports three label modes:
        - "original": Show class name with unique identifier (e.g., "table_1", "chair_2")
        - "numeric": Show sequential numbers (1, 2, 3, ...)
        - "alphabetic": Show letters (A, B, C, ..., Z, AA, AB, ...)
        
        Optionally appends confidence score if cfg.show_confidence enabled.
        
        Args:
            label: Raw object label (may have numeric suffix)
            score: Detection confidence score (0.0-1.0)
            obj_index: Zero-based object index for numeric/alphabetic modes
        
        Returns:
            Formatted label string ready for display
        
        Example:
            >>> viz.cfg.label_mode = "numeric"
            >>> viz._format_label_text("person_1", 0.95, 0)
            "1"
            >>> viz.cfg.show_confidence = True
            >>> viz._format_label_text("car", 0.87, 3)
            "4 (87%)"
        """
        mode = self.cfg.label_mode
        if mode == "numeric":
            text = str(obj_index + 1)
        elif mode == "alphabetic":
            n = obj_index
            alphabet = ""
            while True:
                alphabet = chr(65 + (n % 26)) + alphabet
                n //= 26
                if n == 0:
                    break
                n -= 1
            text = alphabet
        else:
            # In "original" mode, keep the full label with unique identifier
            text = label
        if self.cfg.show_confidence:
            text = f"{text} ({score * 100:.0f}%)"
        return text

    # ===========================================================
    # LABEL PLACEMENT LOGIC
    # ===========================================================
    def _is_large_object(self, image: Image.Image, box: Sequence[float]) -> bool:
        """
        Return True if the object is large enough to allow inside labels.
        """
        W, H = image.size
        x1, y1, x2, y2 = map(int, box[:4])
        w, h = max(1, x2 - x1), max(1, y2 - y1)
        area_ratio = float(w * h) / float(W * H)
        if area_ratio >= float(self.cfg.large_object_area_ratio):
            return True
        if min(w, h) >= int(self.cfg.large_object_min_side_px):
            return True
        return False

    def _can_draw_label_inside(
        self,
        image: Image.Image,
        box: Sequence[float],
        mask_dict: Optional[Dict[str, Any]],
        label_text: str,
        ax=None,
    ) -> bool:
        """
        Determine if label can be placed inside the object without excessive coverage.
        
        This method implements several checks to ensure inside labels are readable
        and don't obscure the object:
        
        1. Object area threshold: Object must occupy minimum fraction of image
        2. Text size check: Label must not cover >50% of object area
        3. Minimum box size: Boxes smaller than 40px on any side get outside labels
        4. Mask solidity: If mask available, check shape compactness
        5. Circular clearance: Ensure space for rotated label within object bounds
        
        Args:
            image: Original PIL Image for size reference
            box: Bounding box [x1, y1, x2, y2]
            mask_dict: Optional dict with 'segmentation' mask for precise area calculation
            label_text: Text to be placed (used for size estimation)
            ax: Optional matplotlib axes for accurate text measurement
        
        Returns:
            True if label can be safely placed inside object, False otherwise
        
        Notes:
            - Uses estimated or measured text dimensions
            - Considers both bounding box and mask areas if available
            - Applies conservative thresholds to prevent object occlusion
            - Eroded mask check ensures label fits within object core
        """
        W, H = image.size
        area_img = float(W * H)
        x1, y1, x2, y2 = map(int, box[:4])
        w, h = max(1, x2 - x1), max(1, y2 - y1)
        area_bbox = float(w * h)

        mask_bool = None
        area_obj = area_bbox
        solidity = min(w, h) / float(max(1, max(w, h)))

        if mask_dict is not None and mask_dict.get("segmentation") is not None:
            m = mask_dict["segmentation"].astype(bool)
            area_mask = int(m.sum())
            if area_mask > 0:
                area_obj = float(area_mask)
                solidity = area_mask / max(1.0, area_bbox)
            mask_bool = m

        # Check 1: Object must be large enough relative to image
        if (area_obj / area_img) < float(self.cfg.min_area_ratio_inside):
            return False

        w_txt, h_txt = self._estimate_text_px(ax, label_text, self.cfg.obj_fontsize_inside)
        
        # Check 2: Label must not cover more than 50% of object
        # This prevents completely hiding small objects with large labels
        label_padding = 8  # Label bbox padding in pixels
        label_w = w_txt + 2 * label_padding
        label_h = h_txt + 2 * label_padding
        label_area = label_w * label_h
        
        if label_area > (0.50 * area_obj):
            return False

        # Check 2b: Label dimensions must fit comfortably within the box
        # Avoid placing labels inside small objects where they would dominate the view
        if label_w > (0.80 * w) or label_h > (0.60 * h):
            return False
        
        # Check 3: Minimum box size requirement
        # Very small objects always get outside labels for clarity
        min_box_size_for_inside = 40  # Minimum pixels per side
        if min(w, h) < min_box_size_for_inside:
            return False
        
        # Check 4: Circular clearance for rotated label
        half_diag = 0.5 * ((w_txt**2 + h_txt**2) ** 0.5)
        margin_px = float(self.cfg.inside_label_margin_px)

        if mask_bool is not None and cv2 is not None:
            # Check 5: Eroded mask still contains circular clearance
            m = (mask_bool.astype(np.uint8) * 255)
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
            dist = cv2.distanceTransform(m, cv2.DIST_L2, 5)
            r_max = float(dist.max())
        else:
            r_max = 0.5 * min(w, h) * 0.7

        if r_max < (half_diag + margin_px):
            return False
        if solidity < float(self.cfg.min_solidity_inside):
            return False
        return True

    def _estimate_text_px(self, ax, text: str, fontsize_px: int) -> Tuple[float, float]:
        """
        Estimate text bounding box dimensions in pixels.
        
        Args:
            ax: Matplotlib axes (used for renderer-based measurement if enabled)
            text: Text string to measure
            fontsize_px: Font size in pixels
        
        Returns:
            Tuple (width, height) in pixels
        
        Methods:
            1. Renderer-based (if cfg.measure_text_with_renderer):
               - Creates temporary invisible text object
               - Measures actual rendered dimensions
               - More accurate but slower
            2. Heuristic (default):
               - width ≈ 0.55 * fontsize * len(text)
               - height ≈ 1.6 * fontsize
               - Fast approximation for typical fonts
        
        Notes:
            - Renderer method requires canvas draw (slow for many labels)
            - Heuristic works well for monospace and sans-serif fonts
            - Use renderer for publication-quality spacing
        """
        if self.cfg.measure_text_with_renderer and ax is not None:
            t = ax.text(0, 0, text, fontsize=fontsize_px, fontfamily=self.cfg.font_family, alpha=0)
            fig = ax.figure
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            bb = t.get_window_extent(renderer=renderer)
            t.remove()
            return bb.width, bb.height
        w_txt = 0.55 * fontsize_px * max(1, len(text))
        h_txt = 1.6 * fontsize_px
        return w_txt, h_txt

    # ===========================================================
    # RELATION LABEL GEOMETRY
    # ===========================================================
    def _get_optimal_relation_label_position(self, ax, arrow, text: str) -> Tuple[float, float]:
        """
        Put the relation label on the arrow path, near the midpoint.
        If the arrow is too short, shift slightly along the normal direction.
        """
        verts = self._arrow_vertices_disp(arrow)
        if len(verts) < 2:
            return (0.0, 0.0)

        to_data = ax.transData.inverted().transform
        # midpoint in display coords
        mid_disp = np.mean(verts, axis=0)
        mid_data = to_data(mid_disp)

        # vector along arrow in display space
        v_disp = verts[-1] - verts[0]
        norm = np.linalg.norm(v_disp)
        if norm < 1e-3:
            return tuple(mid_data)

        # normal in display space
        v_disp = v_disp / norm
        normal_disp = np.array([-v_disp[1], v_disp[0]])

        # estimated text size
        w_txt, h_txt = self._estimate_text_px(ax, text, self.cfg.rel_fontsize)
        text_diag = float(np.sqrt(w_txt ** 2 + h_txt ** 2))

        # if the arrow is short relative to the label, shift it out a bit
        if norm < text_diag * 1.1:
            offset_disp = normal_disp * (text_diag * 0.6)
            off_data = to_data(mid_disp + offset_disp)
            return tuple(off_data)

        return tuple(mid_data)


    def _get_arrow_length_px(self, ax, arrow) -> float:
        """
        Compute total arrow path length in pixel coordinates.
        
        Args:
            ax: Matplotlib axes (unused but kept for API consistency)
            arrow: Matplotlib FancyArrow patch object
        
        Returns:
            Total path length in pixels, or 0.0 on error
        
        Notes:
            - Sums distances between consecutive vertices
            - Accounts for curved arrows via vertex interpolation
            - Used for label placement decisions (short vs long arrows)
        """
        try:
            verts = self._arrow_vertices_disp(arrow)
            if len(verts) < 2:
                return 0.0
            return float(sum(np.linalg.norm(verts[i + 1] - verts[i]) for i in range(len(verts) - 1)))
        except Exception:
            return 0.0

    def _get_arrow_center(self, ax, arrow) -> Tuple[float, float]:
        """
        Compute arrow center point in data coordinates.
        
        Args:
            ax: Matplotlib axes for coordinate transformation
            arrow: Matplotlib FancyArrow patch object
        
        Returns:
            (x, y) center position in data coordinates, or (0, 0) on error
        
        Algorithm:
            1. Extract arrow vertices in display coordinates
            2. Compute mean position of all vertices
            3. Transform back to data coordinates
        
        Notes:
            - Center is geometric mean, not necessarily visual midpoint
            - Used for curved arrow label placement
            - Gracefully handles errors by returning origin
        """
        try:
            verts = self._arrow_vertices_disp(arrow)
            if len(verts) == 0:
                return (0.0, 0.0)
            center_disp = np.mean(verts, axis=0)
            to_data = ax.transData.inverted().transform
            return tuple(to_data(center_disp))
        except Exception:
            return (0.0, 0.0)

    # ===========================================================
    # OVERLAP RESOLUTION
    # ===========================================================
    def _resolve_object_overlaps_only(
        self,
        ax,
        obj_texts: List[Any],
        obj_anchors: List[Tuple[float, float]],
        *,
        fixed_texts: Sequence[Any] = (),
        avoid_objects: Sequence[Any] = (),
    ) -> None:
        """
        Resolve overlapping object labels using intelligent repositioning.
        
        Delegates to _resolve_overlaps() with only object labels to avoid
        cluttered text where multiple object labels would otherwise overlap.
        
        Args:
            ax: Matplotlib axes object
            obj_texts: List of matplotlib text objects (object labels)
            obj_anchors: List of (x, y) anchor points for each label
        
        Notes:
            - Uses adjustText library if available
            - Falls back to no adjustment if adjustText unavailable
            - Labels are moved to minimize overlaps while staying near anchors
        """
        if not obj_texts:
            return
        self._resolve_overlaps(
            ax,
            movable_texts=obj_texts,
            movable_anchors=obj_anchors,
            fixed_texts=fixed_texts,
            arrows=avoid_objects,
        )
        if self.cfg.obj_label_max_dist_px and self.cfg.obj_label_max_dist_px > 0:
            to_px = ax.transData.transform
            to_data = ax.transData.inverted().transform
            default_max = float(self.cfg.obj_label_max_dist_px)
            for i, t in enumerate(obj_texts):
                if i >= len(obj_anchors):
                    continue
                max_dist = float(getattr(t, "_gom_max_dist_px", default_max))
                anchor_px = np.array(to_px(obj_anchors[i]))
                pos_px = np.array(to_px(t.get_position()))
                delta = pos_px - anchor_px
                dist = np.linalg.norm(delta)
                if dist > max_dist and dist > 1e-6:
                    pos_px = anchor_px + (delta / dist) * max_dist
                    t.set_position(tuple(to_data(pos_px)))

        # Final greedy pass to reduce any remaining overlaps for object labels.
        try:
            fig = ax.figure
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            to_px = ax.transData.transform
            to_data = ax.transData.inverted().transform
            default_max = float(self.cfg.obj_label_max_dist_px) if self.cfg.obj_label_max_dist_px > 0 else 60.0

            fixed_bbs = [t.get_window_extent(renderer=renderer).expanded(1.03, 1.08) for t in fixed_texts]
            if avoid_objects:
                fixed_bbs.extend(
                    [o.get_window_extent(renderer=renderer).expanded(1.02, 1.06) for o in avoid_objects]
                )

            def _bb_area(bb):
                return float(bb.width * bb.height)

            bbs = [t.get_window_extent(renderer=renderer) for t in obj_texts]
            order = sorted(range(len(obj_texts)), key=lambda i: -_bb_area(bbs[i]))
            placed_bbs = []
            directions = [
                (0.0, 0.0),
                (1.0, 0.0),
                (-1.0, 0.0),
                (0.0, 1.0),
                (0.0, -1.0),
                (1.0, 1.0),
                (1.0, -1.0),
                (-1.0, 1.0),
                (-1.0, -1.0),
            ]

            for i in order:
                if i >= len(obj_texts):
                    continue
                max_dist = float(getattr(obj_texts[i], "_gom_max_dist_px", default_max))
                anchor = obj_anchors[i] if i < len(obj_anchors) else obj_texts[i].get_position()
                anchor_px = np.array(to_px(anchor))
                bb0 = bbs[i]
                step = max(6.0, min(bb0.width, bb0.height) * 0.6)
                max_steps = int(max_dist / step) + 1
                chosen_bb = None
                for k in range(max_steps + 1):
                    for dx, dy in directions:
                        if k == 0 and (dx != 0.0 or dy != 0.0):
                            continue
                        pos_px = anchor_px + np.array([dx, dy]) * step * k
                        pos = tuple(to_data(pos_px))
                        obj_texts[i].set_position(pos)
                        bb = obj_texts[i].get_window_extent(renderer=renderer).expanded(1.03, 1.08)
                        if any(bb.overlaps(b) for b in fixed_bbs):
                            continue
                        if any(bb.overlaps(b) for b in placed_bbs):
                            continue
                        chosen_bb = bb
                        break
                    if chosen_bb is not None:
                        break
                if chosen_bb is None:
                    chosen_bb = obj_texts[i].get_window_extent(renderer=renderer).expanded(1.03, 1.08)
                placed_bbs.append(chosen_bb)
        except Exception:
            pass

    def _resolve_relation_vs_relation_overlaps(
        self,
        ax,
        rel_texts: List[Any],
        arrows: List[Any],
        max_dist_px: float,
        *,
        anchors: Optional[List[Tuple[float, float]]] = None,
        arrow_dirs: Optional[List[Tuple[float, float]]] = None,
        fixed_texts: List[Any] = None,
        avoid_objects: Sequence[Any] = (),
    ) -> None:
        """
        Resolve overlaps between relationship labels and with object labels.
        
        Uses iterative physics-based repulsion to push overlapping labels apart
        while respecting distance constraints. This ensures relationship labels
        remain readable even in dense visualizations.
        
        Args:
            ax: Matplotlib axes object
            rel_texts: List of relationship label text objects (movable)
            arrows: List of arrow patch objects (for reference)
            max_dist_px: Maximum distance in pixels labels can move from arrows
        fixed_texts: Optional list of object labels (immovable obstacles)
        avoid_objects: Optional patches (e.g., object boxes) to avoid
        
        Algorithm:
            1. Iterate up to 30 times for convergence
            2. Detect overlaps between relation labels
            3. Compute repulsion vectors (push strength: 12px)
            4. Apply forces to separate overlapping labels
            5. Also push relation labels away from fixed object labels
            6. Clamp movement to max_dist_px to keep labels near arrows
            7. Stop early if no movement occurs
        
        Notes:
            - Uses bounding box expansion (5% width, 10% height) for padding
            - Bidirectional repulsion (both labels move for relation-relation)
            - Unidirectional repulsion (only relation moves for relation-object)
            - Respects maximum distance to prevent labels drifting too far
        """
        if len(rel_texts) < 1:
            return
        
        if fixed_texts is None:
            fixed_texts = []

        fig = ax.figure
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()

        max_iter = 30
        push_strength = 12.0
        to_px = ax.transData.transform
        to_data = ax.transData.inverted().transform

        if anchors is None:
            anchors = [t.get_position() for t in rel_texts]
        anchor_px = [np.array(to_px(a)) for a in anchors]

        normals_px = []
        if arrow_dirs is None:
            arrow_dirs = [(1.0, 0.0)] * len(rel_texts)
        for i, (dx, dy) in enumerate(arrow_dirs):
            a0 = anchors[i]
            p0 = np.array(to_px(a0))
            p1 = np.array(to_px((a0[0] + dx, a0[1] + dy)))
            v = p1 - p0
            norm = np.hypot(v[0], v[1])
            if norm < 1e-6:
                n = np.array([0.0, 1.0])
            else:
                v = v / norm
                n = np.array([-v[1], v[0]])
            normals_px.append(n)

        rel_bbs = [t.get_window_extent(renderer=renderer).expanded(1.05, 1.1) for t in rel_texts]

        def _rel_max_dist(idx: int) -> float:
            t = rel_texts[idx] if idx < len(rel_texts) else None
            if t is None:
                return float(max_dist_px)
            return float(getattr(t, "_gom_max_dist_px", max_dist_px))

        def _greedy_place():
            placed_bbs = []
            fixed_bbs = [t.get_window_extent(renderer=renderer).expanded(1.05, 1.1) for t in fixed_texts]
            if avoid_objects:
                fixed_bbs.extend(
                    [o.get_window_extent(renderer=renderer).expanded(1.02, 1.06) for o in avoid_objects]
                )
            arrow_bbs = self._arrow_bboxes_px(arrows, renderer)
            order = sorted(
                range(len(rel_texts)),
                key=lambda i: (-(rel_bbs[i].width * rel_bbs[i].height), -rel_bbs[i].height),
            )
            for i in order:
                a_px = anchor_px[i]
                n = normals_px[i]
                # Step based on text height for stable spacing
                h = max(6.0, rel_bbs[i].height * 0.75)
                max_steps = int(_rel_max_dist(i) / h) + 1
                offsets = [0.0]
                for k in range(1, max_steps + 1):
                    offsets.append(k * h)
                    offsets.append(-k * h)
                chosen_bb = None
                chosen_pos = None
                for off in offsets:
                    p = a_px + n * off
                    pos = tuple(to_data(p))
                    rel_texts[i].set_position(pos)
                    bb = rel_texts[i].get_window_extent(renderer=renderer).expanded(1.05, 1.1)
                    if any(bb.overlaps(b) for b in fixed_bbs):
                        continue
                    if any(bb.overlaps(b) for b in arrow_bbs):
                        continue
                    if any(bb.overlaps(b) for b in placed_bbs):
                        continue
                    chosen_bb = bb
                    chosen_pos = pos
                    break
                if chosen_pos is None:
                    rel_texts[i].set_position(tuple(to_data(a_px)))
                    chosen_bb = rel_texts[i].get_window_extent(renderer=renderer).expanded(1.05, 1.1)
                placed_bbs.append(chosen_bb)

        _greedy_place()
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        rel_bbs = [t.get_window_extent(renderer=renderer).expanded(1.05, 1.1) for t in rel_texts]

        def _clamp_to_anchor(idx: int, pos_data: Tuple[float, float]) -> Tuple[float, float]:
            p = np.array(to_px(pos_data))
            delta = p - anchor_px[idx]
            dist = np.linalg.norm(delta)
            if dist <= _rel_max_dist(idx):
                return pos_data
            if dist < 1e-6:
                return pos_data
            p_clamped = anchor_px[idx] + delta / dist * _rel_max_dist(idx)
            return tuple(to_data(p_clamped))

        for _ in range(max_iter):
            moved = False
            rel_bbs = [t.get_window_extent(renderer=renderer).expanded(1.05, 1.1) for t in rel_texts]
            
            # Get bounding boxes for fixed texts (object labels)
            fixed_bbs = [t.get_window_extent(renderer=renderer).expanded(1.05, 1.1) for t in fixed_texts]
            if avoid_objects:
                fixed_bbs.extend(
                    [o.get_window_extent(renderer=renderer).expanded(1.02, 1.06) for o in avoid_objects]
                )
            arrow_bbs = self._arrow_bboxes_px(arrows, renderer)

            # Resolve overlaps between relation labels
            for i in range(len(rel_bbs)):
                for j in range(i + 1, len(rel_bbs)):
                    if rel_bbs[i].overlaps(rel_bbs[j]):
                        ci = np.array([(rel_bbs[i].x0 + rel_bbs[i].x1) / 2, (rel_bbs[i].y0 + rel_bbs[i].y1) / 2])
                        cj = np.array([(rel_bbs[j].x0 + rel_bbs[j].x1) / 2, (rel_bbs[j].y0 + rel_bbs[j].y1) / 2])
                        sep = cj - ci
                        dist = max(np.linalg.norm(sep), 1e-6)
                        sep_dir = sep / dist

                        n_i = normals_px[i]
                        n_j = normals_px[j]
                        sign_i = 1.0 if np.dot(sep_dir, n_i) >= 0 else -1.0
                        sign_j = 1.0 if np.dot(-sep_dir, n_j) >= 0 else -1.0
                        sep_i = n_i * sign_i * push_strength
                        sep_j = n_j * sign_j * push_strength

                        dx_i, dy_i = self._pixels_to_data(ax, -sep_i[0], -sep_i[1])
                        dx_j, dy_j = self._pixels_to_data(ax, -sep_j[0], -sep_j[1])

                        pos_i = rel_texts[i].get_position()
                        pos_j = rel_texts[j].get_position()
                        new_i = (pos_i[0] + dx_i, pos_i[1] + dy_i)
                        new_j = (pos_j[0] + dx_j, pos_j[1] + dy_j)
                        rel_texts[i].set_position(_clamp_to_anchor(i, new_i))
                        rel_texts[j].set_position(_clamp_to_anchor(j, new_j))
                        moved = True
            
            # Also push relation labels away from object labels and arrows
            for i, rel_bb in enumerate(rel_bbs):
                for fixed_bb in fixed_bbs + arrow_bbs:
                    if rel_bb.overlaps(fixed_bb):
                        ci = np.array([(rel_bb.x0 + rel_bb.x1) / 2, (rel_bb.y0 + rel_bb.y1) / 2])
                        cf = np.array([(fixed_bb.x0 + fixed_bb.x1) / 2, (fixed_bb.y0 + fixed_bb.y1) / 2])
                        sep = ci - cf  # Push relation label away from fixed box
                        dist = max(np.linalg.norm(sep), 1e-6)
                        sep_dir = sep / dist
                        n = normals_px[i]
                        sign = 1.0 if np.dot(sep_dir, n) >= 0 else -1.0
                        sep = n * sign * push_strength

                        dx, dy = self._pixels_to_data(ax, sep[0], sep[1])
                        pos = rel_texts[i].get_position()
                        new_pos = (pos[0] + dx, pos[1] + dy)
                        rel_texts[i].set_position(_clamp_to_anchor(i, new_pos))
                        moved = True

            if not moved:
                break
            fig.canvas.draw_idle()

        # Final greedy pass: try to re-seat remaining overlaps with wider search.
        try:
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            rel_bbs = [t.get_window_extent(renderer=renderer).expanded(1.05, 1.1) for t in rel_texts]
            fixed_bbs = [t.get_window_extent(renderer=renderer).expanded(1.05, 1.1) for t in fixed_texts]
            if avoid_objects:
                fixed_bbs.extend(
                    [o.get_window_extent(renderer=renderer).expanded(1.02, 1.06) for o in avoid_objects]
                )
            arrow_bbs = self._arrow_bboxes_px(arrows, renderer)

            def _bb_area(bb):
                return float(bb.width * bb.height)

            order = sorted(range(len(rel_texts)), key=lambda i: -_bb_area(rel_bbs[i]))
            placed_bbs = []
            for i in order:
                a_px = anchor_px[i]
                n = normals_px[i]
                v = np.array(arrow_dirs[i], dtype=float)
                v_norm = np.hypot(v[0], v[1])
                if v_norm < 1e-6:
                    tdir = np.array([1.0, 0.0])
                else:
                    tdir = v / v_norm
                directions = [
                    n,
                    -n,
                    tdir,
                    -tdir,
                    n + tdir,
                    n - tdir,
                    -n + tdir,
                    -n - tdir,
                ]
                h = max(6.0, rel_bbs[i].height * 0.75)
                max_steps = int(_rel_max_dist(i) / h) + 1
                chosen_bb = None
                for k in range(max_steps + 1):
                    for d in directions:
                        if k == 0 and np.linalg.norm(d) > 1e-6:
                            continue
                        dn = d / (np.linalg.norm(d) + 1e-6)
                        p = a_px + dn * (k * h)
                        pos = tuple(to_data(p))
                        rel_texts[i].set_position(pos)
                        bb = rel_texts[i].get_window_extent(renderer=renderer).expanded(1.05, 1.1)
                        if any(bb.overlaps(b) for b in fixed_bbs):
                            continue
                        if any(bb.overlaps(b) for b in arrow_bbs):
                            continue
                        if any(bb.overlaps(b) for b in placed_bbs):
                            continue
                        chosen_bb = bb
                        break
                    if chosen_bb is not None:
                        break
                if chosen_bb is None:
                    chosen_bb = rel_texts[i].get_window_extent(renderer=renderer).expanded(1.05, 1.1)
                placed_bbs.append(chosen_bb)
        except Exception:
            pass


    def _resolve_overlaps(
        self,
        ax,
        movable_texts: List[Any],
        movable_anchors: List[Tuple[float, float]],
        fixed_texts: Sequence[Any] = (),
        arrows: Sequence[Any] = (),
    ) -> None:
        """
        Automatically reposition text labels to minimize overlaps using adjustText.
        
        This is the core overlap resolution method that uses the adjustText library
        to intelligently move labels away from each other while keeping them close
        to their anchor points.
        
        Args:
            ax: Matplotlib axes object
            movable_texts: Text objects that can be repositioned
            movable_anchors: Original (x, y) positions for each movable text
            fixed_texts: Text objects that cannot move (act as obstacles)
            arrows: Arrow patches to avoid overlapping with
        
        Notes:
            - Requires adjustText library (gracefully degrades if unavailable)
            - Uses profile parameters (_profile_params) for tuning
            - Respects both text-text and text-object spacing
            - Labels stay near anchors while avoiding collisions
        
        Algorithm:
            - Iterative optimization with spring forces
            - Attractive forces pull labels toward anchors
            - Repulsive forces push labels away from overlaps
            - Converges to locally optimal non-overlapping layout
        """
        if adjust_text is None or not movable_texts:
            return

        prof = self._profile_params()

        adjust_text(
            movable_texts,
            x=[p[0] for p in movable_anchors],
            y=[p[1] for p in movable_anchors],
            ax=ax,
            only_move={"points": "xy", "text": "xy"},
            force_text=prof["force_text"],
            expand_text=prof["expand_text"],
            expand_points=prof["expand_points"],
            expand_objects=prof["expand_objects"],
            add_objects=list(arrows) + list(fixed_texts),
        )

        fig = ax.figure
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()

        for _ in range(self.cfg.micro_push_iters):
            moved = False
            mov_bbs = [t.get_window_extent(renderer=renderer).expanded(1.02, 1.08) for t in movable_texts]
            fix_bbs = [t.get_window_extent(renderer=renderer).expanded(1.02, 1.08) for t in fixed_texts] if fixed_texts else []
            arrow_bbs = self._arrow_bboxes_px(arrows, renderer)

            for i in range(len(mov_bbs)):
                for j in range(i + 1, len(mov_bbs)):
                    if mov_bbs[i].overlaps(mov_bbs[j]):
                        dx_px = (mov_bbs[j].x1 - mov_bbs[i].x0) * prof["push_tt"]
                        dy_px = (mov_bbs[j].y1 - mov_bbs[i].y0) * prof["push_tt"]
                        dx, dy = self._pixels_to_data(ax, dx_px, dy_px)
                        xi, yi = movable_texts[i].get_position()
                        xj, yj = movable_texts[j].get_position()
                        movable_texts[i].set_position((xi - dx * 0.5, yi - dy * 0.5))
                        movable_texts[j].set_position((xj + dx * 0.5, yj + dy * 0.5))
                        moved = True

            if not moved:
                break
            fig.canvas.draw_idle()

    # ===========================================================
    # SMALL GEOM UTILS
    # ===========================================================
    def _profile_params(self):
        """
        Get adjustText parameters based on visualization density profile.
        
        Returns dict with tuning parameters for the adjustText library that
        controls how aggressively labels are repositioned to avoid overlaps.
        
        Returns:
            Dictionary with keys:
                - force_text: Repulsion strength between text labels
                - expand_text: Padding around text bounding boxes (x, y)
                - expand_points: Padding around anchor points (x, y)
                - expand_objects: Padding around fixed objects (x, y)
                - push_tt: Text-to-text push strength
        
        Profiles:
            - "dense": Aggressive repositioning for crowded scenes
              * force_text: 0.8, expansions: 1.45-1.55
            - default: Moderate repositioning for typical scenes
              * force_text: 0.4, expansions: 1.05
        
        Notes:
            - Profile selected via cfg.adjust_text_profile
            - Dense profile useful for visualizations with many objects
            - Default profile preserves label positions better
        """
        dense = self.cfg.adjust_text_profile == "dense"
        return dict(
            force_text=0.8 if dense else 0.4,
            expand_text=(1.55, 1.55) if dense else (1.05, 1.05),
            expand_points=(1.45, 1.45) if dense else (1.05, 1.05),
            expand_objects=(1.45, 1.45) if dense else (1.05, 1.05),
            push_tt=0.15 if dense else 0.08,
        )

    def _build_avoid_patches(
        self,
        ax: plt.Axes,
        boxes: Sequence[Sequence[float]],
    ) -> List[Any]:
        """
        Create invisible patches for object boxes to avoid label overlap.
        """
        avoid: List[Any] = []
        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            w, h = max(1, x2 - x1), max(1, y2 - y1)
            rect = patches.Rectangle(
                (x1, y1),
                w,
                h,
                linewidth=0,
                edgecolor="none",
                facecolor="none",
                alpha=0.0,
            )
            ax.add_patch(rect)
            avoid.append(rect)
        return avoid

    def _adjust_position(
        self,
        candidate: Tuple[float, float],
        placed_positions: List[Tuple[float, float]],
        overlap_thresh: float,
        max_iterations: int = 10,
    ) -> Tuple[float, float]:
        """
        Iteratively adjust a position to avoid overlapping with placed positions.
        
        Uses physics-based repulsion simulation to push the candidate position
        away from all placed positions that are within the overlap threshold.
        
        Args:
            candidate: Initial (x, y) position to adjust
            placed_positions: List of already placed positions to avoid
            overlap_thresh: Minimum distance to maintain from placed positions
            max_iterations: Maximum optimization iterations (default: 10)
        
        Returns:
            Adjusted (x, y) position with minimal overlaps
        
        Algorithm:
            1. For each iteration, compute repulsion vectors from all nearby positions
            2. Sum repulsion forces (strength proportional to overlap amount)
            3. Move candidate in direction of net repulsion
            4. Stop early if movement becomes negligible (< 1e-3)
        
        Notes:
            - Converges to locally optimal non-overlapping position
            - May not find global optimum for complex configurations
            - Used for incremental label placement
        """
        new_pos = np.array(candidate, dtype=float)
        for _ in range(max_iterations):
            disp = np.zeros(2, dtype=float)
            for p in placed_positions:
                diff = new_pos - np.array(p)
                dist = np.linalg.norm(diff)
                if dist < overlap_thresh:
                    disp += (overlap_thresh - dist) * (diff / (dist + 1e-6))
            if np.linalg.norm(disp) < 1e-3:
                break
            new_pos += disp
        return tuple(new_pos)

    def _shrink_segment_px(self, p0, p1, shrink_px, ax):
        """
        Shrink line segment by specified pixels from both ends.
        
        Useful for preventing arrows from overlapping with object boundaries.
        Converts to pixel space, shrinks, then converts back to data coordinates.
        
        Args:
            p0: Start point in data coordinates (x, y)
            p1: End point in data coordinates (x, y)
            shrink_px: Number of pixels to shrink from each end
            ax: Matplotlib axes for coordinate transformation
        
        Returns:
            Tuple of (new_p0, new_p1) in data coordinates
        
        Notes:
            - Returns original points if segment length < 1 pixel
            - Preserves segment direction
            - Useful for arrow placement to avoid box boundaries
        """
        to_px = ax.transData.transform
        to_data = ax.transData.inverted().transform
        P0, P1 = np.array(to_px(p0)), np.array(to_px(p1))
        v = P1 - P0
        L = np.linalg.norm(v)
        if L < 1:
            return p0, p1
        v /= L
        return tuple(to_data(P0 + v * shrink_px)), tuple(to_data(P1 - v * shrink_px))

    def _pixels_to_data(self, ax, dx_px, dy_px):
        """
        Convert pixel displacement to data coordinate displacement.
        
        Args:
            ax: Matplotlib axes object
            dx_px: Horizontal displacement in pixels
            dy_px: Vertical displacement in pixels
        
        Returns:
            Tuple (dx_data, dy_data) in data coordinates
        
        Notes:
            - Accounts for axes scaling and aspect ratio
            - Essential for consistent spacing in different plot sizes
        """
        inv = ax.transData.inverted()
        x0, y0 = inv.transform((0, 0))
        x1, y1 = inv.transform((dx_px, dy_px))
        return x1 - x0, y1 - y0

    def _arrow_bboxes_px(self, arrows: Sequence[Any], renderer):
        """
        Compute bounding boxes in pixel coordinates for arrow patches.
        
        Args:
            arrows: Sequence of matplotlib FancyArrow patches
            renderer: Matplotlib renderer for coordinate transformations
        
        Returns:
            List of bounding box objects in pixel coordinates
        
        Notes:
            - Used for overlap detection with labels
            - Skips arrows that fail transformation (returns fewer boxes)
            - Applies DPI scaling for accurate pixel measurements
        """
        bbs = []
        for a in arrows:
            try:
                path = a.get_path().transformed(a.get_transform())
                bb = path.get_extents()
                bb_px = bb.transformed(a.axes.transData + a.figure.dpi_scale_trans)
                bbs.append(bb_px)
            except Exception:
                pass
        return bbs

    @staticmethod
    def _arrow_vertices_disp(arrow) -> np.ndarray:
        """
        Extract arrow vertices in display coordinates.
        
        Args:
            arrow: Matplotlib FancyArrow patch
        
        Returns:
            Numpy array of vertices with shape (N, 2)
        
        Notes:
            - Applies arrow's transform to get final positions
            - Used for geometric computations (intersections, distances)
        """
        path = arrow.get_path().transformed(arrow.get_transform())
        return np.asarray(path.vertices, dtype=float)

    # ===========================================================
    # RELATION FILTERS
    # ===========================================================
    def _filter_redundant_relations(self, relationships: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove duplicate relationships between the same object pairs.
        
        When multiple relationships exist between the same pair of objects
        (e.g., "left of" and "near"), keeps only the best one based on
        priority and confidence.
        
        Args:
            relationships: List of relationship dictionaries with src_idx and tgt_idx
        
        Returns:
            Filtered list with at most one relationship per object pair
        
        Algorithm:
            1. Group relationships by unordered object pair (src, tgt)
            2. For each pair with multiple relationships, choose best via _choose_best_relation
            3. Return one relationship per pair
        
        Notes:
            - Considers (A, B) and (B, A) as different pairs (directional)
            - Sorts pair indices to treat symmetric relationships as duplicates
            - Reduces visual clutter in dense scenes
        """
        if not relationships:
            return list(relationships)
        from collections import defaultdict
        pair_relations: Dict[Tuple[int, int], List[Dict[str, Any]]] = defaultdict(list)
        for rel in relationships:
            s0, t0 = int(rel["src_idx"]), int(rel["tgt_idx"])
            pair_relations[tuple(sorted([s0, t0]))].append(dict(rel))
        filtered: List[Dict[str, Any]] = []
        for _, rels in pair_relations.items():
            filtered.append(rels[0] if len(rels) == 1 else self._choose_best_relation(rels))
        return filtered

    def _cap_relations_per_object(self, relationships: Sequence[Dict[str, Any]], boxes: Sequence[Sequence[float]]) -> List[Dict[str, Any]]:
        """
        Limit number of relationships per object to prevent visual clutter.
        
        Enforces both maximum and minimum relationship counts per object,
        prioritizing important relationships (spatial > descriptive) and
        closer objects over distant ones.
        
        Args:
            relationships: List of relationship dictionaries
            boxes: Bounding boxes for computing object distances
        
        Returns:
            Filtered list respecting min/max relationship constraints
        
        Algorithm:
            1. Group relationships by source object
            2. Sort by priority (relation type) then distance (closer first)
            3. Keep top max_relations_per_object per source
            4. Ensure at least min_relations_per_object per source by adding closest
        
        Configuration:
            - cfg.max_relations_per_object: Upper limit per object
            - cfg.min_relations_per_object: Lower limit per object
        
        Notes:
            - Prevents overwhelming visualizations with too many arrows
            - Ensures important objects have at least some relationships shown
            - Distance computed from bounding box centers
        """
        if not relationships:
            return list(relationships)

        centers = [((float(b[0]) + float(b[2])) / 2.0, (float(b[1]) + float(b[3])) / 2.0) for b in boxes]

        def _dist_for(rel: Dict[str, Any]) -> float:
            if "distance" in rel:
                try:
                    return float(rel["distance"])
                except Exception:
                    pass
            s = int(rel.get("src_idx", -1))
            t = int(rel.get("tgt_idx", -1))
            if 0 <= s < len(centers) and 0 <= t < len(centers):
                dx = centers[t][0] - centers[s][0]
                dy = centers[t][1] - centers[s][1]
                return float((dx * dx + dy * dy) ** 0.5)
            return 1e9

        from collections import Counter, defaultdict
        rels_by_src: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for r in relationships:
            rels_by_src[int(r.get("src_idx", -1))].append(r)

        kept: List[Dict[str, Any]] = []
        max_per = max(0, int(self.cfg.max_relations_per_object))
        min_per = max(0, int(self.cfg.min_relations_per_object))

        for s, rlist in rels_by_src.items():
            rlist_sorted = sorted(rlist, key=lambda r: (-self._get_relation_priority(str(r.get("relation", ""))), _dist_for(r)))
            kept.extend(rlist_sorted[:max_per])

        counts = Counter(int(r.get("src_idx", -1)) for r in kept)
        for s, rlist in rels_by_src.items():
            cur = counts.get(s, 0)
            if cur < min_per:
                leftovers = [r for r in sorted(rlist, key=_dist_for) if r not in kept]
                need = min_per - cur
                kept.extend(leftovers[:need])

        out: List[Dict[str, Any]] = []
        seen = set()
        for r in kept:
            key = (int(r.get("src_idx", -1)), int(r.get("tgt_idx", -1)), str(r.get("relation", "")))
            if key in seen:
                continue
            seen.add(key)
            out.append(r)
        return out

    def _choose_best_relation(self, relations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Select the best relationship from multiple candidates between same object pair.
        
        When multiple relationships exist between the same two objects (e.g., "near",
        "left_of", "touching"), chooses the most informative one based on priority
        and confidence.
        
        Args:
            relations: List of relationship dicts for the same object pair
        
        Returns:
            Single best relationship dict
        
        Selection Criteria:
            1. Higher priority relation type (see _get_relation_priority)
            2. If equal priority, higher confidence score
        
        Example:
            Input: ["near" (priority=2, conf=0.8), "left_of" (priority=1, conf=0.9)]
            Output: "near" (higher priority wins despite lower confidence)
        """
        best = relations[0]
        best_priority = self._get_relation_priority(best["relation"])
        best_conf = self._get_relation_confidence(best)
        for r in relations[1:]:
            p = self._get_relation_priority(r["relation"])
            c = self._get_relation_confidence(r)
            if p > best_priority or (p == best_priority and c > best_conf):
                best, best_priority, best_conf = r, p, c
        return best

    def _get_relation_priority(self, relation: str) -> int:
        """
        Assign priority level to relationship type for visualization.
        
        Prioritizes more specific, semantically meaningful relationships over
        generic spatial relationships.
        
        Args:
            relation: Relationship type string (e.g., "on_top_of", "near", "left_of")
        
        Returns:
            Priority level (0-4, higher is more important)
        
        Priority Levels:
            4 - Contact/support: on_top_of, under, holding, wearing, riding, sitting_on, carrying
            3 - Proximity: touching, adjacent
            2 - Distance: near, close
            1 - Directional: left_of, right_of, above, below, in_front_of, behind
            0 - Generic/other: all other relationships
        
        Notes:
            - Case-insensitive matching
            - Uses substring matching (e.g., "on_top_of_table" matches priority 4)
            - Contact relationships most informative for scene understanding
        """
        rel_name = str(relation).lower()
        if any(k in rel_name for k in {"on_top_of", "under", "holding", "wearing", "riding", "sitting_on", "carrying"}):
            return 4
        if any(k in rel_name for k in {"touching", "adjacent"}):
            return 3
        if any(k in rel_name for k in {"near", "close"}):
            return 2
        if any(k in rel_name for k in {"left_of", "right_of", "above", "below", "in_front_of", "behind"}):
            return 1
        return 0

    def _get_relation_confidence(self, relation: Dict[str, Any]) -> float:
        """
        Extract or estimate confidence score for a relationship.
        
        Args:
            relation: Relationship dictionary with optional confidence metrics
        
        Returns:
            Confidence score in range [0.0, 1.0]
        
        Sources (in priority order):
            1. clip_sim: CLIP semantic similarity score (most reliable)
            2. distance: Inverse distance (closer = higher confidence)
            3. Default: 0.5 if no confidence metric available
        
        Notes:
            - Distance-based confidence: 1.0 / (1.0 + distance/100)
            - Assumes distance in pixels, scaled by 100 for normalization
        """
        if "clip_sim" in relation:
            return float(relation["clip_sim"])
        if "distance" in relation:
            dist = float(relation["distance"])
            return 1.0 / (1.0 + dist / 100.0)
        return 0.5

    # ===========================================================
    # MISC
    # ===========================================================
    @staticmethod
    def _humanize_relation(rel: str) -> str:
        """
        Convert relationship name to human-readable format.
        
        Transforms machine-readable relationship strings into natural language
        for display in visualizations.
        
        Args:
            rel: Relationship string (e.g., "on_top_of", "LeftOf", "near_to")
        
        Returns:
            Human-readable string (e.g., "On Top Of", "Left Of", "Near To")
        
        Transformations:
            1. CamelCase → space-separated (LeftOf → Left Of)
            2. Underscores → spaces (on_top_of → on top of)
            3. Title case (on top of → On Top Of)
        
        Examples:
            - "on_top_of" → "On Top Of"
            - "LeftOf" → "Left Of"
            - "sitting_on" → "Sitting On"
        """
        s = str(rel)
        if any(c.isupper() for c in s):
            import re as _re
            s = _re.sub(r"(?<!^)(?=[A-Z])", " ", s)
        return s.replace("_", " ").strip().title()
