# ===============================================================
# Visual overlay utilities for detections, masks, and relationships
# Modernized / clearer / API-compatible version
# ===============================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# Optional dependencies
try:
    import cv2  # type: ignore
except Exception:
    cv2 = None

try:
    from adjustText import adjust_text  # type: ignore
except Exception:
    adjust_text = None

# Color helpers
try:
    from igp.utils.colors import color_for_label, text_color_for_bg  # type: ignore
except Exception:
    from igp.utils.colors import ColorCycler, text_color_for_bg  # type: ignore
    _color_cycler = ColorCycler()

    def color_for_label(
        label: str,
        idx: int = 0,
        sat_boost: float = 1.3,
        val_boost: float = 1.15,
        cache: Optional[dict] = None,
    ) -> str:
        return _color_cycler.color_for_label(label)

# Rendering optimizations
try:
    from igp.viz.rendering_opt import (
        VectorizedMaskRenderer,
        BatchTextRenderer,
        GeometricOptimizer,
    )
    RENDERING_OPT_AVAILABLE = True
except ImportError:
    RENDERING_OPT_AVAILABLE = False


# ===============================================================
# CONFIG
# ===============================================================

@dataclass
class VisualizerConfig:
    # WHAT to show
    display_labels: bool = True
    display_relationships: bool = True
    display_relation_labels: bool = True
    display_legend: bool = True

    # HOW to draw objects
    show_segmentation: bool = True
    fill_segmentation: bool = True
    show_bboxes: bool = True
    # Use optimized vectorized mask blending when available
    use_vectorized_masks: bool = False
    # Use batch text renderer to reduce text draw overhead
    use_batch_text_renderer: bool = False

    # Typography / styling
    obj_fontsize_inside: int = 12
    obj_fontsize_outside: int = 12
    rel_fontsize: int = 10
    legend_fontsize: int = 8
    seg_fill_alpha: float = 0.6
    bbox_linewidth: float = 1.0
    rel_arrow_linewidth: float = 1.5
    rel_arrow_mutation_scale: float = 22.0
    # Make contours/labels/relations more prominent by default
    seg_fill_alpha: float = 0.75
    bbox_linewidth: float = 2.0
    rel_arrow_linewidth: float = 2.5
    rel_arrow_mutation_scale: float = 26.0
    # Label/legend styling
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

    # Overlap resolution
    resolve_overlaps: bool = True
    adjust_text_profile: str = "dense"
    micro_push_iters: int = 100

    # Depth handling
    use_depth_ordering: bool = True
    depth_key: str = "depth"

    # Relation label placement
    relation_label_placement: str = "midpoint"
    relation_label_offset_px: float = 10.0
    relation_label_max_dist_px: float = 50.0

    # Global color tweaks
    color_sat_boost: float = 1.30
    color_val_boost: float = 1.15

    # Special knobs
    on_top_gap_px: int = 8
    on_top_horiz_overlap: float = 0.35


# ===============================================================
# VISUALIZER
# ===============================================================

class Visualizer:
    """
    High-level renderer:
      - draws SAM masks and/or boxes,
      - places object labels (inside if feasible, otherwise outside with connectors),
      - plots relationship arrows and optional labels,
      - composes a small legend (top-right).
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
        self.cfg = config or VisualizerConfig()
        self._label2color_cache: Dict[str, str] = {}

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
        dpi: int = 200,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Render full overlay visualization.

        Parameters
        ----------
        image : PIL.Image
        boxes : list[list[float]]
        labels : list[str]
        scores : list[float]
        relationships : list[dict]
        masks : list, optional
        save_path : str, optional
        draw_background : bool
        bg_color : tuple
        dpi : int

        Returns
        -------
        fig, ax
        """
        fig, ax = self._create_canvas(image, draw_background, bg_color)

        # 1) preprocess relations
        relations = self._preprocess_relations(relationships, boxes)

        # 2) assign colors
        colors = self._assign_colors(labels)

        # 3) draw passes
        self._draw_objects(ax, boxes, masks, labels, scores, colors, image)
        self._draw_relationships(ax, relations, boxes, colors)
        self._draw_labels(ax, boxes, labels, scores, masks, colors, image)
        self._draw_legend(ax, labels, colors)

        # 4) finalize
        self._finalize_figure(fig, save_path, draw_background, bg_color, dpi)
        return fig, ax

    # -----------------------------------------------------------
    # CANVAS
    # -----------------------------------------------------------
    def _create_canvas(
        self, image: Image.Image, draw_background: bool, bg_color: Tuple[float, float, float, float]
    ) -> Tuple[plt.Figure, plt.Axes]:
        W, H = image.size
        fig, ax = plt.subplots(figsize=(W / 100, H / 100))
        ax.axis("off")
        if draw_background:
            ax.imshow(image)
        else:
            ax.set_xlim(0, W)
            ax.set_ylim(H, 0)
            ax.set_facecolor(bg_color)
            if len(bg_color) == 4 and bg_color[3] == 0:
                fig.patch.set_alpha(0)
        return fig, ax

    def _finalize_figure(
        self, fig: plt.Figure, save_path: Optional[str], draw_background: bool,
        bg_color: Tuple[float, float, float, float], dpi: int
    ) -> None:
        fig.tight_layout()
        if save_path:
            fig.savefig(
                save_path,
                bbox_inches="tight",
                transparent=(not draw_background and (len(bg_color) == 4 and bg_color[3] == 0)),
                dpi=dpi,
            )
            plt.close(fig)
        else:
            plt.show()

    # -----------------------------------------------------------
    # RELATIONS PREPROCESS + COLORS
    # -----------------------------------------------------------
    def _preprocess_relations(
        self, relationships: Sequence[Dict[str, Any]], boxes: Sequence[Sequence[float]]
    ) -> List[Dict[str, Any]]:
        rels = list(relationships)
        if self.cfg.filter_redundant_relations:
            rels = self._filter_redundant_relations(rels)
        if self.cfg.cap_relations_per_object:
            rels = self._cap_relations_per_object(rels, boxes)
        return rels

    def _assign_colors(self, labels: Sequence[str]) -> List[str]:
        return [self._pick_color(lbl, i) for i, lbl in enumerate(labels)]

    def _pick_color(self, label: str, idx: int) -> str:
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
    # OBJECT DRAWING
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
        cfg = self.cfg
        if not boxes:
            return

        # depth-sorted
        ordered = []
        for i, box in enumerate(boxes):
            depth_idx = self._extract_depth_index(labels[i], i)
            ordered.append((i, box, depth_idx))
        ordered.sort(key=lambda x: x[2], reverse=True)

        # If vectorized mask rendering is enabled and available, produce a
        # blended image once and then draw contours/boxes as outlines.
        if cfg.use_vectorized_masks and RENDERING_OPT_AVAILABLE and masks:
            # Collect masks and colors matching original order
            masks_list = []
            colors_list = []
            for (original_idx, box, depth) in ordered:
                m = self._get_mask_for_index(original_idx, masks)
                if m is not None and m.get("segmentation") is not None:
                    masks_list.append(m["segmentation"])
                    colors_list.append(colors[original_idx])
                else:
                    # keep alignment with None for objects without mask
                    masks_list.append(None)
                    colors_list.append(colors[original_idx])

            # Convert PIL image to numpy background
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

            # Draw contours or bbox outlines for each object
            for (original_idx, box, depth) in ordered:
                color = colors[original_idx]
                x1, y1, x2, y2 = map(int, box[:4])
                mask_info = self._get_mask_for_index(original_idx, masks)
                z_order = 2 + (len(boxes) - min(depth, len(boxes))) * 0.1
                if mask_info is not None and mask_info.get("segmentation") is not None:
                    # draw contour stroke on top of blended image
                    self._draw_segmentation(ax, mask_info["segmentation"], color, cfg.bbox_linewidth, z_order)
                elif cfg.show_bboxes:
                    self._draw_bbox(ax, x1, y1, x2, y2, color, cfg.bbox_linewidth, z_order)
        else:
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
        """Draw mask (filled + opaque border)."""
        if cv2 is None:
            ax.imshow(mask.astype(float), alpha=self.cfg.seg_fill_alpha, extent=(0, mask.shape[1], mask.shape[0], 0))
            return

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
            if self.cfg.fill_segmentation:
                ax.fill(cnt[:, 0], cnt[:, 1], color=color, alpha=self.cfg.seg_fill_alpha, zorder=zorder)
            # Opaque border
            ax.plot(cnt[:, 0], cnt[:, 1], color=color, linewidth=linewidth, alpha=1.0, zorder=zorder + 0.1)

    # ===========================================================
    # RELATIONSHIPS
    # ===========================================================
    def _draw_relationships(
        self,
        ax: plt.Axes,
        relationships: Sequence[Dict[str, Any]],
        boxes: Sequence[Sequence[float]],
        colors: Sequence[str],
    ) -> None:
        """
        Draw arrows and labels for object relationships (opaque color),
        keeping relation labels centered on the arrow and resolving overlaps.
        """
        cfg = self.cfg
        if not cfg.display_relationships or not relationships:
            return

        centers = [
            ((float(b[0]) + float(b[2])) / 2.0, (float(b[1]) + float(b[3])) / 2.0)
            for b in boxes
        ]

        arrow_patches: List[patches.FancyArrowPatch] = []
        rel_texts: List[Any] = []

        arrow_counts: Dict[Tuple[int, int], int] = {}

        for rel in relationships:
            src, tgt = int(rel["src_idx"]), int(rel["tgt_idx"])
            if not (0 <= src < len(centers) and 0 <= tgt < len(centers)):
                continue

            start, end = centers[src], centers[tgt]
            relation_name = str(rel.get("relation", "")).lower()
            color = colors[src]

            arrow_counts[(src, tgt)] = arrow_counts.get((src, tgt), 0) + 1
            curvature = 0.2 + 0.1 * (arrow_counts[(src, tgt)] - 1)

            # shrink endpoints to avoid covering the center points
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
                # posizione iniziale: centro arco
                pos = self._get_optimal_relation_label_position(ax, arrow, readable)
                t = ax.text(
                    pos[0],
                    pos[1],
                    readable,
                    fontsize=cfg.rel_fontsize,
                    ha="center",
                    va="center",
                    color="black",
                    bbox=dict(
                        boxstyle="round,pad=0.2",
                        facecolor="white",
                        alpha=1.0,
                        edgecolor=color,
                        linewidth=cfg.relation_label_bbox_linewidth,
                    ),
                    zorder=5,
                )
                rel_texts.append(t)

        # risolvi overlap tra label di relazione
        if cfg.resolve_overlaps and rel_texts:
            fig = ax.figure
            fig.canvas.draw()
            self._resolve_relation_vs_relation_overlaps(ax, rel_texts, arrow_patches, cfg.relation_label_max_dist_px)

    # ===========================================================
    # LABELS
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
    ) -> None:
        """
        Place labels inside objects if feasible, otherwise on the box border,
        and only then slightly outside, trying to avoid collisions.
        """
        cfg = self.cfg
        if not cfg.display_labels:
            return

        W, H = image.size
        placed_texts: List[Any] = []
        placed_anchors: List[Tuple[float, float]] = []

        # Optionally collect outside labels to render them in batch
        batch_renderer = None
        batch_out_specs = []  # list of (border_pos, label_text, color)
        if cfg.use_batch_text_renderer and RENDERING_OPT_AVAILABLE:
            batch_renderer = BatchTextRenderer()

        for i, box in enumerate(boxes):
            color = colors[i]
            x1, y1, x2, y2 = map(int, box[:4])
            w, h = max(1, x2 - x1), max(1, y2 - y1)
            label_text = self._format_label_text(labels[i], scores[i], obj_index=i)
            mask_info = self._get_mask_for_index(i, masks)

            # 1) tenta inside
            if self._can_draw_label_inside(image, box, mask_info, label_text, ax if cfg.measure_text_with_renderer else None):
                txt_col = text_color_for_bg(color)
                t = ax.text(
                    (x1 + x2) / 2,
                    (y1 + y2) / 2,
                    label_text,
                    ha="center",
                    va="center",
                    fontsize=cfg.obj_fontsize_inside,
                    color=txt_col,
                    bbox=dict(
                            facecolor=color,
                            alpha=1.0,
                            edgecolor=color,
                            linewidth=cfg.label_bbox_linewidth,
                            boxstyle="round,pad=0.25",
                    ),
                    zorder=7,
                )
                placed_texts.append(t)
                placed_anchors.append(((x1 + x2) / 2, (y1 + y2) / 2))
                continue

            # 2) prova sul bordo: scegli lato più corto per essere più leggibile
            #    e muovi leggermente verso l'esterno
            #    prendiamo il centro del lato top di default
            border_x = (x1 + x2) / 2
            border_y = y1
            # shift verso l'alto di 4px
            dx_data, dy_data = self._pixels_to_data(ax, 0, -6)
            border_pos = (border_x + dx_data, border_y + dy_data)

            font_col = text_color_for_bg(color)
            if batch_renderer is not None:
                # Defer rendering; store spec for connector annotation later
                batch_renderer.add_text(
                    border_pos[0],
                    border_pos[1],
                    label_text,
                    fontsize=cfg.obj_fontsize_outside,
                    color=font_col,
                    bbox_params=dict(facecolor=color, alpha=1.0, edgecolor=color, linewidth=cfg.label_bbox_linewidth, boxstyle="round,pad=0.25"),
                    ha="center",
                    va="bottom",
                    zorder=7,
                )
                batch_out_specs.append((border_x, border_y))
            else:
                t = ax.text(
                    border_pos[0],
                    border_pos[1],
                    label_text,
                    fontsize=cfg.obj_fontsize_outside,
                    color=font_col,
                    ha="center",
                    va="bottom",
                    bbox=dict(
                        facecolor=color,
                        alpha=1.0,
                        edgecolor=color,
                        linewidth=cfg.label_bbox_linewidth,
                        boxstyle="round,pad=0.25",
                    ),
                    zorder=7,
                )
                placed_texts.append(t)
                placed_anchors.append((border_x, border_y))

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
            # created aligns with batch_out_specs order
            for t, (bx, by) in zip(created, batch_out_specs):
                placed_texts.append(t)
                placed_anchors.append((bx, by))
                # connector
                ax.annotate("", xy=(bx, by), xytext=t.get_position(), arrowprops=dict(arrowstyle="-", color="gray", alpha=0.45, shrinkA=4, shrinkB=4, linewidth=1, linestyle="-"), zorder=6)

        # 3) risolvi overlap tra label di oggetti
        if placed_texts and cfg.resolve_overlaps:
            fig = ax.figure
            fig.canvas.draw()
            self._resolve_object_overlaps_only(ax, placed_texts, placed_anchors)


    # ===========================================================
    # LEGEND
    # ===========================================================
    def _draw_legend(self, ax: plt.Axes, labels: Sequence[str], colors: Sequence[str]) -> None:
        cfg = self.cfg
        if not cfg.display_legend or not labels:
            return
        uniq_base = sorted({lab.rsplit("_", 1)[0] for lab in labels})
        handles = [patches.Patch(color=self._pick_color(lb, 0), label=lb) for lb in uniq_base[:10]]
        if handles:
            ax.legend(handles=handles, fontsize=cfg.legend_fontsize, loc="upper right")

    # ===========================================================
    # HELPERS (MASK, DEPTH, LABEL FORMAT)
    # ===========================================================
    def _get_mask_for_index(
        self, i: int, masks: Optional[Sequence[np.ndarray | Dict[str, Any]]]
    ) -> Optional[Dict[str, Any]]:
        if masks is None or i >= len(masks) or masks[i] is None:
            return None
        m = masks[i]
        if isinstance(m, dict):
            return m
        if isinstance(m, np.ndarray):
            return {"segmentation": m}
        return None

    def _extract_depth_index(self, label: str, fallback_index: int, metadata: Optional[Dict[str, Any]] = None) -> int:
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
        mode = self.cfg.label_mode
        base = label.rsplit("_", 1)[0]
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
            text = base
        if self.cfg.show_confidence:
            text = f"{text} ({score * 100:.0f}%)"
        return text

    # ===========================================================
    # LABEL PLACEMENT CHECK
    # ===========================================================
    def _can_draw_label_inside(
        self,
        image: Image.Image,
        box: Sequence[float],
        mask_dict: Optional[Dict[str, Any]],
        label_text: str,
        ax=None,
    ) -> bool:
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

        if (area_obj / area_img) < float(self.cfg.min_area_ratio_inside):
            return False

        w_txt, h_txt = self._estimate_text_px(ax, label_text, self.cfg.obj_fontsize_inside)
        half_diag = 0.5 * ((w_txt**2 + h_txt**2) ** 0.5)
        margin_px = float(self.cfg.inside_label_margin_px)

        if mask_bool is not None and cv2 is not None:
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
        if self.cfg.measure_text_with_renderer and ax is not None:
            t = ax.text(0, 0, text, fontsize=fontsize_px, alpha=0)
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

        # testo stimato
        w_txt, h_txt = self._estimate_text_px(ax, text, self.cfg.rel_fontsize)
        text_diag = float(np.sqrt(w_txt ** 2 + h_txt ** 2))

        # se la freccia è corta rispetto alla label, sposta un po' fuori
        if norm < text_diag * 1.1:
            offset_disp = normal_disp * (text_diag * 0.6)
            off_data = to_data(mid_disp + offset_disp)
            return tuple(off_data)

        return tuple(mid_data)


    def _get_arrow_length_px(self, ax, arrow) -> float:
        try:
            verts = self._arrow_vertices_disp(arrow)
            if len(verts) < 2:
                return 0.0
            return float(sum(np.linalg.norm(verts[i + 1] - verts[i]) for i in range(len(verts) - 1)))
        except Exception:
            return 0.0

    def _get_arrow_center(self, ax, arrow) -> Tuple[float, float]:
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
    def _resolve_object_overlaps_only(self, ax, obj_texts: List[Any], obj_anchors: List[Tuple[float, float]]) -> None:
        if not obj_texts:
            return
        self._resolve_overlaps(ax, movable_texts=obj_texts, movable_anchors=obj_anchors)

    def _resolve_relation_vs_relation_overlaps(
        self, ax, rel_texts: List[Any], arrows: List[Any], max_dist_px: float
    ) -> None:
        if len(rel_texts) < 2:
            return

        fig = ax.figure
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()

        max_iter = 30
        push_strength = 12.0

        for _ in range(max_iter):
            moved = False
            rel_bbs = [t.get_window_extent(renderer=renderer).expanded(1.05, 1.1) for t in rel_texts]

            for i in range(len(rel_bbs)):
                for j in range(i + 1, len(rel_bbs)):
                    if rel_bbs[i].overlaps(rel_bbs[j]):
                        ci = np.array([(rel_bbs[i].x0 + rel_bbs[i].x1) / 2, (rel_bbs[i].y0 + rel_bbs[i].y1) / 2])
                        cj = np.array([(rel_bbs[j].x0 + rel_bbs[j].x1) / 2, (rel_bbs[j].y0 + rel_bbs[j].y1) / 2])
                        sep = cj - ci
                        dist = max(np.linalg.norm(sep), 1e-6)
                        sep = sep / dist * push_strength

                        # clamp: non andare oltre max_dist_px
                        if dist > max_dist_px:
                            continue

                        dx_i, dy_i = self._pixels_to_data(ax, -sep[0] * 0.5, -sep[1] * 0.5)
                        dx_j, dy_j = self._pixels_to_data(ax, sep[0] * 0.5, sep[1] * 0.5)

                        pos_i = rel_texts[i].get_position()
                        pos_j = rel_texts[j].get_position()
                        rel_texts[i].set_position((pos_i[0] + dx_i, pos_i[1] + dy_i))
                        rel_texts[j].set_position((pos_j[0] + dx_j, pos_j[1] + dy_j))
                        moved = True

            if not moved:
                break
            fig.canvas.draw_idle()


    def _resolve_overlaps(
        self,
        ax,
        movable_texts: List[Any],
        movable_anchors: List[Tuple[float, float]],
        fixed_texts: Sequence[Any] = (),
        arrows: Sequence[Any] = (),
    ) -> None:
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
        dense = self.cfg.adjust_text_profile == "dense"
        return dict(
            force_text=0.8 if dense else 0.4,
            expand_text=(1.55, 1.55) if dense else (1.05, 1.05),
            expand_points=(1.45, 1.45) if dense else (1.05, 1.05),
            expand_objects=(1.45, 1.45) if dense else (1.05, 1.05),
            push_tt=0.15 if dense else 0.08,
        )

    def _adjust_position(
        self,
        candidate: Tuple[float, float],
        placed_positions: List[Tuple[float, float]],
        overlap_thresh: float,
        max_iterations: int = 10,
    ) -> Tuple[float, float]:
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
        inv = ax.transData.inverted()
        x0, y0 = inv.transform((0, 0))
        x1, y1 = inv.transform((dx_px, dy_px))
        return x1 - x0, y1 - y0

    def _arrow_bboxes_px(self, arrows: Sequence[Any], renderer):
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
        path = arrow.get_path().transformed(arrow.get_transform())
        return np.asarray(path.vertices, dtype=float)

    # ===========================================================
    # RELATION FILTERS
    # ===========================================================
    def _filter_redundant_relations(self, relationships: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
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

        from collections import defaultdict, Counter
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
        s = str(rel)
        if any(c.isupper() for c in s):
            import re as _re
            s = _re.sub(r"(?<!^)(?=[A-Z])", " ", s)
        return s.replace("_", " ").strip().title()