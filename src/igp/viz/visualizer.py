from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from PIL import Image

# opzionali (degradano con fallback)
try:
    import cv2  # type: ignore
except Exception:
    cv2 = None

try:
    from adjustText import adjust_text  # type: ignore
except Exception:
    adjust_text = None  # fallback

try:
    from igp.utils.colors import color_for_label, text_color_for_bg
except ImportError:
    # Fallback: usa ColorCycler esistente
    from igp.utils.colors import ColorCycler, text_color_for_bg  # type: ignore

    _color_cycler = ColorCycler()

    def color_for_label(
        label: str,
        idx: int = 0,
        sat_boost: float = 1.3,
        val_boost: float = 1.15,
        cache: Optional[dict] = None,
    ) -> str:
        """Wrapper minimal per ColorCycler.color_for_label."""
        return _color_cycler.color_for_label(label)


@dataclass
class VisualizerConfig:
    # cosa mostrare
    display_labels: bool = True
    display_relationships: bool = True
    display_relation_labels: bool = True
    display_legend: bool = True

    # cosa disegnare per gli oggetti
    show_segmentation: bool = True
    fill_segmentation: bool = True
    show_bboxes: bool = True

    # stile oggetti/relazioni
    obj_fontsize_inside: int = 12
    obj_fontsize_outside: int = 12
    rel_fontsize: int = 10
    legend_fontsize: int = 8
    seg_fill_alpha: float = 0.2
    bbox_linewidth: float = 2.0
    rel_arrow_linewidth: float = 2.0
    rel_arrow_mutation_scale: float = 22.0

    # gestione relazioni
    filter_redundant_relations: bool = True
    cap_relations_per_object: bool = False
    max_relations_per_object: int = 1
    min_relations_per_object: int = 1

    # etichette
    label_mode: str = "original"  # "original" | "numeric" | "alphabetic"
    show_confidence: bool = False

    # posizionamento etichette interne
    min_area_ratio_inside: float = 0.006  # 0.6% dell'immagine
    inside_label_margin_px: int = 6
    min_solidity_inside: float = 0.45
    measure_text_with_renderer: bool = True  # renderer preciso di default

    # risoluzione sovrapposizioni
    resolve_overlaps: bool = True
    adjust_text_profile: str = "dense"  # "balanced" | "dense"
    micro_push_iters: int = 60
    
    # controllo profondità
    use_depth_ordering: bool = True
    depth_key: str = "depth"

    # posizionamento etichette relazioni
    relation_label_placement: str = "near_arrow"  # "near_arrow" | "midpoint"
    relation_label_offset_px: float = 10.0
    relation_label_max_dist_px: float = 30.0

    # color tweaks
    color_sat_boost: float = 1.30
    color_val_boost: float = 1.15

    # relazione speciale
    on_top_gap_px: int = 8
    on_top_horiz_overlap: float = 0.35


class Visualizer:
    """
    Visualizza:
      - maschere SAM (opzionale, con riempimento/contorno)
      - bounding boxes (opzionale)
      - etichette dentro/fuori gli oggetti con contrasto automatico
      - frecce di relazione (con eventuali etichette)
      - legenda per classi (max 10 voci)
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

    # ------------------------------------------------------------------ public

    def draw(
        self,
        image: Image.Image,
        boxes: Sequence[Sequence[float]],
        labels: Sequence[str],
        scores: Sequence[float],
        relationships: Sequence[Dict[str, Any]],
        masks: Optional[Sequence[Dict[str, Any]]] = None,
        save_path: Optional[str] = None,
        draw_background: bool = True,
        bg_color: Tuple[float, float, float, float] = (1, 1, 1, 0),
    ) -> None:
        """Disegna l'overlay e salva/mostra il risultato."""
        cfg = self.cfg  # alias locale per evitare getattr ripetuti
        fig, ax = plt.subplots(figsize=(10, 8))
        W, H = image.size

        # 0) filtro relazioni
        rels = list(relationships)
        if cfg.filter_redundant_relations:
            rels = self._filter_redundant_relations(rels)
        if cfg.cap_relations_per_object:
            rels = self._cap_relations_per_object(rels, boxes)

        # 1) sfondo
        if draw_background:
            ax.imshow(image)
            ax.axis("off")
        else:
            ax.set_xlim(0, W)
            ax.set_ylim(H, 0)
            ax.axis("off")
            ax.set_facecolor(bg_color)
            if len(bg_color) == 4 and bg_color[3] == 0:
                fig.patch.set_alpha(0)

        # 2) colori per oggetti
        obj_colors = [self._pick_color(labels[i], i) for i in range(len(boxes))]

        # 3) oggetti + label oggetto (ordinati per profondità)
        detection_labels_info: List[Tuple[Tuple[float, float], str, str]] = []
        placed_positions: List[Tuple[float, float]] = []
        overlap_threshold = 30

        centers: List[Tuple[float, float]] = []
        
        # Crea lista di oggetti con i loro indici per ordinamento per profondità
        objects_with_depth = []
        for i, box in enumerate(boxes):
            # Estrai l'indice di profondità se presente nel label, altrimenti usa l'indice dell'oggetto
            depth_index = self._extract_depth_index(labels[i], i)
            objects_with_depth.append((i, box, depth_index))
        
        # Ordina per profondità (più lontano prima = indice più alto)
        objects_with_depth.sort(key=lambda x: x[2], reverse=True)
        
        # Prima pass: disegna solo segmentazioni/bbox in ordine di profondità
        for original_idx, box, depth_idx in objects_with_depth:
            col = obj_colors[original_idx]
            x1, y1, x2, y2 = map(int, box[:4])
            
            best_mask = self._best_mask(original_idx, masks)
            
            # Disegna segmentazione/bbox con z-order basato sulla profondità
            z_order_seg = 1 + (len(boxes) - depth_idx) * 0.1  # più vicino = z-order più alto
            
            if cfg.show_segmentation and best_mask is not None and best_mask.get("segmentation") is not None:
                self._draw_mask(ax, best_mask["segmentation"], color=col, linewidth=cfg.bbox_linewidth, zorder=z_order_seg)
            elif cfg.show_bboxes:
                self._draw_box(ax, x1, y1, x2, y2, color=col, linewidth=cfg.bbox_linewidth, zorder=z_order_seg)
        
        # Seconda pass: calcola centri e prepara etichette (in ordine originale)
        for i, box in enumerate(boxes):
            col = obj_colors[i]
            x1, y1, x2, y2 = map(int, box[:4])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            centers.append((cx, cy))

            if cfg.display_labels:
                best_mask = self._best_mask(i, masks)
                label_text = self._format_label_text(labels[i], scores[i], obj_index=i)
                place_inside = self._can_place_inside(
                    image, box, best_mask, label_text, ax if cfg.measure_text_with_renderer else None
                )
                if place_inside:
                    txt_col = text_color_for_bg(col)
                    ax.text(
                        (x1 + x2) / 2,
                        (y1 + y2) / 2,
                        label_text,
                        ha="center",
                        va="center",
                        fontsize=cfg.obj_fontsize_inside,
                        color=txt_col,
                        bbox=dict(
                            facecolor=col,
                            alpha=0.6,
                            edgecolor=col,
                            linewidth=2.0,
                            boxstyle="round,pad=0.25",
                        ),
                        zorder=7,
                    )
                else:
                    center_pt = self._adjust_position((cx, cy), placed_positions, overlap_threshold)
                    placed_positions.append(center_pt)
                    detection_labels_info.append((center_pt, label_text, col))

        # 4) prepara relazioni
        arrow_patches: List[patches.FancyArrowPatch] = []
        rel_texts: List[Any] = []
        rel_anchors: List[Tuple[float, float]] = []
        rel_draw_data: List[Dict[str, Any]] = []
        rel_label_specs: List[Dict[str, Any]] = []

        if cfg.display_relationships and rels:
            arrow_counts: Dict[Tuple[int, int], int] = {}
            for rel in rels:
                s0, t0 = int(rel["src_idx"]), int(rel["tgt_idx"])
                name = str(rel.get("relation", "")).lower()
                s, t = (t0, s0) if any(k in name for k in self.SPATIAL_KEYS) else (s0, t0)
                if not (0 <= s < len(centers) and 0 <= t < len(centers)):
                    continue

                start, end = centers[s], centers[t]
                col = obj_colors[s]

                arrow_counts[(s, t)] = arrow_counts.get((s, t), 0) + 1
                rad_offset = 0.2 + 0.1 * (arrow_counts[(s, t)] - 1)

                # anchor "ideale" per testo relazione (midpoint + offset perpendicolare se curvata)
                mid_x = (start[0] + end[0]) / 2.0
                mid_y = (start[1] + end[1]) / 2.0
                if rad_offset != 0:
                    dx = end[0] - start[0]
                    dy = end[1] - start[1]
                    length = max(1e-6, (dx ** 2 + dy ** 2) ** 0.5)
                    perp_x = -dy / length
                    perp_y = dx / length
                    mid_x += perp_x * 15 * (1 if rad_offset > 0 else -1)
                    mid_y += perp_y * 15 * (1 if rad_offset > 0 else -1)

                if cfg.display_relation_labels:
                    raw = self._humanize_relation(rel.get("relation", "near"))
                    rel_label_specs.append({"text": raw, "query_pt": (mid_x, mid_y), "color": col})

                rel_draw_data.append({"src_pt": start, "tgt_pt": end, "color": col, "rad": rad_offset})

        # STEP 1: disegna label oggetti esterne e risolvi solo le loro sovrapposizioni
        obj_texts: List[Any] = []
        obj_anchors: List[Tuple[float, float]] = []
        for (pt, text, color) in detection_labels_info:
            font_col = text_color_for_bg(color)
            t = ax.text(
                pt[0],
                pt[1],
                text,
                fontsize=cfg.obj_fontsize_outside,
                color=font_col,
                bbox=dict(facecolor=color, alpha=0.6, edgecolor=color, linewidth=2.0, boxstyle="round,pad=0.25"),
                zorder=7,
            )
            obj_texts.append(t)
            obj_anchors.append(pt)

        if obj_texts and cfg.resolve_overlaps:
            fig.canvas.draw()
            self._resolve_object_overlaps_only(ax, obj_texts, obj_anchors)

        # STEP 2: disegna frecce (solo frecce, senza etichette) e verifica non overlapping
        if cfg.display_relationships and rel_draw_data:
            fig.canvas.draw()
            SHRINK_PX = 6
            for d in rel_draw_data:
                p0, p1 = self._shrink_segment_px(d["src_pt"], d["tgt_pt"], SHRINK_PX, ax)
                arrow = patches.FancyArrowPatch(
                    p0,
                    p1,
                    arrowstyle="->",
                    color=d["color"],
                    linewidth=cfg.rel_arrow_linewidth,
                    connectionstyle=f"arc3,rad={d['rad']}",
                    mutation_scale=cfg.rel_arrow_mutation_scale,
                    zorder=4,
                )
                ax.add_patch(arrow)
                arrow_patches.append(arrow)

            # Verifica e correggi sovrapposizioni tra frecce
            if cfg.resolve_overlaps:
                fig.canvas.draw()
                self._resolve_arrow_overlaps(ax, arrow_patches, rel_draw_data)

            # STEP 3: aggiungi etichette relazioni al centro delle frecce
            if cfg.display_relation_labels and rel_label_specs:
                for spec, arrow in zip(rel_label_specs, arrow_patches):
                    # Posiziona al centro della freccia o leggermente spostata se troppo piccola
                    center_pos = self._get_optimal_relation_label_position(ax, arrow, spec["text"])
                    
                    tr = ax.text(
                        center_pos[0],
                        center_pos[1],
                        spec["text"],
                        fontsize=cfg.rel_fontsize,
                        ha="center",
                        va="center",
                        color="black",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85, edgecolor=spec["color"], linewidth=2),
                        zorder=5,
                    )
                    rel_texts.append(tr)
                    rel_anchors.append(center_pos)

                # STEP 4a: risolvi sovrapposizioni etichette relazioni vs etichette oggetti
                if cfg.resolve_overlaps:
                    fig.canvas.draw()
                    self._resolve_relation_vs_object_overlaps(ax, obj_texts, rel_texts, arrow_patches, cfg.relation_label_max_dist_px)

                # STEP 4b: risolvi sovrapposizioni tra etichette relazioni
                if cfg.resolve_overlaps:
                    fig.canvas.draw()
                    self._resolve_relation_vs_relation_overlaps(ax, rel_texts, arrow_patches, cfg.relation_label_max_dist_px)

        # 8) connector lines
        for t, pt in zip(obj_texts, obj_anchors):
            ax.annotate("", xy=pt, xytext=t.get_position(),
                        arrowprops=dict(arrowstyle="-", color="gray", alpha=0.45, shrinkA=4, shrinkB=4, linewidth=1, linestyle="-"),
                        zorder=6)

        if cfg.display_relationships and cfg.display_relation_labels and rel_texts and arrow_patches:
            for t_rel, arrow in zip(rel_texts, arrow_patches):
                xt, yt = t_rel.get_position()
                near_pt = self._nearest_point_on_arrow(ax, arrow, xt, yt)
                ax.annotate("", xy=near_pt, xytext=(xt, yt),
                            arrowprops=dict(arrowstyle="-", color="gray", alpha=0.45, shrinkA=4, shrinkB=4, linewidth=1, linestyle="dotted"),
                            zorder=6)

        # 9) legenda
        if cfg.display_legend and len(labels) > 0:
            uniq_base = sorted({lab.rsplit("_", 1)[0] for lab in labels})
            handles = [patches.Patch(color=self._pick_color(lb, 0), label=lb) for lb in uniq_base[:10]]
            if handles:
                ax.legend(handles=handles, fontsize=cfg.legend_fontsize, loc="upper right")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches="tight", transparent=(not draw_background and (len(bg_color) == 4 and bg_color[3] == 0)))
            plt.close(fig)
        else:
            plt.show()

    # ------------------------------------------------------------------ pipeline methods

    def _extract_depth_index(self, label: str, fallback_index: int, metadata: Optional[Dict[str, Any]] = None) -> int:
        """Estrae l'indice di profondità dal label, metadati o usa l'indice di fallback."""
        # Prima prova con metadati espliciti
        if metadata and self.cfg.depth_key in metadata:
            try:
                return int(metadata[self.cfg.depth_key])
            except (ValueError, TypeError):
                pass
        
        # Poi cerca pattern nel label come "object_1", "person_2", etc.
        import re
        match = re.search(r'_(\d+)$', label)
        if match:
            return int(match.group(1))
        
        # Se non trova pattern, usa l'indice dell'oggetto
        return fallback_index

    def _draw_box(self, ax, x1: int, y1: int, x2: int, y2: int, color: str, linewidth: float, zorder: float = 2) -> None:
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

    def _draw_mask(self, ax, mask: np.ndarray, color: str, linewidth: float, zorder: float = 2) -> None:
        """Contorno (sempre) e riempimento (se attivo) con z-order specificato."""
        if cv2 is None:
            ax.imshow(mask.astype(float), alpha=self.cfg.seg_fill_alpha, extent=(0, mask.shape[1], mask.shape[0], 0), zorder=zorder)
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
            
            # Riempimento con z-order per la profondità
            if self.cfg.fill_segmentation:
                ax.fill(cnt[:, 0], cnt[:, 1], color=color, alpha=self.cfg.seg_fill_alpha, zorder=zorder)
            
            # Contorno sempre sopra il riempimento
            ax.plot(cnt[:, 0], cnt[:, 1], color=color, linewidth=linewidth, alpha=0.95, zorder=zorder + 0.05)

    def _resolve_object_overlaps_only(
        self,
        ax,
        obj_texts: List[Any],
        obj_anchors: List[Tuple[float, float]]
    ) -> None:
        """STEP 1: Risolve solo sovrapposizioni tra etichette degli oggetti."""
        if not obj_texts:
            return
            
        # Usa il metodo esistente solo per gli oggetti
        self._resolve_overlaps(ax, movable_texts=obj_texts, movable_anchors=obj_anchors)

    def _resolve_arrow_overlaps(
        self,
        ax,
        arrows: List[Any],
        arrow_data: List[Dict[str, Any]]
    ) -> None:
        """STEP 2: Verifica e corregge sovrapposizioni tra frecce."""
        if not arrows:
            return
            
        fig = ax.figure
        renderer = fig.canvas.get_renderer()
        max_iterations = 10
        
        for iteration in range(max_iterations):
            moved = False
            arrow_bbs = self._arrow_bboxes_px(arrows, renderer)
            
            # Controlla sovrapposizioni tra frecce
            for i in range(len(arrow_bbs)):
                for j in range(i + 1, len(arrow_bbs)):
                    if arrow_bbs[i].overlaps(arrow_bbs[j]):
                        # Modifica leggermente la curvatura delle frecce sovrapposte
                        # Aumenta il raggio di curvatura per separarle
                        if i < len(arrow_data):
                            arrow_data[i]["rad"] += 0.05
                        if j < len(arrow_data):
                            arrow_data[j]["rad"] -= 0.05
                        
                        # Rimuovi le frecce esistenti e ricreale
                        arrows[i].remove()
                        arrows[j].remove()
                        
                        # Ricrea frecce con nuova curvatura
                        for idx, arrow_idx in enumerate([i, j]):
                            d = arrow_data[arrow_idx]
                            p0, p1 = self._shrink_segment_px(d["src_pt"], d["tgt_pt"], 6, ax)
                            new_arrow = patches.FancyArrowPatch(
                                p0, p1,
                                arrowstyle="->",
                                color=d["color"],
                                linewidth=self.cfg.rel_arrow_linewidth,
                                connectionstyle=f"arc3,rad={d['rad']}",
                                mutation_scale=self.cfg.rel_arrow_mutation_scale,
                                zorder=4,
                            )
                            ax.add_patch(new_arrow)
                            arrows[arrow_idx] = new_arrow
                        
                        moved = True
                        break
                if moved:
                    break
            
            if not moved:
                break
            
            fig.canvas.draw()

    def _get_optimal_relation_label_position(
        self,
        ax,
        arrow,
        text: str
    ) -> Tuple[float, float]:
        """STEP 3: Calcola posizione ottimale per etichetta relazione (centro o spostata se freccia piccola)."""
        # Calcola la lunghezza della freccia
        arrow_length_px = self._get_arrow_length_px(ax, arrow)
        
        # Stima dimensioni del testo
        text_width_px, text_height_px = self._estimate_text_px(ax, text, self.cfg.rel_fontsize)
        text_diagonal_px = np.sqrt(text_width_px**2 + text_height_px**2)
        
        # Se la freccia è abbastanza lunga, posiziona al centro
        if arrow_length_px > text_diagonal_px * 1.5:
            return self._get_arrow_center(ax, arrow)
        else:
            # Freccia piccola: sposta leggermente di lato
            center_pos = self._get_arrow_center(ax, arrow)
            offset_px = text_diagonal_px * 0.7
            
            # Trova la direzione perpendicolare alla freccia
            verts_disp = self._arrow_vertices_disp(arrow)
            if len(verts_disp) >= 2:
                # Calcola vettore tangente
                tangent = verts_disp[-1] - verts_disp[0]
                tangent_norm = max(np.linalg.norm(tangent), 1e-9)
                tangent /= tangent_norm
                
                # Vettore normale (perpendicolare)
                normal = np.array([-tangent[1], tangent[0]])
                
                # Converti offset in coordinate dati
                to_data = ax.transData.inverted().transform
                to_disp = ax.transData.transform
                
                center_disp = to_disp(center_pos)
                offset_disp = center_disp + normal * offset_px
                offset_data = to_data(offset_disp)
                
                return tuple(offset_data)
            
            return center_pos

    def _get_arrow_length_px(self, ax, arrow) -> float:
        """Calcola la lunghezza della freccia in pixel."""
        try:
            verts_disp = self._arrow_vertices_disp(arrow)
            if len(verts_disp) < 2:
                return 0.0
            
            # Somma delle distanze tra vertici consecutivi
            total_length = 0.0
            for i in range(len(verts_disp) - 1):
                segment_vec = verts_disp[i + 1] - verts_disp[i]
                total_length += np.linalg.norm(segment_vec)
            
            return total_length
        except Exception:
            return 0.0

    def _resolve_relation_vs_object_overlaps(
        self,
        ax,
        obj_texts: List[Any],
        rel_texts: List[Any],
        arrows: List[Any],
        max_dist_px: float
    ) -> None:
        """STEP 4a: Risolve sovrapposizioni tra etichette relazioni e oggetti."""
        if not obj_texts or not rel_texts:
            return
            
        fig = ax.figure
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        
        max_iterations = 15
        to_disp = ax.transData.transform
        to_data = ax.transData.inverted().transform
        
        for iteration in range(max_iterations):
            moved = False
            
            obj_bbs = [t.get_window_extent(renderer=renderer).expanded(1.05, 1.1) for t in obj_texts]
            rel_bbs = [t.get_window_extent(renderer=renderer).expanded(1.05, 1.1) for t in rel_texts]
            
            # Controlla ogni etichetta relazione contro ogni etichetta oggetto
            for rel_idx, rel_bb in enumerate(rel_bbs):
                for obj_bb in obj_bbs:
                    if rel_bb.overlaps(obj_bb):
                        # Sposta solo l'etichetta della relazione
                        rel_center = ((rel_bb.x0 + rel_bb.x1) / 2, (rel_bb.y0 + rel_bb.y1) / 2)
                        obj_center = ((obj_bb.x0 + obj_bb.x1) / 2, (obj_bb.y0 + obj_bb.y1) / 2)
                        
                        # Vettore di allontanamento
                        push_x = rel_center[0] - obj_center[0]
                        push_y = rel_center[1] - obj_center[1]
                        push_dist = max(np.sqrt(push_x**2 + push_y**2), 1e-6)
                        
                        # Normalizza e applica spinta
                        push_strength = 10.0
                        push_x = (push_x / push_dist) * push_strength
                        push_y = (push_y / push_dist) * push_strength
                        
                        # Converti in coordinate dati e sposta
                        dx, dy = self._pixels_to_data(ax, push_x, push_y)
                        current_pos = rel_texts[rel_idx].get_position()
                        new_pos = (current_pos[0] + dx, current_pos[1] + dy)
                        rel_texts[rel_idx].set_position(new_pos)
                        moved = True
                        
                        # Verifica vincolo distanza dalla freccia
                        self._clamp_relation_to_arrow(ax, rel_texts[rel_idx], arrows[rel_idx] if rel_idx < len(arrows) else None, max_dist_px)
            
            if not moved:
                break
                
            fig.canvas.draw_idle()

    def _resolve_relation_vs_relation_overlaps(
        self,
        ax,
        rel_texts: List[Any],
        arrows: List[Any],
        max_dist_px: float
    ) -> None:
        """STEP 4b: Risolve sovrapposizioni tra etichette delle relazioni."""
        if len(rel_texts) < 2:
            return
            
        fig = ax.figure
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        
        max_iterations = 15
        
        for iteration in range(max_iterations):
            moved = False
            
            rel_bbs = [t.get_window_extent(renderer=renderer).expanded(1.05, 1.1) for t in rel_texts]
            
            # Controlla sovrapposizioni tra etichette relazioni
            for i in range(len(rel_bbs)):
                for j in range(i + 1, len(rel_bbs)):
                    if rel_bbs[i].overlaps(rel_bbs[j]):
                        # Sposta entrambe le etichette in direzioni opposte
                        center_i = ((rel_bbs[i].x0 + rel_bbs[i].x1) / 2, (rel_bbs[i].y0 + rel_bbs[i].y1) / 2)
                        center_j = ((rel_bbs[j].x0 + rel_bbs[j].x1) / 2, (rel_bbs[j].y0 + rel_bbs[j].y1) / 2)
                        
                        # Vettore di separazione
                        sep_x = center_j[0] - center_i[0]
                        sep_y = center_j[1] - center_i[1]
                        sep_dist = max(np.sqrt(sep_x**2 + sep_y**2), 1e-6)
                        
                        # Normalizza e applica spinta
                        push_strength = 8.0
                        sep_x = (sep_x / sep_dist) * push_strength
                        sep_y = (sep_y / sep_dist) * push_strength
                        
                        # Converti e sposta
                        dx_i, dy_i = self._pixels_to_data(ax, -sep_x * 0.5, -sep_y * 0.5)
                        dx_j, dy_j = self._pixels_to_data(ax, sep_x * 0.5, sep_y * 0.5)
                        
                        pos_i = rel_texts[i].get_position()
                        pos_j = rel_texts[j].get_position()
                        
                        new_pos_i = (pos_i[0] + dx_i, pos_i[1] + dy_i)
                        new_pos_j = (pos_j[0] + dx_j, pos_j[1] + dy_j)
                        
                        rel_texts[i].set_position(new_pos_i)
                        rel_texts[j].set_position(new_pos_j)
                        moved = True
                        
                        # Verifica vincoli distanza dalle frecce
                        if i < len(arrows):
                            self._clamp_relation_to_arrow(ax, rel_texts[i], arrows[i], max_dist_px)
                        if j < len(arrows):
                            self._clamp_relation_to_arrow(ax, rel_texts[j], arrows[j], max_dist_px)
            
            if not moved:
                break
                
            fig.canvas.draw_idle()

    def _clamp_relation_to_arrow(
        self,
        ax,
        rel_text,
        arrow,
        max_dist_px: float
    ) -> None:
        """Mantiene un'etichetta relazione vicina alla sua freccia."""
        if arrow is None:
            return
            
        to_disp = ax.transData.transform
        to_data = ax.transData.inverted().transform
        
        current_pos = rel_text.get_position()
        rel_pos_disp = to_disp(current_pos)
        verts_disp = self._arrow_vertices_disp(arrow)
        
        if len(verts_disp) > 0:
            _, dist_sq = self._nearest_point_on_polyline_disp(verts_disp, np.array(rel_pos_disp))
            current_dist_px = np.sqrt(dist_sq)
            
            # Se troppo lontana, riportala più vicina
            if current_dist_px > max_dist_px:
                proj, _ = self._nearest_point_on_polyline_disp(verts_disp, np.array(rel_pos_disp))
                direction = np.array(rel_pos_disp) - proj
                dir_norm = max(np.linalg.norm(direction), 1e-6)
                
                clamped_pos_disp = proj + (direction / dir_norm) * max_dist_px
                clamped_pos_data = to_data(clamped_pos_disp)
                rel_text.set_position(clamped_pos_data)

    # ------------------------------------------------------------------ internals (refactor)

    def _get_arrow_center(self, ax, arrow) -> Tuple[float, float]:
        """Calcola il centro geometrico della freccia."""
        try:
            verts_disp = self._arrow_vertices_disp(arrow)
            if len(verts_disp) == 0:
                return (0.0, 0.0)
            
            # Centro geometrico dei vertici
            center_disp = np.mean(verts_disp, axis=0)
            
            # Converti in coordinate dati
            to_data = ax.transData.inverted().transform
            center_data = to_data(center_disp)
            return tuple(center_data)
        except Exception:
            return (0.0, 0.0)

    # profilo parametri centralizzato
    def _profile_params(self):
        dense = (self.cfg.adjust_text_profile == "dense")
        return dict(
            force_text=0.8 if dense else 0.4,
            expand_text=(1.55, 1.55) if dense else (1.05, 1.05),
            expand_points=(1.45, 1.45) if dense else (1.05, 1.05),
            expand_objects=(1.45, 1.45) if dense else (1.05, 1.05),
            push_tt=0.15 if dense else 0.08,  # testo↔testo
            push_ta=0.12 if dense else 0.06,  # testo↔arrow
            push_to=0.14 if dense else 0.08,  # testo↔objlabel
        )

    def _adjust_position(
        self,
        candidate: Tuple[float, float],
        placed_positions: List[Tuple[float, float]],
        overlap_thresh: float,
        max_iterations: int = 10,
    ) -> Tuple[float, float]:
        """Soft-avoidance tra etichette esterne (semplice, locale)."""
        new_candidate = np.array(candidate, dtype=float)
        eps = 1e-6
        for _ in range(max_iterations):
            displacement = np.zeros(2, dtype=float)
            for p in placed_positions:
                diff = new_candidate - np.array(p, dtype=float)
                dist = np.linalg.norm(diff)
                if dist < overlap_thresh:
                    push = (overlap_thresh - dist) * (diff / (dist + eps))
                    displacement += push
            if np.linalg.norm(displacement) < 1e-3:
                break
            new_candidate += displacement
        return tuple(new_candidate)

    def _shrink_segment_px(self, p0, p1, shrink_px, ax):
        """Accorcia p0→p1 di 'shrink_px' pixel alle estremità in spazio display."""
        to_px = ax.transData.transform
        to_data = ax.transData.inverted().transform
        P0 = np.array(to_px(p0))
        P1 = np.array(to_px(p1))
        v = P1 - P0
        L = np.linalg.norm(v)
        if L < 1:
            return p0, p1
        v_norm = v / L
        P0n = P0 + v_norm * shrink_px
        P1n = P1 - v_norm * shrink_px
        return tuple(to_data(P0n)), tuple(to_data(P1n))

    # ---------- overlap handling (UNIFICATO)

    def _resolve_overlaps(
        self,
        ax,
        movable_texts: List[Any],
        movable_anchors: List[Tuple[float, float]],
        fixed_texts: Sequence[Any] = (),
        arrows: Sequence[Any] = (),
    ) -> None:
        """Sistema sovrapposizioni spostando SOLO 'movable_texts'.
        'fixed_texts' e 'arrows' sono ostacoli."""
        if adjust_text is None or not movable_texts:
            return

        prof = self._profile_params()

        # pass 1: adjust_text con ostacoli
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

        # pass 2: micro-push locale
        for _ in range(self.cfg.micro_push_iters):
            moved = False

            mov_bbs = [t.get_window_extent(renderer=renderer).expanded(1.02, 1.08) for t in movable_texts]
            fix_bbs = [t.get_window_extent(renderer=renderer).expanded(1.02, 1.08) for t in fixed_texts] if fixed_texts else []
            arrow_bbs = self._arrow_bboxes_px(arrows, renderer)

            # (a) mobile ↔ mobile
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

            # (b) mobile ↔ arrow
            for k, bb in enumerate(mov_bbs):
                for abb in arrow_bbs:
                    if bb.overlaps(abb):
                        dx_px = (bb.x1 - abb.x0) * prof["push_ta"]
                        dy_px = (bb.y1 - abb.y0) * prof["push_ta"]
                        dx, dy = self._pixels_to_data(ax, dx_px, dy_px)
                        x, y = movable_texts[k].get_position()
                        movable_texts[k].set_position((x + dx, y + dy))
                        moved = True

            # (c) mobile ↔ fixed text
            for k, bb in enumerate(mov_bbs):
                for fbb in fix_bbs:
                    if bb.overlaps(fbb):
                        dx_px = (bb.x1 - fbb.x0) * prof["push_to"]
                        dy_px = (bb.y1 - fbb.y0) * prof["push_to"]
                        dx, dy = self._pixels_to_data(ax, dx_px, dy_px)
                        x, y = movable_texts[k].get_position()
                        movable_texts[k].set_position((x + dx, y + dy))
                        moved = True

            if not moved:
                break
            fig.canvas.draw_idle()

    def _pixels_to_data(self, ax, dx_px, dy_px):
        inv = ax.transData.inverted()
        x0, y0 = inv.transform((0, 0))
        x1, y1 = inv.transform((dx_px, dy_px))
        return x1 - x0, y1 - y0

    def _arrow_bboxes_px(self, arrows: Sequence[Any], renderer):
        """BBox (in pixel) delle frecce come oggetti di collisione."""
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

    # ---------- geometria frecce / punti più vicini

    @staticmethod
    def _arrow_vertices_disp(arrow) -> np.ndarray:
        """Ritorna i vertici della freccia in display-space."""
        path = arrow.get_path().transformed(arrow.get_transform())
        return np.asarray(path.vertices, dtype=float)

    @staticmethod
    def _nearest_point_on_polyline_disp(verts_disp: np.ndarray, P: np.ndarray) -> Tuple[np.ndarray, float]:
        """Punto più vicino su una polilinea (display-space)."""
        best_d2 = float("inf")
        best = verts_disp[0]
        for i in range(len(verts_disp) - 1):
            a = verts_disp[i]
            b = verts_disp[i + 1]
            ab = b - a
            ab2 = float(np.dot(ab, ab))
            if ab2 <= 1e-9:
                continue
            t = float(np.clip(np.dot(P - a, ab) / ab2, 0.0, 1.0))
            proj = a + t * ab
            d2 = float(np.sum((P - proj) ** 2))
            if d2 < best_d2:
                best_d2 = d2
                best = proj
        return best, best_d2

    def _nearest_point_on_arrow(self, ax, arrow, x, y):
        """Compat: dato (x,y) in data-space, ritorna il punto più vicino sulla freccia (data-space)."""
        try:
            to_disp = ax.transData.transform
            to_data = ax.transData.inverted().transform
            verts_disp = self._arrow_vertices_disp(arrow)
            P = np.asarray(to_disp((x, y)), dtype=float)
            proj, _ = self._nearest_point_on_polyline_disp(verts_disp, P)
            return tuple(to_data(tuple(proj)))
        except Exception:
            return (x, y)

    def _label_pos_near_arrow(
        self, ax, arrow, query_pt: Tuple[float, float], offset_px: float
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Trova il punto più vicino sulla freccia a 'query_pt' e posiziona l'etichetta
        a offset_px pixel lungo la normale in display-space. Ritorna (pos_data, anchor_data)."""
        to_disp = ax.transData.transform
        to_data = ax.transData.inverted().transform
        verts_disp = self._arrow_vertices_disp(arrow)
        if len(verts_disp) < 2:
            anchor_data = self._nearest_point_on_arrow(ax, arrow, *query_pt)
            dx, dy = self._pixels_to_data(ax, 0, -offset_px)
            return (anchor_data[0] + dx, anchor_data[1] + dy), anchor_data

        P = np.asarray(to_disp(query_pt), dtype=float)
        proj, _ = self._nearest_point_on_polyline_disp(verts_disp, P)

        # tangente locale (prendi il segmento più vicino)
        idx = np.argmin(np.linalg.norm(verts_disp[:-1] + 0.5 * (verts_disp[1:] - verts_disp[:-1]) - proj, axis=1))
        a = verts_disp[idx]
        b = verts_disp[idx + 1]
        tangent = b - a
        L = max(np.linalg.norm(tangent), 1e-9)
        tangent /= L

        normal = np.array([-tangent[1], tangent[0]], dtype=float)
        normal /= max(np.linalg.norm(normal), 1e-9)

        pos_disp = proj + normal * float(offset_px)
        pos_data = tuple(to_data(tuple(pos_disp)))
        anchor_data = tuple(to_data(tuple(proj)))
        return pos_data, anchor_data

    def _clamp_relation_labels_near_arrows(self, ax, texts: Sequence[Any], arrows: Sequence[Any], max_dist_px: float) -> None:
        """Se una label è troppo lontana dalla sua freccia, riportala entro max_dist_px."""
        if not texts:
            return
        to_disp = ax.transData.transform
        to_data = ax.transData.inverted().transform
        for t, arrow in zip(texts, arrows):
            verts_disp = self._arrow_vertices_disp(arrow)
            P = np.asarray(to_disp(t.get_position()), dtype=float)
            proj, d2 = self._nearest_point_on_polyline_disp(verts_disp, P)
            dpx = float(np.sqrt(d2))
            if dpx <= max_dist_px:
                continue
            v = P - proj
            n = max(np.linalg.norm(v), 1e-9)
            newP = proj + v / n * float(max_dist_px)
            x_new, y_new = to_data(tuple(newP))
            t.set_position((x_new, y_new))

    # ---------- altre utility

    def _pick_color(self, label: str, idx: int) -> str:
        """Colore consistente per classe base (label senza suffisso _k)."""
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

    def _best_mask(self, i: int, masks: Optional[Sequence[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
        if masks is None or i >= len(masks) or masks[i] is None:
            return None
        return masks[i]

    def _format_label_text(self, label: str, score: float, obj_index: int = 0) -> str:
        """Testo etichetta oggetto."""
        mode = self.cfg.label_mode
        base = label.rsplit("_", 1)[0]
        if mode == "numeric":
            text = str(obj_index + 1)  # Solo il numero
        elif mode == "alphabetic":
            n = obj_index
            alphabet_label = ""
            while True:
                alphabet_label = chr(65 + (n % 26)) + alphabet_label
                n //= 26
                if n == 0:
                    break
                n -= 1
            text = alphabet_label  # Solo la lettera
        else:  # mode == "original"
            text = base  # Solo il nome base senza suffisso numerico
        
        if self.cfg.show_confidence:
            text = f"{text} ({score * 100:.0f}%)"
        return text

    def _estimate_text_px(self, ax, text: str, fontsize_px: int) -> Tuple[float, float]:
        """Stima dimensioni del testo (renderer se disponibile)."""
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

    def _can_place_inside(
        self,
        image: Image.Image,
        box: Sequence[float],
        mask_dict: Optional[Dict[str, Any]],
        label_text: str,
        ax=None,
    ) -> bool:
        """Decide se posizionare l'etichetta all'interno dell'oggetto."""
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
        half_diag = 0.5 * ((w_txt ** 2 + h_txt ** 2) ** 0.5)
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

    # ------------------------------------------------------------------ rel filtering

    def _filter_redundant_relations(self, relationships: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Mantiene la relazione più importante per coppia (priorità+confidence)."""
        if not relationships:
            return list(relationships)

        from collections import defaultdict

        pair_relations: Dict[Tuple[int, int], List[Dict[str, Any]]] = defaultdict(list)
        for rel in relationships:
            s0, t0 = int(rel["src_idx"]), int(rel["tgt_idx"])
            pair_key = tuple(sorted([s0, t0]))
            pair_relations[pair_key].append(dict(rel))

        filtered_relations: List[Dict[str, Any]] = []
        for _, rels in pair_relations.items():
            filtered_relations.append(rels[0] if len(rels) == 1 else self._choose_best_relation(rels))
        return filtered_relations

    def _choose_best_relation(self, relations: List[Dict[str, Any]]) -> Dict[str, Any]:
        best_rel = relations[0]
        best_priority = self._get_relation_priority(best_rel["relation"])  # type: ignore[index]
        best_confidence = self._get_relation_confidence(best_rel)
        for rel in relations[1:]:
            priority = self._get_relation_priority(rel["relation"])  # type: ignore[index]
            confidence = self._get_relation_confidence(rel)
            if priority > best_priority or (priority == best_priority and confidence > best_confidence):
                best_rel, best_priority, best_confidence = rel, priority, confidence
        return best_rel

    def _cap_relations_per_object(self, relationships: Sequence[Dict[str, Any]], boxes: Sequence[Sequence[float]]) -> List[Dict[str, Any]]:
        """Limita relazioni in uscita per oggetto (priorità alta, distanza corta)."""
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
            return float(relation["clip_sim"])  # type: ignore[return-value]
        if "distance" in relation:
            dist = float(relation["distance"])  # type: ignore[assignment]
            return 1.0 / (1.0 + dist / 100.0)
        return 0.5

    # ------------------------------------------------------------------ utils

    @staticmethod
    def _humanize_relation(rel: str) -> str:
        s = str(rel)
        if any(c.isupper() for c in s):
            import re as _re
            s = _re.sub(r"(?<!^)(?=[A-Z])", " ", s)
        return s.replace("_", " ").strip().title()