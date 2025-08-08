# igp/viz/visualizer.py
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
    from igp.utils.colors import ColorCycler, text_color_for_bg
    
    # Crea istanza globale
    _color_cycler = ColorCycler()
    
    def color_for_label(
        label: str, 
        idx: int = 0, 
        sat_boost: float = 1.3, 
        val_boost: float = 1.15, 
        cache: dict = None
    ) -> str:
        """Wrapper per ColorCycler.color_for_label"""
        return _color_cycler.color_for_label(label)


@dataclass
class VisualizerConfig:
    # cosa mostrare
    display_labels: bool = True
    display_relationships: bool = True
    display_relation_labels: bool = False
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

    # etichette
    label_mode: str = "original"  # "original" | "numeric" | "alphabetic"
    show_confidence: bool = False

    # posizionamento etichette interne
    min_area_ratio_inside: float = 0.006  # 0.6% dell'immagine
    inside_label_margin_px: int = 6
    min_solidity_inside: float = 0.45
    measure_text_with_renderer: bool = False

    # risoluzione sovrapposizioni
    resolve_overlaps: bool = True

    # color tweaks
    color_sat_boost: float = 1.30
    color_val_boost: float = 1.15

    # relazione speciale: "on_top_of" robusto (bande contatto ecc.)
    on_top_gap_px: int = 8
    on_top_horiz_overlap: float = 0.35  # overlap orizzontale minimo (ratio)


class Visualizer:
    """
    Visualizza:
      - maschere SAM (opzionale, con riempimento/contorno)
      - bounding boxes (opzionale)
      - etichette dentro/fuori gli oggetti con contrasto automatico
      - frecce di relazione (con eventuali etichette)
      - legenda per classi (max 10 voci)
    Input atteso:
      - boxes: list[[x1,y1,x2,y2]]
      - labels: list[str]   (già canonizzate ed opzionalmente con suffisso _k)
      - scores: list[float]
      - relationships: list[dict] con chiavi 'src_idx', 'tgt_idx', 'relation' (opzionale 'distance')
      - masks: list[dict] con 'segmentation': np.ndarray(bool, H, W)  e 'bbox' xywh
    """
    SPATIAL_KEYS = ("left_of", "right_of", "above", "below", "on_top_of", "under", "in_front_of", "behind")

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
        """
        Disegna l'overlay e salva/mostra il risultato.
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        W, H = image.size

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

        # colori per ciascun oggetto (per classe base)
        obj_colors = [self._pick_color(labels[i], i) for i in range(len(boxes))]

        # etichette da posizionare esternamente (pt, text, color)
        external_labels: List[Tuple[Tuple[float, float], str, str]] = []

        # 1) Disegna oggetti (maschere/box + etichette)
        centers = []
        for i, box in enumerate(boxes):
            col = obj_colors[i]
            x1, y1, x2, y2 = map(int, box[:4])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            centers.append((cx, cy))

            best_mask = self._best_mask(i, masks)
            if self.cfg.show_segmentation and best_mask is not None and best_mask.get("segmentation") is not None:
                self._draw_mask(ax, best_mask["segmentation"], color=col, linewidth=self.cfg.bbox_linewidth)
            elif self.cfg.show_bboxes:
                self._draw_box(ax, x1, y1, x2, y2, color=col, linewidth=self.cfg.bbox_linewidth)

            # testo oggetto
            if self.cfg.display_labels:
                label_text = self._format_label_text(labels[i], scores[i], obj_index=i)
                place_inside = self._can_place_inside(
                    image, box, best_mask, label_text, ax if self.cfg.measure_text_with_renderer else None
                )
                if place_inside:
                    txt_col = text_color_for_bg(col)
                    ax.text(
                        (x1 + x2) / 2,
                        (y1 + y2) / 2,
                        label_text,
                        ha="center",
                        va="center",
                        fontsize=self.cfg.obj_fontsize_inside,
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
                    external_labels.append(((cx, cy), label_text, col))

        # 2) Relazioni
        arrow_patches: List[patches.FancyArrowPatch] = []
        rel_text_artists: List[Any] = []
        rel_text_anchors: List[Tuple[float, float]] = []

        if self.cfg.display_relationships and len(relationships) > 0:
            arrow_counts: Dict[Tuple[int, int], int] = {}

            for rel in relationships:
                s0, t0 = int(rel["src_idx"]), int(rel["tgt_idx"])
                name = str(rel.get("relation", "")).lower()

                # direzione freccia per relazioni spaziali (inverti)
                if any(k in name for k in self.SPATIAL_KEYS):
                    s, t = t0, s0
                else:
                    s, t = s0, t0

                if s >= len(centers) or t >= len(centers) or s < 0 or t < 0:
                    continue

                start = centers[s]
                end = centers[t]
                col = obj_colors[s]

                # offset radiale in caso di più frecce tra la stessa coppia
                arrow_counts[(s, t)] = arrow_counts.get((s, t), 0) + 1
                rad_offset = 0.2 + 0.1 * (arrow_counts[(s, t)] - 1)

                p0, p1 = self._shrink_segment(ax, start, end, shrink_px=6)
                arr = patches.FancyArrowPatch(
                    p0,
                    p1,
                    arrowstyle="->",
                    color=col,
                    linewidth=self.cfg.rel_arrow_linewidth,
                    connectionstyle=f"arc3,rad={rad_offset}",
                    mutation_scale=self.cfg.rel_arrow_mutation_scale,
                    zorder=4,
                )
                ax.add_patch(arr)
                arrow_patches.append(arr)

                if self.cfg.display_relation_labels:
                    midx = (start[0] + end[0]) / 2.0
                    midy = (start[1] + end[1]) / 2.0
                    # piccola deviazione ortogonale per airflow
                    dx = end[0] - start[0]
                    dy = end[1] - start[1]
                    ln = max(1e-6, (dx ** 2 + dy ** 2) ** 0.5)
                    ox = -dy / ln
                    oy = dx / ln
                    midx += ox * 8
                    midy += oy * 8

                    raw = rel.get("relation", "near")
                    raw = self._humanize_relation(raw)
                    tr = ax.text(
                        midx,
                        midy,
                        raw,
                        fontsize=self.cfg.rel_fontsize,
                        ha="center",
                        va="center",
                        color="black",
                        bbox=dict(
                            boxstyle="round,pad=0.2",
                            facecolor="white",
                            alpha=0.85,
                            edgecolor=col,
                            linewidth=2,
                        ),
                        zorder=5,
                    )
                    rel_text_artists.append(tr)
                    rel_text_anchors.append(((start[0] + end[0]) / 2.0, (start[1] + end[1]) / 2.0))

        # 3) Etichette esterne + risoluzione overlap
        obj_text_artists: List[Any] = []
        obj_text_anchors: List[Tuple[float, float]] = []
        for (pt, txt, col) in external_labels:
            txt_col = text_color_for_bg(col)
            t = ax.text(
                pt[0],
                pt[1],
                txt,
                fontsize=self.cfg.obj_fontsize_outside,
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
            obj_text_artists.append(t)
            obj_text_anchors.append(pt)

        # Regolazione delle sovrapposizioni (prima senza frecce, poi con)
        if self.cfg.resolve_overlaps and (obj_text_artists or rel_text_artists) and adjust_text is not None:
            # 1) muove solo i testi, con ancore come punti
            all_texts = rel_text_artists + obj_text_artists
            all_x = [p[0] for p in rel_text_anchors] + [p[0] for p in obj_text_anchors]
            all_y = [p[1] for p in rel_text_anchors] + [p[1] for p in obj_text_anchors]

            adjust_text(
                all_texts,
                x=all_x,
                y=all_y,
                ax=ax,
                only_move={"points": "y", "text": "xy"},
                # ✅ RIDOTTO: parametri più conservativi per le etichette di relazione
                force_text=0.5,              # Ridotto da 0.8 - meno spinta tra etichette
                force_points=0.8,            # Aumentato da 0.6 - più attrazione verso punti originali
                expand_text=(1.1, 1.1),      # Ridotto da (1.3, 1.3) - meno espansione
                expand_points=(1.05, 1.05),  # Ridotto da (1.2, 1.2)
                expand_objects=(1.05, 1.05), # Ridotto da (1.1, 1.1)
                # ✅ CONTROLLO QUALITÀ più conservativo
                lim=100,                     # Ridotto ulteriormente per meno movimenti
                precision=0.05,              # Meno preciso per convergenza più veloce
                add_objects=arrow_patches,
                arrowprops=dict(
                    arrowstyle="-",
                    color="gray",
                    alpha=0.45,
                    shrinkA=4,
                    shrinkB=4,
                    linewidth=1,
                    linestyle="dotted",
                ),
            )

        # 4) Connector lines (punti → etichette)
        #    - per oggetti: linea piena
        #    - per relazioni: linea tratteggiata verso il punto più vicino sulla freccia
        for t, pt in zip(obj_text_artists, obj_text_anchors):
            ax.annotate(
                "",
                xy=pt,
                xytext=t.get_position(),
                arrowprops=dict(
                    arrowstyle="-",
                    color="gray",
                    alpha=0.45,
                    shrinkA=4,
                    shrinkB=4,
                    linewidth=1,
                    linestyle="-",
                ),
                zorder=6,
            )

        if self.cfg.display_relationships and self.cfg.display_relation_labels and rel_text_artists and arrow_patches:
            for tr in rel_text_artists:
                xt, yt = tr.get_position()
                near_pt = self._nearest_point_on_any_arrow(ax, arrow_patches, xt, yt)
                ax.annotate(
                    "",
                    xy=near_pt,
                    xytext=(xt, yt),
                    arrowprops=dict(
                        arrowstyle="-",
                        color="gray",
                        alpha=0.45,
                        shrinkA=4,
                        shrinkB=4,
                        linewidth=1,
                        linestyle="dotted",
                    ),
                    zorder=6,
                )

        # 5) Legenda (max 10 classi base)
        if self.cfg.display_legend and len(labels) > 0:
            uniq_base = sorted({lab.rsplit("_", 1)[0] for lab in labels})
            handles = [
                patches.Patch(color=self._pick_color(lb, 0), label=lb) for lb in uniq_base[:10]
            ]
            if handles:
                ax.legend(handles=handles, fontsize=self.cfg.legend_fontsize, loc="upper right")

        plt.tight_layout()
        if save_path:
            plt.savefig(
                save_path,
                bbox_inches="tight",
                transparent=(not draw_background and (len(bg_color) == 4 and bg_color[3] == 0)),
            )
            plt.close(fig)
        else:
            plt.show()

    # ------------------------------------------------------------------ internals

    def _pick_color(self, label: str, idx: int) -> str:
        """
        Colore consistente per classe di base (label senza suffisso _k),
        con boost di saturazione/valore configurabile.
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

    def _best_mask(self, i: int, masks: Optional[Sequence[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
        if masks is None or i >= len(masks) or masks[i] is None:
            return None
        return masks[i]

    def _draw_box(self, ax, x1: int, y1: int, x2: int, y2: int, color: str, linewidth: float) -> None:
        rect = patches.Rectangle(
            (x1, y1),
            max(1, x2 - x1),
            max(1, y2 - y1),
            linewidth=linewidth,
            edgecolor=color,
            facecolor="none",
            zorder=2,
        )
        ax.add_patch(rect)

    def _draw_mask(self, ax, mask: np.ndarray, color: str, linewidth: float) -> None:
        """
        Disegna contorno (sempre) e riempimento (se config.fill_segmentation).
        Richiede OpenCV per l’estrazione dei contorni; se non disponibile,
        effettua un overlay semitrasparente senza contorni.
        """
        if cv2 is None:
            # fallback: overlay pieno
            ax.imshow(
                mask.astype(float),
                cmap=None,
                alpha=self.cfg.seg_fill_alpha,
                extent=(0, mask.shape[1], mask.shape[0], 0),
            )
            return

        mask_uint8 = (mask.astype(np.uint8) * 255) if mask.dtype != np.uint8 else mask.copy()
        if mask_uint8.max() == 1:
            mask_uint8 *= 255

        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return

        # riempimento + contorni
        for cnt in contours:
            cnt = cnt.squeeze()
            if cnt.ndim != 2 or len(cnt) < 3:
                continue
            if self.cfg.fill_segmentation:
                ax.fill(
                    cnt[:, 0],
                    cnt[:, 1],
                    color=color,
                    alpha=self.cfg.seg_fill_alpha,
                    zorder=1,
                )
            ax.plot(
                cnt[:, 0],
                cnt[:, 1],
                color=color,
                linewidth=linewidth,
                alpha=0.95,
                zorder=2,
            )

    def _format_label_text(self, label: str, score: float, obj_index: int = 0) -> str:
        """
        Formatta il testo dell'etichetta con numerazione automatica per mode="original".
        """
        mode = self.cfg.label_mode
        base = label.rsplit("_", 1)[0]  # rimuovi suffissi esistenti _k
        
        if mode == "numeric":
            text = f"{obj_index + 1}"  # solo numero
        elif mode == "alphabetic":
            # Converte in lettere: 0→A, 1→B, ..., 25→Z, 26→AA, etc.
            alphabet_label = ""
            n = obj_index
            while True:
                alphabet_label = chr(65 + (n % 26)) + alphabet_label
                n = n // 26
                if n == 0:
                    break
            text = alphabet_label
        else:  # mode == "original"
            text = f"{base}_{obj_index + 1}"
            
        if self.cfg.show_confidence:
            text = f"{text} ({score * 100:.0f}%)"
        return text

    def _estimate_text_px(self, ax, text: str, fontsize_px: int) -> Tuple[float, float]:
        """
        Stima dimensioni del testo in pixel. Con renderer attivo è precisa.
        """
        if self.cfg.measure_text_with_renderer and ax is not None:
            t = ax.text(0, 0, text, fontsize=fontsize_px, alpha=0)
            fig = ax.figure
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            bb = t.get_window_extent(renderer=renderer)
            t.remove()
            return bb.width, bb.height
        # euristica
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
        """
        Decide se posizionare l’etichetta all’interno del box/maschera.
        Richiede OpenCV per DT su mask; in assenza usa fallback prudente.
        """
        W, H = image.size
        area_img = float(W * H)

        x1, y1, x2, y2 = map(int, box[:4])
        w, h = max(1, x2 - x1), max(1, y2 - y1)
        area_bbox = float(w * h)

        mask_bool = None
        area_obj = area_bbox
        solidity = min(w, h) / float(max(1, max(w, h)))  # proxy

        if mask_dict is not None and mask_dict.get("segmentation") is not None:
            m = mask_dict["segmentation"].astype(bool)
            area_mask = int(m.sum())
            if area_mask > 0:
                area_obj = float(area_mask)
                solidity = area_mask / max(1.0, area_bbox)
            mask_bool = m

        # (1) area normalizzata minima
        if (area_obj / area_img) < float(self.cfg.min_area_ratio_inside):
            return False

        # (2) r_max vs diagonale testo
        w_txt, h_txt = self._estimate_text_px(ax, label_text, self.cfg.obj_fontsize_inside)
        half_diag = 0.5 * ((w_txt ** 2 + h_txt ** 2) ** 0.5)
        margin_px = float(self.cfg.inside_label_margin_px)

        if mask_bool is not None and cv2 is not None:
            m = (mask_bool.astype(np.uint8) * 255)
            # closing leggero per stabilità DT
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
            dist = cv2.distanceTransform(m, cv2.DIST_L2, 5)
            r_max = float(dist.max())
        else:
            r_max = 0.5 * min(w, h) * 0.7  # fallback prudente

        if r_max < (half_diag + margin_px):
            return False

        # (3) oggetti sottili/poco compatti → meglio fuori
        if solidity < float(self.cfg.min_solidity_inside):
            return False

        return True

    def _shrink_segment(self, ax, p0: Tuple[float, float], p1: Tuple[float, float], shrink_px: float):
        """
        Accorcia il segmento di 'shrink_px' pixel alle estremità lavorando in spazio “pixel”,
        poi riporta in data coords.
        """
        to_px = ax.transData.transform
        to_data = ax.transData.inverted().transform
        P0 = np.array(to_px(p0))
        P1 = np.array(to_px(p1))
        v = P1 - P0
        L = float(np.linalg.norm(v))
        if L < 1:
            return p0, p1
        v_norm = v / L
        P0n = P0 + v_norm * shrink_px
        P1n = P1 - v_norm * shrink_px
        return tuple(to_data(P0n)), tuple(to_data(P1n))

    def _nearest_point_on_any_arrow(self, ax, arrows: List[patches.FancyArrowPatch], x: float, y: float) -> Tuple[float, float]:
        """
        Ritorna il punto più vicino sulle curve delle frecce ad (x, y).
        """
        best = None
        best_d2 = float("inf")
        for arr in arrows:
            try:
                path = arr.get_path().transformed(arr.get_transform())
                verts_disp = path.vertices
                verts_data = ax.transData.inverted().transform(verts_disp)
                p = np.array([x, y], dtype=float)
                for i in range(len(verts_data) - 1):
                    a = verts_data[i]
                    b = verts_data[i + 1]
                    ab = b - a
                    ab2 = float(np.dot(ab, ab))
                    if ab2 <= 0:
                        proj = a
                    else:
                        t = float(np.clip(np.dot(p - a, ab) / ab2, 0.0, 1.0))
                        proj = a + t * ab
                    d2 = float(np.sum((p - proj) ** 2))
                    if d2 < best_d2:
                        best_d2 = d2
                        best = proj
            except Exception:
                continue
        return tuple(best) if best is not None else (x, y)

    @staticmethod
    def _humanize_relation(rel: str) -> str:
        """
        Converte snake/camel in 'Title Case' per la label delle relazioni.
        """
        s = str(rel)
        if any(c.isupper() for c in s):
            # split camel
            import re as _re
            s = _re.sub(r"(?<!^)(?=[A-Z])", " ", s)
        s = s.replace("_", " ").strip().title()
        return s
