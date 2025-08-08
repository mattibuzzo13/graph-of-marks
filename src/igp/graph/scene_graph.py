# igp/graph/scene_graph.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple, Optional, Dict, Any

import math
import json
import gzip
from pathlib import Path

import numpy as np
import networkx as nx
from PIL import Image

from igp.utils.boxes import iou as iou_xyxy, to_xywh, center, union as union_box
from igp.utils.clip_utils import CLIPWrapper
from igp.utils.caption import Captioner
from igp.utils.depth import DepthEstimator


@dataclass
class SceneGraphConfig:
    """
    Soglie e flag per la costruzione del grafo.
    """
    # Edge pruning
    max_dist_norm: float = 0.4          # scarta coppie molto lontane (dist_norm > 0.4)
    min_iou_keep: float = 0.01          # richiede un po' di overlap se simil. CLIP è bassa
    min_clip_sim_keep: float = 0.20     # se iou < min_iou_keep, serve almeno questa simil. CLIP

    # Nodi “scene”
    add_scene_node: bool = True
    add_caption: bool = True

    # Feature
    store_clip_embeddings: bool = True
    store_depth: bool = True
    store_color: bool = True

    # Dominant color (best-effort)
    kmeans_clusters: int = 3

    # Device fallback è gestito dai wrapper (CLIP/Caption/Depth)


class SceneGraphBuilder:
    """
    Costruisce un grafo scena (NetworkX DiGraph) con attributi compatibili
    con il codice originale:
      - nodi 0..N-1: oggetti (label, score, clip_emb, bbox_norm, area_norm, color, depth_norm)
      - un nodo 'scene' opzionale con caption
      - archi i->j con attributi geometrici (dx_norm, dy_norm, dist_norm, angle_deg, iou, clip_sim, depth_delta)
    """
    def __init__(
        self,
        clip: Optional[CLIPWrapper] = None,
        captioner: Optional[Captioner] = None,
        depth: Optional[DepthEstimator] = None,
        config: Optional[SceneGraphConfig] = None,
    ) -> None:
        self.clip = clip
        self.captioner = captioner
        self.depth = depth
        self.cfg = config or SceneGraphConfig()

    # ------------------------------------------------------------------ public

    def build(
        self,
        image: Image.Image,
        boxes_xyxy: Sequence[Sequence[float]],
        labels: Sequence[str],
        scores: Sequence[float],
    ) -> nx.DiGraph:
        """
        Crea il grafo scena a partire da detections fuse (post-WBF/NMS).
        """
        G = nx.DiGraph()
        W, H = image.size

        # 1) Node features ------------------------------------------------------
        clip_embs = self._compute_clip_embeddings(image, boxes_xyxy)  # (N, D) or None
        dom_colors = self._dominant_colors(image, boxes_xyxy) if self.cfg.store_color else ["unknown"] * len(boxes_xyxy)

        # depth: calcolata ai centroidi
        centres = [center(b) for b in boxes_xyxy]
        depths = self._relative_depth(image, centres) if self.cfg.store_depth else [0.5] * len(boxes_xyxy)

        # Aggiungi nodi oggetto
        for idx, (box, lab, sc) in enumerate(zip(boxes_xyxy, labels, scores)):
            x1, y1, x2, y2 = box[:4]
            area_norm = ((x2 - x1) * (y2 - y1)) / float(max(1, W * H))
            node_attrs: Dict[str, Any] = {
                "label": lab,
                "score": float(sc),
                "bbox_norm": [x1 / W, y1 / H, x2 / W, y2 / H],
                "area_norm": float(area_norm),
            }
            if self.cfg.store_clip_embeddings and clip_embs is not None:
                node_attrs["clip_emb"] = clip_embs[idx]  # list[float]
            if self.cfg.store_color:
                node_attrs["color"] = dom_colors[idx]
            if self.cfg.store_depth:
                node_attrs["depth_norm"] = float(depths[idx])

            G.add_node(idx, **node_attrs)

        # Nodo scena + caption
        scene_id: Optional[int] = None
        if self.cfg.add_scene_node:
            scene_id = len(G)
            cap = self._caption(image) if self.cfg.add_caption else ""
            G.add_node(scene_id, label="scene", caption=cap)
            # collegalo agli oggetti (non diretto)
            for i in range(len(boxes_xyxy)):
                G.add_edge(scene_id, i)

        # 2) Edge features ------------------------------------------------------
        # Prepara simmetria CLIP dagli embedding nodo-nodo
        for i in range(len(boxes_xyxy)):
            for j in range(len(boxes_xyxy)):
                if i == j:
                    continue
                self._maybe_add_edge(G, i, j, boxes_xyxy, W, H)

        return G

    # ------------------------------------------------------------------ io utils

    @staticmethod
    def save_gpickle(G: nx.DiGraph, path: str | Path, compress: bool | None = None) -> None:
        """
        Salva il grafo su disco. Se l'estensione è .gz o compress=True usa gzip.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        use_gz = compress if compress is not None else (path.suffix == ".gz")
        if use_gz:
            with gzip.open(str(path), "wb") as f:
                nx.write_gpickle(G, f)
        else:
            nx.write_gpickle(G, str(path))

    @staticmethod
    def save_json(G: nx.DiGraph, path: str | Path) -> None:
        """
        Salva il grafo in formato node-link JSON (serializzabile).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        def _np_converter(o):
            import numpy as _np
            if isinstance(o, _np.generic):
                return o.item()
            raise TypeError(f"Not JSON serializable: {type(o)}")

        data = nx.node_link_data(G)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, default=_np_converter, indent=2)

    # ------------------------------------------------------------------ internals

    def _compute_clip_embeddings(
        self,
        image: Image.Image,
        boxes_xyxy: Sequence[Sequence[float]],
    ) -> Optional[List[List[float]]]:
        """
        Calcola embedding CLIP per i crop di ciascun box. Ritorna lista di vettori (list[float]) o None.
        """
        if self.clip is None or not self.clip.available():
            return None

        crops: List[Image.Image] = []
        for b in boxes_xyxy:
            x1, y1, x2, y2 = map(int, b[:4])
            x2 = max(x1 + 1, x2)
            y2 = max(y1 + 1, y2)
            crop = image.crop((x1, y1, x2, y2)).convert("RGB")
            crops.append(crop)

        feats = self.clip.image_features(crops)  # torch.Tensor [N, D]
        if feats is None:
            return None
        # normalizzati dal wrapper; converti in Python list
        return feats.detach().cpu().tolist()

    def _dominant_colors(
        self, image: Image.Image, boxes_xyxy: Sequence[Sequence[float]]
    ) -> List[str]:
        """
        Stima "colore dominante" grossolano per ogni box.
        Best-effort: usa KMeans se disponibile, altrimenti "unknown".
        """
        try:
            import cv2  # type: ignore
            from sklearn.cluster import KMeans  # type: ignore
        except Exception:
            return ["unknown"] * len(boxes_xyxy)

        out: List[str] = []
        for b in boxes_xyxy:
            x1, y1, x2, y2 = map(int, b[:4])
            x2 = max(x1 + 1, x2)
            y2 = max(y1 + 1, y2)
            np_crop = np.array(image.crop((x1, y1, x2, y2)).convert("RGB"))
            if np_crop.size < 3 * 50:  # troppi pochi pixel
                out.append("unknown")
                continue

            flat = np_crop.reshape(-1, 3).astype(np.float32)
            try:
                k = max(1, int(self.cfg.kmeans_clusters))
                km = KMeans(n_clusters=k, n_init="auto").fit(flat)
                centers = km.cluster_centers_.astype(np.uint8)  # RGB
                # lab euristico per mappare a nomi brevi
                labs = cv2.cvtColor(centers[np.newaxis, :, :], cv2.COLOR_RGB2LAB)[0]
                L, a, b2 = labs.mean(axis=0)
                # mapping molto grezzo
                if L > 200:
                    out.append("white")
                elif L < 50:
                    out.append("black")
                elif a > 140 and b2 < 120:
                    out.append("red")
                elif a < 120 and b2 > 140:
                    out.append("yellow")
                elif a < 120 and b2 < 120:
                    out.append("green")
                else:
                    out.append("unknown")
            except Exception:
                out.append("unknown")
        return out

    def _relative_depth(self, image: Image.Image, centers: Sequence[Tuple[float, float]]) -> List[float]:
        if self.depth is None or not self.depth.available():
            return [0.5] * len(centers)
        return self.depth.relative_depth_at(image, centers)

    def _caption(self, image: Image.Image) -> str:
        if self.captioner is None:
            return ""
        try:
            return self.captioner.caption(image) or ""
        except Exception:
            return ""

    def _clip_sim_nodes(self, G: nx.DiGraph, i: int, j: int) -> float:
        """
        Similarità CLIP nodo-nodo (dot tra embedding normalizzati).
        """
        try:
            ei = G.nodes[i].get("clip_emb")
            ej = G.nodes[j].get("clip_emb")
            if ei is None or ej is None:
                return 0.0
            # prodotto scalare tra vettori normalizzati
            s = float(np.dot(np.asarray(ei, dtype=np.float32), np.asarray(ej, dtype=np.float32)))
            return s
        except Exception:
            return 0.0

    def _maybe_add_edge(
        self,
        G: nx.DiGraph,
        i: int,
        j: int,
        boxes_xyxy: Sequence[Sequence[float]],
        W: int,
        H: int,
    ) -> None:
        b1 = boxes_xyxy[i]
        b2 = boxes_xyxy[j]

        c1x, c1y = center(b1)
        c2x, c2y = center(b2)

        dx = c2x - c1x
        dy = c2y - c1y
        dist = math.hypot(dx, dy)
        dist_norm = dist / float(max(W, H))

        if dist_norm > float(self.cfg.max_dist_norm):
            return

        iou_val = float(iou_xyxy(b1, b2))
        clip_sim = float(self._clip_sim_nodes(G, i, j))

        # logica di pruning come nel monolite:
        # se iou è molto piccolo e simil CLIP è bassa, scarta
        if (iou_val < float(self.cfg.min_iou_keep)) and (clip_sim < float(self.cfg.min_clip_sim_keep)):
            return

        angle = (math.degrees(math.atan2(dy, dx)) + 360.0) % 360.0

        # depth delta (opzionale)
        d_i = G.nodes[i].get("depth_norm", None)
        d_j = G.nodes[j].get("depth_norm", None)
        depth_delta = (float(d_j) - float(d_i)) if (d_i is not None and d_j is not None) else 0.0

        G.add_edge(
            i,
            j,
            dx_norm=dx / float(W),
            dy_norm=dy / float(H),
            dist_norm=dist_norm,
            angle_deg=angle,
            iou=iou_val,
            clip_sim=clip_sim,
            depth_delta=depth_delta,
        )

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def union_crop(image: Image.Image, box_a: Sequence[float], box_b: Sequence[float]) -> Image.Image:
        """
        Croppa l'area che abbraccia entrambi i box. Utile per CLIP-relations, se servisse.
        """
        W, H = image.size
        x1, y1, x2, y2 = union_box(box_a, box_b)
        x1 = int(max(0, min(x1, W - 1)))
        y1 = int(max(0, min(y1, H - 1)))
        x2 = int(max(0, min(x2, W - 1)))
        y2 = int(max(0, min(y2, H - 1)))
        if x2 <= x1 or y2 <= y1:
            return image.crop((0, 0, 1, 1)).convert("RGB")
        return image.crop((x1, y1, x2, y2)).convert("RGB")
