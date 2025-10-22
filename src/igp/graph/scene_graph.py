# igp/graph/scene_graph.py
# Builds a scene graph (NetworkX DiGraph) from fused detections.
# Nodes carry object attributes; edges encode geometric/semantic relations.
# Includes JSON/gpickle IO and robust, batched CLIP embedding extraction.

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

from igp.utils.boxes import iou as iou_xyxy, center, union as union_box
from igp.utils.clip_utils import CLIPWrapper
from igp.utils.depth import DepthEstimator


@dataclass
class SceneGraphConfig:
    """
    Configuration flags and thresholds for graph construction.
    """
    # Pair pruning
    max_dist_norm: float = 0.4          # drop pairs very far apart (dist_norm > 0.4)
    min_iou_keep: float = 0.01          # require minimal overlap if CLIP similarity is low
    min_clip_sim_keep: float = 0.20     # if IoU < min_iou_keep, keep only if CLIP sim ≥ this
    max_neighbors: int = 32             # keep at most this many nearest neighbors per node (after distance filter)

    # "scene" node
    add_scene_node: bool = True

    # Node features
    store_clip_embeddings: bool = True
    store_depth: bool = True
    store_color: bool = True

    # Dominant color (best-effort)
    kmeans_clusters: int = 3

    # CLIP batching
    clip_batch_size: int = 32


class SceneGraphBuilder:
    """
    Build a scene graph (NetworkX DiGraph) with attributes compatible with the
    original codebase:

      - object nodes 0..N-1: (label, score, clip_emb, bbox_norm, area_norm,
                               color, depth_norm)
      - optional 'scene' node
      - directed edges i->j with geometric/semantic attributes:
        (dx_norm, dy_norm, dist_norm, angle_deg, iou, clip_sim, depth_delta)
    """
    def __init__(
        self,
        clip: Optional[CLIPWrapper] = None,
        depth: Optional[DepthEstimator] = None,
        config: Optional[SceneGraphConfig] = None,
    ) -> None:
        self.clip = clip
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
        Build the scene graph from fused detections (post-WBF/NMS).
        """
        G = nx.DiGraph()
        W, H = image.size
        N = len(boxes_xyxy)

        # 1) Node features ------------------------------------------------------
        clip_embs = self._compute_clip_embeddings(image, boxes_xyxy) if self.cfg.store_clip_embeddings else None
        dom_colors = self._dominant_colors(image, boxes_xyxy) if self.cfg.store_color else ["unknown"] * N

        # Depth: sampled at box centroids
        centres = [center(b) for b in boxes_xyxy]
        depths = self._relative_depth(image, centres) if self.cfg.store_depth else [0.5] * N

        # Add object nodes
        for idx, (box, lab, sc) in enumerate(zip(boxes_xyxy, labels, scores)):
            x1, y1, x2, y2 = box[:4]
            area_norm = ((x2 - x1) * (y2 - y1)) / float(max(1, W * H))
            node_attrs: Dict[str, Any] = {
                "label": str(lab),
                "score": float(sc),
                "bbox_norm": [x1 / W, y1 / H, x2 / W, y2 / H],
                "area_norm": float(area_norm),
            }
            if clip_embs is not None:
                node_attrs["clip_emb"] = clip_embs[idx]  # list[float]
            if self.cfg.store_color:
                node_attrs["color"] = dom_colors[idx]
            if self.cfg.store_depth:
                node_attrs["depth_norm"] = float(depths[idx])

            G.add_node(idx, **node_attrs)

        # Optional scene node
        scene_id: Optional[int] = None
        if self.cfg.add_scene_node:
            scene_id = len(G)
            G.add_node(scene_id, label="scene")
            for i in range(N):
                G.add_edge(scene_id, i)

        # 2) Edge features ------------------------------------------------------
        # Build neighbor lists to reduce O(N^2) blow-up
        neighbors = self._candidate_neighbors(boxes_xyxy, W, H)

        for i in range(N):
            for j in neighbors[i]:
                if i == j:
                    continue
                self._maybe_add_edge(G, i, j, boxes_xyxy, W, H)

        return G

    # ------------------------------------------------------------------ io utils

    @staticmethod
    def save_gpickle(G: nx.DiGraph, path: str | Path, compress: bool | None = None) -> None:
        """
        Save the graph to disk. If extension is .gz or compress=True, use gzip.
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
        Save the graph as node-link JSON (serializable).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        def _np_converter(o):
            import numpy as _np
            if isinstance(o, _np.generic):
                return o.item()
            if isinstance(o, _np.ndarray):
                return o.tolist()
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
        Compute CLIP embeddings for each box crop in batches.
        Returns a list of vectors (list[float]) or None when unavailable.
        """
        if self.clip is None or not self.clip.available():
            return None

        crops: List[Image.Image] = []
        for b in boxes_xyxy:
            x1, y1, x2, y2 = map(int, b[:4])
            x2 = max(x1 + 1, x2)
            y2 = max(y1 + 1, y2)
            crops.append(image.crop((x1, y1, x2, y2)).convert("RGB"))

        feats_all: List[List[float]] = []
        B = max(1, int(self.cfg.clip_batch_size))
        # Prefer encode_image; fallback to get_image_features/image_features for compatibility
        encode = getattr(self.clip, "encode_image", None) or getattr(self.clip, "get_image_features", None) or getattr(self.clip, "image_features", None)
        if encode is None:
            return None

        for s in range(0, len(crops), B):
            batch = crops[s:s + B]
            try:
                feats = encode(batch)  # torch.Tensor [b, d]
                if feats is None:
                    return None
                feats_all.extend(feats.detach().cpu().tolist())
            except Exception:
                return None
        return feats_all

    def _dominant_colors(
        self, image: Image.Image, boxes_xyxy: Sequence[Sequence[float]]
    ) -> List[str]:
        """
        Rough "dominant color" estimation per box.
        Best-effort: uses KMeans if available, otherwise HSV heuristic.
        """
        try:
            import cv2  # type: ignore
            from sklearn.cluster import KMeans  # type: ignore
        except Exception:
            # Fallback: HSV-based basic color names
            return [self._hsv_color_name(np.array(image.crop(tuple(map(int, b[:4]))).convert("RGB"))) for b in boxes_xyxy]

        out: List[str] = []
        for b in boxes_xyxy:
            x1, y1, x2, y2 = map(int, b[:4])
            x2 = max(x1 + 1, x2)
            y2 = max(y1 + 1, y2)
            np_crop = np.array(image.crop((x1, y1, x2, y2)).convert("RGB"))
            if np_crop.size < 3 * 50:  # too few pixels
                out.append("unknown")
                continue

            flat = np_crop.reshape(-1, 3).astype(np.float32)
            try:
                k = max(1, int(self.cfg.kmeans_clusters))
                km = KMeans(n_clusters=k, n_init="auto", random_state=0).fit(flat)
                centers = km.cluster_centers_.astype(np.uint8)  # RGB
                out.append(self._hsv_color_name(centers))
            except Exception:
                out.append(self._hsv_color_name(np_crop))
        return out

    @staticmethod
    def _hsv_color_name(rgb: np.ndarray) -> str:
        """
        Map an RGB array (pixels or cluster centers) to a coarse color name.
        """
        import colorsys
        arr = rgb.reshape(-1, 3).astype(np.float32) / 255.0
        h, s, v = colorsys.rgb_to_hsv(float(arr[:, 0].mean()), float(arr[:, 1].mean()), float(arr[:, 2].mean()))
        if v > 0.92 and s < 0.15:
            return "white"
        if v < 0.15:
            return "black"
        if s < 0.2:
            return "gray"
        deg = h * 360.0
        if 345 <= deg or deg < 15:
            return "red"
        if 15 <= deg < 45:
            return "orange"
        if 45 <= deg < 70:
            return "yellow"
        if 70 <= deg < 170:
            return "green"
        if 170 <= deg < 255:
            return "cyan"
        if 255 <= deg < 290:
            return "blue"
        if 290 <= deg < 345:
            return "magenta"
        return "unknown"

    def _relative_depth(self, image: Image.Image, centers: Sequence[Tuple[float, float]]) -> List[float]:
        # Return normalized relative depth at given centers; default to 0.5 if unavailable.
        if self.depth is None or not self.depth.available():
            return [0.5] * len(centers)
        return self.depth.relative_depth_at(image, centers)

    def _clip_sim_nodes(self, G: nx.DiGraph, i: int, j: int) -> float:
        """
        Node-to-node CLIP similarity (dot product between normalized embeddings).
        """
        try:
            ei = G.nodes[i].get("clip_emb")
            ej = G.nodes[j].get("clip_emb")
            if ei is None or ej is None:
                return 0.0
            s = float(np.dot(np.asarray(ei, dtype=np.float32), np.asarray(ej, dtype=np.float32)))
            return s
        except Exception:
            return 0.0

    def _candidate_neighbors(self, boxes_xyxy: Sequence[Sequence[float]], W: int, H: int) -> List[List[int]]:
        """
        For each node i, return up to max_neighbors nearest neighbors within max_dist_norm.
        """
        N = len(boxes_xyxy)
        centers = np.array([center(b) for b in boxes_xyxy], dtype=np.float32)
        # Compute distances
        neighs: List[List[int]] = [[] for _ in range(N)]
        for i in range(N):
            dx = centers[:, 0] - centers[i, 0]
            dy = centers[:, 1] - centers[i, 1]
            dist = np.hypot(dx, dy)
            dist_norm = dist / float(max(W, H))
            idxs = [j for j in np.argsort(dist) if j != i and dist_norm[j] <= float(self.cfg.max_dist_norm)]
            if self.cfg.max_neighbors > 0:
                idxs = idxs[: int(self.cfg.max_neighbors)]
            neighs[i] = list(map(int, idxs))
        return neighs

    def _maybe_add_edge(
        self,
        G: nx.DiGraph,
        i: int,
        j: int,
        boxes_xyxy: Sequence[Sequence[float]],
        W: int,
        H: int,
    ) -> None:
        # Compute basic geometry between object i and j and decide whether to add the edge.
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

        # Pruning:
        if (iou_val < float(self.cfg.min_iou_keep)) and (clip_sim < float(self.cfg.min_clip_sim_keep)):
            return

        angle = (math.degrees(math.atan2(dy, dx)) + 360.0) % 360.0

        # Depth delta (optional)
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
        Crop the minimal region covering both boxes. Useful for CLIP-relations, if needed.
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