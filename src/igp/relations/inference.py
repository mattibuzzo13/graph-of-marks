# igp/relations/inference.py
# Combines geometric heuristics and CLIP-based scoring to infer relationships
# between detected objects. Keeps comments concise for paper readability.
# 🚀 SOTA: Added LLM-guided, 3D spatial, and physics-informed reasoning

from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple
from dataclasses import dataclass

import math
import numpy as np
from PIL import Image

# Parallel processing
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

from .clip_rel import ClipRelScorer
from .geometry import (
    as_xyxy,
    build_precise_nearest_relation,
    center_distance,
    iou,
    is_below_of,
    is_on_top_of,
    is_in_front_of,
    is_behind_of,
    orientation_label,
    horizontal_overlap,
    vertical_overlap,
)

try:
    from .semantic_filter import filter_impossible_relations as _semantic_filter
    _SEMANTIC_FILTER_AVAILABLE = True
except Exception:
    _semantic_filter = None  # type: ignore
    _SEMANTIC_FILTER_AVAILABLE = False

# 🚀 SOTA modules (optional)
try:
    from .llm_guided import LLMRelationInferencer, LLMRelationsConfig
    LLM_AVAILABLE = True
except ImportError:
    LLMRelationInferencer = None  # type: ignore
    LLMRelationsConfig = None  # type: ignore
    LLM_AVAILABLE = False

try:
    from .spatial_3d import Spatial3DReasoner, Spatial3DConfig
    SPATIAL_3D_AVAILABLE = True
except ImportError:
    Spatial3DReasoner = None  # type: ignore
    Spatial3DConfig = None  # type: ignore
    SPATIAL_3D_AVAILABLE = False

try:
    from .physics import PhysicsReasoner, PhysicsConfig
    PHYSICS_AVAILABLE = True
except ImportError:
    PhysicsReasoner = None  # type: ignore
    PhysicsConfig = None  # type: ignore
    PHYSICS_AVAILABLE = False

__all__ = [
    "RelationsConfig",
    "RelationInferencer",
]


@dataclass
class RelationsConfig:
    """Configuration for relationship inference."""
    enabled: bool = True
    max_relations: int = 10
    max_relations_per_object: int = 3
    min_relations_per_object: int = 1
    relationship_types: tuple = ("spatial", "semantic", "action")
    confidence_threshold: float = 0.5
    use_clip_relations: bool = True
    use_geometric_relations: bool = True
    clip_threshold: float = 0.23
    margin_px: int = 20
    min_distance: float = 5.0
    max_distance: float = 20000.0
    filter_redundant: bool = True
    filter_relations_by_question: bool = True
    threshold_relation_similarity: float = 0.50
    
    # 🚀 SOTA: LLM-guided relations (optional)
    use_llm_relations: bool = False
    llm_backend: str = "gpt4v"  # "gpt4v" | "llava" | "mock"
    llm_api_key: Optional[str] = None
    llm_confidence_threshold: float = 0.6
    
    # 🚀 SOTA: 3D spatial reasoning (optional)
    use_3d_reasoning: bool = False
    depth_threshold: float = 0.1
    use_occlusion: bool = True
    
    # 🚀 SOTA: Physics-informed filtering (optional)
    use_physics_filtering: bool = False
    filter_impossible: bool = True
    check_support: bool = True
    check_stability: bool = True

_SPATIAL_KEYS = (
    "left_of",
    "right_of",
    "above",
    "below",
    "on_top_of",
    "under",
    "in_front_of",
    "behind",
)

_INVERSE = {
    # Basic spatial relations
    "left_of": "right_of",
    "right_of": "left_of",
    "above": "below",
    "below": "above",
    "in_front_of": "behind",
    "behind": "in_front_of",
    "on_top_of": "below",
    "under": "on_top_of",
    
    # Composite touching relations
    "touching_left_of": "touching_right_of",
    "touching_right_of": "touching_left_of",
    "touching_above": "touching_below",
    "touching_below": "touching_above",
    
    # Other proximity-based composite relations
    "close_left_of": "close_right_of",
    "close_right_of": "close_left_of",
    "close_above": "close_below",
    "close_below": "close_above",
    
    "very_close_left_of": "very_close_right_of",
    "very_close_right_of": "very_close_left_of",
    "very_close_above": "very_close_below",
    "very_close_below": "very_close_above",
}


class RelationInferencer:
    """
    Combines geometric heuristics and CLIP scoring to derive object relations.
    Returns a list of dicts:
      { "src_idx", "tgt_idx", "relation", "distance", ["relation_raw", "clip_sim"] }
    
    Supports parallel inference for improved performance on multi-core systems.
    """

    def __init__(
        self,
        clip_scorer: Optional[ClipRelScorer] = None,
        *,
        margin_px: int = 20,
        min_distance: float = 5.0,
        max_distance: float = 20000.0,
        enable_parallel: bool = True,
        max_workers: Optional[int] = None,
    ) -> None:
        self.clip = clip_scorer
        self.margin_px = int(margin_px)
        self.min_distance = float(min_distance)
        self.max_distance = float(max_distance)
        self.enable_parallel = enable_parallel
        # Use all available CPUs by default unless an explicit max_workers provided.
        self.max_workers = int(max_workers) if max_workers is not None else (os.cpu_count() or 1)
    
    def _compute_directional_relation_pair(
        self,
        i: int,
        j: int,
        boxes: Sequence[Sequence[float]],
        centers: List[Tuple[float, float]],
    ) -> List[dict]:
        """
        Compute directional relations (above/below/left/right) for a pair (i, j).
        Returns list of relations (can be 0, 1, or 2 relations for bidirectional).
        """
        rels = []
        
        cx1, cy1 = centers[i]
        cx2, cy2 = centers[j]
        dx, dy = cx2 - cx1, cy2 - cy1
        dist = math.hypot(dx, dy)
        
        if dist < self.min_distance or dist > self.max_distance:
            return rels
        
        # Calculate box dimensions for scale-aware thresholds
        box_i = boxes[i]
        box_j = boxes[j]
        w_i = box_i[2] - box_i[0]
        h_i = box_i[3] - box_i[1]
        w_j = box_j[2] - box_j[0]
        h_j = box_j[3] - box_j[1]
        avg_size = (w_i + h_i + w_j + h_j) / 4.0
        
        # Scale-aware margin
        margin = max(self.margin_px, avg_size * 0.15)
        
        # Check for significant overlap
        iou_val = iou(box_i, box_j)
        if iou_val > 0.3:
            return rels  # Skip highly overlapping boxes
        
        # Determina la direzione primaria
        if abs(dy) >= abs(dx) and abs(dy) > margin:
            # Relazione verticale
            relation = "above" if cy1 < cy2 else "below"  # i sopra j se y1 < y2
            v_overlap = vertical_overlap(box_i, box_j)
            if v_overlap > max(h_i, h_j) * 0.5:
                return rels  # Troppa sovrapposizione verticale
        elif abs(dx) > margin:
            # Relazione orizzontale
            relation = "left_of" if cx1 < cx2 else "right_of"  # i a sinistra di j se x1 < x2
            h_overlap = horizontal_overlap(box_i, box_j)
            if h_overlap > max(w_i, w_j) * 0.5:
                return rels  # Troppa sovrapposizione orizzontale
        else:
            return rels

        # Aggiungi la relazione primaria (i -> j)
        rels.append(
            {"src_idx": i, "tgt_idx": j, "relation": relation, "distance": dist}
        )

        # Aggiungi la relazione inversa (j -> i)
        inverse_relation = {
            "left_of": "right_of",
            "right_of": "left_of",
            "above": "below",
            "below": "above",
        }.get(relation)

        if inverse_relation:
            rels.append(
                {"src_idx": j, "tgt_idx": i, "relation": inverse_relation, "distance": dist}
            )

        return rels

    def infer(
        self,
        image_pil: Image.Image,
        boxes: Sequence[Sequence[float]],
        labels: Optional[Sequence[str]] = None,
        *,
        masks: Optional[Sequence[dict]] = None,
        depths: Optional[Sequence[float]] = None,
        depth_map: Optional[np.ndarray] = None,
        use_geometry: bool = True,
        use_clip: bool = True,
        clip_threshold: float = 0.23,
        filter_redundant: bool = True,
    ) -> List[dict]:
        """
        Compute candidate relations (geometry + CLIP).
        """
        n = len(boxes)
        if n <= 1:
            return []

        if labels is None:
            labels = [f"obj{i}" for i in range(n)]

        rels: List[dict] = []

        # ---------- 1) Geometry: on_top_of / below (symmetric) ----------
        if use_geometry:
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    ok = is_on_top_of(
                        boxes[i],
                        boxes[j],
                        mask_a=(masks[i]["segmentation"] if masks else None),
                        mask_b=(masks[j]["segmentation"] if masks else None),
                        depth_a=(depths[i] if depths else None),
                        depth_b=(depths[j] if depths else None),
                        depth_map=depth_map,
                    )
                    if ok:
                        dist_ij = center_distance(boxes[i], boxes[j])
                        rels.append(
                            {"src_idx": i, "tgt_idx": j, "relation": "on_top_of", "distance": dist_ij}
                        )
                        rels.append(
                            {"src_idx": j, "tgt_idx": i, "relation": "below", "distance": dist_ij}
                        )

        # ---------- 2) Geometry: in_front_of / behind (depth-based) ----------
        if use_geometry and (depths is not None or depth_map is not None):
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    ok = is_in_front_of(
                        boxes[i],
                        boxes[j],
                        mask_a=(masks[i]["segmentation"] if masks else None),
                        mask_b=(masks[j]["segmentation"] if masks else None),
                        depth_a=(depths[i] if depths else None),
                        depth_b=(depths[j] if depths else None),
                        depth_map=depth_map,
                    )
                    if ok:
                        dist_ij = center_distance(boxes[i], boxes[j])
                        rels.append(
                            {"src_idx": i, "tgt_idx": j, "relation": "in_front_of", "distance": dist_ij}
                        )
                        rels.append(
                            {"src_idx": j, "tgt_idx": i, "relation": "behind", "distance": dist_ij}
                        )

        # ---------- 3) Geometry: above/below/left/right with improved criteria ----------
        if use_geometry:
            # Try a vectorized path to compute directional relations for many boxes.
            try:
                boxes_np = np.asarray(boxes, dtype=float)
                x1 = boxes_np[:, 0]
                y1 = boxes_np[:, 1]
                x2 = boxes_np[:, 2]
                y2 = boxes_np[:, 3]

                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0

                # pairwise deltas: [i,j] = coord_j - coord_i
                dx = cx[None, :] - cx[:, None]
                dy = cy[None, :] - cy[:, None]
                dist = np.hypot(dx, dy)

                # distance masks
                mask = (dist >= self.min_distance) & (dist <= self.max_distance)
                np.fill_diagonal(mask, False)

                # sizes
                w = (x2 - x1)
                h = (y2 - y1)
                avg_size = (w[:, None] + h[:, None] + w[None, :] + h[None, :]) / 4.0
                margin = np.maximum(self.margin_px, avg_size * 0.15)

                # pairwise intersection dims
                inter_w = np.minimum(x2[:, None], x2[None, :]) - np.maximum(x1[:, None], x1[None, :])
                inter_h = np.minimum(y2[:, None], y2[None, :]) - np.maximum(y1[:, None], y1[None, :])
                inter_w = np.maximum(inter_w, 0.0)
                inter_h = np.maximum(inter_h, 0.0)

                area = w * h
                union = area[:, None] + area[None, :] - (inter_w * inter_h)
                iou_mat = np.zeros_like(union)
                nz = union > 0
                if np.any(nz):
                    iou_mat[nz] = (inter_w * inter_h)[nz] / union[nz]

                # ignore pairs with strong IoU
                mask &= (iou_mat <= 0.3)

                abs_dx = np.abs(dx)
                abs_dy = np.abs(dy)

                # vertical candidate: |dy| >= |dx| and |dy| > margin
                vertical_mask = (abs_dy >= abs_dx) & (abs_dy > margin) & mask
                # exclude if vertical overlap too large
                vertical_mask &= (inter_h <= (np.maximum(h[:, None], h[None, :]) * 0.5))

                # horizontal candidate: |dx| > margin
                horizontal_mask = (abs_dx > margin) & mask
                horizontal_mask &= (inter_w <= (np.maximum(w[:, None], w[None, :]) * 0.5))

                # iterate only over i < j to add primary+inverse relations (same semantics as before)
                n_idx = boxes_np.shape[0]
                for i in range(n_idx):
                    for j in range(i + 1, n_idx):
                        if vertical_mask[i, j] or horizontal_mask[i, j]:
                            if vertical_mask[i, j]:
                                relation = "above" if cy[i] < cy[j] else "below"
                            else:
                                relation = "left_of" if cx[i] < cx[j] else "right_of"
                            dist_ij = float(dist[i, j])
                            rels.append({"src_idx": i, "tgt_idx": j, "relation": relation, "distance": dist_ij})
                            inverse_relation = _INVERSE.get(relation)
                            if inverse_relation:
                                rels.append({"src_idx": j, "tgt_idx": i, "relation": inverse_relation, "distance": dist_ij})
            except Exception:
                # If any error in vectorized path, fallback to previous pairwise logic
                centers = [((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0) for b in boxes]
                pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
                if self.enable_parallel and len(pairs) > 1:
                    with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                        futures = {
                            executor.submit(
                                self._compute_directional_relation_pair,
                                i, j, boxes, centers
                            ): (i, j)
                            for i, j in pairs
                        }
                        for future in as_completed(futures):
                            try:
                                pair_rels = future.result()
                                rels.extend(pair_rels)
                            except Exception as e:
                                i, j = futures[future]
                                print(f"Warning: Error computing relation for pair ({i}, {j}): {e}")
                else:
                    for i, j in pairs:
                        pair_rels = self._compute_directional_relation_pair(i, j, boxes, centers)
                        rels.extend(pair_rels)

        # ---------- 4) CLIP scoring (batched) ----------
        if use_clip and self.clip is not None:
            # Build list of directed pairs
            clip_pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
            try:
                for i, j, rel_canon, rel_raw, score in self.clip.batch_best_relations(
                    image_pil=image_pil, boxes=boxes, labels=labels, pairs=clip_pairs
                ):
                    if score > clip_threshold:
                        dist = center_distance(boxes[i], boxes[j])
                        rels.append(
                            {
                                "src_idx": i,
                                "tgt_idx": j,
                                "relation": rel_canon,
                                "relation_raw": rel_raw,
                                "clip_sim": float(score),
                                "distance": dist,
                            }
                        )
            except Exception:
                # Fallback to original per-pair scoring if batch fails
                for i in range(n):
                    for j in range(n):
                        if i == j:
                            continue
                        try:
                            rel_canon, rel_raw, score = self.clip.best_relation(
                                image_pil, boxes[i], boxes[j], labels[i], labels[j]
                            )
                        except Exception:
                            continue
                        if score > clip_threshold:
                            dist = center_distance(boxes[i], boxes[j])
                            rels.append(
                                {
                                    "src_idx": i,
                                    "tgt_idx": j,
                                    "relation": rel_canon,
                                    "relation_raw": rel_raw,
                                    "clip_sim": float(score),
                                    "distance": dist,
                                }
                            )

        rels = self._unify_pair_relations(rels)
        # Filter out semantically impossible relations (e.g., inanimate wearing objects)
        try:
            rels = self._filter_impossible_relations(rels, labels)
        except Exception:
            # best-effort: if filter fails, keep original relations
            pass

        if filter_redundant:
            rels = self._filter_redundant_relations(rels)
            
        return rels

    # ---------------------------------------------------------------------
    # Post-processing / utilities
    # ---------------------------------------------------------------------

    def limit_relationships_per_object(
        self,
        relationships: List[dict],
        boxes: Sequence[Sequence[float]],
        *,
        max_relations_per_object: int = 3,
        min_relations_per_object: int = 1,
        question_rel_terms: Optional[Set[str]] = None,
    ) -> List[dict]:
        """
        Ensure at least `min_relations_per_object` per node (via nearest),
        and cap at `max_relations_per_object`, prioritizing question-requested
        relations when `question_rel_terms` is provided.
        """
        from collections import defaultdict

        rels_by_src: Dict[int, List[dict]] = defaultdict(list)
        for r in relationships:
            rels_by_src[r["src_idx"]].append(r)

        # Guarantee a minimum per object
        n = len(boxes)
        centers = [((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0) for b in boxes]
        for i in range(n):
            if len(rels_by_src[i]) >= min_relations_per_object:
                continue
            # Find nearest neighbor
            best_j, best_d = None, float("inf")
            for j in range(n):
                if j == i:
                    continue
                d = math.hypot(centers[i][0] - centers[j][0], centers[i][1] - centers[j][1])
                if d < best_d:
                    best_j, best_d = j, d
            if best_j is not None:
                rels_by_src[i].append(
                    build_precise_nearest_relation(i, best_j, boxes, margin_px=self.margin_px)
                )

        # Cap per-object; prioritize relations mentioned in the question
        def _is_question_rel(rel_label: str) -> bool:
            if not question_rel_terms:
                return False
            r = rel_label.lower()
            return any(t in r for t in question_rel_terms)

        final: List[dict] = []
        for i, rlist in rels_by_src.items():
            # Ordina per confidenza/score se presente, altrimenti per distanza
            def rel_sort_key(r):
                # Priorità: question term, poi score/confidenza, poi distanza
                q_priority = 0 if _is_question_rel(r.get("relation", "")) else 1
                # Usa clip_sim, score, o distance
                score = r.get("clip_sim", None)
                if score is None:
                    score = r.get("score", None)
                if score is not None:
                    # Score negativo per ordinare decrescente
                    return (q_priority, -score, r.get("distance", 1e9))
                else:
                    return (q_priority, r.get("distance", 1e9))
            rlist_sorted = sorted(rlist, key=rel_sort_key)
            # Mantieni solo le top-N relazioni per oggetto
            final.extend(rlist_sorted[:max_relations_per_object])
        return final

    def _filter_redundant_relations(self, relationships: List[dict]) -> List[dict]:
        """
        For each unordered pair of objects, keep only the most informative relation.
        When multiple relations exist (e.g., "left_of" + "touching_left"),
        choose according to a priority scheme.
        """
        if not relationships:
            return relationships

        # Group by object pair (order-invariant)
        pair_relations: Dict[Tuple[int, int], List[dict]] = {}
        
        for rel in relationships:
            s0, t0 = rel["src_idx"], rel["tgt_idx"]
            pair_key = tuple(sorted([s0, t0]))  # ordered pair for symmetry
            
            if pair_key not in pair_relations:
                pair_relations[pair_key] = []
            pair_relations[pair_key].append(rel)
        
        # Select one relation per pair
        filtered_relations = []
        for pair_key, rels in pair_relations.items():
            if len(rels) == 1:
                filtered_relations.append(rels[0])
            else:
                best_rel = self._choose_best_relation(rels)
                filtered_relations.append(best_rel)
        
        return filtered_relations

    def _choose_best_relation(self, relations: List[dict]) -> dict:
        """
        Pick the most informative relation among candidates for the same pair.
        Priority: semantic > specific spatial (contact/adjacency) > generic spatial > directional.
        """
        # Priority tiers for relation types
        semantic_relations = {"on_top_of", "under", "holding", "wearing", "riding", "sitting_on", "carrying"}
        spatial_specific = {"touching", "adjacent", "near", "close"}
        spatial_directional = {"left_of", "right_of", "above", "below", "in_front_of", "behind"}
        
        best_rel = relations[0]
        best_priority = self._get_relation_priority(best_rel["relation"])
        best_confidence = self._get_relation_confidence(best_rel)
        
        for rel in relations[1:]:
            priority = self._get_relation_priority(rel["relation"])
            confidence = self._get_relation_confidence(rel)
            
            # Compare by priority first, then by confidence
            if (priority > best_priority or 
                (priority == best_priority and confidence > best_confidence)):
                best_rel = rel
                best_priority = priority
                best_confidence = confidence
        
        return best_rel

    def _get_relation_priority(self, relation: str) -> int:
        """Assign a numeric priority to a relation."""
        rel_name = str(relation).lower()
        
        # 4: strong semantic relations
        semantic_strong = {"on_top_of", "under", "holding", "wearing", "riding", "sitting_on", "carrying"}
        if any(sem in rel_name for sem in semantic_strong):
            return 4
            
        # 3: contact/adjacency
        spatial_contact = {"touching", "adjacent"}
        if any(contact in rel_name for contact in spatial_contact):
            return 3
            
        # 2: generic proximity
        spatial_generic = {"near", "close"}
        if any(gen in rel_name for gen in spatial_generic):
            return 2
            
        # 1: directional spatial cues
        spatial_directional = {"left_of", "right_of", "above", "below", "in_front_of", "behind"}
        if any(dir_rel in rel_name for dir_rel in spatial_directional):
            return 1
            
        # 0: others
        return 0

    def _get_relation_confidence(self, relation: dict) -> float:
        """Extract a confidence score: prefer CLIP similarity; else inverse distance; else default."""
        if "clip_sim" in relation:
            return float(relation["clip_sim"])
        elif "distance" in relation:
            # Inverse distance (closer ⇒ higher)
            dist = float(relation["distance"])
            return 1.0 / (1.0 + dist / 100.0)
        else:
            return 0.5  # default for purely geometric relations

    def _filter_impossible_relations(self, relationships: List[dict], labels: Sequence[str]) -> List[dict]:
        """Delegate semantic filtering to the optional semantic filter helper.

        The heavy lifting is done in `src.igp.relations.semantic_filter`. If
        that module or WordNet is unavailable, we still behave conservatively
        by falling back to a lightweight heuristic.
        """
        if not relationships:
            return relationships

        # Prefer the richer semantic filter when available
        if _SEMANTIC_FILTER_AVAILABLE and _semantic_filter is not None:
            try:
                return _semantic_filter(relationships, labels)
            except Exception:
                # best-effort: fall through to conservative heuristic
                pass

        # Conservative fallback (previous heuristics)
        animates = {
            "person",
            "man",
            "woman",
            "child",
            "boy",
            "girl",
            "human",
            "people",
        }
        require_animate_subj = {"wearing", "holding", "riding", "sitting_on", "carrying"}

        out: List[dict] = []
        for r in relationships:
            rel = str(r.get("relation", "")).lower()
            s_idx = int(r["src_idx"]) if "src_idx" in r else None
            t_idx = int(r["tgt_idx"]) if "tgt_idx" in r else None

            subj_label = labels[s_idx] if s_idx is not None and s_idx < len(labels) else ""
            obj_label = labels[t_idx] if t_idx is not None and t_idx < len(labels) else ""

            subj_norm = str(subj_label).lower()
            obj_norm = str(obj_label).lower()

            subj_is_animate = any(tok in subj_norm for tok in animates) or subj_norm in animates

            if rel in require_animate_subj and not subj_is_animate:
                continue

            if rel == "wearing":
                wearable_tokens = ("hat", "cap", "glasses", "shirt", "jacket", "coat", "shoe", "shoes", "pants", "skirt", "dress", "tie", "scarf", "watch")
                if not any(tok in obj_norm for tok in wearable_tokens):
                    continue

            out.append(r)

        return out

    def drop_inverse_duplicates(
        self,
        relationships: List[dict],
        *,
        question_subject_idxs: Optional[Set[int]] = None,
        max_relations_per_object: int = 3,
        min_relations_per_object: int = 1,
    ) -> List[dict]:
        """
        ✅ FIXED: Remove inverse duplicate relations correctly.
        Keep only one direction per pair (i,j), preferring:
        1. Relations involving question subjects (if provided)
        2. Higher CLIP confidence
        3. Relations on objects with fewer existing relations
        
        Example: If we have both "A left_of B" and "B right_of A",
        keep only one based on the priority above.
        """
        seen_pairs = {}  # (min_idx, max_idx) -> best relation dict
        count_per_src = {}  # src_idx -> count
        
        subj = question_subject_idxs or set()
        
        for rel in relationships:
            i, j = rel["src_idx"], rel["tgt_idx"]
            rel_type = rel["relation"]
            
            # Normalize to canonical pair (smaller index first)
            canonical_i, canonical_j = (i, j) if i < j else (j, i)
            pair_key = (canonical_i, canonical_j)
            
            # Check if we already have a relation for this pair
            if pair_key in seen_pairs:
                # We have a duplicate (potentially inverse)
                existing = seen_pairs[pair_key]
                
                # Priority 1: Question subjects as SOURCE (stronger signal)
                r_src_is_subj = i in subj
                e_src_is_subj = existing["src_idx"] in subj
                
                if r_src_is_subj and not e_src_is_subj:
                    # New relation has subject as source, existing doesn't: replace
                    count_per_src[existing["src_idx"]] = count_per_src.get(existing["src_idx"], 1) - 1
                    seen_pairs[pair_key] = rel
                    count_per_src[i] = count_per_src.get(i, 0) + 1
                elif e_src_is_subj and not r_src_is_subj:
                    # Existing has subject as source, new doesn't: keep existing
                    continue
                else:
                    # Both or neither have subject as source
                    # Priority 1b: Any involvement of question subjects
                    r_hits = (i in subj) or (j in subj)
                    e_hits = (existing["src_idx"] in subj) or (existing["tgt_idx"] in subj)
                    
                    if r_hits and not e_hits:
                        # New relation involves subjects, existing doesn't: replace
                        count_per_src[existing["src_idx"]] = count_per_src.get(existing["src_idx"], 1) - 1
                        seen_pairs[pair_key] = rel
                        count_per_src[i] = count_per_src.get(i, 0) + 1
                    elif e_hits and not r_hits:
                        # Existing involves subjects, new doesn't: keep existing
                        continue
                    else:
                        # Priority 2: CLIP confidence
                        existing_conf = existing.get("clip_sim", 0.0)
                        new_conf = rel.get("clip_sim", 0.0)
                        
                        if new_conf > existing_conf:
                            # Higher confidence: replace
                            count_per_src[existing["src_idx"]] = count_per_src.get(existing["src_idx"], 1) - 1
                            seen_pairs[pair_key] = rel
                            count_per_src[i] = count_per_src.get(i, 0) + 1
                        elif new_conf < existing_conf:
                            # Lower confidence: keep existing
                            continue
                        else:
                            # Priority 3: Balance per-object counts
                            cnt_i = count_per_src.get(i, 0)
                            cnt_e = count_per_src.get(existing["src_idx"], 0)
                            
                            if cnt_i < max_relations_per_object and cnt_e >= max_relations_per_object:
                                # New has room, existing is full: replace
                                count_per_src[existing["src_idx"]] = count_per_src.get(existing["src_idx"], 1) - 1
                                seen_pairs[pair_key] = rel
                                count_per_src[i] = count_per_src.get(i, 0) + 1
                            else:
                                # Keep existing
                                continue
            else:
                # First relation for this pair
                seen_pairs[pair_key] = rel
                count_per_src[i] = count_per_src.get(i, 0) + 1
        
        return list(seen_pairs.values())

    def filter_by_question(
        self,
        relationships: List[dict],
        *,
        question_terms: Optional[Set[str]] = None,
        similarity_fn: Optional[Callable[[str, str], float]] = None,
        threshold: float = 0.5,
    ) -> List[dict]:
        """
        Keep only relations consistent with the question terms.
        - If `similarity_fn` is provided, use it for fuzzy matching (e.g., spaCy similarity).
        - Otherwise perform exact/substring matching.
        """
        if not question_terms:
            return relationships

        out: List[dict] = []
        for r in relationships:
            label = str(r["relation"]).lower().replace("_", " ")
            keep = False
            for t in question_terms:
                t_norm = t.lower().replace("_", " ")
                if label == t_norm or t_norm in label:
                    keep = True
                    break
                if similarity_fn is not None:
                    if similarity_fn(label, t_norm) >= threshold:
                        keep = True
                        break
            if keep:
                out.append(r)
        return out

    # unify_spatial_direction rimossa: le relazioni spaziali mantengono la direzione originale src_idx → tgt_idx

    # -------------------- internals --------------------

    @staticmethod
    def _unify_pair_relations(relationships: List[dict]) -> List[dict]:
        """
        Keep at most one relation per directed pair (src, tgt),
        choosing the one with the smallest distance (or first encountered).
        """
        best_for_pair: Dict[Tuple[int, int], dict] = {}
        for r in relationships:
            key = (r["src_idx"], r["tgt_idx"])
            if key not in best_for_pair:
                best_for_pair[key] = r
            else:
                if r.get("distance", 1e9) < best_for_pair[key].get("distance", 1e9):
                    best_for_pair[key] = r
        return list(best_for_pair.values())
