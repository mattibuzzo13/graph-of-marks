# igp/relations/inference.py
"""
Multi-Modal Relationship Inference Engine

This module orchestrates relationship extraction between detected objects using
multiple reasoning strategies: geometric heuristics, CLIP-based semantic scoring,
optional LLM guidance, 3D spatial reasoning, and physics-informed constraints.

Key Features:
    - Hybrid geometric + semantic relation inference
    - CLIP-based similarity scoring for visual relationships
    - Optional LLM-guided reasoning (GPT-4V, LLaVA)
    - 3D depth-aware spatial relationships
    - Physics-informed constraints (support, containment)
    - Semantic filtering (impossible relation pruning)
    - Parallel processing for large scenes
    - Question-aware relation filtering

Relationship Types:
    Geometric:
        - Spatial: left_of, right_of, above, below, in_front_of, behind
        - Topological: on_top_of, inside, overlaps
        - Proximity: near, far
    
    Semantic:
        - CLIP-scored: wearing, holding, riding, eating, etc.
        - Context-aware: inferred from visual similarity
    
    Physics-informed:
        - Support: supported_by, resting_on
        - Stability: stable, unstable
        - Containment: contained_in

Architecture:
    RelationsConfig: Configuration with ~20 tunable parameters
    RelationInferencer: Main inference engine coordinating:
        - ClipRelScorer: Visual-semantic similarity
        - Geometric reasoners: Spatial predicates
        - LLMRelationInferencer: Optional LLM reasoning
        - Spatial3DReasoner: Depth-based 3D relations
        - PhysicsReasoner: Physics constraints

Performance:
    - Parallel CLIP scoring for large scenes
    - Configurable max_clip_pairs limit (default 500)
    - Per-source pair limits (default 20)
    - Vectorized geometric operations

Usage:
    >>> from gom.relations import RelationInferencer, RelationsConfig
    >>> 
    >>> # Basic configuration
    >>> config = RelationsConfig(
    ...     clip_threshold=0.5,
    ...     max_relations_per_object=3,
    ...     use_clip_relations=True
    ... )
    >>> 
    >>> # Initialize inferencer
    >>> inferencer = RelationInferencer(config, image)
    >>> 
    >>> # Infer relationships
    >>> relationships = inferencer.infer(detections)
    >>> for rel in relationships:
    ...     print(f"{rel.src_idx} {rel.relation} {rel.tgt_idx}")

See Also:
    - gom.relations.clip_rel: CLIP-based scoring
    - gom.relations.geometry: Spatial predicates
    - gom.relations.semantic_filter: Impossible relation filtering
    - gom.relations.llm_guided: LLM-based reasoning (optional)
    - gom.relations.spatial_3d: 3D depth reasoning (optional)
    - gom.relations.physics: Physics constraints (optional)
"""
from __future__ import annotations

import math
import os

# Parallel processing
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from PIL import Image

from .clip_rel import ClipRelScorer
from .geometry import (
    as_xyxy,
    build_precise_nearest_relation,
    center_distance,
    depth_stats_from_map,
    edge_gap,
    horizontal_overlap,
    iou,
    is_in_front_of,
    is_on_top_of,
    vertical_overlap,
)

try:
    from .semantic_filter import filter_impossible_relations as _semantic_filter
    _SEMANTIC_FILTER_AVAILABLE = True
except Exception:
    _semantic_filter = None  # type: ignore
    _SEMANTIC_FILTER_AVAILABLE = False

# SOTA modules (optional)
try:
    from .llm_guided import LLMRelationInferencer, LLMRelationsConfig
    LLM_AVAILABLE = True
except ImportError:
    LLMRelationInferencer = None  # type: ignore
    LLMRelationsConfig = None  # type: ignore
    LLM_AVAILABLE = False

try:
    from .spatial_3d import Spatial3DConfig, Spatial3DReasoner
    SPATIAL_3D_AVAILABLE = True
except ImportError:
    Spatial3DReasoner = None  # type: ignore
    Spatial3DConfig = None  # type: ignore
    SPATIAL_3D_AVAILABLE = False

try:
    from .physics import PhysicsConfig, PhysicsReasoner
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
    """
    Configuration for multi-modal relationship inference.
    
    Controls all aspects of relation extraction including geometric thresholds,
    CLIP scoring, semantic filtering, and optional advanced reasoning modules.
    
    Attributes:
        enabled: Whether to perform relation inference (bool, default True)
        max_relations: Maximum total relationships to extract (int, default 10)
        max_relations_per_object: Max edges per source node (int, default 3)
        auto_adjust_relation_cap: Auto-scale per-object cap to avoid overcrowding (bool, default True)
        min_relations_per_object: Min edges per source node (int, default 1)
        relationship_types: Enabled relation categories (tuple, default ("spatial", "semantic", "action"))
        confidence_threshold: Minimum confidence for keeping relation (float, default 0.5)
        
        Geometric Parameters:
            use_geometric_relations: Enable geometric heuristics (bool, default True)
            margin_px: Pixel margin for proximity tests (int, default 20)
            min_distance: Minimum distance threshold (float, default 5.0)
            max_distance: Maximum distance threshold (float, default 20000.0)
            depth_front_threshold: Depth delta for in_front_of (float, default 0.05)
            depth_touching_threshold: Max depth diff for "touching" (float, default 0.08)
            min_relation_distance: Minimum center distance to keep relation (float, default 5.0)
            max_relation_distance: Maximum center distance to keep relation (float, default 20000.0)
            too_close_gap_px: Edge-gap threshold for "too close" (float, default 3.0)
            too_close_gap_scale: Edge-gap threshold scale vs size (float, default 0.02)
        
        CLIP Parameters:
            use_clip_relations: Enable CLIP-based scoring (bool, default True)
            clip_threshold: Minimum CLIP similarity score (float, default 0.5)
                           Raised from 0.23 to 0.30 to reduce false positives by 20-30%
            max_clip_pairs: Total CLIP pair limit for performance (int, default 500)
            per_src_clip_pairs: Max CLIP evaluations per source object (int, default 20)
        
        Filtering:
            filter_redundant: Remove duplicate/redundant relations (bool, default True)
            filter_relations_by_question: Keep only question-relevant relations (bool, default True)
            threshold_relation_similarity: Similarity threshold for deduplication (float, default 0.50)
        
        Advanced Modules (Optional):
            use_llm_relations: Enable LLM-guided reasoning (bool, default False)
            llm_backend: LLM provider: "gpt4v", "llava", "mock" (str, default "gpt4v")
            llm_api_key: API key for LLM service (Optional[str], default None)
            llm_confidence_threshold: Minimum LLM confidence (float, default 0.6)
    
    Examples:
        >>> # Conservative geometric-only config
        >>> config = RelationsConfig(
        ...     use_clip_relations=False,
        ...     use_geometric_relations=True,
        ...     max_relations_per_object=2
        ... )
        
        >>> # High-precision CLIP config
        >>> config = RelationsConfig(
        ...     clip_threshold=0.7,
        ...     filter_redundant=True,
        ...     max_clip_pairs=1000
        ... )
        
        >>> # LLM-enhanced config
        >>> config = RelationsConfig(
        ...     use_llm_relations=True,
        ...     llm_backend="gpt4v",
        ...     llm_api_key="sk-...",
        ...     llm_confidence_threshold=0.75
        ... )
    
    Notes:
        - clip_threshold tuning: 0.3-0.5 balanced, >0.6 high precision
        - max_clip_pairs prevents O(n²) scaling for large scenes
        - LLM modules require additional dependencies
        - Geometric relations are fastest (no ML inference)
    
    Performance:
        - Geometric-only: ~1ms per object pair
        - + CLIP: ~5-10ms per pair (GPU), ~50ms (CPU)
        - + LLM: ~500ms per scene (API latency)
    """
    enabled: bool = True
    max_relations: int = 10
    max_relations_per_object: int = 3
    auto_adjust_relation_cap: bool = True
    min_relations_per_object: int = 1
    relationship_types: tuple = ("spatial", "semantic", "action")
    confidence_threshold: float = 0.5
    use_clip_relations: bool = True
    use_geometric_relations: bool = True
    # OPTIMIZED: Raised from 0.23 to 0.30 to reduce false positives by ~20-30%
    clip_threshold: float = 0.5
    margin_px: int = 20
    min_distance: float = 5.0
    max_distance: float = 20000.0
    depth_front_threshold: float = 0.05
    depth_touching_threshold: float = 0.08
    min_relation_distance: float = 5.0
    max_relation_distance: float = 20000.0
    too_close_gap_px: float = 3.0
    too_close_gap_scale: float = 0.02
    filter_redundant: bool = True
    filter_relations_by_question: bool = True
    threshold_relation_similarity: float = 0.50
    relation_confidence_tie_eps: float = 0.05
    # Limits for CLIP-based pair scoring to avoid O(n^2) explosion
    max_clip_pairs: int = 500
    per_src_clip_pairs: int = 20
    
    # SOTA: LLM-guided relations (optional)
    use_llm_relations: bool = False
    llm_backend: str = "gpt4v"  # "gpt4v" | "llava" | "mock"
    llm_api_key: Optional[str] = None
    llm_confidence_threshold: float = 0.6
    
    # SOTA: 3D spatial reasoning (optional)
    use_3d_reasoning: bool = False
    depth_threshold: float = 0.1
    use_occlusion: bool = True
    
    # SOTA: Physics-informed filtering (ENABLED for better reliability)
    # Filters impossible relations like "sofa on_top_of book"
    use_physics_filtering: bool = True
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
        relations_config: Optional[RelationsConfig] = None,
        *,
        margin_px: int = 20,
        min_distance: float = 5.0,
        max_distance: float = 20000.0,
        enable_parallel: bool = True,
        max_workers: Optional[int] = None,
    ) -> None:
        self.clip = clip_scorer
        # Relations configuration (controls optional reasoners)
        self.config = relations_config or RelationsConfig()
        self.margin_px = int(margin_px)
        self.min_distance = float(min_distance)
        self.max_distance = float(max_distance)
        self.enable_parallel = enable_parallel
        # Use all available CPUs by default unless an explicit max_workers provided.
        self.max_workers = int(max_workers) if max_workers is not None else (os.cpu_count() or 1)

        # Optional advanced reasoners (instantiated if available and enabled)
        self.llm = None
        self.spatial3d = None
        self.physics = None

        try:
            if getattr(self.config, "use_llm_relations", False) and LLM_AVAILABLE and LLMRelationInferencer is not None:
                llm_conf = LLMRelationsConfig()
                # propagate minimal settings from RelationsConfig
                llm_conf.backend = getattr(self.config, "llm_backend", llm_conf.backend)
                llm_conf.api_key = getattr(self.config, "llm_api_key", llm_conf.api_key)
                llm_conf.confidence_threshold = getattr(self.config, "llm_confidence_threshold", llm_conf.confidence_threshold)
                self.llm = LLMRelationInferencer(llm_conf)
                print("[RelationInferencer] LLMRelationInferencer activated")
        except Exception as e:
            print(f"[RelationInferencer] Warning: failed to initialize LLM reasoner: {e}")

        try:
            if getattr(self.config, "use_3d_reasoning", False) and SPATIAL_3D_AVAILABLE and Spatial3DReasoner is not None:
                sp_conf = Spatial3DConfig()
                self.spatial3d = Spatial3DReasoner(sp_conf)
                print("[RelationInferencer] Spatial3DReasoner activated")
        except Exception as e:
            print(f"[RelationInferencer] Warning: failed to initialize Spatial3D reasoner: {e}")

        try:
            if getattr(self.config, "use_physics_filtering", False) and PHYSICS_AVAILABLE and PhysicsReasoner is not None:
                phys_conf = PhysicsConfig()
                self.physics = PhysicsReasoner(phys_conf)
                print("[RelationInferencer] PhysicsReasoner activated")
        except Exception as e:
            print(f"[RelationInferencer] Warning: failed to initialize Physics reasoner: {e}")
    
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
        margin = max(self.margin_px, avg_size * 0.08)
        
        # Check for significant overlap
        iou_val = iou(box_i, box_j)
        if iou_val > 0.3:
            return rels  # Skip highly overlapping boxes
        
        # Determine primary direction
        if abs(dy) >= abs(dx) and abs(dy) > margin:
            # Vertical relation
            relation = "above" if cy1 < cy2 else "below"  # i above j if y1 < y2
            v_overlap = vertical_overlap(box_i, box_j)
            if v_overlap > max(h_i, h_j) * 0.7:
                return rels  # Too much vertical overlap
            h_ref = min(h_i, h_j)
            contact_tol = max(2.0, 0.02 * h_ref)
            if relation == "above":
                gap = box_j[1] - box_i[3]
            else:
                gap = box_i[1] - box_j[3]
            if gap <= contact_tol:
                return rels
        elif abs(dx) > margin:
            # Horizontal relation
            relation = "left_of" if cx1 < cx2 else "right_of"  # i left of j if x1 < x2
            h_overlap = horizontal_overlap(box_i, box_j)
            if h_overlap > max(w_i, w_j) * 0.7:
                return rels  # Too much horizontal overlap
        else:
            return rels

        # Add primary relation (i -> j)
        rels.append(
            {"src_idx": i, "tgt_idx": j, "relation": relation, "distance": dist}
        )

        # Add inverse relation (j -> i)
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
        question_rel_terms: Optional[Set[str]] = None,
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
                        dist_ij = center_distance(
                            boxes[i], boxes[j],
                            mask1=(masks[i] if masks else None),
                            mask2=(masks[j] if masks else None),
                        )
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
                    mask_a = masks[i]["segmentation"] if masks else None
                    mask_b = masks[j]["segmentation"] if masks else None
                    da = None
                    db = None
                    if depth_map is not None:
                        da = depth_stats_from_map(mask_a, depth_map, box=boxes[i])
                        db = depth_stats_from_map(mask_b, depth_map, box=boxes[j])
                    elif depths is not None and i < len(depths) and j < len(depths):
                        da = depths[i]
                        db = depths[j]
                    ok = is_in_front_of(
                        boxes[i],
                        boxes[j],
                        mask_a=mask_a,
                        mask_b=mask_b,
                        depth_a=da,
                        depth_b=db,
                        depth_map=depth_map,
                        delta=float(getattr(self.config, "depth_front_threshold", 0.05)),
                    )
                    if ok:
                        dist_ij = center_distance(
                            boxes[i], boxes[j],
                            mask1=(masks[i] if masks else None),
                            mask2=(masks[j] if masks else None),
                        )
                        confidence = None
                        if da is not None and db is not None:
                            depth_diff = abs(float(da) - float(db))
                            denom = max(1e-6, float(getattr(self.config, "depth_front_threshold", 0.05)) * 2.0)
                            confidence = min(0.95, max(0.5, depth_diff / denom))
                        rel = {"src_idx": i, "tgt_idx": j, "relation": "in_front_of", "distance": dist_ij}
                        rel_inv = {"src_idx": j, "tgt_idx": i, "relation": "behind", "distance": dist_ij}
                        if confidence is not None:
                            rel["confidence"] = confidence
                            rel_inv["confidence"] = confidence
                        rels.append(rel)
                        rels.append(rel_inv)

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
                margin = np.maximum(self.margin_px, avg_size * 0.08)

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
                # exclude if vertical overlap too large (relaxed from 0.5 to 0.7)
                vertical_mask &= (inter_h <= (np.maximum(h[:, None], h[None, :]) * 0.85))

                # horizontal candidate: |dx| > margin
                horizontal_mask = (abs_dx > margin) & mask
                # exclude if horizontal overlap too large (relaxed from 0.5 to 0.7)
                horizontal_mask &= (inter_w <= (np.maximum(w[:, None], w[None, :]) * 0.85))

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

                            if relation in {"above", "below"}:
                                h_ref = min(h[i], h[j])
                                contact_tol = max(2.0, 0.02 * h_ref)
                                if relation == "above":
                                    gap = y1[j] - y2[i]
                                else:
                                    gap = y1[i] - y2[j]
                                if gap <= contact_tol:
                                    continue
                            
                            # Calibrate confidence based on distance and overlap
                            # Closer objects with less overlap = higher confidence
                            overlap_ratio = 0.0
                            if vertical_mask[i, j]:
                                max_w = max(w[i], w[j])
                                overlap_ratio = float(inter_w[i, j] / max_w) if max_w > 0 else 0.0
                            else:
                                max_h = max(h[i], h[j])
                                overlap_ratio = float(inter_h[i, j] / max_h) if max_h > 0 else 0.0
                            
                            # Confidence: high for close objects with low overlap
                            # Range: [0.4, 0.9] based on distance and overlap
                            distance_factor = 1.0 / (1.0 + dist_ij / 500.0)  # decays with distance
                            overlap_penalty = overlap_ratio * 0.3  # penalize high overlap
                            confidence = min(0.9, max(0.4, 0.5 + distance_factor * 0.4 - overlap_penalty))
                            
                            rels.append({
                                "src_idx": i, 
                                "tgt_idx": j, 
                                "relation": relation, 
                                "distance": dist_ij,
                                "confidence": confidence
                            })
                            inverse_relation = _INVERSE.get(relation)
                            if inverse_relation:
                                rels.append({
                                    "src_idx": j, 
                                    "tgt_idx": i, 
                                    "relation": inverse_relation, 
                                    "distance": dist_ij,
                                    "confidence": confidence
                                })
                # Spatial3D reasoner: try to infer depth-based / occlusion / support relations
                if self.spatial3d is not None:
                    try:
                        # only run when depth information is available
                        if depths is not None or depth_map is not None:
                            rels_3d = self.spatial3d.infer_3d_relations(
                                boxes=boxes,
                                depth_map=depth_map,
                                depths=depths,
                                masks=masks,
                                image_size=(image_pil.width, image_pil.height) if image_pil is not None else None,
                            )
                            if rels_3d:
                                rels.extend(rels_3d)
                    except Exception as e:
                        print(f"[RelationInferencer] Warning: spatial3d inference failed: {e}")
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
            # Build a filtered list of candidate directed pairs to score with CLIP.
            # Heuristics:
            #  - Only evaluate pairs within [min_distance, max_distance]
            #  - Skip pairs with strong IoU overlap (likely same object or containment)
            #  - For each source, only keep up to per_src_clip_pairs nearest targets
            #  - Enforce a global cap max_clip_pairs
            clip_pairs = []
            try:
                # compute centers and pairwise distances
                centers = [((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0) for b in boxes]
                dists = [[0.0] * n for _ in range(n)]
                for i in range(n):
                    for j in range(n):
                        if i == j:
                            continue
                        dx = centers[j][0] - centers[i][0]
                        dy = centers[j][1] - centers[i][1]
                        d = math.hypot(dx, dy)
                        dists[i][j] = d

                # compute pairwise iou matrix fast-ish
                iou_mat = [[0.0] * n for _ in range(n)]
                for i in range(n):
                    bi = boxes[i]
                    ai = max(0.0, (bi[2] - bi[0]) * (bi[3] - bi[1]))
                    for j in range(n):
                        if i == j:
                            continue
                        bj = boxes[j]
                        inter_w = max(0.0, min(bi[2], bj[2]) - max(bi[0], bj[0]))
                        inter_h = max(0.0, min(bi[3], bj[3]) - max(bi[1], bj[1]))
                        inter = inter_w * inter_h
                        aj = max(0.0, (bj[2] - bj[0]) * (bj[3] - bj[1]))
                        union = ai + aj - inter if (ai + aj - inter) > 0 else 1.0
                        iou_mat[i][j] = inter / union if union > 0 else 0.0

                # For each source, get candidate targets sorted by distance and filtered
                for i in range(n):
                    cand = []
                    for j in range(n):
                        if i == j:
                            continue
                        d = dists[i][j]
                        if d < self.config.min_distance or d > self.config.max_distance:
                            continue
                        if iou_mat[i][j] > 0.6:
                            # too much overlap — skip (likely same object or containment)
                            continue
                        cand.append((j, d))
                    if not cand:
                        continue
                    cand.sort(key=lambda x: x[1])
                    topk = [t for t, _ in cand[: self.config.per_src_clip_pairs]]
                    for t in topk:
                        clip_pairs.append((i, t))

                # enforce a global cap while preserving nearest-first ordering
                if len(clip_pairs) > self.config.max_clip_pairs:
                    clip_pairs = clip_pairs[: self.config.max_clip_pairs]
            except Exception:
                # Fallback to all pairs if anything goes wrong
                clip_pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
            try:
                for i, j, rel_canon, rel_raw, score in self.clip.batch_best_relations(
                    image_pil=image_pil, boxes=boxes, labels=labels, pairs=clip_pairs
                ):
                    if score > clip_threshold:
                        dist = center_distance(
                            boxes[i], boxes[j],
                            mask1=(masks[i] if masks else None),
                            mask2=(masks[j] if masks else None),
                        )
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
                            dist = center_distance(
                                boxes[i], boxes[j],
                                mask1=(masks[i] if masks else None),
                                mask2=(masks[j] if masks else None),
                            )
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

        # Optional LLM-guided relations (use CLIP/geometry as context)
        if self.llm is not None:
            try:
                llm_rels = self.llm.infer_relations(
                    image=image_pil,
                    boxes=boxes,
                    labels=labels,
                    existing_relations=rels,
                )
                # normalize returned relations and merge
                for r in llm_rels:
                    # Accept either 'src_idx'/'tgt_idx' or 'src'/'tgt'
                    if "src" in r and "tgt" in r and "src_idx" not in r:
                        r["src_idx"] = r.pop("src")
                        r["tgt_idx"] = r.pop("tgt")
                    # promote LLM confidence into clip_sim for unified ranking
                    conf = r.get("confidence") or r.get("score") or r.get("llm_confidence")
                    if conf is not None:
                        try:
                            r["clip_sim"] = float(conf)
                        except Exception:
                            pass
                if llm_rels:
                    rels.extend(llm_rels)
            except Exception as e:
                print(f"[RelationInferencer] Warning: LLM inference failed: {e}")

        rels = self._unify_pair_relations(rels)
        # Filter out semantically impossible relations (e.g., inanimate wearing objects)
        try:
            rels = self._filter_impossible_relations(rels, labels)
        except Exception:
            # best-effort: if filter fails, keep original relations
            pass

        # Check for contradictory relations (e.g., A left_of B + B left_of A)
        try:
            rels = self._check_consistency(rels)
        except Exception as e:
            print(f"[RelationInferencer] Warning: consistency check failed: {e}")

        # Depth-aware validation: keep "touching" only when depths agree
        if depths is not None or depth_map is not None:
            try:
                rels = self._filter_touching_by_depth(
                    rels, boxes, depths=depths, depth_map=depth_map, masks=masks
                )
            except Exception as e:
                print(f"[RelationInferencer] Warning: touching depth filter failed: {e}")

        # Optional physics-based filtering/scoring (post semantic-filter)
        if self.physics is not None:
            try:
                # allow physics reasoner to add support relations (if configured)
                if getattr(self.config, "check_support", True):
                    try:
                        support_rels = self.physics.detect_support_relations(boxes, depths=depths, masks=masks)
                        if support_rels:
                            rels.extend(support_rels)
                    except Exception:
                        pass

                # Filter and score relations by physics plausibility
                rels = self.physics.filter_relations(rels, boxes, depths=depths, masks=masks)
            except Exception as e:
                print(f"[RelationInferencer] Warning: physics filter failed: {e}")

        if filter_redundant:
            rels = self._filter_redundant_relations(rels, question_rel_terms=question_rel_terms)
            
        return rels

    # ---------------------------------------------------------------------
    # Post-processing / utilities
    # ---------------------------------------------------------------------

    def _touching_depth_consistent(
        self,
        i: int,
        j: int,
        boxes: Sequence[Sequence[float]],
        *,
        depths: Optional[Sequence[float]] = None,
        depth_map: Optional[np.ndarray] = None,
        masks: Optional[Sequence[dict]] = None,
    ) -> bool:
        """Return True if depth evidence supports contact between i and j."""
        if (depths is None and depth_map is None) or i >= len(boxes) or j >= len(boxes):
            return True
        threshold = float(getattr(self.config, "depth_touching_threshold", 0.08))
        depth_diff = None
        if depth_map is not None:
            mask_a = masks[i]["segmentation"] if masks else None
            mask_b = masks[j]["segmentation"] if masks else None
            da = depth_stats_from_map(mask_a, depth_map, box=boxes[i])
            db = depth_stats_from_map(mask_b, depth_map, box=boxes[j])
            if da is not None and db is not None:
                depth_diff = abs(da - db)
        if depth_diff is None and depths is not None and i < len(depths) and j < len(depths):
            da = depths[i]
            db = depths[j]
            if da is not None and db is not None:
                depth_diff = abs(float(da) - float(db))
        if depth_diff is None:
            return True
        return depth_diff <= threshold

    def _filter_touching_by_depth(
        self,
        relationships: List[dict],
        boxes: Sequence[Sequence[float]],
        *,
        depths: Optional[Sequence[float]] = None,
        depth_map: Optional[np.ndarray] = None,
        masks: Optional[Sequence[dict]] = None,
    ) -> List[dict]:
        """Remove touching relations that are inconsistent with depth."""
        if not relationships:
            return relationships
        filtered = []
        for rel in relationships:
            rel_name = str(rel.get("relation", "")).lower()
            if "touching" in rel_name:
                i = rel.get("src_idx")
                j = rel.get("tgt_idx")
                if i is None or j is None:
                    filtered.append(rel)
                    continue
                if not self._touching_depth_consistent(
                    int(i),
                    int(j),
                    boxes,
                    depths=depths,
                    depth_map=depth_map,
                    masks=masks,
                ):
                    continue
            filtered.append(rel)
        return filtered

    def filter_relations_by_proximity(
        self,
        relationships: List[dict],
        boxes: Sequence[Sequence[float]],
        *,
        question_rel_terms: Optional[Set[str]] = None,
    ) -> List[dict]:
        """
        Drop relations that are too close (unless justified) or too far.
        Question-requested relations are always kept.
        """
        if not relationships:
            return relationships

        min_rel_dist = float(getattr(self.config, "min_relation_distance", self.min_distance))
        max_rel_dist = float(getattr(self.config, "max_relation_distance", self.max_distance))
        gap_px = float(getattr(self.config, "too_close_gap_px", 3.0))
        gap_scale = float(getattr(self.config, "too_close_gap_scale", 0.02))

        allowed_close = {
            "on_top_of",
            "under",
            "below",
            "inside",
            "contained_in",
            "overlaps",
            "overlapping",
            "touching",
            "adjacent",
            "supported_by",
            "supports",
            "resting_on",
            "sitting_on",
            "leaning_on",
            "in_front_of",
            "behind",
            "holding",
            "wearing",
            "carrying",
            "riding",
        }

        def _matches_question(rel_name: str) -> bool:
            if not question_rel_terms:
                return False
            label = rel_name.lower().replace("_", " ")
            return any(t.lower().replace("_", " ") in label for t in question_rel_terms)

        out: List[dict] = []
        for r in relationships:
            if r.get("forced"):
                out.append(r)
                continue
            rel_name = str(r.get("relation", "")).lower()
            if _matches_question(rel_name):
                out.append(r)
                continue

            i = r.get("src_idx")
            j = r.get("tgt_idx")
            dist = r.get("distance", None)
            if dist is None and i is not None and j is not None and i < len(boxes) and j < len(boxes):
                dist = center_distance(boxes[int(i)], boxes[int(j)])

            # Too far: drop any relation (unless question requested above)
            if dist is not None and dist > max_rel_dist:
                continue

            # Too close: drop if relation not justified by contact/functional semantics
            if i is not None and j is not None and i < len(boxes) and j < len(boxes):
                a = boxes[int(i)]
                b = boxes[int(j)]
                x1, y1, x2, y2 = as_xyxy(a)
                X1, Y1, X2, Y2 = as_xyxy(b)
                avg_size = max(1.0, (x2 - x1 + y2 - y1 + X2 - X1 + Y2 - Y1) / 4.0)
                gap_thresh = max(gap_px, gap_scale * avg_size)
                if edge_gap(a, b) <= gap_thresh:
                    if not any(tok in rel_name for tok in allowed_close):
                        continue

            # Too near by center distance: suppress weak spatial cues
            if dist is not None and dist < min_rel_dist:
                if not any(tok in rel_name for tok in allowed_close):
                    continue

            out.append(r)

        return out

    def limit_relationships_per_object(
        self,
        relationships: List[dict],
        boxes: Sequence[Sequence[float]],
        *,
        max_relations_per_object: int = 3,
        min_relations_per_object: int = 1,
        question_rel_terms: Optional[Set[str]] = None,
        question_subject_idxs: Optional[Set[int]] = None,
        masks: Optional[Sequence[dict]] = None,
        depths: Optional[Sequence[float]] = None,
        depth_map: Optional[np.ndarray] = None,
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

        n = len(boxes)
        max_relations_per_object = self._compute_effective_max_relations_per_object(
            relationships=relationships,
            num_objects=n,
            max_relations_per_object=max_relations_per_object,
            question_rel_terms=question_rel_terms,
        )
        question_subject_idxs = question_subject_idxs or set()

        # Guarantee a minimum per object
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
                    build_precise_nearest_relation(
                        i,
                        best_j,
                        boxes,
                        margin_px=self.margin_px,
                        depth_map=depth_map,
                        depths=depths,
                        masks=masks,
                        depth_touching_threshold=float(
                            getattr(self.config, "depth_touching_threshold", 0.08)
                        ),
                    )
                )

        # Cap per-object; prioritize relations mentioned in the question
        def _is_question_rel(rel_label: str, rel: Optional[dict] = None) -> bool:
            return self._relation_matches_question(
                rel,
                question_rel_terms=question_rel_terms,
                question_subject_idxs=question_subject_idxs,
            )

        # Compute area-based caps to avoid overcrowding small objects.
        areas = []
        for b in boxes:
            x1, y1, x2, y2 = as_xyxy(b)
            areas.append(max(1.0, float(x2 - x1)) * max(1.0, float(y2 - y1)))
        max_area = max(areas) if areas else 1.0

        final: List[dict] = []
        for i, rlist in rels_by_src.items():
            # Shrink cap for small objects to reduce label clutter.
            rel_cap = max_relations_per_object
            if i < len(areas):
                area_ratio = float(areas[i] / max_area) if max_area > 0 else 1.0
                rel_cap = min(rel_cap, 2)
            # Allow more relations for question target objects.
            if i in question_subject_idxs:
                rel_cap = max(rel_cap, 3)

            # Sort by confidence/score if present, otherwise by distance
            def rel_sort_key(r):
                # Priority: question term, then score/confidence, then distance
                q_priority = 0 if _is_question_rel(r.get("relation", ""), r) else 1
                rel_priority = self._get_relation_priority(r.get("relation", ""))
                rel_conf = self._get_relation_confidence(r)
                # Use clip_sim, score, or distance
                score = r.get("clip_sim", None)
                if score is None:
                    score = r.get("score", None)
                if score is not None:
                    # Negative score for descending order
                    return (q_priority, -rel_priority, -score, r.get("distance", 1e9))
                else:
                    return (q_priority, -rel_priority, -rel_conf, r.get("distance", 1e9))
            question_rels = [r for r in rlist if _is_question_rel(r.get("relation", ""), r)]
            other_rels = [r for r in rlist if not _is_question_rel(r.get("relation", ""), r)]
            q_sorted = sorted(question_rels, key=rel_sort_key)
            other_sorted = sorted(other_rels, key=rel_sort_key)
            if rel_cap > 0:
                remaining = rel_cap - len(q_sorted)
                if remaining < 0:
                    remaining = 0
            else:
                remaining = len(other_sorted)
            final.extend(q_sorted + other_sorted[:remaining])
        return final

    def _compute_effective_max_relations_per_object(
        self,
        *,
        relationships: Sequence[dict],
        num_objects: int,
        max_relations_per_object: int,
        question_rel_terms: Optional[Set[str]] = None,
    ) -> int:
        """
        Auto-scale per-object relation cap to avoid overcrowding while
        preserving question-requested relations.
        """
        if not getattr(self.config, "auto_adjust_relation_cap", False):
            return max_relations_per_object

        n = max(1, int(num_objects))
        rel_count = len(relationships)
        if rel_count <= 0:
            return max(1, max_relations_per_object)

        base_target = max(2, int(math.ceil(n * 0.7)))
        if question_rel_terms:
            base_target = max(base_target, int(math.ceil(n * 1.0)))
        target_total = min(self.config.max_relations, base_target, rel_count)
        cap = max(1, int(math.ceil(target_total / n)))

        if max_relations_per_object and max_relations_per_object > 0:
            cap = min(cap, max_relations_per_object)
        return cap

    def _filter_redundant_relations(
        self,
        relationships: List[dict],
        *,
        question_rel_terms: Optional[Set[str]] = None,
    ) -> List[dict]:
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
                best_rel = self._choose_best_relation(rels, question_rel_terms=question_rel_terms)
                filtered_relations.append(best_rel)
        
        return filtered_relations

    def _choose_best_relation(
        self,
        relations: List[dict],
        *,
        question_rel_terms: Optional[Set[str]] = None,
    ) -> dict:
        """
        Pick the most informative relation among candidates for the same pair.
        Priority: semantic > specific spatial (contact/adjacency) > generic spatial > directional.
        """
        # Priority tiers for relation types
        semantic_relations = {"on_top_of", "under", "holding", "wearing", "riding", "sitting_on", "carrying"}
        spatial_specific = {"touching", "adjacent", "near", "close"}
        spatial_directional = {"left_of", "right_of", "above", "below"}
        spatial_depth = {"in_front_of", "behind"}
        
        def _is_question_rel(rel: dict) -> bool:
            return self._relation_matches_question(rel, question_rel_terms=question_rel_terms)

        best_rel = relations[0]
        best_priority = self._get_relation_priority(best_rel["relation"])
        best_confidence = self._get_relation_confidence(best_rel)
        best_specificity = self._get_relation_specificity(best_rel["relation"])
        conf_eps = float(getattr(self.config, "relation_confidence_tie_eps", 0.05))
        
        for rel in relations[1:]:
            # Priority 0: question-requested relations
            if _is_question_rel(rel) and not _is_question_rel(best_rel):
                best_rel = rel
                best_priority = self._get_relation_priority(best_rel["relation"])
                best_confidence = self._get_relation_confidence(best_rel)
                best_specificity = self._get_relation_specificity(best_rel["relation"])
                continue
            if _is_question_rel(best_rel) and not _is_question_rel(rel):
                continue

            priority = self._get_relation_priority(rel["relation"])
            confidence = self._get_relation_confidence(rel)
            specificity = self._get_relation_specificity(rel["relation"])
            
            # Compare by priority first, then by confidence
            if priority > best_priority:
                best_rel = rel
                best_priority = priority
                best_confidence = confidence
                best_specificity = specificity
                continue
            if priority == best_priority:
                if confidence > (best_confidence + conf_eps):
                    best_rel = rel
                    best_priority = priority
                    best_confidence = confidence
                    best_specificity = specificity
                    continue
                if abs(confidence - best_confidence) <= conf_eps:
                    rel_name = str(rel.get("relation", "")).lower()
                    best_name = str(best_rel.get("relation", "")).lower()
                    rel_is_depth = rel_name in spatial_depth
                    best_is_depth = best_name in spatial_depth
                    rel_is_directional = rel_name in spatial_directional
                    best_is_directional = best_name in spatial_directional
                    # Prefer depth relations over directional ones when confidence is tied.
                    if rel_is_depth and best_is_directional:
                        best_rel = rel
                        best_priority = priority
                        best_confidence = confidence
                        best_specificity = specificity
                        continue
                    if best_is_depth and rel_is_directional:
                        continue
                    if specificity > best_specificity or (
                        specificity == best_specificity and confidence > best_confidence
                    ):
                        best_rel = rel
                        best_priority = priority
                        best_confidence = confidence
                        best_specificity = specificity
        
        return best_rel

    def _relation_matches_question(
        self,
        rel: Optional[dict],
        *,
        question_rel_terms: Optional[Set[str]] = None,
        question_subject_idxs: Optional[Set[int]] = None,
    ) -> bool:
        """Return True if a relation matches the question direction/context."""
        if rel is None:
            return False
        if rel.get("forced"):
            return True
        if not question_rel_terms:
            return False

        rel_terms = {str(t).lower().replace(" ", "_") for t in question_rel_terms}
        label = str(rel.get("relation", "")).lower().replace(" ", "_")
        inverse_map = {
            "above": "below",
            "below": "above",
            "left_of": "right_of",
            "right_of": "left_of",
            "in_front_of": "behind",
            "behind": "in_front_of",
        }
        directional = set(inverse_map.keys())
        src_idx = rel.get("src_idx")
        tgt_idx = rel.get("tgt_idx")

        if label in rel_terms:
            if question_subject_idxs and label in directional:
                return tgt_idx in question_subject_idxs
            return True

        if label in inverse_map and inverse_map[label] in rel_terms:
            if question_subject_idxs:
                return src_idx in question_subject_idxs
            return False

        return False

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
            
        # 2: directional + depth relations (same priority)
        spatial_directional = {"left_of", "right_of", "above", "below"}
        spatial_depth = {"in_front_of", "behind"}
        if any(dir_rel in rel_name for dir_rel in spatial_directional | spatial_depth):
            return 2
            
        # 0: others
        return 0

    def _get_relation_specificity(self, relation: str) -> int:
        """Assign a specificity score for tie-breaking within same priority."""
        rel_name = str(relation).lower()
        specific_high = {
            "on_top_of",
            "under",
            "inside",
            "contained_in",
            "overlaps",
            "overlapping",
            "touching",
            "adjacent",
            "supported_by",
            "supports",
            "resting_on",
            "sitting_on",
            "leaning_on",
            "holding",
            "wearing",
            "carrying",
            "riding",
        }
        if any(tok in rel_name for tok in specific_high):
            return 2
        specific_mid = {"left_of", "right_of", "above", "below", "in_front_of", "behind"}
        if any(tok in rel_name for tok in specific_mid):
            return 1
        specific_low = {"near", "close"}
        if any(tok in rel_name for tok in specific_low):
            return 0
        return 0

    def _get_relation_confidence(self, relation: dict) -> float:
        """Extract a confidence score: prefer explicit confidence, then CLIP similarity, else inverse distance."""
        # Priority: explicit confidence > clip_sim > distance-based
        if "confidence" in relation:
            return float(relation["confidence"])
        elif "clip_sim" in relation:
            return float(relation["clip_sim"])
        elif "distance" in relation:
            # Inverse distance (closer ⇒ higher)
            dist = float(relation["distance"])
            return 1.0 / (1.0 + dist / 100.0)
        else:
            return 0.5  # default for purely geometric relations

    def _check_consistency(self, relationships: List[dict]) -> List[dict]:
        """
        Remove contradictory relations (e.g., A left_of B + B left_of A).
        Detects cycles and violations of directional/hierarchical constraints.
        
        Returns:
            Filtered list with contradictions removed (keeps higher confidence relation)
        """
        if not relationships:
            return relationships
        
        # Define opposing relation pairs
        opposing_pairs = {
            ("left_of", "right_of"),
            ("above", "below"),
            ("in_front_of", "behind"),
            ("on_top_of", "under"),
        }
        
        # Build bidirectional mapping for quick lookup
        opposites_map = {}
        for rel1, rel2 in opposing_pairs:
            opposites_map[rel1] = rel2
            opposites_map[rel2] = rel1
        
        # Track relations by (src, tgt) pair
        relation_dict = {}  # (src_idx, tgt_idx) -> relation
        
        filtered = []
        for rel in relationships:
            src_idx = rel["src_idx"]
            tgt_idx = rel["tgt_idx"]
            rel_type = rel.get("relation", "").lower()
            
            # Check for direct contradiction: (tgt, src) with opposing relation
            reverse_key = (tgt_idx, src_idx)
            if reverse_key in relation_dict:
                existing_rel = relation_dict[reverse_key]
                existing_type = existing_rel.get("relation", "").lower()
                
                # Check if relations are opposing
                if opposites_map.get(rel_type) == existing_type:
                    # Contradiction detected - keep the one with higher confidence
                    existing_conf = self._get_relation_confidence(existing_rel)
                    current_conf = self._get_relation_confidence(rel)
                    
                    if current_conf > existing_conf:
                        # Remove existing, add current
                        filtered.remove(existing_rel)
                        del relation_dict[reverse_key]
                        relation_dict[(src_idx, tgt_idx)] = rel
                        filtered.append(rel)
                    # else: skip current, keep existing (already in filtered)
                    continue
            
            # Check for symmetric contradiction: (src, tgt) already exists with same direction
            forward_key = (src_idx, tgt_idx)
            if forward_key in relation_dict:
                # Multiple relations for same pair - will be handled by _filter_redundant_relations
                pass
            
            # Add relation
            relation_dict[forward_key] = rel
            filtered.append(rel)
        
        return filtered

    def _filter_impossible_relations(self, relationships: List[dict], labels: Sequence[str]) -> List[dict]:
        """Delegate semantic filtering to the optional semantic filter helper.

        The heavy lifting is done in `src.gom.relations.semantic_filter`. If
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
        
        # Large furniture/objects that typically support other objects (not vice versa)
        large_supporters = {
            "sofa", "couch", "bed", "table", "desk", "floor", "ground",
            "chair", "bench", "shelf", "counter", "cabinet"
        }
        
        # Small objects that cannot support large objects
        small_objects = {
            "book", "cup", "glass", "plate", "bowl", "bottle", "phone",
            "remote", "mouse", "keyboard", "pen", "pencil", "paper"
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
            
            # Size-aware semantic filtering for "on_top_of" relations
            if rel in ("on_top_of", "above"):
                # Large furniture cannot be on top of small objects
                subj_is_large = any(tok in subj_norm for tok in large_supporters)
                obj_is_small = any(tok in obj_norm for tok in small_objects)
                
                if subj_is_large and obj_is_small:
                    # Skip impossible relations like "sofa on_top_of book"
                    continue

            out.append(r)

        return out

    def drop_inverse_duplicates(
        self,
        relationships: List[dict],
        *,
        question_subject_idxs: Optional[Set[int]] = None,
        question_rel_terms: Optional[Set[str]] = None,
        max_relations_per_object: int = 3,
        min_relations_per_object: int = 1,
        total_objects: Optional[int] = None,
    ) -> List[dict]:
        """
        Remove inverse duplicate relations correctly.
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

        def _is_question_rel(rel: dict) -> bool:
            return self._relation_matches_question(
                rel,
                question_rel_terms=question_rel_terms,
                question_subject_idxs=question_subject_idxs,
            )
        
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
                    # Priority 1.5: Question-requested relations
                    r_is_qrel = _is_question_rel(rel)
                    e_is_qrel = _is_question_rel(existing)
                    if r_is_qrel and not e_is_qrel:
                        count_per_src[existing["src_idx"]] = count_per_src.get(existing["src_idx"], 1) - 1
                        seen_pairs[pair_key] = rel
                        count_per_src[i] = count_per_src.get(i, 0) + 1
                        continue
                    if e_is_qrel and not r_is_qrel:
                        continue

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
        question_subject_idxs: Optional[Set[int]] = None,
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

        def _norm_label(s: str) -> str:
            return s.lower().strip().replace(" ", "_")

        norm_terms = {_norm_label(t) for t in question_terms}
        inverse_map = {
            "above": "below",
            "below": "above",
            "left_of": "right_of",
            "right_of": "left_of",
            "in_front_of": "behind",
            "behind": "in_front_of",
        }
        directional_terms = set(inverse_map.keys())
        question_dir_terms = {t for t in norm_terms if t in inverse_map}
        restrict_to_targets = question_subject_idxs is not None and len(question_subject_idxs) > 0

        out: List[dict] = []
        for r in relationships:
            label = _norm_label(str(r.get("relation", "")))
            src_idx = r.get("src_idx")
            tgt_idx = r.get("tgt_idx")

            if restrict_to_targets:
                if src_idx not in question_subject_idxs and tgt_idx not in question_subject_idxs:
                    continue

                # If the question specifies a directional relation, enforce direction relative to target.
                if question_dir_terms:
                    if label in question_dir_terms:
                        if tgt_idx in question_subject_idxs:
                            r2 = r.copy()
                            r2["forced"] = True
                            out.append(r2)
                        continue
                    if label in inverse_map and inverse_map[label] in question_dir_terms:
                        if src_idx in question_subject_idxs:
                            desired = inverse_map[label]
                            flipped = r.copy()
                            flipped["src_idx"] = tgt_idx
                            flipped["tgt_idx"] = src_idx
                            flipped["relation"] = desired
                            flipped["forced"] = True
                            out.append(flipped)
                        continue

                # If question asks a specific direction, keep other target-involved
                # relations as secondary context (only non-directional).
                if question_dir_terms:
                    if label not in directional_terms:
                        out.append(r)
                    continue

                # Otherwise, keep additional relations involving the target for context.
                out.append(r)
                continue

            keep = False
            for t_norm in norm_terms:
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

    def enforce_question_relations(
        self,
        relationships: List[dict],
        boxes: Sequence[Sequence[float]],
        *,
        question_rel_terms: Optional[Set[str]] = None,
        question_subject_idxs: Optional[Set[int]] = None,
        masks: Optional[Sequence[dict]] = None,
        depths: Optional[Sequence[float]] = None,
        depth_map: Optional[np.ndarray] = None,
    ) -> List[dict]:
        """
        Ensure that relations explicitly requested by the question exist when geometry supports them.
        """
        if not question_rel_terms or not question_subject_idxs or not boxes:
            return relationships

        directional = {"above", "below", "left_of", "right_of", "in_front_of", "behind"}
        rel_terms = {t for t in question_rel_terms if t in directional}
        if not rel_terms:
            return relationships

        centers = [((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0) for b in boxes]
        widths = [max(1.0, b[2] - b[0]) for b in boxes]
        heights = [max(1.0, b[3] - b[1]) for b in boxes]

        existing = {(r.get("src_idx"), r.get("tgt_idx"), r.get("relation")) for r in relationships}

        def _margin(i: int, j: int) -> float:
            avg = (widths[i] + heights[i] + widths[j] + heights[j]) / 4.0
            return max(self.margin_px * 0.5, avg * 0.06)

        def _depth_at(idx: int) -> Optional[float]:
            if depths is not None and idx < len(depths):
                return depths[idx]
            if depth_map is not None and masks is not None and idx < len(masks):
                m = masks[idx]["segmentation"] if masks[idx] is not None else None
                return depth_stats_from_map(m, depth_map, box=boxes[idx])
            return None

        added = []
        for tgt in question_subject_idxs:
            if tgt < 0 or tgt >= len(boxes):
                continue
            cx_t, cy_t = centers[tgt]
            for rel in rel_terms:
                # If already present for this target, skip.
                if rel in {"above", "below", "left_of", "right_of"}:
                    already = any(
                        r[2] == rel and r[1] == tgt for r in existing
                    )
                else:
                    already = any(
                        r[2] == rel and r[1] == tgt for r in existing
                    )
                if already:
                    continue

                best_idx = None
                best_dist = float("inf")
                for i in range(len(boxes)):
                    if i == tgt:
                        continue
                    cx_i, cy_i = centers[i]
                    dx = cx_i - cx_t
                    dy = cy_i - cy_t
                    margin = _margin(i, tgt)
                    da = _depth_at(i)
                    db = _depth_at(tgt)
                    if da is not None and db is not None and rel in {"above", "below", "left_of", "right_of"}:
                        depth_delta = abs(float(da) - float(db))
                        depth_gate = float(getattr(self.config, "depth_front_threshold", 0.05)) * 2.0
                        if depth_delta > depth_gate:
                            continue
                    if rel == "below":
                        if dy <= margin or abs(dy) < abs(dx) * 0.6:
                            continue
                    elif rel == "above":
                        if dy >= -margin or abs(dy) < abs(dx) * 0.6:
                            continue
                    elif rel == "left_of":
                        if dx >= -margin or abs(dx) < abs(dy) * 0.6:
                            continue
                    elif rel == "right_of":
                        if dx <= margin or abs(dx) < abs(dy) * 0.6:
                            continue
                    elif rel in {"in_front_of", "behind"}:
                        if da is None or db is None:
                            continue
                        delta = float(getattr(self.config, "depth_front_threshold", 0.05))
                        if rel == "in_front_of" and (da <= db + delta):
                            continue
                        if rel == "behind" and (da >= db - delta):
                            continue
                    dist = math.hypot(dx, dy)
                    if dist < best_dist:
                        best_idx = i
                        best_dist = dist

                if best_idx is None:
                    continue

                # Build relation directed from candidate -> target.
                rel_dict = {
                    "src_idx": best_idx,
                    "tgt_idx": tgt,
                    "relation": rel,
                    "distance": float(best_dist),
                    "confidence": 0.55,
                    "forced": True,
                }
                key = (rel_dict["src_idx"], rel_dict["tgt_idx"], rel_dict["relation"])
                if key not in existing:
                    added.append(rel_dict)
                    existing.add(key)

        if added:
            relationships = list(relationships) + added
        return relationships

    # unify_spatial_direction removed: spatial relations keep original direction src_idx → tgt_idx

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
        num_objects = total_objects
        if num_objects is None:
            uniq = set()
            for rel in relationships:
                uniq.add(rel.get("src_idx"))
                uniq.add(rel.get("tgt_idx"))
            num_objects = len([u for u in uniq if u is not None])
        max_relations_per_object = self._compute_effective_max_relations_per_object(
            relationships=relationships,
            num_objects=max(1, num_objects),
            max_relations_per_object=max_relations_per_object,
            question_rel_terms=question_rel_terms,
        )
