# igp/relations/inference.py
from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple
from dataclasses import dataclass

import math
from PIL import Image

from .clip_rel import ClipRelScorer
from .geometry import (
    as_xyxy,
    build_precise_nearest_relation,
    center_distance,
    iou,
    is_below_of,
    is_on_top_of,
    orientation_label,
)

@dataclass
class RelationsConfig:
    """Configuration for relationship inference"""
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
    "left_of": "right_of",
    "right_of": "left_of",
    "above": "below",
    "below": "above",
    "in_front_of": "behind",
    "behind": "in_front_of",
    "on_top_of": "below",
    "under": "on_top_of",
}


class RelationInferencer:
    """
    Combina euristiche geometriche e scoring CLIP per derivare relazioni tra oggetti.
    Restituisce una lista di dict:
      { "src_idx", "tgt_idx", "relation", "distance", ["relation_raw", "clip_sim"] }
    """

    def __init__(
        self,
        clip_scorer: Optional[ClipRelScorer] = None,
        *,
        margin_px: int = 20,
        min_distance: float = 5.0,
        max_distance: float = 20000.0,
    ) -> None:
        self.clip = clip_scorer
        self.margin_px = int(margin_px)
        self.min_distance = float(min_distance)
        self.max_distance = float(max_distance)

    def infer(
        self,
        image_pil: Image.Image,
        boxes: Sequence[Sequence[float]],
        labels: Optional[Sequence[str]] = None,
        *,
        masks: Optional[Sequence[dict]] = None,
        depths: Optional[Sequence[float]] = None,
        use_geometry: bool = True,
        use_clip: bool = True,
        clip_threshold: float = 0.23,
    ) -> List[dict]:
        """
        Calcola relazioni candidate (geometriche + CLIP).
        """
        n = len(boxes)
        if n <= 1:
            return []

        if labels is None:
            labels = [f"obj{i}" for i in range(n)]

        rels: List[dict] = []

        # ---------- 1) Geometria: on_top_of / below simmetrici ----------
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
                    )
                    if ok:
                        dist_ij = center_distance(boxes[i], boxes[j])
                        rels.append(
                            {"src_idx": i, "tgt_idx": j, "relation": "on_top_of", "distance": dist_ij}
                        )
                        rels.append(
                            {"src_idx": j, "tgt_idx": i, "relation": "below", "distance": dist_ij}
                        )

        # ---------- 2) Geometria: above/below/left/right con margine e distanza ----------
        if use_geometry:
            centers = [((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0) for b in boxes]
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    cx1, cy1 = centers[i]
                    cx2, cy2 = centers[j]
                    dx, dy = cx2 - cx1, cy2 - cy1
                    dist = math.hypot(dx, dy)
                    if dist < self.min_distance or dist > self.max_distance:
                        continue
                    if abs(dy) >= abs(dx) and abs(dy) > self.margin_px:
                        relation = "above" if dy < 0 else "below"
                    elif abs(dx) > self.margin_px:
                        relation = "right_of" if dx > 0 else "left_of"
                    else:
                        continue
                    rels.append(
                        {"src_idx": i, "tgt_idx": j, "relation": relation, "distance": dist}
                    )

        # ---------- 3) CLIP ----------#
        if use_clip and self.clip is not None:
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    rel_canon, rel_raw, score = self.clip.best_relation(
                        image_pil, boxes[i], boxes[j], labels[i], labels[j]
                    )
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

        # deduplica primordiale (mantieni la più vicina per ogni coppia direzionata)
        rels = self._unify_pair_relations(rels)
        return rels

    # ---------------------------------------------------------------------
    # Post-processing/utility
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
        Garantisce almeno `min_relations_per_object` relazioni per nodo (con nearest),
        e limita a `max_relations_per_object` privilegiando quelle richieste dalla domanda
        (se `question_rel_terms` è fornito).
        """
        from collections import defaultdict

        rels_by_src: Dict[int, List[dict]] = defaultdict(list)
        for r in relationships:
            rels_by_src[r["src_idx"]].append(r)

        # assicura il minimo per-oggetto
        n = len(boxes)
        centers = [((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0) for b in boxes]
        for i in range(n):
            if len(rels_by_src[i]) >= min_relations_per_object:
                continue
            # trova vicino migliore
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

        # limita al massimo per-oggetto, con priorità relazione richiesta
        def _is_question_rel(rel_label: str) -> bool:
            if not question_rel_terms:
                return False
            r = rel_label.lower()
            return any(t in r for t in question_rel_terms)

        final: List[dict] = []
        for i, rlist in rels_by_src.items():
            rlist_sorted = sorted(
                rlist,
                key=lambda r: (0 if _is_question_rel(r["relation"]) else 1, r.get("distance", 1e9)),
            )
            final.extend(rlist_sorted[: max_relations_per_object])
        return final

    def drop_inverse_duplicates(
        self,
        relationships: List[dict],
        *,
        question_subject_idxs: Optional[Set[int]] = None,
        max_relations_per_object: int = 3,
        min_relations_per_object: int = 1,
    ) -> List[dict]:
        """
        Rimuove relazioni inverse ridondanti. In caso di conflitto, privilegia quelle
        che coinvolgono il soggetto della domanda (se fornito), poi rispetta i limiti
        min/max per oggetto.
        """
        kept: List[dict] = []
        from collections import defaultdict

        count_per_src = defaultdict(int)
        subj = question_subject_idxs or set()

        for r in relationships:
            i, j, rel = r["src_idx"], r["tgt_idx"], str(r["relation"]).lower()
            inv_rel = _INVERSE.get(rel)
            existing = None
            if inv_rel:
                for rr in kept:
                    if rr["src_idx"] == j and rr["tgt_idx"] == i and str(rr["relation"]).lower() == inv_rel:
                        existing = rr
                        break

            if existing:
                r_hits = (i in subj) or (j in subj)
                e_hits = (existing["src_idx"] in subj) or (existing["tgt_idx"] in subj)
                if r_hits and not e_hits:
                    kept.remove(existing)
                    count_per_src[existing["src_idx"]] -= 1
                    kept.append(r)
                    count_per_src[i] += 1
                elif e_hits and not r_hits:
                    pass
                else:
                    cnt_i, cnt_j = count_per_src[i], count_per_src[j]
                    if cnt_i >= max_relations_per_object and cnt_j < max_relations_per_object:
                        pass
                    elif cnt_j >= max_relations_per_object and cnt_i < max_relations_per_object:
                        kept.remove(existing)
                        count_per_src[existing["src_idx"]] -= 1
                        kept.append(r)
                        count_per_src[i] += 1
                continue

            kept.append(r)
            count_per_src[i] += 1

        return kept

    def filter_by_question(
        self,
        relationships: List[dict],
        *,
        question_terms: Optional[Set[str]] = None,
        similarity_fn: Optional[Callable[[str, str], float]] = None,
        threshold: float = 0.5,
    ) -> List[dict]:
        """
        Filtra le relazioni mantenendo solo quelle coerenti con i termini della domanda.
        - Se `similarity_fn` è fornita, la usa per matching fuzzy (es. spaCy similarity).
        - Altrimenti fa solo match esatto/sottostringa.
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

    def unify_spatial_direction(self, relationships: List[dict]) -> List[dict]:
        """
        Per le relazioni spaziali, inverte la direzione così che la freccia punti
        verso l'oggetto di riferimento (coerente con il rendering grafico).
        """
        out: List[dict] = []
        for r in relationships:
            rel_name = str(r["relation"]).lower()
            if any(k in rel_name for k in _SPATIAL_KEYS):
                out.append(
                    {
                        "src_idx": r["tgt_idx"],
                        "tgt_idx": r["src_idx"],
                        "relation": r["relation"],
                        "distance": r.get("distance", 0.0),
                        **({k: r[k] for k in ("relation_raw", "clip_sim") if k in r}),
                    }
                )
            else:
                out.append(r)
        return out

    # -------------------- internals --------------------

    @staticmethod
    def _unify_pair_relations(relationships: List[dict]) -> List[dict]:
        """
        Mantiene al più una relazione per coppia direzionata (src, tgt),
        scegliendo quella con distanza minore (o la prima).
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
