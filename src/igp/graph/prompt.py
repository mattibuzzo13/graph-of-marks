# igp/graph/prompt.py
# Utilities to serialize a scene graph into prompt-like text and compact triples.
# Heuristics infer missing edge relations (overlaps/near/front_of/behind and basic spatial).

from __future__ import annotations

from typing import List, Tuple
import re
import math
import networkx as nx


def graph_to_prompt(G: nx.DiGraph) -> str:
    """
    Convert the graph into a prompt-like string compatible with the monolith:
      scene:"<caption>"; 0:<color> <label> (area=..); ...; (i)-<rel?>->(j)

    - If an edge relation is missing, infer it heuristically
      (overlaps / near / front_of / behind) using IoU and depth delta.
    """
    # Find an optional “scene” node
    scene_id = next((n for n, d in G.nodes(data=True) if d.get("label") == "scene"), None)

    # Nodes
    nodes_txt: List[str] = []
    for idx, data in G.nodes(data=True):
        if data.get("label") == "scene":
            caption = data.get("caption", "")
            nodes_txt.append(f'scene:"{caption}"')
            continue
        desc_color = (data.get("color", "") + " ").strip()
        area = float(data.get("area_norm", 0.0))
        nodes_txt.append(f'{idx}:{desc_color} {data.get("label","unknown")} (area={area:.2f})'.strip())

    # Edges
    edges_txt: List[str] = []
    for u, v, e in G.edges(data=True):
        if scene_id is not None and (u == scene_id or v == scene_id):
            continue
        rel = e.get("relation", None)
        if rel is None:
            # Monolith-inspired fallback
            rels = []
            if float(e.get("iou", 0.0)) > 0.25:
                rels.append("overlaps")
            dd = float(e.get("depth_delta", 0.0))
            if abs(dd) > 0.1:
                rels.append("front_of" if dd < 0.0 else "behind")
            if not rels:
                rels.append("near")
            rel = "/".join(rels)
        edges_txt.append(f"({u})-{rel}->({v})")

    return "; ".join(nodes_txt + edges_txt)


def _fmt_triple(src: str, rel: str, tgt: str) -> str:
    # Compact textual triple representation.
    return f"{src} ---> ({rel}) --> {tgt}"


def graph_to_triples_text(G: nx.DiGraph) -> str:
    """
    Return a 'Triples:' block with one triple per line.

    If the relation is spatial ('left_of', 'right_of', 'above', 'below',
    'on_top_of', 'under', 'in_front_of', 'behind'), swap source/target
    to mirror the original arrow direction semantics.
    """
    SPATIAL_KEYS = (
        "left_of", "right_of", "above", "below",
        "on_top_of", "under", "in_front_of", "behind"
    )

    # Ignore edges connected to 'scene'
    scene_ids = {n for n, d in G.nodes(data=True) if d.get("label") == "scene"}

    def lab(i: int) -> str:
        return G.nodes[i].get("label", "unknown")

    lines: List[str] = []
    for u, v, e in G.edges(data=True):
        if u in scene_ids or v in scene_ids:
            continue

        rel = e.get("relation")
        if not rel:
            # Heuristic reconstruction as in the monolith
            dx = e.get("dx_norm", None)
            dy = e.get("dy_norm", None)
            iou_val = float(e.get("iou", 0.0) or 0.0)
            if iou_val > 0.25:
                rel = "overlaps"
            elif dx is not None and dy is not None:
                if abs(dx) >= abs(dy):
                    rel = "right_of" if dx > 0 else "left_of"
                else:
                    rel = "below" if dy > 0 else "above"
            else:
                rel = "near"

        rel_l = str(rel).lower()
        if any(k in rel_l for k in SPATIAL_KEYS):
            src_label, tgt_label = lab(v), lab(u)  # invert
        else:
            src_label, tgt_label = lab(u), lab(v)

        lines.append(_fmt_triple(src_label, rel, tgt_label))

    return "Triples:\n" + "\n".join(lines) + ("\n" if lines else "")


def save_triples_text(G: nx.DiGraph, path: str) -> None:
    """
    Serialize 'Triples:' text to a file at the given path (UTF-8).
    """
    txt = graph_to_triples_text(G)
    with open(path, "w", encoding="utf-8") as f:
        f.write(txt)
