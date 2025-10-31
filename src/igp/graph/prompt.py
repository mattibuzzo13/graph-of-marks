# igp/graph/prompt.py
# Utilities to serialize a scene graph into prompt-like text and compact triples.
# Heuristics infer missing edge relations (overlaps/near/front_of/behind and basic spatial).

from __future__ import annotations

from typing import List, Tuple
import html
import math
import networkx as nx


def _sanitize(s: str) -> str:
    # Escape quotes/newlines for inline prompt fields
    return html.escape(str(s).replace("\n", " ").replace("\r", " ")).strip()


def _infer_relation_from_attrs(attrs: dict) -> str:
    """
    Best-effort relation inference from edge attributes:
    - overlaps if IoU high
    - front_of / behind by depth_delta sign (smaller depth => in front)
    - else orientation by dx/dy; final fallback: near
    """
    iou = float(attrs.get("iou", 0.0) or 0.0)
    if iou >= 0.25:
        return "overlaps"

    dd = attrs.get("depth_delta", None)
    if dd is not None:
        dd = float(dd)
        if abs(dd) > 0.10:
            return "front_of" if dd < 0.0 else "behind"

    dx = attrs.get("dx_norm", None)
    dy = attrs.get("dy_norm", None)
    if dx is not None and dy is not None:
        # Note: dx/dy are stored as (target_center - source_center).
        # Therefore dx > 0 means target is to the right of source ->
        # the *source* is left_of the target. Similarly, dy > 0 means
        # target has larger y (is lower) -> the *source* is above the target.
        dx = float(dx)
        dy = float(dy)
        if abs(dx) >= abs(dy):
            # Horizontal dominant: dx>0 -> source is left_of target
            return "left_of" if dx > 0 else "right_of"
        else:
            # Vertical dominant: dy>0 -> source is above target
            return "above" if dy > 0 else "below"

    dist = float(attrs.get("dist_norm", 0.0) or 0.0)
    return "near" if dist <= 0.4 else "far"


def graph_to_prompt(G: nx.DiGraph) -> str:
    """
    Convert the graph into a prompt-like string:
      scene:"<caption>"; 0:<color> <label> (area=..); ...; (i)-<rel?>->(j)

    - If an edge relation is missing, infer it heuristically
      (overlaps / near / front_of / behind and basic spatial).
    """
    # Optional “scene” node
    scene_id = next((n for n, d in G.nodes(data=True) if d.get("label") == "scene"), None)

    # Nodes (sorted by id for stability)
    nodes_txt: List[str] = []
    if scene_id is not None:
        caption = _sanitize(G.nodes[scene_id].get("caption", ""))
        nodes_txt.append(f'scene:"{caption}"')

    for idx in sorted(n for n in G.nodes if n != scene_id):
        data = G.nodes[idx]
        desc_color = (str(data.get("color", "")).strip() + " ").strip()
        area = float(data.get("area_norm", 0.0))
        label = str(data.get("label", "unknown"))
        node_str = f'{idx}:{desc_color} {label} (area={area:.2f})'.replace("  ", " ").strip()
        nodes_txt.append(node_str)

    # Edges (skip edges touching scene)
    edges_txt: List[str] = []
    for u, v, e in G.edges(data=True):
        if scene_id is not None and (u == scene_id or v == scene_id):
            continue
        rel = e.get("relation")
        if not rel:
            rel = _infer_relation_from_attrs(e)
        edges_txt.append(f"({u})-{rel}->({v})")

    return "; ".join(nodes_txt + edges_txt)


def _fmt_triple(src: str, rel: str, tgt: str) -> str:
    # Compact textual triple representation.
    return f"{src} ---> ({rel}) --> {tgt}"


def graph_to_triples_text(G: nx.DiGraph) -> str:
    """
    Return a 'Triples:' block with one triple per line.

    Uses the natural direction: if edge is (u -> v) with relation 'left_of',
    the triple will be 'u ---> (left_of) --> v', meaning "u is left_of v".
    """
    # Ignore edges connected to 'scene'
    scene_ids = {n for n, d in G.nodes(data=True) if d.get("label") == "scene"}

    def lab(i: int) -> str:
        return str(G.nodes[i].get("label", "unknown"))

    lines: List[str] = []
    # Deterministic order
    for u, v in sorted(G.edges()):
        if u in scene_ids or v in scene_ids:
            continue
        e = G.edges[u, v]
        rel = e.get("relation")
        if not rel:
            rel = _infer_relation_from_attrs(e)

        # Use natural direction: u ---> (relation) --> v
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