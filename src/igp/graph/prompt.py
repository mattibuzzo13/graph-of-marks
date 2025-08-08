# igp/graph/prompt.py
from __future__ import annotations

from typing import List, Tuple
import re
import math
import networkx as nx


def graph_to_prompt(G: nx.DiGraph) -> str:
    """
    Converte il grafo in una stringa “prompt-like” compatibile col monolite:
      scene:"<caption>"; 0:<color> <label> (area=..); ...; (i)-<rel?>->(j)
    - Se manca la relazione sugli archi, la deduce euristicamente (overlaps/near/front_of/behind).
    """
    # trova eventuale nodo “scene”
    scene_id = next((n for n, d in G.nodes(data=True) if d.get("label") == "scene"), None)

    # Nodi
    nodes_txt: List[str] = []
    for idx, data in G.nodes(data=True):
        if data.get("label") == "scene":
            caption = data.get("caption", "")
            nodes_txt.append(f'scene:"{caption}"')
            continue
        desc_color = (data.get("color", "") + " ").strip()
        area = float(data.get("area_norm", 0.0))
        nodes_txt.append(f'{idx}:{desc_color} {data.get("label","unknown")} (area={area:.2f})'.strip())

    # Archi
    edges_txt: List[str] = []
    for u, v, e in G.edges(data=True):
        if scene_id is not None and (u == scene_id or v == scene_id):
            continue
        rel = e.get("relation", None)
        if rel is None:
            # fallback dal monolite
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
    return f"{src} ---> ({rel}) --> {tgt}"


def graph_to_triples_text(G: nx.DiGraph) -> str:
    """
    Restituisce un blocco 'Triples:' con una tripla per riga.
    Se la relazione è spaziale (‘left_of’, ‘right_of’, ‘above’, ‘below’,
    ‘on_top_of’, ‘under’, ‘in_front_of’, ‘behind’), inverte sorgente/destinazione
    per riflettere la direzione delle frecce come nel codice originale.
    """
    SPATIAL_KEYS = (
        "left_of", "right_of", "above", "below",
        "on_top_of", "under", "in_front_of", "behind"
    )

    # ignora archi collegati a 'scene'
    scene_ids = {n for n, d in G.nodes(data=True) if d.get("label") == "scene"}

    def lab(i: int) -> str:
        return G.nodes[i].get("label", "unknown")

    lines: List[str] = []
    for u, v, e in G.edges(data=True):
        if u in scene_ids or v in scene_ids:
            continue

        rel = e.get("relation")
        if not rel:
            # ricostruzione euristica come nel monolite
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
            src_label, tgt_label = lab(v), lab(u)  # inverti
        else:
            src_label, tgt_label = lab(u), lab(v)

        lines.append(_fmt_triple(src_label, rel, tgt_label))

    return "Triples:\n" + "\n".join(lines) + ("\n" if lines else "")


def save_triples_text(G: nx.DiGraph, path: str) -> None:
    """
    Salva il testo 'Triples:' su file.
    """
    txt = graph_to_triples_text(G)
    with open(path, "w", encoding="utf-8") as f:
        f.write(txt)
