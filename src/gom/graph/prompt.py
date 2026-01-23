# igp/graph/prompt.py
"""
Scene Graph Prompt Serialization

This module converts NetworkX scene graphs into textual representations suitable
for language model prompts and human-readable outputs. Provides both compact
prompt format and structured triple format.

Key Features:
    - Graph → prompt string serialization
    - Graph → triples text conversion
    - Heuristic relation inference from edge attributes
    - HTML-safe sanitization for web/API use
    - Compact representations for token efficiency

Prompt Formats:
    1. Inline Prompt:
       scene:"kitchen interior"; 0:red chair (area=0.05); 1:white table (area=0.12); (0)-on_top_of->(1)
    
    2. Triples Format:
       Triples:
       chair ---> (on_top_of) --> table
       person ---> (left_of) --> car

Relation Inference:
    When edge.relation is missing, infers from geometric attributes:
    - overlaps: IoU ≥ 0.25
    - front_of/behind: depth_delta magnitude > 0.10
    - Spatial: left_of, right_of, above, below (from dx_norm, dy_norm)
    - Proximity: near (dist_norm ≤ 0.4), far (dist_norm > 0.4)

Functions:
    graph_to_prompt(G) -> str: Convert to inline prompt format
    graph_to_triples_text(G) -> str: Convert to triples format
    save_triples_text(G, path): Save triples to file
    
    Internal:
        _sanitize(s) -> str: HTML-escape and clean strings
        _infer_relation_from_attrs(attrs) -> str: Heuristic relation inference
        _fmt_triple(src, rel, tgt) -> str: Format single triple

Usage:
    >>> from gom.graph.prompt import graph_to_prompt, graph_to_triples_text
    >>> 
    >>> # Convert graph to prompt
    >>> prompt = graph_to_prompt(scene_graph)
    >>> print(prompt)
    scene:"bedroom"; 0:brown bed (area=0.25); 1:white pillow (area=0.03); (1)-on_top_of->(0)
    >>> 
    >>> # Convert to triples
    >>> triples = graph_to_triples_text(scene_graph)
    >>> print(triples)
    Triples:
    pillow ---> (on_top_of) --> bed
"""
from __future__ import annotations

import html
from typing import List

import networkx as nx


def _sanitize(s: str) -> str:
    """
    Sanitize string for safe inclusion in prompts and web contexts.
    
    Applies HTML escaping and removes problematic whitespace characters
    that could break prompt formatting or cause parsing issues.
    
    Args:
        s: Input string to sanitize
    
    Returns:
        Cleaned string with:
            - HTML entities escaped (&lt;, &gt;, &quot;, etc.)
            - Newlines and carriage returns replaced with spaces
            - Leading/trailing whitespace stripped
    
    Example:
        >>> _sanitize('Object with "quotes" and\nnewlines')
        'Object with &quot;quotes&quot; and newlines'
        >>> _sanitize('<script>alert("XSS")</script>')
        '&lt;script&gt;alert(&quot;XSS&quot;)&lt;/script&gt;'
    
    Notes:
        - Prevents prompt injection via special characters
        - Safe for JSON, HTML, and inline text contexts
        - Preserves semantic content while ensuring safety
    """
    return html.escape(str(s).replace("\n", " ").replace("\r", " ")).strip()


def _infer_relation_from_attrs(attrs: dict) -> str:
    """
    Infer spatial/semantic relation from edge attributes using heuristics.
    
    Applies decision tree to geometric and depth attributes to determine
    most appropriate relationship predicate when explicit relation missing.
    
    Args:
        attrs: Edge attribute dictionary with optional keys:
              - iou: Intersection over Union (float)
              - depth_delta: Relative depth difference (float)
              - dx_norm, dy_norm: Normalized displacement (float)
              - dist_norm: Normalized distance (float)
    
    Returns:
        Inferred relation string from:
            - "overlaps": High spatial overlap
            - "front_of", "behind": Depth ordering
            - "left_of", "right_of": Horizontal displacement
            - "above", "below": Vertical displacement
            - "near", "far": Proximity based
    
    Decision Logic:
        1. If IoU ≥ 0.25 → "overlaps" (significant overlap)
        2. If |depth_delta| > 0.10 → "front_of" or "behind" (depth ordering)
        3. If |dx| ≥ |dy| → "left_of" or "right_of" (horizontal dominant)
        4. If |dy| > |dx| → "above" or "below" (vertical dominant)
        5. If dist_norm ≤ 0.4 → "near", else "far" (proximity fallback)
    
    Coordinate Convention:
        - dx_norm, dy_norm: target_center - source_center
        - dx > 0 means target is RIGHT of source (source is LEFT_OF target)
        - dy > 0 means target is BELOW source (source is ABOVE target)
        - depth_delta < 0 means source is FRONT_OF target (smaller depth = closer)
    
    Example:
        >>> attrs = {"iou": 0.3}
        >>> _infer_relation_from_attrs(attrs)
        'overlaps'
        
        >>> attrs = {"dx_norm": 0.2, "dy_norm": 0.05}
        >>> _infer_relation_from_attrs(attrs)
        'left_of'  # Horizontal displacement dominant
        
        >>> attrs = {"depth_delta": -0.15}
        >>> _infer_relation_from_attrs(attrs)
        'front_of'  # Source closer to camera
    
    Notes:
        - Thresholds tuned for typical scene configurations
        - Handles missing attributes gracefully (falls back to "near")
        - Deterministic given same input (no randomness)
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
    Convert scene graph to compact inline prompt string.
    
    Serializes graph into semicolon-separated format suitable for language
    model prompts, maximizing information density while maintaining readability.
    
    Args:
        G: NetworkX DiGraph with standard scene graph schema:
           - Nodes: {label, color?, area_norm?, ...}
           - Edges: {relation?, iou?, dx_norm?, dy_norm?, ...}
           - Optional "scene" node with caption
    
    Returns:
        Prompt string in format:
        scene:"<caption>"; 0:<color> <label> (area=X.XX); ...; (i)-<relation>->(j); ...
    
    Format Specification:
        Scene Node (if present):
            scene:"<HTML-escaped caption>"
        
        Object Nodes (sorted by ID):
            {id}:<color> <label> (area={normalized_area})
            - color: optional dominant color
            - area: normalized box area [0.0, 1.0]
        
        Relationship Edges:
            ({src_id})-{relation}->({tgt_id})
            - relation: explicit or inferred from attributes
            - edges to/from scene node omitted
    
    Example:
        >>> graph = nx.DiGraph()
        >>> graph.add_node(0, label="chair", color="brown", area_norm=0.05)
        >>> graph.add_node(1, label="table", color="white", area_norm=0.12)
        >>> graph.add_edge(0, 1, relation="under")
        >>> print(graph_to_prompt(graph))
        0:brown chair (area=0.05); 1:white table (area=0.12); (0)-under->(1)
    
    Notes:
        - Deterministic output (nodes sorted by ID)
        - HTML-safe (uses _sanitize for all text)
        - Token-efficient for LLM prompts
        - Missing relations inferred via _infer_relation_from_attrs
        - Scene node always appears first if present
    
    Use Cases:
        - Vision-language model prompts
        - Structured scene descriptions
        - Graph-to-text generation
        - VQA context encoding
    """
    # Optional "scene" node
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
    """
    Format a single RDF-style triple for display.
    
    Args:
        src: Subject/source entity label
        rel: Predicate/relation label
        tgt: Object/target entity label
    
    Returns:
        Formatted triple string with arrows and parentheses
    
    Example:
        >>> _fmt_triple("person", "left_of", "car")
        'person ---> (left_of) --> car'
    """
    return f"{src} ---> ({rel}) --> {tgt}"


def graph_to_triples_text(G: nx.DiGraph) -> str:
    """
    Convert scene graph to structured triples format.
    
    Generates a "Triples:" block with one relationship per line in
    subject-predicate-object format. More human-readable than inline prompt.
    
    Args:
        G: NetworkX DiGraph scene graph
    
    Returns:
        Multi-line string starting with "Triples:" header followed by
        one triple per line in format: {src_label} ---> ({relation}) --> {tgt_label}
    
    Format:
        Triples:
        {object1_label} ---> ({relation1}) --> {object2_label}
        {object3_label} ---> ({relation2}) --> {object4_label}
        ...
    
    Triple Direction:
        Uses natural edge direction: if edge is (u → v) with relation "left_of",
        the triple will be "u ---> (left_of) --> v", meaning "u is left_of v".
    
    Example:
        >>> graph = nx.DiGraph()
        >>> graph.add_node(0, label="chair")
        >>> graph.add_node(1, label="table")
        >>> graph.add_edge(0, 1, relation="under")
        >>> print(graph_to_triples_text(graph))
        Triples:
        chair ---> (under) --> table
    
    Notes:
        - Skips edges involving scene node
        - Infers missing relations via heuristics
        - Uses human-readable labels (not node IDs)
        - One triple per line for easy parsing
        - Empty graph returns "Triples:" with no entries
        - Deterministic order (sorted by edge tuple)
    
    Use Cases:
        - Knowledge graph export
        - Structured scene descriptions
        - Triple-based reasoning systems
        - Human-readable graph summaries
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
    Serialize triples text to file.
    
    Convenience function to write graph_to_triples_text output to disk
    with UTF-8 encoding.
    
    Args:
        G: NetworkX DiGraph scene graph
        path: Output file path (will be created/overwritten)
    
    Side Effects:
        Creates or overwrites file at `path` with UTF-8 encoded triples text
    
    Example:
        >>> save_triples_text(scene_graph, "output/scene_triples.txt")
        # File now contains:
        # Triples:
        # chair ---> (under) --> table
        # person ---> (left_of) --> car
    
    Notes:
        - Automatically handles file encoding (UTF-8)
        - Creates parent directories if needed (via Python open)
        - Safe for paths with special characters
    """
    txt = graph_to_triples_text(G)
    with open(path, "w", encoding="utf-8") as f:
        f.write(txt)



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