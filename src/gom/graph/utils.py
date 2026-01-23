"""
Graph Utilities

Helper functions for working with scene graphs and JSON exports.
"""
from typing import List, Tuple


def boxes_from_scene_graph_json(
    scene_graph_json: dict,
    image_size: Tuple[int, int],
) -> Tuple[List[List[float]], List[str], List[float]]:
    """
    Extract pixel-space bounding boxes, labels and scores from scene_graph_json.

    The JSON uses normalized coordinates in [0, 1]:
        bbox_norm = [x1, y1, x2, y2]

    This function converts them to absolute pixel coordinates (x1,y1,x2,y2).

    Returns:
        (boxes, labels, scores)
    """
    W, H = image_size
    boxes: List[List[float]] = []
    labels: List[str] = []
    scores: List[float] = []

    if not scene_graph_json:
        return boxes, labels, scores

    nodes = scene_graph_json.get("nodes", None)
    if nodes is None:
        return boxes, labels, scores

    # Support both dict-of-dicts and list-of-nodes formats
    if isinstance(nodes, dict):
        node_items = nodes.items()
    elif isinstance(nodes, list):
        node_items = [(n.get("id", i), n) for i, n in enumerate(nodes)]
    else:
        return boxes, labels, scores

    for _, node_data in node_items:
        label = node_data.get("label", "")
        if label == "scene":
            # Skip global scene node
            continue

        bbox_norm = node_data.get("bbox_norm", None)
        score = float(node_data.get("score", 0.0))

        if bbox_norm is not None:
            x1 = float(bbox_norm[0]) * W
            y1 = float(bbox_norm[1]) * H
            x2 = float(bbox_norm[2]) * W
            y2 = float(bbox_norm[3]) * H
        else:
            # Fallback: some exports may store pixel-space bbox
            bbox = node_data.get("bbox", None)
            if bbox is None:
                continue
            x1, y1, x2, y2 = bbox

        boxes.append([x1, y1, x2, y2])
        labels.append(label)
        scores.append(score)

    return boxes, labels, scores
