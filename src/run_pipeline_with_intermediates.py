#!/usr/bin/env python
"""
Run the Graph-of-Marks preprocessing pipeline on a single image
WITHOUT using the high-level API (gom.api.GraphOfMarks).

This script:
    1) Runs object detection and exports an image with raw detections
    2) Runs the full preprocessing pipeline (with question filtering)
    3) Exports an image with detections AFTER question filtering
    4) Computes segmentation masks (SAM2) for filtered objects
    5) Computes and exports a depth map (grayscale)
    6) Uses the built-in visualizer to export the full scene graph image
       (segmentation + relations + labels)

Usage (example from Colab):

    !python /content/graph-of-marks/src/run_pipeline_with_intermediates_no_api.py \
        --image "/content/living.png" \
        --question "Where is the vase?" \
        --output "/content/gom_outputs"
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, List, Tuple, Sequence, Optional

import numpy as np
from PIL import Image

# Core pipeline & config (no gom.api)
from gom.config import default_config
from gom.pipeline.preprocessor import ImageGraphPreprocessor, PreprocessorConfig
from gom.viz.visualizer import Visualizer, VisualizerConfig


# -------------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------------
def load_image(image_path: Path) -> Image.Image:
    """Load an RGB image from disk."""
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    return Image.open(image_path).convert("RGB")


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


def draw_detections_only(
    image: Image.Image,
    boxes: Sequence[Sequence[float]],
    labels: Sequence[str],
    scores: Sequence[float],
    save_path: Path,
) -> None:
    """
    Render an image with only detection bounding boxes + labels (no masks, no relations).
    """
    cfg = VisualizerConfig()
    cfg.display_labels = True
    cfg.display_relationships = False
    cfg.display_relation_labels = False
    cfg.show_segmentation = False
    cfg.show_bboxes = True
    cfg.display_legend = True

    viz = Visualizer(cfg)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    viz.draw(
        image=image,
        boxes=boxes,
        labels=labels,
        scores=scores,
        relationships=[],   # no relations
        masks=None,
        save_path=str(save_path),
        draw_background=True,
        bg_color=(1, 1, 1, 1),
    )


def draw_segmentation_only(
    image: Image.Image,
    boxes: Sequence[Sequence[float]],
    labels: Sequence[str],
    scores: Sequence[float],
    masks: Sequence[np.ndarray],
    save_path: Path,
) -> None:
    """
    Render segmentation masks (and optional boxes/labels) for a set of objects.
    """
    cfg = VisualizerConfig()
    cfg.display_labels = True
    cfg.display_relationships = False
    cfg.display_relation_labels = False
    cfg.show_segmentation = True
    cfg.fill_segmentation = True
    cfg.show_bboxes = True
    cfg.display_legend = True

    viz = Visualizer(cfg)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    viz.draw(
        image=image,
        boxes=boxes,
        labels=labels,
        scores=scores,
        relationships=[],
        masks=masks,
        save_path=str(save_path),
        draw_background=True,
        bg_color=(1, 1, 1, 1),
    )


def save_depth_map(
    depth_estimator: Any,
    image: Image.Image,
    save_path: Path,
) -> None:
    """
    Compute and save a full-image depth map in grayscale [0..255].

    Uses DepthEstimator.infer_map(image) which returns a normalized depth map
    in [0,1] where higher = closer to the camera.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if not hasattr(depth_estimator, "infer_map"):
        raise RuntimeError("Depth estimator does not expose 'infer_map(image)'.")

    depth = depth_estimator.infer_map(image)  # H x W float array in [0,1] (or None)
    if depth is None:
        # Fallback: uniform mid-depth
        depth = np.full((image.height, image.width), 0.5, dtype=np.float32)
    else:
        depth = np.array(depth, dtype=np.float32)

    # Map [0,1] → [0,255] uint8
    depth_img = (np.clip(depth, 0.0, 1.0) * 255.0).astype(np.uint8)
    depth_pil = Image.fromarray(depth_img, mode="L")
    depth_pil.save(save_path)


# -------------------------------------------------------------------------
# Main pipeline
# -------------------------------------------------------------------------
def run_pipeline_with_intermediates(
    image_path: Path,
    question: str,
    output_dir: Path,
) -> None:
    """
    Run the full Graph-of-Marks preprocessing pipeline without gom.api,
    and export intermediate visualizations:

        1) Raw detections (no question filtering)
        2) Detections after question-driven filtering
        3) Segmentation (SAM2) of filtered objects
        4) Full-image depth map
        5) Full scene visualization with relations (standard _output image)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    image = load_image(image_path)
    W, H = image.size
    image_name = image_path.stem

    # ------------------------------------------------------------------
    # 0) Build PreprocessorConfig (no GraphOfMarks)
    # ------------------------------------------------------------------
    cfg: PreprocessorConfig = default_config(
        # I/O
        input_path=str(image_path),
        output_folder=str(output_dir),

        # Question-guided filtering
        question=question,
        apply_question_filter=True,
        filter_relations_by_question=True,

        # Segmentation: force SAM2 to avoid SAM1 issues
        sam_version="2",

        # Visualization settings for final export
        show_segmentation=True,
        display_relationships=True,
        display_relation_labels=False,
        display_labels=True,
        show_bboxes=True,
        display_legend=True,

        # Do NOT skip heavy steps: we want full pipeline
        skip_graph=False,
        skip_prompt=False,
        skip_visualization=False,
        skip_relations_when_unused=False,
        skip_depth_when_unused=False,
        skip_segmentation_when_unused=False,

        # Verbose logs (optional)
        verbose=False,
    )

    preproc = ImageGraphPreprocessor(cfg)

    # ------------------------------------------------------------------
    # 1) Raw detections (no question filtering)
    #    → use internal detector directly, without question filter
    # ------------------------------------------------------------------
    det_raw = preproc._run_detectors(image)
    boxes_det: List[List[float]] = det_raw.get("boxes", [])
    labels_det: List[str] = det_raw.get("labels", [])
    scores_det: List[float] = det_raw.get("scores", [])

    det_raw_path = output_dir / f"{image_name}_01_detections_raw.png"
    if boxes_det:
        draw_detections_only(
            image=image,
            boxes=boxes_det,
            labels=labels_det,
            scores=scores_det,
            save_path=det_raw_path,
        )
    else:
        print("WARNING: No detections found for raw detection stage.")

    # ------------------------------------------------------------------
    # 2) Full preprocessing (question-aware) via process_single_image
    #    This will:
    #      - run detection, segmentation, depth, relations
    #      - build scene graph
    #      - save {image_name}_graph.json + {image_name}_output.{ext}
    # ------------------------------------------------------------------
    print("\n[PIPELINE] Running full preprocessing (with question)...")
    preproc.process_single_image(
        image_pil=image,
        image_name=image_name,
        custom_question=question,
    )

    graph_json_path = output_dir / f"{image_name}_graph.json"
    if not graph_json_path.exists():
        raise RuntimeError(
            f"Scene graph JSON not found at {graph_json_path}. "
            f"Check that skip_graph=False in PreprocessorConfig."
        )

    with open(graph_json_path, "r") as f:
        scene_graph_json = json.load(f)

    # Extract filtered detections (AFTER question filtering) from graph
    boxes_filt, labels_filt, scores_filt = boxes_from_scene_graph_json(
        scene_graph_json,
        image_size=(W, H),
    )

    det_filtered_path = output_dir / f"{image_name}_02_detections_filtered.png"
    if boxes_filt:
        draw_detections_only(
            image=image,
            boxes=boxes_filt,
            labels=labels_filt,
            scores=scores_filt,
            save_path=det_filtered_path,
        )
    else:
        print("WARNING: No nodes/detections found in scene_graph_json.")

    # ------------------------------------------------------------------
    # 3) Segmentation of filtered objects (SAM2) + visualization
    # ------------------------------------------------------------------
    if not hasattr(preproc, "segmenter") or preproc.segmenter is None:
        raise RuntimeError("Segmenter is not initialized. Check sam_version / SAM2 config.")

    masks = preproc.segmenter.segment(image, boxes_filt) if boxes_filt else []

    seg_path = output_dir / f"{image_name}_03_segmentation.png"
    if masks:
        draw_segmentation_only(
            image=image,
            boxes=boxes_filt,
            labels=labels_filt,
            scores=scores_filt,
            masks=masks,
            save_path=seg_path,
        )
    else:
        print("WARNING: No masks produced for segmentation stage (no boxes or SAM failure).")

    # ------------------------------------------------------------------
    # 4) Full-image depth map
    # ------------------------------------------------------------------
    if not hasattr(preproc, "depth_est") or preproc.depth_est is None:
        raise RuntimeError("Depth estimator is not initialized.")

    depth_path = output_dir / f"{image_name}_04_depth.png"
    save_depth_map(
        depth_estimator=preproc.depth_est,
        image=image,
        save_path=depth_path,
    )

    # ------------------------------------------------------------------
    # 5) Final visualization with relations
    #     Already generated by process_single_image:
    #       {image_name}_output.{ext}
    # ------------------------------------------------------------------
    # Infer extension from config (default "jpg")
    ext = cfg.output_format if getattr(cfg, "output_format", "jpg") in ["jpg", "png", "svg"] else "jpg"
    final_output_path = output_dir / f"{image_name}_output.{ext}"

    if not final_output_path.exists():
        print(
            f"WARNING: Final visualization not found at {final_output_path}.\n"
            f"Check PreprocessorConfig.skip_visualization and output_format."
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n=== Exported intermediate outputs (no gom.api) ===")
    print(f"1) Raw detections (no question filtering): {det_raw_path}")
    print(f"2) Question-filtered detections:           {det_filtered_path}")
    print(f"3) Segmentation (filtered objects):        {seg_path}")
    print(f"4) Depth map:                              {depth_path}")
    print(f"5) Full scene visualization:               {final_output_path}")
    print("===============================================\n")


# -------------------------------------------------------------------------
# CLI entry point
# -------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Graph-of-Marks preprocessing on a single image "
                    "and export intermediate visualizations (no gom.api)."
    )
    parser.add_argument(
        "--image",
        "-i",
        type=str,
        required=True,
        help="Path to the input image.",
    )
    parser.add_argument(
        "--question",
        "-q",
        type=str,
        required=True,
        help="Natural language question for VQA-aware filtering.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="gom_intermediate_outputs",
        help="Output directory for all generated images.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline_with_intermediates(
        image_path=Path(args.image),
        question=args.question,
        output_dir=Path(args.output),
    )
