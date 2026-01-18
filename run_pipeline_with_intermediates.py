#!/usr/bin/env python
"""
Run the Graph-of-Marks preprocessing pipeline on a single image
WITHOUT using the high-level API (gom.api.GraphOfMarks).

This script:
    1) Runs object detection and exports an image with raw detections
    2) Runs the full preprocessing pipeline (with question filtering)
    3) Exports an image with detections AFTER question filtering
    4) Computes segmentation masks (SAM-HQ) for filtered objects
    5) Computes and exports a depth map (grayscale)
    6) Uses the built-in visualizer to export the full scene graph image
       (segmentation + relations + labels)

Usage (example from Colab):

    !python /content/graph-of-marks/src/run_pipeline_with_intermediates_no_api.py \
        --image "/content/living.png" \
        --question "Where is the light?" \
        --output "/content/gom_outputs"
"""

import argparse
import json
from pathlib import Path
from typing import Any, List, Tuple, Sequence

import numpy as np
from PIL import Image

# Core pipeline & config (no gom.api)
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
    cfg.display_legend = False

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
    cfg.display_legend = False
    cfg.seg_fill_alpha = 0.25

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
    Run the full Graph-of-Marks preprocessing pipeline (igp) and export:

        1) Raw detections (no question filtering)
        2) Detections after question-driven filtering
        3) Segmentation (SAM-HQ) of filtered objects
        4) Full-image depth map
        5) Full scene visualization with relations (standard _output image)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    image = load_image(image_path)
    W, H = image.size
    image_name = image_path.stem

    # ------------------------------------------------------------------
    # 0) Build PreprocessorConfig using your settings
    # ------------------------------------------------------------------
    cfg = PreprocessorConfig()

    # =============================================================================
    # 📥 INPUT / OUTPUT
    # =============================================================================
    cfg.input_path = str(image_path)
    cfg.output_folder = str(output_dir)
    cfg.output_format = "png"

    cfg.save_image_only = False
    cfg.skip_graph = False
    cfg.skip_visualization = False
    cfg.skip_prompt = False

    # =============================================================================
    # ❓ QUESTION-GUIDED FILTERING
    # =============================================================================
    cfg.question = question
    cfg.apply_question_filter = True
    cfg.aggressive_pruning = True
    cfg.filter_relations_by_question = True

    cfg.threshold_object_similarity = 0.40
    cfg.threshold_relation_similarity = 0.30
    cfg.clip_pruning_threshold = 0.20
    cfg.semantic_boost_weight = 0.30

    cfg.context_expansion_enabled = True
    cfg.context_expansion_radius = 1.5
    cfg.context_min_iou = 0.10

    # =============================================================================
    # 🔍 DETECTORS
    # =============================================================================
    cfg.detectors_to_use = ["owlvit", "yolov8", "detectron2"]

    cfg.threshold_owl = 0.9
    cfg.threshold_yolo = 0.9
    cfg.threshold_detectron = 0.9

    cfg.max_detections_total = 200
    cfg.min_detection_conf = 0.30

    cfg.enable_detection_cache = True

    cfg.detection_resize = True
    cfg.detection_max_side = 1200   # HQ
    cfg.detection_hash_method = "full"

    # =============================================================================
    # 🔄 FUSION / MERGE
    # =============================================================================
    cfg.wbf_iou_threshold = 0.55
    cfg.label_nms_threshold = 0.60
    cfg.seg_iou_threshold = 0.60

    cfg.enable_group_merge = True
    cfg.merge_mask_iou_threshold = 0.55
    cfg.merge_box_iou_threshold = 0.75

    cfg.enable_semantic_dedup = True
    cfg.semantic_dedup_iou_threshold = 0.40

    # =============================================================================
    # 🎭 SEGMENTATION — HQ SAM-HQ
    # =============================================================================
    cfg.sam_version = "hq"
    cfg.sam_hq_model_type = "vit_h"
    cfg.points_per_side = 64

    cfg.pred_iou_thresh = 0.90
    cfg.stability_score_thresh = 0.95

    # =============================================================================
    # 🎨 VISUALIZATION
    # =============================================================================
    cfg.display_labels = True
    cfg.display_relationships = True
    cfg.display_relation_labels = True

    cfg.show_segmentation = True
    cfg.fill_segmentation = True
    cfg.seg_fill_alpha = 0.25

    cfg.display_legend = False
    cfg.resolve_overlaps = True

    cfg.show_bboxes = True
    cfg.bbox_linewidth = 2.0

    cfg.color_sat_boost = 1.1
    cfg.color_val_boost = 1.1

    # =============================================================================
    # ⚙️ SYSTEM
    # =============================================================================
    # cfg.preproc_device = "cuda"   # change to "cpu" if needed
    cfg.preproc_device = "mps"   # change to "cpu" if needed
    
    cfg.verbose = True

    # Instantiate preprocessor
    pre = ImageGraphPreprocessor(cfg)

    # ------------------------------------------------------------------
    # 1) Raw detections (no question filtering)
    #    → run detectors directly on the PIL image
    # ------------------------------------------------------------------
    det_raw = pre._run_detectors(image)  # internal helper, but convenient here
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
    # 2) Full preprocessing via pre.run()
    #    This will:
    #      - run detection, segmentation, depth, relations, CLIP, etc.
    #      - build scene graph JSON
    #      - save {image_name}_graph.json + {image_name}_output.png
    # ------------------------------------------------------------------
    print("\n[PIPELINE] Running full preprocessing (with question filtering)...")
    pre.run()

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
    # 3) Segmentation of filtered objects (SAM-HQ) + visualization
    # ------------------------------------------------------------------
    if not hasattr(pre, "segmenter") or pre.segmenter is None:
        raise RuntimeError("Segmenter is not initialized. Check SAM-HQ configuration.")

    masks = pre.segmenter.segment(image, boxes_filt) if boxes_filt else []

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
    if not hasattr(pre, "depth_est") or pre.depth_est is None:
        print("WARNING: Depth estimator is not initialized, skipping depth map.")
        depth_path = output_dir / f"{image_name}_04_depth.png"
    else:
        depth_path = output_dir / f"{image_name}_04_depth.png"
        save_depth_map(
            depth_estimator=pre.depth_est,
            image=image,
            save_path=depth_path,
        )

    # ------------------------------------------------------------------
    # 5) Final visualization with relations
    #     Already generated by pre.run:
    #       {image_name}_output.png
    # ------------------------------------------------------------------
    final_output_path = output_dir / f"{image_name}_output.png"
    if not final_output_path.exists():
        print(
            f"WARNING: Final visualization not found at {final_output_path}.\n"
            f"Check cfg.skip_visualization and cfg.output_format."
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
