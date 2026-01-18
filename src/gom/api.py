"""
Graph of Marks - High-level API

Simple interface for visual scene understanding. Accepts optional custom functions
for detection, segmentation, and depth estimation. When not provided, uses defaults.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

from .config import PreprocessorConfig
from .pipeline.preprocessor import ImageGraphPreprocessor
from .types import Detection, Relationship
from .viz.visualizer import Visualizer, VisualizerConfig


class Gom:
    """
    Graph of Marks pipeline.

    Processes images to extract objects, masks, depth, and relationships.
    Optionally accepts custom functions for each processing step.

    Custom function signatures:
        detect_fn(image: Image.Image) -> Tuple[List[List[float]], List[str], List[float]]
            Returns (boxes, labels, scores) where boxes are [x1, y1, x2, y2]

        segment_fn(image: Image.Image, boxes: List[List[float]]) -> List[np.ndarray]
            Returns list of binary masks (H, W) for each box

        depth_fn(image: Image.Image) -> np.ndarray
            Returns normalized depth map (H, W) in [0, 1], higher = closer

    Example:
        # Default models
        gom = Gom()
        result = gom.process("scene.jpg")

        # With custom detection
        def my_detector(image):
            boxes, labels, scores = run_yolo(image)
            return boxes, labels, scores

        gom = Gom(detect_fn=my_detector)
        result = gom.process("scene.jpg")
    """

    def __init__(
        self,
        detect_fn: Optional[Callable[[Image.Image], Tuple[List, List, List]]] = None,
        segment_fn: Optional[Callable[[Image.Image, List], List[np.ndarray]]] = None,
        depth_fn: Optional[Callable[[Image.Image], np.ndarray]] = None,
        output_dir: str = "output",
        device: Optional[str] = None,
        **config_overrides,
    ):
        """
        Initialize the pipeline.

        Args:
            detect_fn: Custom detection function. If None, uses default detectors.
            segment_fn: Custom segmentation function. If None, uses SAM-HQ.
            depth_fn: Custom depth function. If None, uses Depth Anything V2.
            output_dir: Directory for output files.
            device: Compute device ("cuda", "mps", "cpu"). Auto-detected if None.
            **config_overrides: Additional PreprocessorConfig overrides.
        """
        self.detect_fn = detect_fn
        self.segment_fn = segment_fn
        self.depth_fn = depth_fn
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Build config with defaults matching run_pipeline_with_intermediates.py
        cfg_dict = {
            "output_folder": str(self.output_dir),
            "output_format": "png",
            # Detection
            "detectors_to_use": ("yolov8",) if detect_fn is None else (),
            "threshold_yolo": 0.5,
            "threshold_owl": 0.5,
            "threshold_detectron": 0.5,
            "max_detections_total": 200,
            "detection_resize": True,
            "detection_max_side": 1200,
            "enable_detection_cache": True,
            # Segmentation
            "sam_version": "hq",
            "sam_hq_model_type": "vit_h",
            "points_per_side": 64,
            "pred_iou_thresh": 0.90,
            "stability_score_thresh": 0.95,
            # Fusion
            "wbf_iou_threshold": 0.55,
            "label_nms_threshold": 0.60,
            "seg_iou_threshold": 0.60,
            "enable_group_merge": True,
            "merge_mask_iou_threshold": 0.55,
            "merge_box_iou_threshold": 0.75,
            "enable_semantic_dedup": True,
            "semantic_dedup_iou_threshold": 0.40,
            # Visualization
            "display_labels": True,
            "display_relationships": True,
            "display_relation_labels": True,
            "show_segmentation": True,
            "fill_segmentation": True,
            "seg_fill_alpha": 0.25,
            "display_legend": False,
            "resolve_overlaps": True,
            "show_bboxes": True,
            "bbox_linewidth": 2.0,
            "color_sat_boost": 1.1,
            "color_val_boost": 1.1,
            # System
            "verbose": False,
            "skip_graph": False,
            "skip_visualization": False,
        }

        if device:
            cfg_dict["preproc_device"] = device

        cfg_dict.update(config_overrides)
        self.config = PreprocessorConfig(**cfg_dict)

        # Always initialize preprocessor for relationships and fallback defaults
        self._preprocessor = ImageGraphPreprocessor(self.config)

        # Initialize visualizer with correct defaults
        self._viz_config = VisualizerConfig(
            display_labels=True,
            display_relationships=True,
            display_relation_labels=True,
            display_legend=False,
            show_segmentation=True,
            fill_segmentation=True,
            show_bboxes=True,
            seg_fill_alpha=0.25,
            bbox_linewidth=2.0,
            color_sat_boost=1.1,
            color_val_boost=1.1,
        )
        self._visualizer = Visualizer(self._viz_config)

    def process(
        self,
        image: Union[str, Path, Image.Image],
        question: Optional[str] = None,
        save: bool = True,
    ) -> Dict[str, Any]:
        """
        Process an image through the pipeline.

        Args:
            image: Image path or PIL Image.
            question: Optional question for VQA-aware filtering.
            save: Whether to save outputs to disk.

        Returns:
            Dictionary with keys:
                - boxes: List of [x1, y1, x2, y2]
                - labels: List of class labels
                - scores: List of confidence scores
                - masks: List of binary masks (H, W)
                - depth: Depth map (H, W) in [0, 1]
                - relationships: List of relationship dicts
                - output_path: Path to visualization (if saved)
        """
        t0 = time.time()

        # Load image
        if isinstance(image, (str, Path)):
            image_path = Path(image)
            image_pil = Image.open(image_path).convert("RGB")
            image_name = image_path.stem
        else:
            image_pil = image.convert("RGB")
            image_name = f"image_{int(time.time())}"

        W, H = image_pil.size

        # Detection
        if self.detect_fn is not None:
            boxes, labels, scores = self.detect_fn(image_pil)
            boxes = [list(b) for b in boxes]
        else:
            det_result = self._preprocessor._run_detectors(image_pil)
            boxes = det_result.get("boxes", [])
            labels = det_result.get("labels", [])
            scores = det_result.get("scores", [])

        # Segmentation
        masks = []
        if boxes:
            if self.segment_fn is not None:
                masks = self.segment_fn(image_pil, boxes)
            elif self._preprocessor.segmenter:
                seg_results = self._preprocessor.segmenter.segment(image_pil, boxes)
                masks = [r.get("segmentation") if isinstance(r, dict) else r for r in seg_results]

        # Depth
        depth = None
        if self.depth_fn is not None:
            depth = self.depth_fn(image_pil)
        elif self._preprocessor.depth_est:
            depth = self._preprocessor.depth_est.infer_map(image_pil)

        # Relationships
        relationships = []
        if boxes:
            depths_at_centers = None
            if depth is not None:
                centers = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in boxes]
                depths_at_centers = []
                for cx, cy in centers:
                    x = int(np.clip(round(cx), 0, W - 1))
                    y = int(np.clip(round(cy), 0, H - 1))
                    depths_at_centers.append(float(depth[y, x]))

            relationships = self._preprocessor.relations_inferencer.infer(
                image_pil=image_pil,
                boxes=boxes,
                labels=labels,
                masks=[{"segmentation": m} for m in masks] if masks else None,
                depths=depths_at_centers,
                use_geometry=True,
                use_clip=False,
            )
            # Apply per-object limits and remove inverse duplicates
            relationships = self._preprocessor.relations_inferencer.limit_relationships_per_object(
                relationships,
                boxes,
                max_relations_per_object=3,
                min_relations_per_object=0,
            )
            relationships = self._preprocessor.relations_inferencer.drop_inverse_duplicates(relationships)

        result = {
            "boxes": boxes,
            "labels": labels,
            "scores": scores,
            "masks": masks,
            "depth": depth,
            "relationships": relationships,
            "processing_time": time.time() - t0,
        }

        # Save outputs
        if save:
            self._save_outputs(image_pil, image_name, result, question)
            result["output_path"] = str(self.output_dir / f"{image_name}_04_output.png")

        return result

    def _save_outputs(
        self,
        image: Image.Image,
        name: str,
        result: Dict[str, Any],
        question: Optional[str] = None,
    ):
        """Save all intermediate and final visualizations."""
        boxes = result["boxes"]
        labels = result["labels"]
        scores = result["scores"]
        masks = result["masks"]
        relationships = result["relationships"]
        depth = result.get("depth")

        W, H = image.size

        # 1. Detections only (no masks, no relations)
        if boxes:
            det_viz = Visualizer(VisualizerConfig(
                display_labels=True,
                display_relationships=False,
                display_relation_labels=False,
                show_segmentation=False,
                show_bboxes=True,
                display_legend=False,
                seg_fill_alpha=0.25,
                color_sat_boost=1.1,
                color_val_boost=1.1,
            ))
            det_viz.draw(
                image=image,
                boxes=boxes,
                labels=labels,
                scores=scores,
                relationships=[],
                masks=None,
                save_path=str(self.output_dir / f"{name}_01_detections.png"),
                draw_background=True,
            )

        # 2. Segmentation only (masks + boxes, no relations)
        if boxes and masks:
            seg_viz = Visualizer(VisualizerConfig(
                display_labels=True,
                display_relationships=False,
                display_relation_labels=False,
                show_segmentation=True,
                fill_segmentation=True,
                show_bboxes=True,
                display_legend=False,
                seg_fill_alpha=0.25,
                color_sat_boost=1.1,
                color_val_boost=1.1,
            ))
            mask_dicts = [{"segmentation": m} for m in masks]
            seg_viz.draw(
                image=image,
                boxes=boxes,
                labels=labels,
                scores=scores,
                relationships=[],
                masks=mask_dicts,
                save_path=str(self.output_dir / f"{name}_02_segmentation.png"),
                draw_background=True,
            )

        # 3. Depth map
        if depth is not None:
            depth_img = (np.clip(depth, 0.0, 1.0) * 255.0).astype(np.uint8)
            depth_pil = Image.fromarray(depth_img, mode="L")
            depth_pil.save(self.output_dir / f"{name}_03_depth.png")

        # 4. Final output (masks + relations + labels)
        if boxes:
            mask_dicts = [{"segmentation": m} for m in masks] if masks else None
            self._visualizer.draw(
                image=image,
                boxes=boxes,
                labels=labels,
                scores=scores,
                relationships=relationships,
                masks=mask_dicts,
                save_path=str(self.output_dir / f"{name}_04_output.png"),
                draw_background=True,
            )

        # 5. Scene graph JSON
        graph_data = {
            "image_size": {"width": W, "height": H},
            "question": question or "",
            "nodes": {},
            "edges": [],
        }

        for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
            node_id = f"obj_{i}"
            graph_data["nodes"][node_id] = {
                "id": i,
                "label": label,
                "bbox": box,
                "bbox_norm": [box[0] / W, box[1] / H, box[2] / W, box[3] / H],
                "score": score,
            }

        for rel in relationships:
            graph_data["edges"].append({
                "source": rel.get("src_idx"),
                "target": rel.get("tgt_idx"),
                "relation": rel.get("relation"),
            })

        with open(self.output_dir / f"{name}_05_graph.json", "w") as f:
            json.dump(graph_data, f, indent=2)


# Alias for backward compatibility
GraphOfMarks = Gom


def create_pipeline(
    detect_fn: Optional[Callable] = None,
    segment_fn: Optional[Callable] = None,
    depth_fn: Optional[Callable] = None,
    **kwargs,
) -> Gom:
    """
    Factory function to create a Gom pipeline.

    Example:
        gom = create_pipeline()
        result = gom.process("scene.jpg")
    """
    return Gom(
        detect_fn=detect_fn,
        segment_fn=segment_fn,
        depth_fn=depth_fn,
        **kwargs
    )
