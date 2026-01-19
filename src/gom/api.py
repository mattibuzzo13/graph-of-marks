"""
Graph of Marks - High-level API

Simple interface for visual scene understanding. Accepts optional custom functions
for detection, segmentation, and depth estimation. When not provided, uses defaults.

GoM Visual Prompting Modes (from AAAI 2026 paper):
    The library supports different visual prompting configurations as described
    in the Graph-of-Mark paper. These can be controlled via:

    1. label_mode: "original" (textual IDs like "oven_1") or "numeric" (1, 2, 3)
    2. display_relationships: True/False to show/hide relation arrows
    3. display_relation_labels: True/False to show/hide labels on arrows

    Paper configurations (Table 2):
    - Segmented objects + Object Text IDs: label_mode="original", display_relationships=False
    - Segmented objects + Object Num IDs: label_mode="numeric", display_relationships=False
    - GoM with Text IDs: label_mode="original", display_relationships=True, display_relation_labels=False
    - GoM with Num IDs: label_mode="numeric", display_relationships=True, display_relation_labels=False
    - GoM with Text IDs + Relation labels: label_mode="original", display_relationships=True, display_relation_labels=True
    - GoM with Num IDs + Relation labels: label_mode="numeric", display_relationships=True, display_relation_labels=True
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
from PIL import Image

from .config import PreprocessorConfig
from .pipeline.preprocessor import ImageGraphPreprocessor
from .types import Detection, Relationship
from .viz.visualizer import Visualizer, VisualizerConfig
from .graph.prompt import graph_to_triples_text, graph_to_prompt


# GoM prompting style presets matching the paper's experimental configurations
GOM_STYLE_PRESETS = {
    # Set-of-Mark style (no relations, just segmented objects with IDs)
    "som_text": {
        "label_mode": "original",
        "display_relationships": False,
        "display_relation_labels": False,
    },
    "som_numeric": {
        "label_mode": "numeric",
        "display_relationships": False,
        "display_relation_labels": False,
    },
    # GoM with relations but no relation labels (arrows only)
    "gom_text": {
        "label_mode": "original",
        "display_relationships": True,
        "display_relation_labels": False,
    },
    "gom_numeric": {
        "label_mode": "numeric",
        "display_relationships": True,
        "display_relation_labels": False,
    },
    # Full GoM with relation labels (paper's best configuration for most tasks)
    "gom_text_labeled": {
        "label_mode": "original",
        "display_relationships": True,
        "display_relation_labels": True,
    },
    "gom_numeric_labeled": {
        "label_mode": "numeric",
        "display_relationships": True,
        "display_relation_labels": True,
    },
}

# Type alias for prompting styles
GomStyle = Literal[
    "som_text", "som_numeric",
    "gom_text", "gom_numeric",
    "gom_text_labeled", "gom_numeric_labeled"
]


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

    GoM Visual Prompting Styles (from AAAI 2026 paper):
        Use the `style` parameter to easily switch between paper configurations:

        - "som_text": Set-of-Mark with textual IDs (oven_1, chair_2)
        - "som_numeric": Set-of-Mark with numeric IDs (1, 2, 3)
        - "gom_text": GoM with textual IDs + relation arrows (no labels)
        - "gom_numeric": GoM with numeric IDs + relation arrows (no labels)
        - "gom_text_labeled": GoM with textual IDs + labeled relations (best for VQA)
        - "gom_numeric_labeled": GoM with numeric IDs + labeled relations (best for REC)

        Or configure manually via label_mode, display_relationships, display_relation_labels.

    Example:
        # Default models with full GoM (textual IDs + labeled relations)
        gom = Gom()
        result = gom.process("scene.jpg")

        # Use specific GoM style from the paper
        gom = Gom(style="gom_numeric_labeled")  # Best for RefCOCO tasks
        result = gom.process("scene.jpg")

        # With custom detection
        def my_detector(image):
            boxes, labels, scores = run_yolo(image)
            return boxes, labels, scores

        gom = Gom(detect_fn=my_detector, style="gom_text_labeled")
        result = gom.process("scene.jpg")

        # Access textual scene graph for multimodal prompting
        print(result["scene_graph_text"])  # Triples format for LLM prompts
    """

    def __init__(
        self,
        detect_fn: Optional[Callable[[Image.Image], Tuple[List, List, List]]] = None,
        segment_fn: Optional[Callable[[Image.Image, List], List[np.ndarray]]] = None,
        depth_fn: Optional[Callable[[Image.Image], np.ndarray]] = None,
        output_dir: str = "output",
        device: Optional[str] = None,
        style: Optional[GomStyle] = None,
        include_textual_sg: bool = True,
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
            style: GoM visual prompting style preset. One of:
                   - "som_text": Set-of-Mark with textual IDs
                   - "som_numeric": Set-of-Mark with numeric IDs
                   - "gom_text": GoM with textual IDs (arrows, no labels)
                   - "gom_numeric": GoM with numeric IDs (arrows, no labels)
                   - "gom_text_labeled": GoM with textual IDs + relation labels
                   - "gom_numeric_labeled": GoM with numeric IDs + relation labels
                   If None, uses default (gom_text_labeled style).
            include_textual_sg: If True, includes textual scene graph representation
                               in the output for multimodal (Visual + Textual SG) prompting.
            **config_overrides: Additional PreprocessorConfig overrides.
        """
        self.detect_fn = detect_fn
        self.segment_fn = segment_fn
        self.depth_fn = depth_fn
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.include_textual_sg = include_textual_sg

        # Apply style preset if specified
        style_config = {}
        if style is not None:
            if style not in GOM_STYLE_PRESETS:
                raise ValueError(
                    f"Unknown GoM style '{style}'. "
                    f"Available styles: {list(GOM_STYLE_PRESETS.keys())}"
                )
            style_config = GOM_STYLE_PRESETS[style].copy()

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

        # Apply style preset first, then allow explicit overrides
        cfg_dict.update(style_config)
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

        # Build scene graph for textual representation
        scene_graph = None
        scene_graph_text = ""
        scene_graph_prompt = ""
        if self.include_textual_sg and boxes:
            try:
                from .graph.scene_graph import SceneGraphBuilder, SceneGraphConfig
                # Use minimal config for fast graph building (no CLIP/depth recomputation)
                sg_config = SceneGraphConfig(
                    store_clip_embeddings=False,
                    store_depth=False,
                    store_color=False,
                    add_scene_node=False,
                )
                builder = SceneGraphBuilder(config=sg_config)
                scene_graph = builder.build(
                    image=image_pil,
                    boxes_xyxy=boxes,
                    labels=labels,
                    scores=scores,
                )
                # Add relationship edges with relation attribute
                for rel in relationships:
                    src_idx = rel.get("src_idx", -1)
                    tgt_idx = rel.get("tgt_idx", -1)
                    relation = rel.get("relation", "related_to")
                    if 0 <= src_idx < len(boxes) and 0 <= tgt_idx < len(boxes):
                        if scene_graph.has_edge(src_idx, tgt_idx):
                            scene_graph[src_idx][tgt_idx]["relation"] = relation
                        else:
                            scene_graph.add_edge(src_idx, tgt_idx, relation=relation)
                # Generate textual representations
                scene_graph_text = graph_to_triples_text(scene_graph)
                scene_graph_prompt = graph_to_prompt(scene_graph)
            except Exception:
                # Gracefully handle if scene graph building fails
                pass

        result = {
            "boxes": boxes,
            "labels": labels,
            "scores": scores,
            "masks": masks,
            "depth": depth,
            "relationships": relationships,
            "scene_graph": scene_graph,
            "scene_graph_text": scene_graph_text,  # Triples format for T^SG prompting
            "scene_graph_prompt": scene_graph_prompt,  # Compact inline format
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
