"""
Graph of Marks

Interface for visual scene understanding. Accepts optional custom functions
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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .config import PreprocessorConfig
from .graph.prompt import graph_to_prompt, graph_to_triples_text
from .pipeline.preprocessor import ImageGraphPreprocessor
from .viz.visualizer import Visualizer, VisualizerConfig

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
    # Alphabetic variants (A, B, C instead of 1, 2, 3)
    "som_alphabetic": {
        "label_mode": "alphabetic",
        "display_relationships": False,
        "display_relation_labels": False,
    },
    "gom_alphabetic": {
        "label_mode": "alphabetic",
        "display_relationships": True,
        "display_relation_labels": False,
    },
    "gom_alphabetic_labeled": {
        "label_mode": "alphabetic",
        "display_relationships": True,
        "display_relation_labels": True,
    },
}

# Type alias for prompting styles
GomStyle = Literal[
    "som_text", "som_numeric", "som_alphabetic",
    "gom_text", "gom_numeric", "gom_alphabetic",
    "gom_text_labeled", "gom_numeric_labeled", "gom_alphabetic_labeled"
]


@dataclass
class ProcessingConfig:
    """
    Runtime configuration for GoM.process() calls.

    This is similar to SamplingParams in LLM inference - it controls how
    processing behaves WITHOUT reloading models. Change these parameters
    freely between process() calls.

    Example:
        gom = GoM()  # Load models once

        # Process with different configs
        result1 = gom.process(image, config=ProcessingConfig(threshold=0.5))
        result2 = gom.process(image, config=ProcessingConfig(threshold=0.8))
        result3 = gom.process(image, config=ProcessingConfig(
            style="gom_numeric_labeled",
            max_detections=10,
            max_relations_per_object=2
        ))

        # VQA-aware filtering
        result4 = gom.process(image, config=ProcessingConfig(
            question="What is on the table?",
            apply_question_filter=True,
            aggressive_pruning=True
        ))

    Attributes:
        # Detection parameters
        threshold: Default detection confidence threshold (applies to all detectors)
        threshold_owl: OWL-ViT detector threshold (overrides threshold if set)
        threshold_yolo: YOLOv8 detector threshold (overrides threshold if set)
        threshold_detectron: Detectron2 detector threshold (overrides threshold if set)
        threshold_grounding_dino: GroundingDINO detector threshold (overrides threshold if set)
        max_detections: Maximum number of detections to keep (top-N by score)

        # Relationship parameters
        max_relations_per_object: Maximum relationships per object
        min_relations_per_object: Minimum relationships per object

        # Visualization style (GoM paper configurations)
        style: Preset style ("som_text", "som_numeric", "gom_text", "gom_numeric",
               "gom_text_labeled", "gom_numeric_labeled")
        label_mode: "original" (oven_1), "numeric" (1, 2, 3), or "alphabetic" (A, B, C)
        display_labels: Show object labels
        display_relationships: Show relationship arrows
        display_relation_labels: Show labels on arrows
        display_legend: Show color legend

        # Visualization appearance
        show_segmentation: Show segmentation masks
        fill_segmentation: Fill masks with color
        seg_fill_alpha: Mask fill opacity (0-1)
        show_bboxes: Show bounding boxes
        bbox_linewidth: Bounding box line width

        # Output options
        output_dir: Directory for saving outputs (overrides GoM default)
        save_intermediates: Save intermediate outputs (detections, depth, etc.)
        include_textual_sg: Include textual scene graph in output

        # VQA question-guided filtering
        question: Question for VQA-aware filtering
        apply_question_filter: Enable question-based filtering
        aggressive_pruning: Remove objects with low relevance to question
        filter_relations_by_question: Filter relationships by question relevance

        # CLIP semantic filtering thresholds
        threshold_object_similarity: Minimum CLIP similarity for object filtering (0-1)
        threshold_relation_similarity: Minimum CLIP similarity for relation filtering (0-1)
        clip_pruning_threshold: Minimum CLIP similarity to question for inclusion
        semantic_boost_weight: Weight for semantic relevance vs raw confidence (0-1)

        # Context expansion (keep nearby objects for spatial context)
        context_expansion_enabled: Include contextually related objects
        context_expansion_radius: Area multiplier for context expansion
        context_min_iou: Minimum IoU to consider objects contextual
    """
    # Detection
    threshold: float = 0.5
    threshold_owl: Optional[float] = None  # OWL-ViT threshold (uses threshold if None)
    threshold_yolo: Optional[float] = None  # YOLOv8 threshold (uses threshold if None)
    threshold_detectron: Optional[float] = None  # Detectron2 threshold (uses threshold if None)
    threshold_grounding_dino: Optional[float] = None  # GroundingDINO threshold (uses threshold if None)
    max_detections: int = 200

    # Relationships
    max_relations_per_object: int = 3
    min_relations_per_object: int = 0

    # Style preset (overrides individual style settings if set)
    style: Optional[GomStyle] = None

    # Visualization style
    label_mode: str = "original"
    display_labels: bool = True
    display_relationships: bool = True
    display_relation_labels: bool = True
    display_legend: bool = False

    # Visualization appearance
    show_segmentation: bool = True
    fill_segmentation: bool = True
    seg_fill_alpha: float = 0.25
    show_bboxes: bool = True
    bbox_linewidth: float = 2.0
    color_sat_boost: float = 1.1
    color_val_boost: float = 1.1

    # Output options
    output_dir: Optional[str] = None
    save_intermediates: bool = True
    include_textual_sg: bool = True

    # VQA question-guided filtering
    question: Optional[str] = None
    apply_question_filter: bool = False
    aggressive_pruning: bool = False
    filter_relations_by_question: bool = True

    # CLIP semantic filtering thresholds
    threshold_object_similarity: float = 0.50
    threshold_relation_similarity: float = 0.45
    clip_pruning_threshold: float = 0.25
    semantic_boost_weight: float = 0.30

    # Context expansion
    context_expansion_enabled: bool = True
    context_expansion_radius: float = 1.5
    context_min_iou: float = 0.10

    def __post_init__(self):
        """Apply style preset if specified."""
        if self.style is not None:
            if self.style not in GOM_STYLE_PRESETS:
                raise ValueError(
                    f"Unknown GoM style '{self.style}'. "
                    f"Available styles: {list(GOM_STYLE_PRESETS.keys())}"
                )
            preset = GOM_STYLE_PRESETS[self.style]
            self.label_mode = preset.get("label_mode", self.label_mode)
            self.display_relationships = preset.get("display_relationships", self.display_relationships)
            self.display_relation_labels = preset.get("display_relation_labels", self.display_relation_labels)

    @classmethod
    def from_style(cls, style: GomStyle, **overrides) -> "ProcessingConfig":
        """Create a ProcessingConfig from a style preset with optional overrides."""
        return cls(style=style, **overrides)


class GoM:
    """
    Graph of Marks pipeline.

    Processes images to extract objects, masks, depth, and relationships.

    Key Design:
        - Models (detector, segmenter, depth estimator) are loaded ONCE at __init__
        - Processing parameters are passed per-call via ProcessingConfig
        - This allows efficient batch processing with different configurations

    Custom function signatures:
        detect_fn(image: Image.Image) -> Tuple[List[List[float]], List[str], List[float]]
            Returns (boxes, labels, scores) where boxes are [x1, y1, x2, y2]

        segment_fn(image: Image.Image, boxes: List[List[float]]) -> List[np.ndarray]
            Returns list of binary masks (H, W) for each box

        depth_fn(image: Image.Image) -> np.ndarray
            Returns normalized depth map (H, W) in [0, 1], higher = closer

    Example:
        # Initialize once (loads models)
        gom = GoM()

        # Process with default config
        result = gom.process("scene.jpg")

        # Process same image with different configurations (no model reload!)
        result_high_thresh = gom.process("scene.jpg", config=ProcessingConfig(threshold=0.8))
        result_numeric = gom.process("scene.jpg", config=ProcessingConfig(style="gom_numeric_labeled"))
        result_minimal = gom.process("scene.jpg", config=ProcessingConfig(
            max_detections=5,
            display_relationships=False
        ))

        # With custom detection
        def my_detector(image):
            boxes, labels, scores = run_yolo(image)
            return boxes, labels, scores

        gom = GoM(detect_fn=my_detector)
        result = gom.process("scene.jpg")
    """

    def __init__(
        self,
        detect_fn: Optional[Callable[[Image.Image], Tuple[List, List, List]]] = None,
        segment_fn: Optional[Callable[[Image.Image, List], List[np.ndarray]]] = None,
        depth_fn: Optional[Callable[[Image.Image], np.ndarray]] = None,
        output_dir: str = "output",
        device: Optional[str] = None,
    ):
        """
        Initialize the pipeline and load models.

        Models are loaded once here and reused for all process() calls.
        Use ProcessingConfig in process() to change runtime parameters.

        Args:
            detect_fn: Custom detection function. If None, uses YOLOv8.
            segment_fn: Custom segmentation function. If None, uses SAM-HQ.
            depth_fn: Custom depth function. If None, uses Depth Anything V2.
            output_dir: Default directory for output files.
            device: Compute device for all preprocessing models ("cuda", "cuda:0",
                "mps", "cpu"). If None, auto-detects CUDA > MPS > CPU.
                All models (detector, segmenter, depth estimator, CLIP) will
                be loaded on this device.
        """
        self.detect_fn = detect_fn
        self.segment_fn = segment_fn
        self.depth_fn = depth_fn
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device

        # Build minimal config for model initialization
        # These are MODEL settings, not processing settings
        cfg_dict = {
            "output_folder": str(self.output_dir),
            "output_format": "png",
            # Detection model config
            "detectors_to_use": ("yolov8",) if detect_fn is None else (),
            "threshold_yolo": 0.5,  # Default, can be overridden per-call
            "detection_resize": True,
            "detection_max_side": 1200,
            "enable_detection_cache": True,
            # Segmentation model config
            "sam_version": "hq",
            "sam_hq_model_type": "vit_h",
            "points_per_side": 64,
            "pred_iou_thresh": 0.90,
            "stability_score_thresh": 0.95,
            # Fusion config
            "wbf_iou_threshold": 0.55,
            "label_nms_threshold": 0.60,
            "seg_iou_threshold": 0.60,
            "enable_group_merge": True,
            "merge_mask_iou_threshold": 0.55,
            "merge_box_iou_threshold": 0.75,
            "enable_semantic_dedup": True,
            "semantic_dedup_iou_threshold": 0.40,
            # System
            "verbose": False,
        }

        if device:
            cfg_dict["preproc_device"] = device

        self._model_config = PreprocessorConfig(**cfg_dict)

        # Initialize preprocessor (loads models)
        self._preprocessor = ImageGraphPreprocessor(self._model_config)

    def process(
        self,
        image: Union[str, Path, Image.Image],
        config: Optional[ProcessingConfig] = None,
        save: bool = True,
    ) -> Dict[str, Any]:
        """
        Process an image through the pipeline.

        Args:
            image: Image path or PIL Image.
            config: Processing configuration. If None, uses defaults.
            save: Whether to save outputs to disk.

        Returns:
            Dictionary with keys:
                - boxes: List of [x1, y1, x2, y2]
                - labels: List of class labels
                - scores: List of confidence scores
                - masks: List of binary masks (H, W)
                - depth: Depth map (H, W) in [0, 1]
                - relationships: List of relationship dicts
                - scene_graph: NetworkX graph object
                - scene_graph_text: Textual scene graph (triples format)
                - scene_graph_prompt: Compact prompt format
                - output_image: PIL Image with the rendered visualization
                - processing_time: Time in seconds
                - output_path: Path to visualization (if save=True)
        """
        t0 = time.time()

        # Use default config if not provided
        if config is None:
            config = ProcessingConfig()

        # Determine output directory
        output_dir = Path(config.output_dir) if config.output_dir else self.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

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
            # Update thresholds for this run (per-detector or fallback to global)
            self._preprocessor.cfg.threshold_owl = config.threshold_owl if config.threshold_owl is not None else config.threshold
            self._preprocessor.cfg.threshold_yolo = config.threshold_yolo if config.threshold_yolo is not None else config.threshold
            self._preprocessor.cfg.threshold_detectron = config.threshold_detectron if config.threshold_detectron is not None else config.threshold
            self._preprocessor.cfg.threshold_grounding_dino = config.threshold_grounding_dino if config.threshold_grounding_dino is not None else config.threshold
            det_result = self._preprocessor._run_detectors(image_pil)
            boxes = det_result.get("boxes", [])
            labels = det_result.get("labels", [])
            scores = det_result.get("scores", [])

        # Apply max_detections limit (keep top-N by score)
        if config.max_detections > 0 and len(boxes) > config.max_detections:
            indexed = sorted(enumerate(scores), key=lambda x: -x[1])
            kept_indices = sorted([i for i, _ in indexed[:config.max_detections]])
            boxes = [boxes[i] for i in kept_indices]
            labels = [labels[i] for i in kept_indices]
            scores = [scores[i] for i in kept_indices]

        # Question-based filtering (VQA)
        obj_terms = set()
        rel_terms = set()
        if config.question and config.apply_question_filter and boxes:
            # Update preprocessor config with question filtering settings
            self._preprocessor.cfg.question = config.question
            self._preprocessor.cfg.apply_question_filter = config.apply_question_filter
            self._preprocessor.cfg.aggressive_pruning = config.aggressive_pruning
            self._preprocessor.cfg.filter_relations_by_question = config.filter_relations_by_question
            self._preprocessor.cfg.threshold_object_similarity = config.threshold_object_similarity
            self._preprocessor.cfg.context_min_iou = config.context_min_iou

            # Extract question terms
            obj_terms, rel_terms = self._preprocessor._parse_question(config.question)

            # Apply question-based filtering
            if config.aggressive_pruning:
                filtered = self._preprocessor._filter_by_question_terms(
                    boxes, labels, scores, obj_terms
                )
                if filtered[0]:  # Only apply if we got results
                    boxes, labels, scores = filtered

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
                # Resize depth map to match image dimensions if needed
                dH, dW = depth.shape[:2]
                if (dH, dW) != (H, W):
                    try:
                        import cv2
                        depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_LINEAR)
                    except ImportError:
                        from PIL import Image as PILImage
                        depth_pil = PILImage.fromarray(depth)
                        depth = np.array(depth_pil.resize((W, H), PILImage.BILINEAR))
                
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

            # Filter relationships by question terms if enabled
            if config.question and config.filter_relations_by_question and rel_terms:
                filtered_rels = []
                for rel in relationships:
                    rel_type = rel.get("relation", "")
                    # Keep relation if it matches any of the question relation terms
                    # or if no specific relation terms were found (keep all)
                    rel_base = rel_type.replace("_", " ").lower()
                    if any(term.replace("_", " ").lower() in rel_base or rel_base in term.replace("_", " ").lower()
                           for term in rel_terms):
                        filtered_rels.append(rel)
                # Only apply filter if we found matches, otherwise keep all
                if filtered_rels:
                    relationships = filtered_rels

            # Apply per-object limits from config
            relationships = self._preprocessor.relations_inferencer.limit_relationships_per_object(
                relationships,
                boxes,
                max_relations_per_object=config.max_relations_per_object,
                min_relations_per_object=config.min_relations_per_object,
            )
            relationships = self._preprocessor.relations_inferencer.drop_inverse_duplicates(relationships)

        # Build scene graph for textual representation
        scene_graph = None
        scene_graph_text = ""
        scene_graph_prompt = ""
        if config.include_textual_sg and boxes:
            try:
                from .graph.scene_graph import SceneGraphBuilder, SceneGraphConfig
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
                for rel in relationships:
                    src_idx = rel.get("src_idx", -1)
                    tgt_idx = rel.get("tgt_idx", -1)
                    relation = rel.get("relation", "related_to")
                    if 0 <= src_idx < len(boxes) and 0 <= tgt_idx < len(boxes):
                        if scene_graph.has_edge(src_idx, tgt_idx):
                            scene_graph[src_idx][tgt_idx]["relation"] = relation
                        else:
                            scene_graph.add_edge(src_idx, tgt_idx, relation=relation)
                scene_graph_text = graph_to_triples_text(scene_graph)
                scene_graph_prompt = graph_to_prompt(scene_graph)
            except Exception:
                pass

        result = {
            "boxes": boxes,
            "labels": labels,
            "scores": scores,
            "masks": masks,
            "depth": depth,
            "relationships": relationships,
            "scene_graph": scene_graph,
            "scene_graph_text": scene_graph_text,
            "scene_graph_prompt": scene_graph_prompt,
            "processing_time": time.time() - t0,
        }

        # Always render the visualization (returned in output_image)
        output_image = self._render_visualization(image_pil, result, config)
        result["output_image"] = output_image

        # Save outputs to disk if requested
        if save:
            self._save_outputs(image_pil, image_name, result, config, output_dir)
            result["output_path"] = str(output_dir / f"{image_name}_04_output.png")

        return result

    def _render_visualization(
        self,
        image: Image.Image,
        result: Dict[str, Any],
        config: ProcessingConfig,
    ) -> Image.Image:
        """
        Render the final visualization and return as PIL Image.
        
        This method creates the annotated output image with bounding boxes,
        segmentation masks, labels, and relationship arrows based on the
        configuration settings.
        
        Args:
            image: Original input image
            result: Processing result dict with boxes, labels, masks, relationships
            config: Processing configuration
            
        Returns:
            PIL Image with the rendered visualization
        """
        import io
        
        boxes = result["boxes"]
        labels = result["labels"]
        scores = result["scores"]
        masks = result["masks"]
        relationships = result["relationships"]
        
        if not boxes:
            # Return original image if no detections
            return image.copy()
        
        # Create visualizer with config settings
        viz_config = VisualizerConfig(
            label_mode=config.label_mode,
            display_labels=config.display_labels,
            display_relationships=config.display_relationships,
            display_relation_labels=config.display_relation_labels,
            display_legend=config.display_legend,
            show_segmentation=config.show_segmentation,
            fill_segmentation=config.fill_segmentation,
            show_bboxes=config.show_bboxes,
            seg_fill_alpha=config.seg_fill_alpha,
            bbox_linewidth=config.bbox_linewidth,
            color_sat_boost=config.color_sat_boost,
            color_val_boost=config.color_val_boost,
        )
        visualizer = Visualizer(viz_config)
        
        mask_dicts = [{"segmentation": m} for m in masks] if masks else None
        fig, ax = visualizer.draw(
            image=image,
            boxes=boxes,
            labels=labels,
            scores=scores,
            relationships=relationships if config.display_relationships else [],
            masks=mask_dicts,
            save_path=None,  # Don't save, just render
            draw_background=True,
        )
        
        # Convert matplotlib figure to PIL Image
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=150)
        buf.seek(0)
        output_image = Image.open(buf).convert("RGB")
        plt.close(fig)
        
        return output_image

    def _save_outputs(
        self,
        image: Image.Image,
        name: str,
        result: Dict[str, Any],
        config: ProcessingConfig,
        output_dir: Path,
    ):
        """Save all intermediate and final visualizations."""
        boxes = result["boxes"]
        labels = result["labels"]
        scores = result["scores"]
        masks = result["masks"]
        relationships = result["relationships"]
        depth = result.get("depth")

        W, H = image.size

        # Create visualizer with config settings
        viz_config = VisualizerConfig(
            label_mode=config.label_mode,
            display_labels=config.display_labels,
            display_relationships=config.display_relationships,
            display_relation_labels=config.display_relation_labels,
            display_legend=config.display_legend,
            show_segmentation=config.show_segmentation,
            fill_segmentation=config.fill_segmentation,
            show_bboxes=config.show_bboxes,
            seg_fill_alpha=config.seg_fill_alpha,
            bbox_linewidth=config.bbox_linewidth,
            color_sat_boost=config.color_sat_boost,
            color_val_boost=config.color_val_boost,
        )
        visualizer = Visualizer(viz_config)

        # Save intermediates if requested
        if config.save_intermediates:
            # 1. Detections only
            if boxes:
                det_viz = Visualizer(VisualizerConfig(
                    label_mode=config.label_mode,
                    display_labels=config.display_labels,
                    display_relationships=False,
                    show_segmentation=False,
                    show_bboxes=True,
                    display_legend=False,
                    seg_fill_alpha=config.seg_fill_alpha,
                    color_sat_boost=config.color_sat_boost,
                    color_val_boost=config.color_val_boost,
                ))
                det_viz.draw(
                    image=image,
                    boxes=boxes,
                    labels=labels,
                    scores=scores,
                    relationships=[],
                    masks=None,
                    save_path=str(output_dir / f"{name}_01_detections.png"),
                    draw_background=True,
                )

            # 2. Segmentation only
            if boxes and masks:
                seg_viz = Visualizer(VisualizerConfig(
                    label_mode=config.label_mode,
                    display_labels=config.display_labels,
                    display_relationships=False,
                    show_segmentation=True,
                    fill_segmentation=config.fill_segmentation,
                    show_bboxes=True,
                    display_legend=False,
                    seg_fill_alpha=config.seg_fill_alpha,
                    color_sat_boost=config.color_sat_boost,
                    color_val_boost=config.color_val_boost,
                ))
                mask_dicts = [{"segmentation": m} for m in masks]
                seg_viz.draw(
                    image=image,
                    boxes=boxes,
                    labels=labels,
                    scores=scores,
                    relationships=[],
                    masks=mask_dicts,
                    save_path=str(output_dir / f"{name}_02_segmentation.png"),
                    draw_background=True,
                )

            # 3. Depth map
            if depth is not None:
                depth_img = (np.clip(depth, 0.0, 1.0) * 255.0).astype(np.uint8)
                depth_pil = Image.fromarray(depth_img, mode="L")
                depth_pil.save(output_dir / f"{name}_03_depth.png")

        # 4. Final output (always saved if save=True)
        if boxes:
            mask_dicts = [{"segmentation": m} for m in masks] if masks else None
            visualizer.draw(
                image=image,
                boxes=boxes,
                labels=labels,
                scores=scores,
                relationships=relationships if config.display_relationships else [],
                masks=mask_dicts,
                save_path=str(output_dir / f"{name}_04_output.png"),
                draw_background=True,
            )

        # 5. Scene graph JSON
        graph_data = {
            "image_size": {"width": W, "height": H},
            "question": config.question or "",
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

        with open(output_dir / f"{name}_05_graph.json", "w") as f:
            json.dump(graph_data, f, indent=2)


# Aliases for backward compatibility
GraphOfMarks = GoM
Gom = GoM


def create_pipeline(
    detect_fn: Optional[Callable] = None,
    segment_fn: Optional[Callable] = None,
    depth_fn: Optional[Callable] = None,
    **kwargs,
) -> GoM:
    """
    Factory function to create a GoM pipeline.

    Example:
        gom = create_pipeline()
        result = gom.process("scene.jpg")
    """
    return GoM(
        detect_fn=detect_fn,
        segment_fn=segment_fn,
        depth_fn=depth_fn,
        **kwargs
    )


def run(
    image: Union[str, Path, Image.Image],
    config: Optional[ProcessingConfig] = None,
    device: Optional[str] = None,
    output_dir: str = "output",
    save: bool = True,
) -> Dict[str, Any]:
    """
    Simple one-shot function to process an image through the GoM pipeline.

    This is a convenience function that creates a GoM instance, processes
    the image, and returns the result. For processing multiple images,
    prefer creating a GoM instance once and calling process() multiple times.

    Args:
        image: Image path or PIL Image to process.
        config: Processing configuration. If None, uses defaults.
        device: Compute device ("cuda", "mps", "cpu"). Auto-detected if None.
        output_dir: Directory for output files.
        save: Whether to save outputs to disk.

    Returns:
        Dictionary with detection, segmentation, depth, and relationship results.

    Example:
        from gom import run, ProcessingConfig

        # Simple usage
        result = run("scene.jpg")

        # With configuration
        result = run("scene.jpg", config=ProcessingConfig(
            threshold=0.7,
            style="gom_text_labeled",
            question="What is on the table?",
            apply_question_filter=True
        ))

        # Explicit device
        result = run("scene.jpg", device="cuda")
    """
    gom = GoM(output_dir=output_dir, device=device)
    return gom.process(image, config=config, save=save)
