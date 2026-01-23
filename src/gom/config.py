# igp/config.py
"""
IGP Configuration Module

Centralized configuration system for the Image Graph Preprocessing pipeline.
Re-exports configuration dataclasses from individual modules and provides
a fallback PreprocessorConfig to prevent circular import issues during startup.

Configuration Architecture:
    
    PreprocessorConfig (pipeline/preprocessor.py):
        Master configuration containing all pipeline parameters
        - I/O paths and dataset settings
        - Detector selection and thresholds
        - Fusion and NMS parameters
        - Relationship extraction config
        - Visualization settings
        - Performance tuning knobs
    
    SegmenterConfig (segmentation/base.py):
        SAM-based segmentation configuration
        - Model selection (SAM1/SAM2/HQ/Fast)
        - Post-processing options
        - Device and precision settings
    
    RelationsConfig (relations/inference.py):
        Relationship extraction configuration
        - Geometric predicates
        - CLIP-based semantic scoring
        - LLM-guided reasoning (optional)
        - Filtering and pruning thresholds
    
    VisualizerConfig (viz/visualizer.py):
        Visualization rendering configuration
        - Box/label/mask rendering
        - Color schemes
        - Font and layout
        - Output format options

Usage Patterns:

    1. Import specific configs:
        >>> from gom.config import RelationsConfig, VisualizerConfig
        >>> rel_cfg = RelationsConfig(clip_threshold=0.6)
        >>> viz_cfg = VisualizerConfig(show_masks=True)
    
    2. Import PreprocessorConfig (full pipeline):
        >>> from gom.config import PreprocessorConfig
        >>> config = PreprocessorConfig(
        ...     detectors_to_use=("yolov8", "owlvit"),
        ...     question="What objects are in the scene?"
        ... )
    
    3. Use with pipeline:
        >>> from gom.pipeline.preprocessor import Preprocessor
        >>> from gom.config import PreprocessorConfig
        >>> config = PreprocessorConfig(output_folder="results/")
        >>> preprocessor = Preprocessor(config)

Fallback Mechanism:
    During initial import, PreprocessorConfig may not be available due to
    circular dependencies. This module provides a lightweight fallback that
    defines only essential fields. Once pipeline.preprocessor fully loads,
    the real PreprocessorConfig replaces the fallback.

Configuration Hierarchy:
    PreprocessorConfig
    ├── I/O: input_path, output_folder, json_file
    ├── Dataset: dataset, split, num_instances
    ├── Filtering: question, apply_question_filter, aggressive_pruning
    ├── Detectors: detectors_to_use, threshold_*, grounding_dino_*
    ├── Fusion: wbf_iou_threshold, label_nms_threshold, cross_class_*
    ├── Relations: max_relations_per_object, relations_max_clip_pairs
    ├── Segmentation: segmenter, sam_checkpoint_path
    ├── Depth: use_depth, depth_model
    ├── Visualization: visualizer_config, show_*, render_*
    └── Performance: use_cache, batch_size, num_workers

See Also:
    - gom.pipeline.preprocessor: Main pipeline implementation
    - gom.segmentation.base: Segmentation config details
    - gom.relations.inference: Relationship config details
    - gom.viz.visualizer: Visualization config details
"""
from __future__ import annotations

from typing import Any

from gom.relations.inference import RelationsConfig
from gom.segmentation.base import SegmenterConfig
from gom.viz.visualizer import VisualizerConfig

# Re-export configuration dataclasses from their respective modules

# PreprocessorConfig is defined in the pipeline module
# Use try/except to provide a lightweight fallback and avoid circular import errors
try:
    from gom.pipeline.preprocessor import PreprocessorConfig
except Exception as _exc:
    # Fallback configuration class used only during early import phases
    # This will be replaced by the actual PreprocessorConfig once the pipeline module loads
    from dataclasses import dataclass, field
    from typing import Any, Dict, Optional, Tuple

    @dataclass
    class PreprocessorConfig:  # type: ignore[no-redef]
        """
        Fallback preprocessing configuration (lightweight import-time version).
        
        This minimal version prevents circular import errors during module initialization.
        The full PreprocessorConfig from gom.pipeline.preprocessor will replace this
        once all dependencies are loaded.
        
        This fallback includes only the most essential fields. For complete documentation,
        see gom.pipeline.preprocessor.PreprocessorConfig.
        
        Essential Fields:
            input_path: Input image or directory path
            output_folder: Output directory for results
            dataset: Dataset name for batch processing
            question: Question for VQA-aware filtering
            detectors_to_use: Tuple of detector names
        
        Notes:
            - This is a bootstrap class, not the production config
            - Missing many fields from full PreprocessorConfig
            - Should not be used directly in application code
            - Automatically replaced after gom.pipeline loads
        """
        # Segmentation
        segmenter: str = "sam2"
        segmenter_kwargs: Dict[str, Any] = field(default_factory=dict)

        # Output configuration
        output_folder: str = "output_images"

        # Dataset configuration and batching
        dataset: Optional[str] = None
        split: str = "train"
        image_column: str = "image"
        num_instances: int = -1  # -1 means process all instances

        # Question-based filtering parameters
        question: str = ""
        apply_question_filter: bool = True
        aggressive_pruning: bool = False  # Keep only objects mentioned in question
        filter_relations_by_question: bool = True
        threshold_object_similarity: float = 0.50  # CLIP similarity threshold for objects
        threshold_relation_similarity: float = 0.50  # CLIP similarity threshold for relations
        clip_pruning_threshold: float = 0.25  # Minimum CLIP score to keep detection
        semantic_boost_weight: float = 0.4  # Weight for CLIP scores vs detection scores
        context_expansion_radius: float = 2.0  # Radius multiplier for context expansion
        context_min_iou: float = 0.1  # Minimum IoU for context inclusion

        # Object detector configuration and confidence thresholds
        detectors_to_use: Tuple[str, ...] = ("owlvit", "yolov8", "detectron2")
        threshold_owl: float = 0.10  # OWL-ViT confidence threshold
        threshold_yolo: float = 0.25  # YOLOv8 confidence threshold
        threshold_detectron: float = 0.50  # Detectron2 confidence threshold
        threshold_grounding_dino: float = 0.30  # GroundingDINO confidence threshold
        grounding_dino_text_threshold: float = 0.25  # GroundingDINO text similarity threshold
        auto_detector_thresholds: bool = True
        auto_threshold_min_default: float = 0.25
        auto_threshold_min_owl: float = 0.25
        auto_threshold_min_yolo: float = 0.25
        auto_threshold_min_detectron: float = 0.25
        auto_threshold_min_grounding_dino: float = 0.15
        auto_threshold_max_per_detector: Optional[int] = None

        # Relationship inference constraints
        max_relations_per_object: int = 3  # Maximum relationships per object
        min_relations_per_object: int = 0  # Minimum relationships per object
        # CLIP-based relationship scoring limits (performance tuning)
        relations_max_clip_pairs: int = 1000  # Global limit on CLIP-scored pairs
        relations_per_src_clip_pairs: int = 50  # Per-source limit on CLIP-scored candidates

        # Non-Maximum Suppression and fusion parameters
        label_nms_threshold: float = 0.25  # IoU threshold for per-label NMS
        seg_iou_threshold: float = 0.50  # IoU threshold for segmentation mask merging
        wbf_iou_threshold: float = 0.10  # IoU threshold for Weighted Boxes Fusion
        skip_box_threshold: float = 0.10  # Skip boxes below this confidence in fusion
        
        # Advanced deduplication parameters
        cross_class_suppression: bool = True  # Remove overlaps between different classes
        cross_class_iou_threshold: float = 0.65  # IoU threshold for cross-class suppression
        same_class_iou_threshold: float = 0.30  # IoU threshold for same-class deduplication
        cross_class_score_diff_threshold: float = 0.80  # Score difference ratio for cross-class dedup
        
        # Advanced deduplication and merging options
        cross_class_suppression: bool = False  # Enable cross-class NMS
        cross_class_iou_threshold: float = 0.75  # IoU threshold for cross-class suppression
        enable_group_merge: bool = False  # Enable semantic grouping and merging
        merge_mask_iou_threshold: float = 0.80  # IoU threshold for mask-based merging
        merge_box_iou_threshold: float = 0.90  # IoU threshold for box-based merging
        mask_union_max_expand_ratio: float = 1.25
        enable_semantic_dedup: bool = False  # Enable CLIP-based semantic deduplication
        semantic_dedup_iou_threshold: float = 0.70  # IoU threshold for semantic dedup
        enable_containment_removal: bool = False  # Remove fully contained detections
        containment_threshold: float = 0.95  # Area overlap threshold for containment
        
        # Cascade and cache management
        cascade_conf_threshold: float = 0.40  # Confidence threshold for cascade fusion
        detection_mask_merge_iou_thr: float = 0.60  # IoU for detection-mask merging
        clip_cache_max_age_days: float = 30.0  # CLIP cache entry expiration (days)
        
        # Non-competing detection recovery (reduces false negatives)
        keep_non_competing_low_scores: bool = True  # Enable low-score recovery
        non_competing_iou_threshold: float = 0.30  # IoU threshold for competition check
        non_competing_min_score: float = 0.05  # Minimum score for recovery

        # geometry
        margin: int = 20
        min_distance: float = 10  # Reduced from 50 to allow closer object relationships
        max_distance: float = 20000

                # SAM settings
        sam_version: str = "1"  # "1" | "2" | "hq"
        sam_hq_model_type: str = "vit_h"
        points_per_side: int = 32
        pred_iou_thresh: float = 0.88
        stability_score_thresh: float = 0.95
        min_mask_region_area: int = 100

        # detection cache
        enable_detection_cache: bool = True
        max_cache_size: int = 100

        # visualization
        label_mode: str = "original"
        display_labels: bool = True
        display_relationships: bool = True
        display_relation_labels: bool = True
        show_segmentation: bool = True
        fill_segmentation: bool = True
        display_legend: bool = False
        seg_fill_alpha: float = 0.25
        bbox_linewidth: float = 2.0
        obj_fontsize_inside: int = 9
        obj_fontsize_outside: int = 10
        rel_fontsize: int = 8
        legend_fontsize: int = 8
        rel_arrow_linewidth: float = 2.0
        rel_arrow_mutation_scale: float = 26.0
        resolve_overlaps: bool = True
        show_bboxes: bool = True
        show_confidence: bool = False

        # mask post-processing
        close_holes: bool = True
        hole_kernel: int = 7
        min_hole_area: int = 100
        remove_small_components: bool = True
        min_component_area: int = 150

        # exports
        save_image_only: bool = False
        skip_graph: bool = False
        skip_prompt: bool = False
        skip_visualization: bool = False
        export_preproc_only: bool = False
        output_format: str = "jpg"  # jpg, png, svg
        save_without_background: bool = False
        verbose: bool = False

        # device
        preproc_device: Optional[str] = None
        # If True, always run full preprocessing per question (ignore detection cache)
        force_preprocess_per_question: bool = False

        # color tweaks
        color_sat_boost: float = 1.1
        color_val_boost: float = 1.1


def default_config(**overrides: Any) -> PreprocessorConfig:
    """
    Create a PreprocessorConfig with sensible defaults and optional overrides.
    """
    cfg = PreprocessorConfig()
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg
