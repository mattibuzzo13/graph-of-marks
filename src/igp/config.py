# igp/config.py
from __future__ import annotations

from typing import Any

# Re-export config dataclasses defined in their modules
from igp.segmentation.base import SegmenterConfig
from igp.relations.inference import RelationsConfig
from igp.viz.visualizer import VisualizerConfig

# PreprocessorConfig lives in the pipeline module
try:
    from igp.pipeline.preprocessor import PreprocessorConfig
except Exception as _exc:
    # Lightweight fallback to avoid early ImportError/circular imports.
    # This gets replaced once the pipeline module is importable.
    from dataclasses import dataclass
    from typing import Optional, Tuple

    @dataclass
    class PreprocessorConfig:  # type: ignore[no-redef]
        # I/O
        input_path: Optional[str] = None
        json_file: str = ""
        output_folder: str = "output_images"

        # dataset / batching
        dataset: Optional[str] = None
        split: str = "train"
        image_column: str = "image"
        num_instances: int = -1

        # question / filtering
        question: str = ""
        apply_question_filter: bool = True
        aggressive_pruning: bool = False
        filter_relations_by_question: bool = True
        threshold_object_similarity: float = 0.50
        threshold_relation_similarity: float = 0.50

        # detectors & thresholds
        detectors_to_use: Tuple[str, ...] = ("owlvit", "yolov8", "detectron2")
        threshold_owl: float = 0.40
        threshold_yolo: float = 0.80
        threshold_detectron: float = 0.80

        # per-object relation limits
        max_relations_per_object: int = 3
        min_relations_per_object: int = 1

        # NMS / fusion
        label_nms_threshold: float = 0.50
        seg_iou_threshold: float = 0.70

        # geometry
        margin: int = 20
        min_distance: float = 50
        max_distance: float = 20000

        # SAM settings
        sam_version: str = "1"           # "1" | "2" | "hq"
        sam_hq_model_type: str = "vit_h"
        points_per_side: int = 32
        pred_iou_thresh: float = 0.90
        stability_score_thresh: float = 0.92
        min_mask_region_area: int = 100

        # device
        preproc_device: Optional[str] = None

        # visualization
        label_mode: str = "original"
        display_labels: bool = True
        display_relationships: bool = True
        display_relation_labels: bool = False
        show_segmentation: bool = True
        fill_segmentation: bool = True
        display_legend: bool = True
        seg_fill_alpha: float = 0.30
        bbox_linewidth: float = 2.0
        obj_fontsize_inside: int = 12
        obj_fontsize_outside: int = 12
        rel_fontsize: int = 10
        legend_fontsize: int = 8
        rel_arrow_linewidth: float = 2.5
        rel_arrow_mutation_scale: float = 22.0
        resolve_overlaps: bool = True
        show_bboxes: bool = True
        show_confidence: bool = False

        # mask post-processing
        close_holes: bool = False
        hole_kernel: int = 7
        min_hole_area: int = 100

        # exports
        save_image_only: bool = False
        skip_graph: bool = False
        skip_prompt: bool = False
        skip_visualization: bool = False
        export_preproc_only: bool = False

        # detection cache
        enable_detection_cache: bool = True
        max_cache_size: int = 500

        # color tweaks
        color_sat_boost: float = 1.30
        color_val_boost: float = 1.15


def default_config(**overrides: Any) -> PreprocessorConfig:
    """
    Create a PreprocessorConfig with sensible defaults and optional overrides.
    """
    cfg = PreprocessorConfig()
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg
