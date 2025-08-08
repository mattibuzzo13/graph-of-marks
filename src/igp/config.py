# igp/config.py
from __future__ import annotations

from typing import Any

# Re-export delle dataclass di configurazione definite nei moduli dedicati
from igp.segmentation.base import SegmenterConfig
from igp.relations.inference import RelationsConfig
from igp.viz.visualizer import VisualizerConfig

# Il PreprocessorConfig vive nella pipeline
try:
    from igp.pipeline.preprocessor import PreprocessorConfig
except Exception as _exc:
    # Fallback leggero per evitare ImportError circolari se si importa troppo presto.
    # Verrà sovrascritto una volta che la pipeline è importabile.
    from dataclasses import dataclass
    from typing import Optional, Tuple

    @dataclass
    class PreprocessorConfig:  # type: ignore[no-redef]
        input_path: Optional[str] = None
        json_file: str = ""
        output_folder: str = "output_images"
        dataset: Optional[str] = None
        split: str = "train"
        image_column: str = "image"
        num_instances: int = -1
        question: str = ""
        apply_question_filter: bool = True
        aggressive_pruning: bool = False
        filter_relations_by_question: bool = True
        threshold_object_similarity: float = 0.50
        threshold_relation_similarity: float = 0.50
        detectors_to_use: Tuple[str, ...] = ("owlvit", "yolov8", "detectron2")
        threshold_owl: float = 0.40
        threshold_yolo: float = 0.80
        threshold_detectron: float = 0.80
        max_relations_per_object: int = 3
        min_relations_per_object: int = 1
        label_nms_threshold: float = 0.50
        seg_iou_threshold: float = 0.70
        margin: int = 20
        min_distance: float = 50
        max_distance: float = 20000
        sam_version: str = "1"
        sam_hq_model_type: str = "vit_h"
        points_per_side: int = 32
        pred_iou_thresh: float = 0.90
        stability_score_thresh: float = 0.92
        min_mask_region_area: int = 100
        preproc_device: Optional[str] = None
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
        close_holes: bool = False
        hole_kernel: int = 7
        min_hole_area: int = 100
        save_image_only: bool = False
        skip_graph: bool = False
        skip_prompt: bool = False
        skip_visualization: bool = False
        export_preproc_only: bool = False
        enable_detection_cache: bool = True
        max_cache_size: int = 100
        color_sat_boost: float = 1.30
        color_val_boost: float = 1.15


def default_config(**overrides: Any) -> PreprocessorConfig:
    """
    Crea un PreprocessorConfig con i default ragionevoli e override opzionali.
    """
    cfg = PreprocessorConfig()
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg
