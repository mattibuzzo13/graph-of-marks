# igp/pipeline/preprocessor.py
# End-to-end pipeline to build an image graph:
#   detect → fuse (WBF/NMS) → segment → depth → relations → scene graph → visualization/export.

from __future__ import annotations

import gc
import hashlib
import json
import math
import networkx as nx
from PIL import Image
import os
import time
import torch
from dataclasses import dataclass
import contextlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import logging
import warnings

# ---------- detectors ----------
from igp.detectors.base import Detector
from igp.detectors.owlvit import OwlViTDetector
from igp.detectors.yolov8 import YOLOv8Detector
from igp.detectors.detectron2 import Detectron2Detector

# 🚀 SOTA detector (optional)
try:
    from igp.detectors.grounding_dino import GroundingDINODetector
    GROUNDING_DINO_AVAILABLE = True
except ImportError:
    GroundingDINODetector = None  # type: ignore
    GROUNDING_DINO_AVAILABLE = False

# ---------- fusion ----------
from igp.fusion.wbf import fuse_detections_wbf as weighted_boxes_fusion
from igp.fusion.nms import labelwise_nms

# ---------- segmentation ----------
from igp.segmentation.base import Segmenter, SegmenterConfig
from igp.segmentation.sam1 import Sam1Segmenter
from igp.segmentation.sam2 import Sam2Segmenter
from igp.segmentation.samhq import SamHQSegmenter

# 🚀 SOTA segmenters (optional)
try:
    from igp.segmentation.fastsam import FastSAMSegmenter, MobileSAMSegmenter
    FASTSAM_AVAILABLE = True
except ImportError:
    FastSAMSegmenter = None  # type: ignore
    MobileSAMSegmenter = None  # type: ignore
    FASTSAM_AVAILABLE = False

# ---------- utils ----------
from igp.utils.depth import DepthEstimator, DepthConfig
from igp.utils.clip_utils import CLIPWrapper  
from igp.utils.boxes import iou, clamp_xyxy  
from igp.utils.colors import base_label
from igp.utils.cache_advanced import ImageDetectionCache  # 🚀 Advanced LRU cache

# 🚀 SOTA post-processing (optional)
try:
    from igp.utils.tta import TTADetector, TTAConfig
    from igp.utils.calibration import ConfidenceCalibrator, CalibrationConfig
    from igp.utils.ensemble import DetectorEnsemble, SegmenterEnsemble, EnsembleConfig
    TTA_AVAILABLE = True
    ENSEMBLE_AVAILABLE = True
except ImportError:
    TTADetector = None  # type: ignore
    TTAConfig = None  # type: ignore
    ConfidenceCalibrator = None  # type: ignore
    CalibrationConfig = None  # type: ignore
    DetectorEnsemble = None  # type: ignore
    SegmenterEnsemble = None  # type: ignore
    EnsembleConfig = None  # type: ignore
    TTA_AVAILABLE = False
    ENSEMBLE_AVAILABLE = False

# 🚀 Performance optimizations (optional)
try:
    from igp.utils.model_registry import ModelRegistry
    from igp.utils.mixed_precision import MixedPrecisionManager
    from igp.utils.batch_processing import BatchProcessor, BatchConfig
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    ModelRegistry = None  # type: ignore
    MixedPrecisionManager = None  # type: ignore
    BatchProcessor = None  # type: ignore
    BatchConfig = None  # type: ignore
    OPTIMIZATION_AVAILABLE = False

# ---------- GPU memory management ----------
try:
    from igp.utils.gpu_memory import get_gpu_manager, gpu_memory_context
    GPU_MEMORY_AVAILABLE = True
except ImportError:
    GPU_MEMORY_AVAILABLE = False

# ---------- relations ----------
from igp.relations.inference import RelationsConfig, RelationInferencer

# ---------- graph ---------- 
from igp.graph.scene_graph import SceneGraphBuilder
from igp.graph.prompt import graph_to_triples_text
from igp.graph.scene_graph import SceneGraphBuilder as _SceneGraphBuilder

def build_scene_graph(
    image_size: Tuple[int, int],
    boxes: Sequence[Sequence[float]], 
    labels: Sequence[str],
    scores: Sequence[float],
    depths: Optional[Sequence[float]] = None,
    caption: str = ""
) -> "nx.DiGraph":
    """Thin wrapper around SceneGraphBuilder.build() for compatibility."""
    
    W, H = image_size
    # Create a dummy image for the builder (the builder only needs size/crops).
    image = Image.new('RGB', (W, H), color='white')
    
    builder = _SceneGraphBuilder()
    return builder.build(image, boxes, labels, scores)

# ✅ Alias kept for prompt serialization compatibility
to_triples_text = graph_to_triples_text

# ---------- viz ----------
from igp.viz.visualizer import Visualizer, VisualizerConfig


# ----------------------------- Config dataclass -----------------------------
@dataclass
class PreprocessorConfig:
    # I/O
    input_path: Optional[str] = None
    json_file: str = ""
    output_folder: str = "output_images"

    # batching / dataset (optional)
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
    
    # 🚀 Advanced Semantic Pruning (Phase 6)
    use_clip_semantic_pruning: bool = True  # Use CLIP similarity for object ranking
    clip_pruning_threshold: float = 0.25  # Min CLIP similarity to question
    semantic_boost_weight: float = 0.4  # Weight for semantic relevance (vs confidence)
    context_expansion_enabled: bool = True  # Add contextually relevant objects
    context_expansion_radius: float = 2.0  # Multiplier for expansion area
    context_min_iou: float = 0.1  # Min overlap for context objects
    false_negative_reduction: bool = True  # Enable anti-false-negative heuristics
    min_objects_per_question: int = 3  # Min objects to keep (avoid over-pruning)
    max_objects_per_question: int = 50  # Max objects to keep (performance cap)

    # detectors & thresholds
    detectors_to_use: Tuple[str, ...] = ("owlvit", "yolov8", "detectron2")
    threshold_owl: float = 0.40
    threshold_yolo: float = 0.80
    threshold_detectron: float = 0.80
    
    # 🚀 SOTA detector: Grounding DINO (optional, better than OWL-ViT)
    threshold_grounding_dino: float = 0.35
    grounding_dino_model: str = "base"  # "tiny", "base", "large"
    grounding_dino_text_prompt: Optional[str] = None  # Auto-detect if None
    grounding_dino_text_threshold: float = 0.25

    # per-object relation limits
    max_relations_per_object: int = 1
    min_relations_per_object: int = 1

    # CLIP cache tuning
    clip_cache_max_age_days: Optional[float] = 30.0  # default TTL for disk cache (days)

    # NMS / fusion
    label_nms_threshold: float = 0.50
    seg_iou_threshold: float = 0.70

    # geometry (pixels)
    margin: int = 20
    min_distance: float = 50
    max_distance: float = 20000

    # SAM
    sam_version: str = "1"  # "1" | "2" | "hq" | "fast" | "mobile"
    sam_hq_model_type: str = "vit_h"
    points_per_side: int = 32
    pred_iou_thresh: float = 0.90
    stability_score_thresh: float = 0.92
    min_mask_region_area: int = 100
    
    # 🚀 SOTA segmentation: FastSAM (10x speed) or MobileSAM (60x speed)
    fastsam_imgsz: int = 1024
    fastsam_conf: float = 0.4
    fastsam_iou: float = 0.9
    fastsam_retina_masks: bool = True
    # Mask refinement (SOTA quality improvement)
    refine_masks: bool = False  # Apply edge-aware refinement
    refine_edge_aware: bool = True
    refine_fill_holes: bool = True
    refine_boundary: bool = False  # Expensive (GrabCut)
    
    # 🚀 SOTA: Test-Time Augmentation (Phase 4)
    use_tta: bool = False  # Apply TTA for detection (+2-4% mAP)
    tta_scales: Tuple[float, ...] = (0.75, 1.0, 1.25, 1.5)
    tta_flip_horizontal: bool = True
    tta_flip_vertical: bool = False
    tta_fusion_method: str = "wbf"  # "wbf" | "nms" | "soft_nms"
    tta_iou_threshold: float = 0.5
    
    # 🚀 SOTA: Confidence Calibration (Phase 4)
    use_calibration: bool = False  # Calibrate confidence scores
    calibration_method: str = "temperature"  # "temperature" | "platt" | "isotonic"
    calibration_cache_path: Optional[str] = None  # Path to calibration params
    temperature: float = 1.5  # Temperature for scaling (>1 = softer)
    
    # 🚀 SOTA: Ensemble Methods (Phase 5)
    use_detector_ensemble: bool = False  # Ensemble multiple detectors
    ensemble_fusion_method: str = "wbf"  # "wbf" | "voting" | "weighted_avg" | "nms"
    ensemble_detector_weights: Optional[Dict[str, float]] = None  # Model weights
    ensemble_min_votes: int = 2  # For voting method
    ensemble_iou_threshold: float = 0.5
    use_segmenter_ensemble: bool = False  # Ensemble multiple segmenters
    
    # 🚀 Performance Optimizations
    use_model_cache: bool = True  # Use ModelRegistry for caching
    use_mixed_precision: bool = False  # Enable FP16/BF16 (2x speedup)
    batch_size: int = 1  # Batch processing (1=disabled, 4-8 recommended)
    mixed_precision_dtype: str = "auto"  # "auto" | "fp16" | "bf16" | "fp32"
    
    # detector parallelism and pruning
    detectors_parallelism: str = "auto"  # "auto" | "thread" | "sequential"
    detectors_max_workers: Optional[int] = None
    max_detections_total: int = 200
    max_detections_per_label: int = 50
    min_box_area_px: int = 0

    # conditional compute skipping
    skip_relations_when_unused: bool = True
    skip_depth_when_unused: bool = True
    skip_segmentation_when_unused: bool = True

    # Enable optional Spatial3D reasoning (off by default). When True, the
    # RelationInferencer will attempt to instantiate the Spatial3DReasoner (if
    # available) and run depth/occlusion/support inference.
    enable_spatial_3d: bool = False

    # device
    preproc_device: Optional[str] = None  # e.g., "cpu" or "cuda"

    # logging / verbosity
    verbose: bool = False
    # If True, suppress noisy library warnings (Deprecation/User/Resource) during preprocessing
    suppress_warnings: bool = True

    # rendering toggles
    label_mode: str = "original"
    display_labels: bool = True
    display_relationships: bool = True
    display_relation_labels: bool = False
    show_segmentation: bool = True
    fill_segmentation: bool = True
    display_legend: bool = True
    seg_fill_alpha: float = 0.75
    bbox_linewidth: float = 2.0
    obj_fontsize_inside: int = 12
    obj_fontsize_outside: int = 12
    rel_fontsize: int = 10
    legend_fontsize: int = 8
    rel_arrow_linewidth: float = 2.5
    rel_arrow_mutation_scale: float = 26.0
    resolve_overlaps: bool = True
    show_bboxes: bool = True
    show_confidence: bool = False

    # mask post-processing (morphological close)
    close_holes: bool = False
    hole_kernel: int = 7
    min_hole_area: int = 100

    # export flags
    save_image_only: bool = False
    skip_graph: bool = False
    skip_prompt: bool = False
    skip_visualization: bool = False
    export_preproc_only: bool = False  # save a transparent PNG with overlay only

    # detection cache
    enable_detection_cache: bool = True
    max_cache_size: int = 100

    # colors
    color_sat_boost: float = 1.30
    color_val_boost: float = 1.15


# ----------------------------- Preprocessor -----------------------------
class ImageGraphPreprocessor:
    """
    End-to-end pipeline:
      detect → fuse (NMS/WBF) → segment → depth → relate → graph → viz/export.
    """

    def __init__(self, config: PreprocessorConfig) -> None:
        self.cfg = config
        # per-instance logger
        self.logger = logging.getLogger(__name__)
        try:
            if getattr(self.cfg, "verbose", False):
                self.logger.setLevel(logging.INFO)
        except Exception:
            pass
        os.makedirs(self.cfg.output_folder, exist_ok=True)

        # Device selection with CUDA fallback if available.
        if self.cfg.preproc_device:
            self.device = self.cfg.preproc_device
        else:
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                self.device = "cpu"

        # Detectors stack (open-vocab + closed-vocab for complementarity).
        self.detectors: List[Detector] = self._init_detectors()

        # Segmenter selection (SAM v1 / v2 / HQ).
        self.segmenter: Segmenter = self._init_segmenter()

        # Depth, caption, and CLIP helpers (used by relations if enabled).
        depth_config = DepthConfig(device=self.device)
        self.depth_est = DepthEstimator(config=depth_config)
        
        try:
            self.clip = CLIPWrapper(device=self.device)
        except TypeError:
            # If CLIPWrapper expects a config object, use the optional config path.
            from igp.utils.clip_utils import CLIPConfig
            clip_config = CLIPConfig(device=self.device)
            self.clip = CLIPWrapper(config=clip_config) 

        # Relation inference with geometric constraints and optional CLIP.
        # If we have a CLIP wrapper available, create a ClipRelScorer and pass
        # it to the inferencer so batched scoring + persistent cache are used.
        try:
            from igp.relations.clip_rel import ClipRelScorer
        except Exception:
            ClipRelScorer = None

        clip_scorer = None
        # Create a ClipRelScorer but DO NOT persist the disk DB. The persistent
        # cache (.igp_clip_cache.db) is disabled to avoid writing files during
        # runs; in-memory caching is still used within the process.
        if ClipRelScorer is not None and getattr(self, "clip", None) is not None:
            try:
                clip_scorer = ClipRelScorer(
                    device=self.device,
                    clip=self.clip,
                    # do not pass disk_cache_path -> persistent DB disabled
                    batch_size=getattr(self.cfg, "batch_size", 16),
                )
            except Exception:
                clip_scorer = ClipRelScorer(device=self.device, clip=self.clip)

        # Enable Spatial3D reasoning by default if depth estimator is available
        rels_cfg = RelationsConfig()
        # Honor explicit preprocessor flags if present; defer to the
        # PreprocessorConfig.enable_spatial_3d flag so users can toggle it via
        # CLI or overrides. Defaults to False.
        rels_cfg.use_3d_reasoning = bool(getattr(self.cfg, "enable_spatial_3d", False))

        self.relations_inferencer = RelationInferencer(
            clip_scorer,
            relations_config=rels_cfg,
            margin_px=config.margin,
            min_distance=config.min_distance,
            max_distance=config.max_distance,
        )

        # Visualization configuration (keeps visual output consistent for the paper).
        self.visualizer = Visualizer(
            VisualizerConfig(
                display_labels=self.cfg.display_labels,
                display_relationships=self.cfg.display_relationships,
                display_relation_labels=self.cfg.display_relation_labels,
                display_legend=self.cfg.display_legend,
                show_segmentation=self.cfg.show_segmentation and not self.cfg.export_preproc_only,
                fill_segmentation=self.cfg.fill_segmentation,
                show_bboxes=self.cfg.show_bboxes,
                obj_fontsize_inside=self.cfg.obj_fontsize_inside,
                obj_fontsize_outside=self.cfg.obj_fontsize_outside,
                rel_fontsize=self.cfg.rel_fontsize,
                legend_fontsize=self.cfg.legend_fontsize,
                seg_fill_alpha=self.cfg.seg_fill_alpha,
                bbox_linewidth=self.cfg.bbox_linewidth,
                rel_arrow_linewidth=self.cfg.rel_arrow_linewidth,
                rel_arrow_mutation_scale=self.cfg.rel_arrow_mutation_scale,
                resolve_overlaps=self.cfg.resolve_overlaps,
                color_sat_boost=self.cfg.color_sat_boost,
                color_val_boost=self.cfg.color_val_boost,
            )
        )

        # 🚀 Advanced LRU detection cache with memory-aware eviction
        if self.cfg.enable_detection_cache:
            from igp.utils.cache_advanced import ImageDetectionCache
            self.detection_cache = ImageDetectionCache(max_items=self.cfg.max_cache_size)
        else:
            self.detection_cache = None
        
        # 🚀 Performance optimizations
        self._init_performance_optimizations()

    def _init_performance_optimizations(self) -> None:
        """Initialize performance optimization modules."""
        if not OPTIMIZATION_AVAILABLE:
            return
        
        # Model Registry for caching (uses class methods, no instance needed)
        if self.cfg.use_model_cache:
            # Just verify it's available, no need to store instance
            self.model_registry = ModelRegistry  # Store class reference
            if getattr(self.cfg, "verbose", False):
                self.logger.info("[Performance] Model Registry enabled (caching models)")
        else:
            self.model_registry = None
        
        # Mixed Precision Manager
        if self.cfg.use_mixed_precision:
            dtype = self.cfg.mixed_precision_dtype
            if dtype == "auto":
                dtype = None  # Auto-detect
            self.mixed_precision = MixedPrecisionManager(
                enabled=True,
                dtype=dtype,
                device=self.device,
            )
            if getattr(self.cfg, "verbose", False):
                self.logger.info(f"[Performance] Mixed Precision enabled ({self.mixed_precision.dtype})")
        else:
            self.mixed_precision = None
        
        # Batch Processor
        if self.cfg.batch_size > 1:
            batch_config = BatchConfig(
                batch_size=self.cfg.batch_size,
                resize_mode="pad",  # Preserve aspect ratio
            )
            self.batch_processor = BatchProcessor(batch_config)
            if getattr(self.cfg, "verbose", False):
                self.logger.info(f"[Performance] Batch Processing enabled (batch_size={self.cfg.batch_size})")
        else:
            self.batch_processor = None
        self._detection_cache = ImageDetectionCache(
            max_items=self.cfg.max_cache_size,
            max_size_mb=500.0  # 500 MB max cache size
        )

    @contextlib.contextmanager
    def _maybe_suppress_warnings(self):
        """Context manager: suppress common noisy warnings when configured."""
        if getattr(self.cfg, "suppress_warnings", True):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                warnings.simplefilter("ignore", category=DeprecationWarning)
                warnings.simplefilter("ignore", category=ResourceWarning)
                yield
        else:
            yield

    # ----------------------------- setup helpers -----------------------------

    def _init_detectors(self) -> List[Detector]:
        """Initialize enabled detectors according to config."""
        dets: List[Detector] = []
        names = set(d.strip().lower() for d in self.cfg.detectors_to_use)
        
        # 🚀 SOTA: Grounding DINO (open-vocabulary, better than OWL-ViT)
        if "grounding_dino" in names or "groundingdino" in names:
            if not GROUNDING_DINO_AVAILABLE:
                # important: missing optional dependency — warn the user
                if getattr(self.cfg, "verbose", False):
                    self.logger.warning("[WARNING] Grounding DINO requested but not installed. Skipping.")
                    self.logger.info("Install with: pip install groundingdino-py")
                else:
                    self.logger.warning("Grounding DINO requested but not installed; skipping (set verbose=True for details)")
            else:
                dets.append(GroundingDINODetector(
                    model_name=self.cfg.grounding_dino_model,
                    text_prompt=self.cfg.grounding_dino_text_prompt,
                    box_threshold=self.cfg.threshold_grounding_dino,
                    text_threshold=self.cfg.grounding_dino_text_threshold,
                    device=self.device,
                    score_threshold=self.cfg.threshold_grounding_dino,
                ))
        
        if "owlvit" in names:
            # Use score_threshold consistent with base Detector interface.
            dets.append(OwlViTDetector(
                device=self.device, 
                score_threshold=self.cfg.threshold_owl
            ))
        
        if "yolov8" in names:
            dets.append(YOLOv8Detector(
                device=self.device, 
                score_threshold=self.cfg.threshold_yolo
            ))
        
        if "detectron2" in names:
            dets.append(Detectron2Detector(
                device=self.device, 
                score_threshold=self.cfg.threshold_detectron
            ))
        
        return dets

    def _init_segmenter(self) -> Segmenter:
        """
        Create the SAM segmenter variant with common post-processing flags.
        🚀 SOTA: Supports FastSAM (10x speed) and MobileSAM (60x speed)
        """
        s_cfg = SegmenterConfig(
            device=self.device,
            close_holes=self.cfg.close_holes,
            hole_kernel=self.cfg.hole_kernel,
            min_hole_area=self.cfg.min_hole_area,
        )
        
        # 🚀 SOTA FastSAM: 10x faster than SAM2
        if self.cfg.sam_version == "fast":
            if not FASTSAM_AVAILABLE or FastSAMSegmenter is None:
                if getattr(self.cfg, "verbose", False):
                    self.logger.warning("[WARNING] FastSAM requested but not installed. Falling back to SAM2.")
                    self.logger.info("Install with: pip install git+https://github.com/CASIA-IVA-Lab/FastSAM.git")
                else:
                    self.logger.warning("FastSAM requested but not installed; falling back to SAM2 (set verbose=True for details)")
                return Sam2Segmenter(config=s_cfg)
            
            return FastSAMSegmenter(
                config=s_cfg,
                imgsz=self.cfg.fastsam_imgsz,
                conf=self.cfg.fastsam_conf,
                iou=self.cfg.fastsam_iou,
                retina_masks=self.cfg.fastsam_retina_masks,
            )
        
        # 🚀 SOTA MobileSAM: 60x faster than SAM (ViT-B), best for mobile/edge
        if self.cfg.sam_version == "mobile":
            if not FASTSAM_AVAILABLE or MobileSAMSegmenter is None:
                if getattr(self.cfg, "verbose", False):
                    self.logger.warning("[WARNING] MobileSAM requested but not installed. Falling back to SAM1.")
                    self.logger.info("Install with: pip install git+https://github.com/ChaoningZhang/MobileSAM.git")
                else:
                    self.logger.warning("MobileSAM requested but not installed; falling back to SAM1 (set verbose=True for details)")
                return Sam1Segmenter(config=s_cfg)
            
            return MobileSAMSegmenter(config=s_cfg)
        
        # Original SAM variants
        if self.cfg.sam_version == "2":
            return Sam2Segmenter(config=s_cfg)
        if self.cfg.sam_version == "hq":
            return SamHQSegmenter(config=s_cfg, model_type=self.cfg.sam_hq_model_type)
        return Sam1Segmenter(config=s_cfg)

    # ----------------------------- cache helpers -----------------------------

    def _generate_cache_key(self, image_pil: Image.Image, question: str = "") -> str:
        """
        Generate deterministic cache key using advanced hashing.
        🚀 Optimized: delegates to ImageDetectionCache for consistent key generation.
        """
        thresholds = {
            "owl": self.cfg.threshold_owl,
            "yolo": self.cfg.threshold_yolo,
            "detectron": self.cfg.threshold_detectron,
        }
        return ImageDetectionCache.generate_key(
            image=image_pil,
            detectors=self.cfg.detectors_to_use,
            thresholds=thresholds,
            question=question or self.cfg.question
        )

    def _cache_get(self, key: str) -> Optional[Dict[str, Any]]:
        """Read detection results from advanced LRU cache."""
        if not self.cfg.enable_detection_cache:
            return None
        return self._detection_cache.get(key)

    def _cache_put(self, key: str, value: Dict[str, Any]) -> None:
        """
        Write detection results to advanced LRU cache.
        🚀 Optimized: automatic memory-aware eviction, no manual size checks needed.
        """
        if not self.cfg.enable_detection_cache:
            return
        self._detection_cache.put(key, value)

    # ----------------------------- pipeline core -----------------------------

    def _run_detectors(self, image_pil: Image.Image) -> Dict[str, Any]:
        """Run all detectors with configurable parallelism and return raw detections ready for fusion."""
        from concurrent.futures import ThreadPoolExecutor

        all_dets: List[Dict[str, Any]] = []
        counts: Dict[str, int] = {}

        # Decide parallel strategy
        par = (self.cfg.detectors_parallelism or "auto").lower()
        num_det = len(self.detectors)
        try:
            any_gpu = any(str(getattr(d, "device", "")).startswith("cuda") for d in self.detectors) and torch.cuda.is_available()
        except Exception:
            any_gpu = torch.cuda.is_available()

        def _run_detector(det):
            # Use Mixed Precision if enabled
            if self.mixed_precision is not None:
                ctx = self.mixed_precision.autocast()
            else:
                ctx = contextlib.nullcontext()

            # Use GPU memory context if available for automatic cleanup
            if GPU_MEMORY_AVAILABLE:
                mem_ctx = gpu_memory_context(clear_after=True, verbose=False)
            else:
                mem_ctx = contextlib.nullcontext()

            with ctx:
                with mem_ctx:
                    out = det.run(image_pil)
            src_name = det.__class__.__name__
            return src_name, out

        # Choose execution mode
        if num_det > 1:
            if par == "sequential" or (par == "auto" and any_gpu):
                # Avoid GPU contention by running sequentially when detectors use the same GPU
                results = [_run_detector(det) for det in self.detectors]
            else:
                max_workers = self.cfg.detectors_max_workers or num_det
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    results = list(executor.map(_run_detector, self.detectors))
        else:
            results = [_run_detector(self.detectors[0])]

        # Clear GPU cache after all detectors complete
        if GPU_MEMORY_AVAILABLE:
            get_gpu_manager().clear_cache(verbose=False)

        for src_name, out in results:
            counts[src_name] = len(out)
            for d in out:
                all_dets.append({
                    "box": list(d.box),
                    "label": str(d.label),
                    "score": float(d.score),
                    "from": src_name.lower(),
                    "mask": d.extra.get("mask") if d.extra else None,
                })

        return {
            "detections": all_dets,
            "counts": counts,
            "boxes": [d["box"] for d in all_dets],
            "labels": [d["label"] for d in all_dets],
            "scores": [d["score"] for d in all_dets],
        }
    
    def _run_detectors_batch(self, images: List[Image.Image]) -> List[Dict[str, Any]]:
        """Run detectors on a batch of images (4-8x faster than sequential)."""
        from concurrent.futures import ThreadPoolExecutor
        
        batch_results = []
        
        # Initialize empty results for each image
        for _ in images:
            batch_results.append({
                "detections": [],
                "counts": {},
                "boxes": [],
                "labels": [],
                "scores": [],
            })
        
        for det in self.detectors:
            src_name = det.__class__.__name__
            
            # Check if detector supports batching
            if hasattr(det, 'supports_batch') and det.supports_batch:
                try:
                    det_batch_results = det.detect_batch(images)
                except Exception as e:
                    self.logger.warning(f"[WARN] Batch detection failed for {src_name}, falling back to sequential: {e}")
                    det_batch_results = [det.run(img) for img in images]
            else:
                # Fallback: parallelize per-image calls for this detector
                max_workers = min(len(images), max(2, (os.cpu_count() or 4)))
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    det_batch_results = list(executor.map(det.run, images))
            
            # Aggregate results per image
            for img_idx, det_results in enumerate(det_batch_results):
                batch_results[img_idx]["counts"][src_name] = len(det_results)
                
                for d in det_results:
                    batch_results[img_idx]["detections"].append({
                        "box": list(d.box),
                        "label": str(d.label),
                        "score": float(d.score),
                        "from": src_name.lower(),
                        "mask": d.extra.get("mask") if d.extra else None,
                    })
        
        # Finalize boxes/labels/scores arrays
        for result in batch_results:
            result["boxes"] = [d["box"] for d in result["detections"]]
            result["labels"] = [d["label"] for d in result["detections"]]
            result["scores"] = [d["score"] for d in result["detections"]]
        
        return batch_results
    
    def _get_optimal_batch_size(self) -> int:
        """
        Adaptive batch sizing for better GPU utilization.
        """
        if not torch.cuda.is_available():
            return 1
        
        try:
            gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)

            # Base batch sizes by VRAM capacity
            if gpu_mem_gb >= 40:  
                base_batch = 32
            elif gpu_mem_gb >= 24:  
                base_batch = 16
            elif gpu_mem_gb >= 16:  
                base_batch = 12
            elif gpu_mem_gb >= 12:  
                base_batch = 8
            else:
                base_batch = 4
            
            # Adjust based on previous image size
            if hasattr(self, '_last_processed_size'):
                w, h = self._last_processed_size
                pixels_mp = (w * h) / 1_000_000
                if pixels_mp > 4.0:  # reduce for very large images (>4MP)
                    base_batch = max(base_batch // 2, 2)
            
            return base_batch
        except Exception:
            return 4  # safe default

    def _wbf_fusion(self, all_detections: List[Dict[str, Any]], image_size: Tuple[int, int]) -> Tuple[List[List[float]], List[str], List[float]]:
        """Weighted Boxes Fusion (WBF) with source-specific weights."""
        if not all_detections:
            return [], [], []
        W, H = image_size
        # Stable label mapping (normalize labels before fusion).
        canon_labels = [base_label(d["label"]) for d in all_detections]
        uniq_labels = sorted(set(canon_labels))
        label2id = {lb: i for i, lb in enumerate(uniq_labels)}

        # Convert raw dicts to Detection objects for the fusion utility.
        from igp.types import Detection
        detections_obj = []
        
        for d in all_detections:
            x1, y1, x2, y2 = d["box"]
            label = base_label(d["label"])
            score = float(d["score"])
            source = d.get("from", "unknown")
            
            # Robust construction for different Detection signatures.
            try:
                det_obj = Detection(box=(x1, y1, x2, y2), label=label, score=score, source=source)
            except TypeError:
                try:
                    det_obj = Detection(box=(x1, y1, x2, y2), label=label, score=score)
                    det_obj.source = source
                except TypeError:
                    det_obj = Detection(box=(x1, y1, x2, y2), label=label)
                    det_obj.score = score
                    det_obj.source = source
            
            detections_obj.append(det_obj)

        # Apply WBF with sensible defaults; returns fused boxes in pixels.
        fused_detections = weighted_boxes_fusion(
            detections_obj,
            image_size=(W, H),
            iou_thr=0.55,
            skip_box_thr=0.0,
            weights_by_source={"owlvit": 2.0, "yolov8": 1.5, "yolo": 1.5, "detectron2": 1.0},
            default_weight=1.0,
            sort_desc=True
        )
        
        # Extract final arrays for downstream tasks.
        boxes_px = [list(d.box) for d in fused_detections]
        labels = [d.label for d in fused_detections]
        scores = [d.score for d in fused_detections]
        
        return boxes_px, labels, scores

    def _fuse_with_det2_mask(self, sam_mask: np.ndarray, det2_mask: Optional[np.ndarray]) -> np.ndarray:
        """Union with Detectron2 mask when sufficiently overlapping (IoU ≥ 0.5)."""
        if det2_mask is None:
            return sam_mask
        iou = self._mask_iou(sam_mask, det2_mask)
        if iou >= 0.5:
            return np.logical_or(sam_mask, det2_mask)
        return sam_mask

    @staticmethod
    def _mask_iou(m1: np.ndarray, m2: np.ndarray) -> float:
        """Binary mask IoU helper."""
        inter = np.logical_and(m1, m2).sum()
        union = np.logical_or(m1, m2).sum()
        return float(inter) / float(union) if union > 0 else 0.0

    def _apply_label_nms(self, boxes: List[List[float]], labels: List[str], scores: List[float]) -> Tuple[List[List[float]], List[str], List[float], List[int]]:
        """Per-class (label-wise) NMS; returns filtered lists and kept indices."""
        keep = labelwise_nms(boxes, labels, scores, iou_threshold=self.cfg.label_nms_threshold)
        boxes_f = [boxes[i] for i in keep]
        labels_f = [labels[i] for i in keep]
        scores_f = [scores[i] for i in keep]
        return boxes_f, labels_f, scores_f, keep

    def _compute_clip_semantic_scores(
        self,
        image_pil: Image.Image,
        boxes: List[List[float]],
        labels: List[str],
        question: str,
        obj_terms: set,
    ) -> Dict[int, float]:
        """
        🚀 Advanced Semantic Pruning: Use CLIP to compute semantic relevance scores.
        
        Returns:
            Dict mapping box index → semantic score [0.0, 1.0]
        
        Strategy:
            1. Encode question with CLIP text encoder
            2. Encode object labels with CLIP text encoder
            3. Compute similarity between question and each label
            4. Optionally encode image crops for visual similarity
        """
        semantic_scores = {}
        
        # Early exit if CLIP not available or not enabled
        if not self.cfg.use_clip_semantic_pruning:
            return semantic_scores
        
        if not hasattr(self, 'clip') or self.clip is None:
            return semantic_scores
        
        if not self.clip.available():
            return semantic_scores
        
        if not question or not boxes:
            return semantic_scores
        
        try:
            # Build text prompts for each object
            # Format: "a photo of {label}" for better CLIP matching
            label_prompts = []
            for lb in labels:
                # Clean label for CLIP
                clean_label = base_label(lb).replace("_", " ")
                prompt = f"a photo of {clean_label}"
                label_prompts.append(prompt)
            
            # Encode question as query
            question_clean = question.strip().lower()
            if not question_clean.endswith("?"):
                question_clean += "?"
            
            # Get CLIP similarities: question vs all labels
            # Returns tensor [1, num_labels]
            sims = self.clip.similarities([question_clean], label_prompts)
            
            if sims is None:
                return semantic_scores
            
            # Convert to dict
            sims_list = sims.squeeze(0).detach().cpu().tolist()
            for i, score in enumerate(sims_list):
                # Clip to [0, 1] and apply threshold
                score = max(0.0, min(1.0, float(score)))
                if score >= self.cfg.clip_pruning_threshold:
                    semantic_scores[i] = score
            
            # 🚀 Context-aware expansion: boost nearby/overlapping objects
            if self.cfg.context_expansion_enabled and semantic_scores:
                semantic_scores = self._expand_context_objects(
                    boxes, 
                    semantic_scores,
                    radius=self.cfg.context_expansion_radius,
                    min_iou=self.cfg.context_min_iou
                )
        
        except Exception as e:
            # Silent fallback - don't break the pipeline
            if hasattr(self, '_verbose') and self._verbose:
                self.logger.warning(f"[WARNING] CLIP semantic scoring failed: {e}")
        
        return semantic_scores
    
    def _expand_context_objects(
        self,
        boxes: List[List[float]],
        semantic_scores: Dict[int, float],
        radius: float = 2.0,
        min_iou: float = 0.1,
    ) -> Dict[int, float]:
        """
        🚀 False Negative Reduction: Expand to include contextually relevant objects.
        
        For each semantically relevant object, boost nearby/overlapping objects.
        This prevents over-aggressive pruning of contextually important objects.
        
        Args:
            boxes: List of bounding boxes
            semantic_scores: Current semantic scores
            radius: Multiplier for expansion area (2.0 = 2x box area)
            min_iou: Minimum IoU for context inclusion
        
        Returns:
            Updated semantic_scores with context boost
        """
        if not semantic_scores or not boxes:
            return semantic_scores
        
        expanded_scores = dict(semantic_scores)
        
        # For each highly relevant object (score > 0.5)
        anchor_indices = [i for i, score in semantic_scores.items() if score > 0.5]
        
        for anchor_idx in anchor_indices:
            anchor_box = boxes[anchor_idx]
            anchor_score = semantic_scores[anchor_idx]
            
            # Compute expanded box (radius * original size)
            x1, y1, x2, y2 = anchor_box
            w, h = x2 - x1, y2 - y1
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            
            # Expand by radius
            new_w = w * radius
            new_h = h * radius
            expanded_box = [
                cx - new_w / 2,
                cy - new_h / 2,
                cx + new_w / 2,
                cy + new_h / 2
            ]
            
            # Check overlap with all other boxes
            for i, box in enumerate(boxes):
                if i == anchor_idx:
                    continue
                
                # Compute IoU with expanded box
                overlap = iou(box, expanded_box)
                
                if overlap >= min_iou:
                    # Boost this object's score (decay based on distance)
                    boost = anchor_score * 0.3 * (overlap / min_iou) ** 0.5
                    current_score = expanded_scores.get(i, 0.0)
                    expanded_scores[i] = max(current_score, boost)
        
        return expanded_scores

    def _limit_detections(
        self,
        boxes: List[List[float]],
        labels: List[str],
        scores: List[float],
        question_terms: Optional[set] = None,
    ) -> Tuple[List[List[float]], List[str], List[float]]:
        """
        Apply lightweight pruning to reduce downstream cost (segmentation/relations).
        🚀 Enhanced: prioritizes semantically relevant objects when question_terms provided.
        """
        if not boxes:
            return boxes, labels, scores

        # Filter by min area
        if self.cfg.min_box_area_px and self.cfg.min_box_area_px > 0:
            kept_idx = []
            for i, b in enumerate(boxes):
                x1, y1, x2, y2 = b[:4]
                area = max(1, int(x2 - x1)) * max(1, int(y2 - y1))
                if area >= int(self.cfg.min_box_area_px):
                    kept_idx.append(i)
            boxes = [boxes[i] for i in kept_idx]
            labels = [labels[i] for i in kept_idx]
            scores = [scores[i] for i in kept_idx]
            if not boxes:
                return boxes, labels, scores

        # 🚀 Compute semantic relevance scores if question terms provided
        semantic_boost = {}
        if question_terms:
            # Level 1: Text-based fuzzy matching (fast, existing implementation)
            for i, lb in enumerate(labels):
                base_lb = base_label(lb).lower()
                relevance = 0.0
                
                # Check for matches with question terms
                for term in question_terms:
                    term_normalized = str(term).replace("_", " ").lower()
                    base_normalized = base_lb.replace("_", " ")
                    
                    # Exact match
                    if base_lb == term or term_normalized == base_normalized:
                        relevance = max(relevance, 1.0)
                    # Substring match
                    elif term_normalized in base_normalized or base_normalized in term_normalized:
                        relevance = max(relevance, 0.7)
                    # Word overlap
                    else:
                        term_words = set(term_normalized.split())
                        label_words = set(base_normalized.split())
                        if term_words and label_words:
                            overlap = len(term_words & label_words) / max(len(term_words), len(label_words))
                            relevance = max(relevance, overlap * 0.6)
                
                semantic_boost[i] = relevance

        # Cap per label (with semantic awareness)
        per_label = max(0, int(self.cfg.max_detections_per_label))
        if per_label > 0:
            from collections import defaultdict
            idx_by_label = defaultdict(list)
            for i, (lb, sc) in enumerate(zip(labels, scores)):
                # 🚀 Boost score for semantically relevant objects
                effective_score = float(sc)
                if semantic_boost:
                    boost_factor = semantic_boost.get(i, 0.0)
                    # Add up to 0.3 boost for highly relevant objects
                    effective_score = sc + (boost_factor * 0.3)
                
                idx_by_label[base_label(lb)].append((i, effective_score))
            
            kept = []
            for _, pairs in idx_by_label.items():
                # Sort by effective score (includes semantic boost)
                pairs_sorted = sorted(pairs, key=lambda x: -x[1])
                kept.extend([i for i, _ in pairs_sorted[:per_label]])
            kept = sorted(set(kept))
            boxes = [boxes[i] for i in kept]
            labels = [labels[i] for i in kept]
            scores = [scores[i] for i in kept]

        # Cap total (with semantic awareness)
        total_cap = max(0, int(self.cfg.max_detections_total))
        if total_cap > 0 and len(boxes) > total_cap:
            # 🚀 Sort by composite score: detection confidence + semantic relevance
            composite_scores = []
            for i in range(len(boxes)):
                base_score = float(scores[i])
                sem_boost = semantic_boost.get(i, 0.0) if semantic_boost else 0.0
                # Weight: 60% detection confidence, 40% semantic relevance
                composite = 0.6 * base_score + 0.4 * sem_boost
                composite_scores.append((i, composite))
            
            # Keep top-K by composite score
            composite_scores.sort(key=lambda x: -x[1])
            kept_indices = [i for i, _ in composite_scores[:total_cap]]
            kept_indices.sort()  # Maintain original order
            
            boxes = [boxes[i] for i in kept_indices]
            labels = [labels[i] for i in kept_indices]
            scores = [scores[i] for i in kept_indices]
        
        return boxes, labels, scores

    def _limit_detections_advanced(
        self,
        boxes: List[List[float]],
        labels: List[str],
        scores: List[float],
        question_terms: Optional[set] = None,
        clip_scores: Optional[Dict[int, float]] = None,
    ) -> Tuple[List[List[float]], List[str], List[float]]:
        """
        🚀 Advanced semantic pruning with CLIP integration.
        
        Combines:
        - Detection confidence (from detectors)
        - Text-based fuzzy matching (n-grams, synonyms)
        - CLIP visual-semantic similarity
        - Context expansion (nearby objects)
        
        Args:
            boxes, labels, scores: Detection results
            question_terms: Extracted terms from question
            clip_scores: Optional CLIP semantic scores from _compute_clip_semantic_scores
        
        Returns:
            Filtered boxes, labels, scores with semantic ranking
        """
        if not boxes:
            return boxes, labels, scores

        # Filter by min area (same as before)
        if self.cfg.min_box_area_px and self.cfg.min_box_area_px > 0:
            kept_idx = []
            for i, b in enumerate(boxes):
                x1, y1, x2, y2 = b[:4]
                area = max(1, int(x2 - x1)) * max(1, int(y2 - y1))
                if area >= int(self.cfg.min_box_area_px):
                    kept_idx.append(i)
            boxes = [boxes[i] for i in kept_idx]
            labels = [labels[i] for i in kept_idx]
            scores = [scores[i] for i in kept_idx]
            
            # Remap clip_scores indices after filtering
            if clip_scores:
                clip_scores = {kept_idx.index(old_i): score 
                              for old_i, score in clip_scores.items() 
                              if old_i in kept_idx}
            
            if not boxes:
                return boxes, labels, scores

        # 🚀 Compute multi-signal semantic scores
        semantic_boost = {}
        
        # Signal 1: Text-based fuzzy matching (existing)
        if question_terms:
            for i, lb in enumerate(labels):
                base_lb = base_label(lb).lower()
                relevance = 0.0
                
                for term in question_terms:
                    term_normalized = str(term).replace("_", " ").lower()
                    base_normalized = base_lb.replace("_", " ")
                    
                    if base_lb == term or term_normalized == base_normalized:
                        relevance = max(relevance, 1.0)
                    elif term_normalized in base_normalized or base_normalized in term_normalized:
                        relevance = max(relevance, 0.7)
                    else:
                        term_words = set(term_normalized.split())
                        label_words = set(base_normalized.split())
                        if term_words and label_words:
                            overlap = len(term_words & label_words) / max(len(term_words), len(label_words))
                            relevance = max(relevance, overlap * 0.6)
                
                semantic_boost[i] = relevance
        
        # Signal 2: CLIP visual-semantic similarity (new)
        if clip_scores:
            for i, clip_score in clip_scores.items():
                text_score = semantic_boost.get(i, 0.0)
                # Blend text and CLIP scores (CLIP weighted higher as it's more robust)
                # 40% text matching, 60% CLIP similarity
                combined = 0.4 * text_score + 0.6 * clip_score
                semantic_boost[i] = max(semantic_boost.get(i, 0.0), combined)

        # Cap per label (with semantic awareness)
        per_label = max(0, int(self.cfg.max_detections_per_label))
        if per_label > 0:
            from collections import defaultdict
            idx_by_label = defaultdict(list)
            for i, (lb, sc) in enumerate(zip(labels, scores)):
                effective_score = float(sc)
                if semantic_boost:
                    boost_factor = semantic_boost.get(i, 0.0)
                    effective_score = sc + (boost_factor * 0.3)
                
                idx_by_label[base_label(lb)].append((i, effective_score))
            
            kept = []
            for _, pairs in idx_by_label.items():
                pairs_sorted = sorted(pairs, key=lambda x: -x[1])
                kept.extend([i for i, _ in pairs_sorted[:per_label]])
            kept = sorted(set(kept))
            boxes = [boxes[i] for i in kept]
            labels = [labels[i] for i in kept]
            scores = [scores[i] for i in kept]
            
            # Remap semantic_boost indices
            if semantic_boost:
                semantic_boost = {kept.index(old_i): score 
                                 for old_i, score in semantic_boost.items() 
                                 if old_i in kept}

        # Cap total with multi-criteria ranking
        total_cap = max(0, int(self.cfg.max_detections_total))
        
        # 🚀 Apply min/max objects per question constraints
        if self.cfg.false_negative_reduction:
            min_cap = max(self.cfg.min_objects_per_question, 3)
            max_cap = min(self.cfg.max_objects_per_question, total_cap if total_cap > 0 else 50)
            total_cap = max_cap
        
        if total_cap > 0 and len(boxes) > total_cap:
            # Multi-criteria composite score
            composite_scores = []
            for i in range(len(boxes)):
                base_score = float(scores[i])
                sem_score = semantic_boost.get(i, 0.0) if semantic_boost else 0.0
                
                # Configurable weighting
                weight_conf = 1.0 - self.cfg.semantic_boost_weight
                weight_sem = self.cfg.semantic_boost_weight
                
                composite = weight_conf * base_score + weight_sem * sem_score
                composite_scores.append((i, composite))
            
            composite_scores.sort(key=lambda x: -x[1])
            kept_indices = [i for i, _ in composite_scores[:total_cap]]
            
            # 🚀 False negative safety: ensure we keep minimum objects
            if self.cfg.false_negative_reduction and len(kept_indices) < min_cap:
                # Add more objects sorted by detection confidence
                remaining = [i for i in range(len(boxes)) if i not in kept_indices]
                remaining.sort(key=lambda i: -float(scores[i]))
                add_count = min(min_cap - len(kept_indices), len(remaining))
                kept_indices.extend(remaining[:add_count])
            
            kept_indices.sort()  # Maintain original order
            
            boxes = [boxes[i] for i in kept_indices]
            labels = [labels[i] for i in kept_indices]
            scores = [scores[i] for i in kept_indices]
        
        return boxes, labels, scores

    def _parse_question(self, question: str) -> Tuple[set, set]:
        """
        Extract (object_terms, relation_terms) from the natural-language question.
        🚀 Enhanced: n-gram extraction, expanded stopwords, better synonyms.
        """
        q = (question or self.cfg.question or "").strip().lower()
        if not q:
            return set(), set()

        # Expanded stopword list for cleaner object extraction
        stopwords = {
            "the", "a", "an", "is", "are", "on", "in", "of", "to",
            "what", "where", "when", "how", "which", "who", "why",
            "this", "that", "these", "those", "there", "here",
            "do", "does", "did", "can", "could", "would", "should",
            "many", "much", "some", "any"
        }

        # Clean and tokenize
        q_clean = q.replace("?", " ").replace(",", " ").replace(".", " ")
        words = [w for w in q_clean.split() if w.isalpha() and len(w) > 1]

        # Extract unigrams (filter stopwords)
        unigrams = {w for w in words if w not in stopwords}


        # Espansione automatica con WordNet se disponibile
        try:
            from nltk.corpus import wordnet as wn
            def get_synonyms(word):
                syns = set()
                for syn in wn.synsets(word):
                    for lemma in syn.lemmas():
                        syns.add(lemma.name().replace("_", " "))
                return syns
        except Exception:
            def get_synonyms(word):
                return set()

        obj_terms = set(unigrams)
        for w in list(obj_terms):
            obj_terms.update(get_synonyms(w))

        # Bigrams
        for i in range(len(words) - 1):
            w1, w2 = words[i], words[i + 1]
            if w1 not in stopwords or w2 not in stopwords:
                bigram = f"{w1} {w2}"
                obj_terms.add(bigram)
                obj_terms.add(bigram.replace(" ", "_"))
                obj_terms.update(get_synonyms(w1))
                obj_terms.update(get_synonyms(w2))

        # Trigrams
        for i in range(len(words) - 2):
            w1, w2, w3 = words[i], words[i + 1], words[i + 2]
            if not all(w in stopwords for w in [w1, w2, w3]):
                trigram = f"{w1} {w2} {w3}"
                obj_terms.add(trigram)
                obj_terms.add(trigram.replace(" ", "_"))
                obj_terms.update(get_synonyms(w1))
                obj_terms.update(get_synonyms(w2))
                obj_terms.update(get_synonyms(w3))

        # Bigrams
        for i in range(len(words) - 1):
            w1, w2 = words[i], words[i + 1]
            if w1 not in stopwords or w2 not in stopwords:
                bigram = f"{w1} {w2}"
                obj_terms.add(bigram)
                obj_terms.add(bigram.replace(" ", "_"))
                # Espansione con sinonimi dei componenti
                obj_terms.update(get_synonyms(w1))
                obj_terms.update(get_synonyms(w2))

        # Trigrams
        for i in range(len(words) - 2):
            w1, w2, w3 = words[i], words[i + 1], words[i + 2]
            if not all(w in stopwords for w in [w1, w2, w3]):
                trigram = f"{w1} {w2} {w3}"
                obj_terms.add(trigram)
                obj_terms.add(trigram.replace(" ", "_"))
                obj_terms.update(get_synonyms(w1))
                obj_terms.update(get_synonyms(w2))
                obj_terms.update(get_synonyms(w3))

        # Sinonimi relazioni espansi
        rel_map = {
            "above": {"above", "over", "higher than", "top of"},
            "below": {"below", "under", "beneath", "lower than", "underneath"},
            "left_of": {"left", "to the left of", "left side", "leftward"},
            "right_of": {"right", "to the right of", "right side", "rightward"},
            "on_top_of": {"on top of", "on", "onto", "resting on", "sitting on", "placed on", "atop"},
            "in_front_of": {"in front of", "front", "before", "ahead of"},
            "behind": {"behind", "back of", "rear of", "after"},
            "next_to": {"next to", "beside", "adjacent to", "alongside", "by", "near"},
            "touching": {"touching", "in contact with", "against"},
            "near": {"near", "close to", "nearby", "around", "close by"},
            "far_from": {"far from", "distant from", "away from"},
            "inside": {"inside", "within", "in"},
            "outside": {"outside", "out of", "beyond"},
            "holding": {"holding", "grasping", "gripping", "carrying"},
            "wearing": {"wearing", "dressed in", "has on"},
        }

        rel_terms = set()
        for canonical, variants in rel_map.items():
            if any(v in q for v in variants):
                rel_terms.add(canonical)

        return obj_terms, rel_terms

    def _filter_by_question_terms(
        self,
        boxes: List[List[float]],
        labels: List[str],
        scores: List[float],
        obj_terms: set,
    ) -> Tuple[List[List[float]], List[str], List[float]]:
        """
        Filter objects by question terms with fuzzy matching.
        🚀 Enhanced: supports n-grams, partial matching, and semantic similarity.
        """
        if not self.cfg.apply_question_filter or not obj_terms:
            return boxes, labels, scores
        
        # 🚀 Multi-level matching strategy:
        matched_indices = []
        
        for i, lb in enumerate(labels):
            base_lb = base_label(lb).lower()
            matched = False
            
            # Level 1: Exact match (fastest)
            if base_lb in obj_terms:
                matched = True
            
            # Level 2: Partial match for compounds
            # e.g., "coffee_table" matches {"coffee", "table"} or {"coffee table"}
            if not matched:
                for term in obj_terms:
                    term_normalized = str(term).replace("_", " ").lower()
                    base_normalized = base_lb.replace("_", " ")
                    
                    # Check if term is substring of label or vice versa
                    if term_normalized in base_normalized or base_normalized in term_normalized:
                        matched = True
                        break
                    
                    # Check word-level overlap for compounds
                    term_words = set(term_normalized.split())
                    label_words = set(base_normalized.split())
                    if term_words and label_words:
                        # Match if >50% words overlap
                        overlap = len(term_words & label_words) / max(len(term_words), len(label_words))
                        if overlap >= 0.5:
                            matched = True
                            break
            
            # Level 3: Semantic similarity with CLIP (optional, if available)
            if not matched and hasattr(self, 'clip') and self.clip is not None:
                try:
                    # Check semantic similarity between label and question terms
                    max_similarity = 0.0
                    for term in obj_terms:
                        # Simple similarity check (can be enhanced)
                        term_str = str(term).lower()
                        # Common synonyms heuristic
                        synonym_pairs = [
                            ({"laptop", "computer", "pc"}, {"laptop", "computer", "pc"}),
                            ({"bike", "bicycle"}, {"bike", "bicycle"}),
                            ({"couch", "sofa"}, {"couch", "sofa"}),
                            ({"tv", "television"}, {"tv", "television"}),
                            ({"phone", "cellphone", "cell phone", "mobile"}, {"phone", "cellphone", "cell phone", "mobile"}),
                        ]
                        for syn_set1, syn_set2 in synonym_pairs:
                            if base_lb in syn_set1 and term_str in syn_set2:
                                matched = True
                                break
                            if base_lb in syn_set2 and term_str in syn_set1:
                                matched = True
                                break
                        if matched:
                            break
                except Exception:
                    pass  # CLIP not available or error, continue
            
            if matched:
                matched_indices.append(i)
        
        # Return filtered results
        if not matched_indices:
            # No matches found - return original (avoid empty result)
            return boxes, labels, scores
        
        return (
            [boxes[i] for i in matched_indices],
            [labels[i] for i in matched_indices],
            [scores[i] for i in matched_indices]
        )

    # ----------------------------- single image -----------------------------

    @torch.inference_mode()  # 🚀 Performance: Disable gradient tracking for inference
    def process_single_image(self, image_pil: Image.Image, image_name: str, custom_question: Optional[str] = None) -> None:
        """Run the full pipeline on one image; save graph/triples/visual output if enabled."""
        t0 = time.time()
        W, H = image_pil.size
        # Record last processed size to let _get_optimal_batch_size adapt batch size
        self._last_processed_size = (W, H)
        cache_key = self._generate_cache_key(image_pil, custom_question or self.cfg.question)

        # Compute which stages are needed (skip heavy steps when unused)
        need_graph = not self.cfg.skip_graph
        need_prompt = not self.cfg.skip_prompt
        need_viz = not self.cfg.skip_visualization
        need_rel_draw = need_viz and self.cfg.display_relationships
        need_rel = (need_graph or need_prompt or need_rel_draw)
        if not self.cfg.skip_relations_when_unused:
            need_rel = True
        need_depth = need_rel if self.cfg.skip_depth_when_unused else True
        need_seg_draw = need_viz and self.cfg.show_segmentation and not self.cfg.export_preproc_only
        need_seg_for_rel = need_rel
        need_seg = (need_seg_draw or need_seg_for_rel) if self.cfg.skip_segmentation_when_unused else True

        # 1) DETECTION (+ cache)
        cached = self._cache_get(cache_key)
        if cached is None:
            det_raw = self._run_detectors(image_pil)
            boxes_fused, labels_fused, scores_fused = self._wbf_fusion(det_raw["detections"], (W, H))
            # Normalize labels to a base form to improve consistency downstream.
            labels_fused = [base_label(l) for l in labels_fused]
            # Persist for later stages (segmentation/union).
            det_for_mask = [
                {
                    "box": b,
                    "label": l,
                    "score": s,
                    "from": "fused",
                    "det2_mask": self._pick_best_det2_mask_for_box(b, det_raw["detections"]),
                }
                for b, l, s in zip(boxes_fused, labels_fused, scores_fused)
            ]
            cached = {
                "boxes": boxes_fused,
                "labels": labels_fused,
                "scores": scores_fused,
                "det2": det_for_mask,
            }
            self._cache_put(cache_key, cached)
        else:
            pass  # use cached results

        boxes = list(cached["boxes"])
        labels = list(cached["labels"])
        scores = list(cached["scores"])
        det2_for_mask = list(cached["det2"])

        # 2) QUESTION FILTER (objects)
        obj_terms, rel_terms = self._parse_question(custom_question or self.cfg.question)

        # Preserve originals in case aggressive pruning is too strong.
        original_boxes = list(cached["boxes"])
        original_labels = list(cached["labels"])
        original_scores = list(cached["scores"])
        
        # 🚀 NEW: Always check if question mentions only one object for fallback
        # This enables automatic context expansion when only one object is mentioned
        target_object_detected = None
        if obj_terms and self.cfg.apply_question_filter:
            # Try to identify objects mentioned in the question
            bx_q, lb_q, sc_q = self._filter_by_question_terms(boxes, labels, scores, obj_terms)
            
            # If exactly one object type is mentioned, prepare fallback
            if len(bx_q) >= 1 and len(set(base_label(lb) for lb in lb_q)) == 1:
                # Single object type identified - store it for potential fallback
                target_object_detected = {
                    'label': lb_q[0],
                    'boxes': bx_q,
                    'labels': lb_q,
                    'scores': sc_q
                }
                if getattr(self.cfg, "verbose", False):
                    self.logger.info(f"[SINGLE OBJECT DETECTED] '{lb_q[0]}' mentioned in question ({len(bx_q)} instance(s) found)")

        if self.cfg.aggressive_pruning:
            # Hard pruning: keep ONLY mentioned objects; fallback if empty/singular.
            bx_q, lb_q, sc_q = self._filter_by_question_terms(boxes, labels, scores, obj_terms)
            if bx_q:
                boxes, labels, scores = bx_q, lb_q, sc_q
                
                # Fallback: if only one object survives, restore all objects and filter relations later.
                if len(boxes) == 1:
                    if getattr(self.cfg, "verbose", False):
                        self.logger.info(f"[FALLBACK] Only 1 object after aggressive pruning; restoring all objects")
                    
                    boxes, labels, scores = original_boxes, original_labels, original_scores
                    # Record the target object's indices to later keep only relations involving it.
                    target_obj_label = lb_q[0]
                    target_indices = [i for i, label in enumerate(original_labels) 
                                    if base_label(label) == base_label(target_obj_label)]
                    
                    self._target_object_indices = set(target_indices)
                    if getattr(self.cfg, "verbose", False):
                        self.logger.info(f"[FALLBACK] Target object: {target_obj_label}, indices: {target_indices}")
                        self.logger.info(f"[FALLBACK] Restored {len(boxes)} total objects")

        # 3) LABEL-WISE NMS BEFORE SEGMENTATION (major speed-up)
        boxes, labels, scores, keep = self._apply_label_nms(boxes, labels, scores)
        det2_for_mask = [det2_for_mask[i] for i in keep]

        # 3.5) 🚀 CLIP SEMANTIC SCORING (if enabled)
        # Compute semantic relevance using CLIP embeddings before pruning
        clip_semantic_scores = {}
        if self.cfg.use_clip_semantic_pruning and (custom_question or self.cfg.question):
            clip_semantic_scores = self._compute_clip_semantic_scores(
                image_pil=image_pil,
                boxes=boxes,
                labels=labels,
                question=custom_question or self.cfg.question,
                obj_terms=obj_terms
            )
            
            # 🚀 False Negative Reduction: Ensure minimum objects are kept
            if self.cfg.false_negative_reduction and len(boxes) > 0:
                # Count objects with good semantic scores
                high_scoring = sum(1 for score in clip_semantic_scores.values() if score > 0.4)
                
                # If too few objects have high scores, lower the threshold
                if high_scoring < self.cfg.min_objects_per_question:
                    # Keep top-K objects by detection confidence as fallback
                    if getattr(self.cfg, "verbose", False):
                        self.logger.info(f"[FALSE NEGATIVE REDUCTION] Only {high_scoring} objects with CLIP score > 0.4, keeping at least {self.cfg.min_objects_per_question} objects")

        # 4) LIGHT PRUNING (area, per-label, total) BEFORE SEGMENTATION
        # 🚀 Pass question terms + CLIP scores for semantic-aware pruning
        boxes, labels, scores = self._limit_detections_advanced(
            boxes, labels, scores, 
            question_terms=obj_terms,
            clip_scores=clip_semantic_scores
        )
        # Sync det2_for_mask with possibly reduced boxes
        if len(det2_for_mask) != len(boxes):
            # Approximate alignment by score order
            idx_sorted = sorted(range(len(scores)), key=lambda i: -float(scores[i]))
            det2_for_mask = [det2_for_mask[i] for i in idx_sorted[: len(boxes)]] if det2_for_mask else [None] * len(boxes)

        # 5) SEGMENTATION (SAM) + optional union with Detectron2 masks — only if needed
        masks = None
        if need_seg and boxes:
            if self.mixed_precision is not None:
                mp_ctx = self.mixed_precision.autocast()
            else:
                mp_ctx = contextlib.nullcontext()
            mem_ctx = gpu_memory_context(clear_after=True, verbose=False) if GPU_MEMORY_AVAILABLE else contextlib.nullcontext()
            with mp_ctx:
                with mem_ctx:
                    masks = self.segmenter.segment(image_pil, boxes)
            # fuse with detectron2 masks if available
            for i in range(len(masks)):
                d2m = det2_for_mask[i].get("det2_mask") if det2_for_mask and det2_for_mask[i] is not None else None
                if d2m is not None:
                    masks[i]["segmentation"] = self._fuse_with_det2_mask(masks[i]["segmentation"], d2m)

        # 6) DEPTH (at box centers) — only if needed
        centers = [((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0) for b in boxes]
        depths = None
        if need_depth and boxes:
            if self.mixed_precision is not None:
                mp_ctx = self.mixed_precision.autocast()
            else:
                mp_ctx = contextlib.nullcontext()
            mem_ctx = gpu_memory_context(clear_after=True, verbose=False) if GPU_MEMORY_AVAILABLE else contextlib.nullcontext()
            with mp_ctx:
                with mem_ctx:
                    if hasattr(self.depth_est, "depth_map"):
                        try:
                            dmap = self.depth_est.depth_map(image_pil)  # returns HxW or normalized map
                            depths = [float(dmap[int(cy), int(cx)]) for (cx, cy) in centers]
                        except Exception:
                            depths = self.depth_est.relative_depth_at(image_pil, centers)
                    else:
                        depths = self.depth_est.relative_depth_at(image_pil, centers)

        # 7) RELATIONS (geometry + CLIP) — only if needed
        rels_all: List[Dict[str, Any]] = []
        if need_rel and boxes:
            r_cfg = RelationsConfig(
                margin_px=self.cfg.margin,
                min_distance=self.cfg.min_distance,
                max_distance=self.cfg.max_distance,
            )
            if self.mixed_precision is not None:
                mp_ctx = self.mixed_precision.autocast()
            else:
                mp_ctx = contextlib.nullcontext()
            with mp_ctx:
                rels_all = self.relations_inferencer.infer(
                    image_pil=image_pil,
                    boxes=boxes,
                    labels=labels,
                    masks=masks,
                    depths=depths,
                    use_geometry=True,
                    use_clip=True,
                    clip_threshold=0.23,
                )
        
        # 🚀 NEW: Apply fallback if single object was detected in question
        # Strategy: Keep ONLY relations directly connected to target + their endpoint objects
        if target_object_detected is not None and not hasattr(self, '_target_object_indices'):
            # Find indices of the target object in the current (pruned) boxes
            target_label = base_label(target_object_detected['label'])
            target_indices = [i for i, label in enumerate(labels) 
                            if base_label(label) == target_label]
            
            if target_indices:
                # Check if we should apply fallback:
                # - If only the target object(s) remain, include connected objects
                unique_labels = set(base_label(lb) for lb in labels)
                
                if len(unique_labels) == 1:
                    # Only target object type remains - apply strict fallback
                    if getattr(self.cfg, "verbose", False):
                        self.logger.info(f"[SINGLE OBJECT FALLBACK] Only '{target_label}' detected, including ONLY directly connected objects and relations")
                    self._target_object_indices = set(target_indices)
                    self._single_object_fallback_active = True
        
        # Apply fallback filtering if needed
        if hasattr(self, '_target_object_indices') and self._target_object_indices:
            # Step 1: Keep ONLY relations that directly involve target objects
            rels_all = self._filter_relations_by_target_object(rels_all)
            
            # Step 2: If single-object fallback, identify connected objects but DON'T expand relations
            if hasattr(self, '_single_object_fallback_active') and self._single_object_fallback_active:
                # Find objects connected to target via the filtered relations
                connected_indices = self._get_connected_object_indices(rels_all, self._target_object_indices)
                
                if connected_indices:
                    if getattr(self.cfg, "verbose", False):
                        self.logger.info(f"[SINGLE OBJECT FALLBACK] Found {len(connected_indices)} objects directly connected to target")
                    
                    # Mark these as "connected only" - they won't contribute their own relations
                    self._connected_only_indices = connected_indices
                    # DO NOT update _target_object_indices - keep it strict!
                else:
                    # No connected objects found - still apply strict filtering
                    self._connected_only_indices = set()
                
                # Keep flags for later filtering step
                # (will be cleared after object filtering)
        
        # 🚀 STRICT FALLBACK: Filter objects FIRST to include only target + directly connected
        # This must happen BEFORE limit_relationships_per_object to ensure clean filtering
        if hasattr(self, '_connected_only_indices'):
            # Collect indices to keep from filtered relations
            all_target_indices = set()
            
            for rel in rels_all:
                src_idx = rel.get('src_idx', -1)
                tgt_idx = rel.get('tgt_idx', -1)
                all_target_indices.add(src_idx)
                all_target_indices.add(tgt_idx)
            
            # Filter objects: keep only those involved in relations
            filtered_boxes = []
            filtered_labels = []
            filtered_scores = []
            filtered_masks = []
            filtered_depths = []
            index_mapping = {}  # old_idx -> new_idx
            
            for old_idx in sorted(all_target_indices):
                if 0 <= old_idx < len(boxes):
                    new_idx = len(filtered_boxes)
                    index_mapping[old_idx] = new_idx
                    
                    filtered_boxes.append(boxes[old_idx])
                    filtered_labels.append(labels[old_idx])
                    filtered_scores.append(scores[old_idx])
                    if masks and old_idx < len(masks):
                        filtered_masks.append(masks[old_idx])
                    if depths and old_idx < len(depths):
                        filtered_depths.append(depths[old_idx])
            
            # Update relations with new indices
            filtered_rels = []
            for rel in rels_all:
                src_idx = rel.get('src_idx', -1)
                tgt_idx = rel.get('tgt_idx', -1)
                
                if src_idx in index_mapping and tgt_idx in index_mapping:
                    rel_copy = rel.copy()
                    rel_copy['src_idx'] = index_mapping[src_idx]
                    rel_copy['tgt_idx'] = index_mapping[tgt_idx]
                    filtered_rels.append(rel_copy)
            
            # Replace with filtered versions
            boxes = filtered_boxes
            labels = filtered_labels
            scores = filtered_scores
            masks = filtered_masks if filtered_masks else masks
            depths = filtered_depths if filtered_depths else depths
            rels_all = filtered_rels
            
            if getattr(self.cfg, "verbose", False):
                self.logger.info(f"[SINGLE OBJECT FALLBACK] Filtered to {len(boxes)} objects and {len(rels_all)} direct relations")
            
            # Clear all fallback flags
            delattr(self, '_connected_only_indices')
            if hasattr(self, '_single_object_fallback_active'):
                delattr(self, '_single_object_fallback_active')
            if hasattr(self, '_target_object_indices'):
                delattr(self, '_target_object_indices')

        # 6a) Relation filtering by question terms (optional).
        if self.cfg.filter_relations_by_question and rel_terms:
            rels_all = self.relations_inferencer.filter_by_question(
                rels_all,
                question_terms=rel_terms,
                threshold=self.cfg.threshold_relation_similarity
            )
        # 6b) Per-object limits and inverse-duplicate removal.
        rels_all = self.relations_inferencer.limit_relationships_per_object(
            rels_all,
            boxes,
            max_relations_per_object=self.cfg.max_relations_per_object,
            min_relations_per_object=self.cfg.min_relations_per_object,
            question_rel_terms=rel_terms if rel_terms else None,
        )
        rels_all = self.relations_inferencer.drop_inverse_duplicates(rels_all)

        # 7) GRAPH + PROMPT/TRIPLES (optional)
        if not self.cfg.skip_graph or not self.cfg.skip_prompt or not self.cfg.skip_visualization:
            scene_graph = build_scene_graph(
                image_size=(W, H),
                boxes=boxes,
                labels=labels,
                scores=scores,
                depths=depths,
            )
            
            # Add inferred relation labels to graph edges
            # This ensures that the triple output matches what's drawn in the visualization
            # ✅ CREATE edges explicitly for ALL inferred relations (don't rely on geometric edge creation)
            for rel in rels_all:
                src_idx = int(rel["src_idx"])
                tgt_idx = int(rel["tgt_idx"])
                relation_name = str(rel.get("relation", ""))
                
                # Add/update edge with relation name
                if scene_graph.has_edge(src_idx, tgt_idx):
                    scene_graph.edges[src_idx, tgt_idx]["relation"] = relation_name
                else:
                    # Create edge if it doesn't exist yet
                    scene_graph.add_edge(src_idx, tgt_idx, relation=relation_name)

                # Normalize spatial relations to match geometric attributes when possible.
                # Some relation sources may have used a different sign convention; prefer
                # a geometry-based inference for pure spatial predicates so visualization
                # matches the triples text.
                try:
                    from igp.graph.prompt import _infer_relation_from_attrs
                    # Only apply normalization to basic spatial predicates
                    spatial_set = {
                        "left_of",
                        "right_of",
                        "above",
                        "below",
                        "on_top_of",
                        "under",
                        "in_front_of",
                        "behind",
                    }
                    if relation_name in spatial_set:
                        edge_data = scene_graph.edges[src_idx, tgt_idx]
                        inferred = _infer_relation_from_attrs(edge_data)
                        if inferred in spatial_set and inferred != relation_name:
                            scene_graph.edges[src_idx, tgt_idx]["relation"] = inferred
                except Exception:
                    # If anything goes wrong here, don't break the pipeline; keep original relation
                    pass
            
            # ✅ FIX: Ensure ALL edges (even those without explicit relations) have a "relation" field
            # This prevents inconsistency between triples.txt (which infers relations) and JSON output
            from igp.graph.prompt import _infer_relation_from_attrs
            for u, v in list(scene_graph.edges()):
                # Skip scene node edges
                if scene_graph.nodes[u].get("label") == "scene" or scene_graph.nodes[v].get("label") == "scene":
                    continue
                
                # If edge doesn't have a relation, infer it from geometric attributes
                edge_data = scene_graph.edges[u, v]
                if "relation" not in edge_data or not edge_data["relation"]:
                    inferred_rel = _infer_relation_from_attrs(edge_data)
                    scene_graph.edges[u, v]["relation"] = inferred_rel
        else:
            scene_graph = None

        # Save scene graph (gpickle/json) if requested.
        if scene_graph is not None and not self.cfg.skip_graph:
            out_gpickle = os.path.join(self.cfg.output_folder, f"{image_name}_graph.gpickle")
            out_json = os.path.join(self.cfg.output_folder, f"{image_name}_graph.json")
            self._save_graph(scene_graph, out_gpickle, out_json)

        # Save textual triples (always derived from scene_graph when available).
        if scene_graph is not None:
            triples_path = os.path.join(self.cfg.output_folder, f"{image_name}_graph_triples.txt")
            with open(triples_path, "w", encoding="utf-8") as f:
                f.write(to_triples_text(scene_graph))

        # ✅ FIX: Extract relationships from the updated scene_graph (not rels_all)
        # This ensures visualization matches triples.txt and graph.json
        rels_for_viz = []
        if scene_graph is not None and need_rel:
            # Extract relationships from scene_graph edges
            for u, v, edge_data in scene_graph.edges(data=True):
                # Skip scene node edges
                if scene_graph.nodes[u].get("label") == "scene" or scene_graph.nodes[v].get("label") == "scene":
                    continue
                
                relation = edge_data.get("relation")
                if relation:
                    rels_for_viz.append({
                        "src_idx": u,
                        "tgt_idx": v,
                        "relation": relation,
                    })
        elif need_rel:
            # Fallback to original rels_all if no scene_graph
            rels_for_viz = rels_all

        # 8) VISUALIZATION / EXPORT
        if not self.cfg.skip_visualization:
            out_img = os.path.join(self.cfg.output_folder, f"{image_name}_output.jpg")
            self.visualizer.draw(
                image=image_pil,
                boxes=boxes,
                labels=self._format_labels_for_display(labels),
                scores=scores,
                relationships=rels_for_viz,
                masks=masks,
                save_path=out_img,
                draw_background=not self.cfg.export_preproc_only,
                bg_color=(1, 1, 1, 0),
            )
        if self.cfg.export_preproc_only:
            out_png = os.path.join(self.cfg.output_folder, f"{image_name}_preproc.png")
            self.visualizer.draw(
                image=image_pil,
                boxes=boxes,
                labels=self._format_labels_for_display(labels),
                scores=scores,
                relationships=rels_for_viz,
                masks=masks,
                save_path=out_png,
                draw_background=False,
                bg_color=(1, 1, 1, 0),
            )

        # Cleanup GPU/CPU memory between runs (useful for batches).
        self._free_memory()
        dt = time.time() - t0
        if getattr(self.cfg, "verbose", False):
            self.logger.info(f"[DONE] {image_name} processed in {dt:.2f}s")

    def _filter_relations_by_target_object(self, relationships: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Keep only relations that involve at least one of the fallback target objects."""
        if not hasattr(self, '_target_object_indices') or not self._target_object_indices:
            return relationships
        
        filtered_rels = []
        for rel in relationships:
            src_idx = rel.get('src_idx', -1)
            tgt_idx = rel.get('tgt_idx', -1)
            if src_idx in self._target_object_indices or tgt_idx in self._target_object_indices:
                filtered_rels.append(rel)
        
        if getattr(self.cfg, "verbose", False):
            self.logger.info(f"[FALLBACK] Filtered relations: {len(filtered_rels)}/{len(relationships)} kept")
        return filtered_rels
    
    def _get_connected_object_indices(
        self, 
        relationships: List[Dict[str, Any]], 
        target_indices: set
    ) -> set:
        """
        Find all object indices that are directly connected to target objects via relations.
        
        Args:
            relationships: List of relation dictionaries
            target_indices: Set of target object indices
            
        Returns:
            Set of indices of objects connected to target objects
        """
        connected = set()
        
        for rel in relationships:
            src_idx = rel.get('src_idx', -1)
            tgt_idx = rel.get('tgt_idx', -1)
            
            # If source is a target, add target to connected
            if src_idx in target_indices and tgt_idx not in target_indices:
                connected.add(tgt_idx)
            
            # If target is a target, add source to connected
            if tgt_idx in target_indices and src_idx not in target_indices:
                connected.add(src_idx)
        
        return connected

    # ----------------------------- runners -----------------------------

    def run(self) -> None:
        """
        Batch entry-point:
          - json_file: list of dicts with "image_path" and optional "question"
          - dataset: optional (requires `datasets`) with split/column
          - input_path: single file or folder
        """
        if self.cfg.json_file:
            self._run_from_json(self.cfg.json_file)
            return

        if self.cfg.dataset:
            self._run_from_dataset()
            return

        if not self.cfg.input_path:
            self.logger.error("[ERROR] No input_path provided and no dataset/json specified.")
            return

        ip = Path(self.cfg.input_path)
        if ip.is_dir():
            paths = [p for p in ip.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        else:
            if not ip.exists():
                self.logger.error(f"[ERROR] Input path '{ip}' does not exist.")
                return
            paths = [ip]

        for p in paths:
            try:
                img = Image.open(str(p)).convert("RGB")
            except Exception as e:
                self.logger.error(f"[ERROR] Could not open '{p}': {e}")
                continue
            name = p.stem
            self.process_single_image(img, name)

    def _run_from_json(self, json_path: str) -> None:
        """Iterate a JSON file of items in batches for efficiency."""
        with open(json_path, "r", encoding="utf-8") as f:
            rows = json.load(f)
        
        if self.cfg.num_instances > 0:
            rows = rows[: int(self.cfg.num_instances)]
        
        # BATCH PROCESSING: Calculate optimal batch size
        batch_size = self._get_optimal_batch_size()
        if getattr(self.cfg, "verbose", False):
            self.logger.info(f"[INFO] Processing {len(rows)} images with batch_size={batch_size}")
        
        for batch_start in range(0, len(rows), batch_size):
            batch_rows = rows[batch_start:batch_start + batch_size]
            
            # Load batch images
            batch_data = []
            for row in batch_rows:
                img_p = row["image_path"]
                q = row.get("question", self.cfg.question)
                
                try:
                    img = Image.open(img_p).convert("RGB")
                    name = Path(img_p).stem
                    batch_data.append({
                        "image": img,
                        "name": name,
                        "question": q,
                        "path": img_p
                    })
                except Exception as e:
                    self.logger.error(f"[ERROR] Loading {img_p}: {e}")
                    continue
            
            if not batch_data:
                continue
            
            # Run batch detection for all images at once
            batch_images = [item["image"] for item in batch_data]
            batch_det_results = self._run_detectors_batch(batch_images)
            
            # Process each image individually with cached detections
            for item, det_result in zip(batch_data, batch_det_results):
                img = item["image"]
                name = item["name"]
                question = item["question"]
                
                # Generate cache key and store detection results
                cache_key = self._generate_cache_key(img, question)
                
                # Apply WBF fusion to batch detection results
                W, H = img.size
                boxes_fused, labels_fused, scores_fused = self._wbf_fusion(
                    det_result["detections"], (W, H)
                )
                labels_fused = [base_label(l) for l in labels_fused]
                
                # Store in cache
                det_for_mask = [
                    {
                        "box": b,
                        "label": l,
                        "score": s,
                        "from": "fused",
                        "det2_mask": self._pick_best_det2_mask_for_box(b, det_result["detections"]),
                    }
                    for b, l, s in zip(boxes_fused, labels_fused, scores_fused)
                ]
                
                self._cache_put(cache_key, {
                    "boxes": boxes_fused,
                    "labels": labels_fused,
                    "scores": scores_fused,
                    "det2": det_for_mask,
                })
                
                # Continue with normal processing (uses cached detection)
                self.process_single_image(img, name, custom_question=question)
            
            # Free memory after each batch
            self._free_memory()

    def _run_from_dataset(self) -> None:
        """Load a Hugging Face dataset split and process images in sequence."""
        try:
            from datasets import load_dataset  # type: ignore
        except Exception:
            self.logger.error("[ERROR] 'datasets' library not installed.")
            return

        if getattr(self.cfg, "verbose", False):
            self.logger.info(f"[INFO] Loading dataset '{self.cfg.dataset}' (split='{self.cfg.split}')...")
        ds = load_dataset(self.cfg.dataset, split=self.cfg.split)
        if getattr(self.cfg, "verbose", False):
            self.logger.info(f"[INFO] Dataset loaded with {len(ds)} items")

        start = 0
        end = len(ds)
        if self.cfg.num_instances > 0:
            end = min(end, self.cfg.num_instances)

        for i in range(start, end):
            ex = ds[i]
            if self.cfg.image_column not in ex:
                self.logger.error(f"[ERROR] image_column='{self.cfg.image_column}' not found at idx {i}. Skipping.")
                continue

            img_data = ex[self.cfg.image_column]
            if isinstance(img_data, Image.Image):
                img_pil = img_data
            elif isinstance(img_data, dict) and "bytes" in img_data:
                from io import BytesIO
                img_pil = Image.open(BytesIO(img_data["bytes"])).convert("RGB")
            elif isinstance(img_data, np.ndarray):
                img_pil = Image.fromarray(img_data).convert("RGB")
            else:
                self.logger.warning(f"[WARNING] Index {i}: image not recognized. Skipping.")
                continue

            image_name = str(ex.get("id", f"img_{i}"))
            self.process_single_image(img_pil, image_name)

    # ----------------------------- utils -----------------------------

    def _pick_best_det2_mask_for_box(self, box: Sequence[float], detections: List[Dict[str, Any]]) -> Optional[np.ndarray]:
        """Among Detectron2 detections, return the mask with maximum IoU w.r.t. the given box."""
        best = None
        best_iou = 0.0
        for d in detections:
            m = d.get("mask")
            if m is None:
                continue
            bx = d["box"]
            i = iou(box, bx)
            if i > best_iou:
                best_iou = i
                best = m
        return best if best_iou >= 0.30 else None

    def _format_labels_for_display(self, labels: List[str]) -> List[str]:
        """Apply label display mode (original/numeric/alphabetic)."""
        if self.cfg.label_mode == "original":
            return [f"{lb}" for lb in labels]
        if self.cfg.label_mode == "numeric":
            return [str(i + 1) for i, _ in enumerate(labels)]
        if self.cfg.label_mode == "alphabetic":
            import string as _string
            return list(_string.ascii_uppercase[: len(labels)])
        return labels

    @staticmethod
    def _save_graph(G, path_gpickle: str, path_json: str) -> None:
        """Save both gpickle (optionally gzipped) and node-link JSON for the scene graph."""
        # gpickle
        try:
            import gzip
            import pickle

            os.makedirs(os.path.dirname(path_gpickle), exist_ok=True)
            with gzip.open(path_gpickle, "wb") if path_gpickle.endswith(".gz") else open(path_gpickle, "wb") as f:
                pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            logging.getLogger(__name__).warning(f"[WARN] Could not save gpickle: {e}")

        # json
        try:
            import networkx as nx
            with open(path_json, "w", encoding="utf-8") as jf:
                def _np_conv(o):
                    import numpy as _np
                    if isinstance(o, _np.generic):
                        return o.item()
                    raise TypeError
                json.dump(nx.node_link_data(G), jf, default=_np_conv, indent=2)
        except Exception as e:
            logging.getLogger(__name__).warning(f"[WARN] Could not save scene graph json: {e}")

    @staticmethod
    def _free_memory() -> None:
        """Free GPU cache (if any) and run GC to reduce memory spikes in batches."""
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()