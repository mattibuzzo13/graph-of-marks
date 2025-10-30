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
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------- detectors ----------
from igp.detectors.base import Detector
from igp.detectors.owlvit import OwlViTDetector
from igp.detectors.yolov8 import YOLOv8Detector
from igp.detectors.detectron2 import Detectron2Detector

# ---------- fusion ----------
from igp.fusion.wbf import fuse_detections_wbf as weighted_boxes_fusion
from igp.fusion.nms import labelwise_nms

# ---------- segmentation ----------
from igp.segmentation.base import Segmenter, SegmenterConfig
from igp.segmentation.sam1 import Sam1Segmenter
from igp.segmentation.sam2 import Sam2Segmenter
from igp.segmentation.samhq import SamHQSegmenter

# ---------- utils ----------
from igp.utils.depth import DepthEstimator, DepthConfig
from igp.utils.clip_utils import CLIPWrapper  
from igp.utils.boxes import iou, clamp_xyxy  
from igp.utils.colors import base_label

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

    # geometry (pixels)
    margin: int = 20
    min_distance: float = 50
    max_distance: float = 20000

    # SAM
    sam_version: str = "1"  # "1" | "2" | "hq"
    sam_hq_model_type: str = "vit_h"
    points_per_side: int = 32
    pred_iou_thresh: float = 0.90
    stability_score_thresh: float = 0.92
    min_mask_region_area: int = 100

    # device
    preproc_device: Optional[str] = None  # e.g., "cpu" or "cuda"

    # rendering toggles
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
        self.relations_inferencer = RelationInferencer(
            margin_px=config.margin,
            min_distance=config.min_distance,
            max_distance=config.max_distance
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

        # In-memory detection cache (per-image + per-params).
        self._detection_cache: Dict[str, Dict[str, Any]] = {}
        self._det_cache: Dict[str, Dict[str, Any]] = {}

    # ----------------------------- setup helpers -----------------------------

    def _init_detectors(self) -> List[Detector]:
        """Initialize enabled detectors according to config."""
        dets: List[Detector] = []
        names = set(d.strip().lower() for d in self.cfg.detectors_to_use)
        
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
        """Create the SAM segmenter variant with common post-processing flags."""
        s_cfg = SegmenterConfig(
            device=self.device,
            close_holes=self.cfg.close_holes,
            hole_kernel=self.cfg.hole_kernel,
            min_hole_area=self.cfg.min_hole_area,
        )
        if self.cfg.sam_version == "2":
            return Sam2Segmenter(config=s_cfg)
        if self.cfg.sam_version == "hq":
            return SamHQSegmenter(config=s_cfg, model_type=self.cfg.sam_hq_model_type)
        return Sam1Segmenter(config=s_cfg)

    # ----------------------------- cache helpers -----------------------------

    def _generate_cache_key(self, image_pil: Image.Image, question: str = "") -> str:
        """Hash the raw image bytes + relevant pipeline params to cache detections."""
        img_hash = hashlib.md5(image_pil.tobytes()).hexdigest()[:16]
        param_str = json.dumps(
            {
                "detectors": sorted(self.cfg.detectors_to_use),
                "owl_thr": self.cfg.threshold_owl,
                "yolo_thr": self.cfg.threshold_yolo,
                "det2_thr": self.cfg.threshold_detectron,
                "question": (question or self.cfg.question).strip().lower(),
            },
            sort_keys=True,
        )
        param_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]
        return f"{img_hash}_{param_hash}"

    def _cache_get(self, key: str) -> Optional[Dict[str, Any]]:
        """Read detection results from cache if enabled."""
        if not self.cfg.enable_detection_cache:
            return None
        return self._detection_cache.get(key)

    def _cache_put(self, key: str, value: Dict[str, Any]) -> None:
        """Write detection results to cache with simple capacity control."""
        if not self.cfg.enable_detection_cache:
            return
        if len(self._detection_cache) >= int(self.cfg.max_cache_size):
            oldest = next(iter(self._detection_cache))
            self._detection_cache.pop(oldest, None)
        self._detection_cache[key] = value

    # ----------------------------- pipeline core -----------------------------

    def _run_detectors(self, image_pil: Image.Image) -> Dict[str, Any]:
        """Run all detectors (in parallel if multiple) and return raw detections ready for fusion."""
        from concurrent.futures import ThreadPoolExecutor
        
        all_dets: List[Dict[str, Any]] = []
        counts: Dict[str, int] = {}
        
        def _run_detector(det):
            out = det.run(image_pil)
            src_name = det.__class__.__name__
            return src_name, out
        
        # Run detectors in parallel if multiple
        if len(self.detectors) > 1:
            with ThreadPoolExecutor(max_workers=len(self.detectors)) as executor:
                results = list(executor.map(_run_detector, self.detectors))
        else:
            results = [_run_detector(self.detectors[0])]
        
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
                    print(f"[WARN] Batch detection failed for {src_name}, falling back to sequential: {e}")
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

    def _parse_question(self, question: str) -> Tuple[set, set]:
        """
        Extract (object_terms, relation_terms) from the natural-language question.
        - Lightweight token filtering for objects (no spaCy dependency).
        - Basic synonym expansion for relations.
        """
        q = (question or self.cfg.question or "").strip().lower()
        if not q:
            return set(), set()

        # Object terms — simple alpha tokens minus common function words.
        tokens = [t for t in q.replace("?", " ").replace(",", " ").split() if t.isalpha()]
        obj_terms = {t for t in tokens if t not in {"the", "a", "an", "is", "are", "on", "in", "of", "to"}}

        # Relation terms (primitive synonym map).
        rel_map = {
            "above": {"above"},
            "below": {"below", "under"},
            "left_of": {"left", "to the left of"},
            "right_of": {"right", "to the right of"},
            "on_top_of": {"on top of", "on", "onto", "resting on", "sitting on"},
            "in_front_of": {"in front of"},
            "behind": {"behind"},
            "next_to": {"next to", "beside"},
            "touching": {"touching"},
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
        """If `obj_terms` is not empty, keep only objects whose base label matches."""
        if not self.cfg.apply_question_filter or not obj_terms:
            return boxes, labels, scores
        idxs = [i for i, lb in enumerate(labels) if base_label(lb) in obj_terms]
        if not idxs:
            return boxes, labels, scores
        return [boxes[i] for i in idxs], [labels[i] for i in idxs], [scores[i] for i in idxs]

    # ----------------------------- single image -----------------------------

    def process_single_image(self, image_pil: Image.Image, image_name: str, custom_question: Optional[str] = None) -> None:
        """Run the full pipeline on one image; save graph/triples/visual output if enabled."""
        t0 = time.time()
        W, H = image_pil.size
        # Record last processed size to let _get_optimal_batch_size adapt batch size
        self._last_processed_size = (W, H)
        cache_key = self._generate_cache_key(image_pil, custom_question or self.cfg.question)

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

        if self.cfg.aggressive_pruning:
            # Hard pruning: keep ONLY mentioned objects; fallback if empty/singular.
            bx_q, lb_q, sc_q = self._filter_by_question_terms(boxes, labels, scores, obj_terms)
            if bx_q:
                boxes, labels, scores = bx_q, lb_q, sc_q
                
                # Fallback: if only one object survives, restore all objects and filter relations later.
                if len(boxes) == 1:
                    print(f"[FALLBACK] Only 1 object after aggressive pruning; restoring all objects")
                    
                    boxes, labels, scores = original_boxes, original_labels, original_scores
                    # Record the target object's indices to later keep only relations involving it.
                    target_obj_label = lb_q[0]
                    target_indices = [i for i, label in enumerate(original_labels) 
                                    if base_label(label) == base_label(target_obj_label)]
                    
                    self._target_object_indices = set(target_indices)
                    print(f"[FALLBACK] Target object: {target_obj_label}, indices: {target_indices}")
                    print(f"[FALLBACK] Restored {len(boxes)} total objects")

        # 3) SEGMENTATION (SAM) + optional union with Detectron2 masks
        masks = self.segmenter.segment(image_pil, boxes)
        for i in range(len(masks)):
            d2m = det2_for_mask[i].get("det2_mask")
            if d2m is not None:
                masks[i]["segmentation"] = self._fuse_with_det2_mask(masks[i]["segmentation"], d2m)

        # 4) LABEL-WISE NMS
        boxes, labels, scores, keep = self._apply_label_nms(boxes, labels, scores)
        masks = [masks[i] for i in keep]

        # 5) DEPTH (at box centers)
        centers = [((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0) for b in boxes]
        # Prefer a single full-depth inference if available
        if hasattr(self.depth_est, "depth_map"):
            try:
                dmap = self.depth_est.depth_map(image_pil)  # returns HxW or normalized map
                depths = [float(dmap[int(cy), int(cx)]) for (cx, cy) in centers]
            except Exception:
                depths = self.depth_est.relative_depth_at(image_pil, centers)
        else:
            depths = self.depth_est.relative_depth_at(image_pil, centers)

        # 6) RELATIONS (geometry + CLIP)
        r_cfg = RelationsConfig(
            margin_px=self.cfg.margin,
            min_distance=self.cfg.min_distance,
            max_distance=self.cfg.max_distance,
        )
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
        
        # If fallback target was set, keep only relations involving those objects.
        if hasattr(self, '_target_object_indices') and self._target_object_indices:
            rels_all = self._filter_relations_by_target_object(rels_all)
            # Clear the flag after use.
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

        # 8) VISUALIZATION / EXPORT
        if not self.cfg.skip_visualization:
            out_img = os.path.join(self.cfg.output_folder, f"{image_name}_output.jpg")
            self.visualizer.draw(
                image=image_pil,
                boxes=boxes,
                labels=self._format_labels_for_display(labels),
                scores=scores,
                relationships=rels_all,
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
                relationships=rels_all,
                masks=masks,
                save_path=out_png,
                draw_background=False,
                bg_color=(1, 1, 1, 0),
            )

        # Cleanup GPU/CPU memory between runs (useful for batches).
        self._free_memory()
        dt = time.time() - t0
        print(f"[DONE] {image_name} processed in {dt:.2f}s")

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
        
        print(f"[FALLBACK] Filtered relations: {len(filtered_rels)}/{len(relationships)} kept")
        return filtered_rels

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
            print("[ERROR] No input_path provided and no dataset/json specified.")
            return

        ip = Path(self.cfg.input_path)
        if ip.is_dir():
            paths = [p for p in ip.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        else:
            if not ip.exists():
                print(f"[ERROR] Input path '{ip}' does not exist.")
                return
            paths = [ip]

        for p in paths:
            try:
                img = Image.open(str(p)).convert("RGB")
            except Exception as e:
                print(f"[ERROR] Could not open '{p}': {e}")
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
        print(f"[INFO] Processing {len(rows)} images with batch_size={batch_size}")
        
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
                    print(f"[ERROR] Loading {img_p}: {e}")
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
            print("[ERROR] 'datasets' library not installed.")
            return

        print(f"[INFO] Loading dataset '{self.cfg.dataset}' (split='{self.cfg.split}')...")
        ds = load_dataset(self.cfg.dataset, split=self.cfg.split)
        print(f"[INFO] Dataset loaded with {len(ds)} items")

        start = 0
        end = len(ds)
        if self.cfg.num_instances > 0:
            end = min(end, self.cfg.num_instances)

        for i in range(start, end):
            ex = ds[i]
            if self.cfg.image_column not in ex:
                print(f"[ERROR] image_column='{self.cfg.image_column}' not found at idx {i}. Skipping.")
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
                print(f"[WARNING] Index {i}: image not recognized. Skipping.")
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
            print(f"[WARN] Could not save gpickle: {e}")

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
            print(f"[WARN] Could not save scene graph json: {e}")

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