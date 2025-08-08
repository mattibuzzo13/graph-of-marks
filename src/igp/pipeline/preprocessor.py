# igp/pipeline/preprocessor.py
from __future__ import annotations

import gc
import hashlib
import json
import math
import networkx as nx
from PIL import Image
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

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
from igp.utils.caption import Captioner
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
    """Wrapper per SceneGraphBuilder.build()"""

    
    W, H = image_size
    # Crea un'immagine dummy per il builder
    image = Image.new('RGB', (W, H), color='white')
    
    builder = _SceneGraphBuilder()
    return builder.build(image, boxes, labels, scores)

# ✅ ALIAS per compatibilità con prompt.py
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

    # batching / dataset (opzionale)
    dataset: Optional[str] = None
    split: str = "train"
    image_column: str = "image"
    num_instances: int = -1

    # domanda / filtri
    question: str = ""
    apply_question_filter: bool = True
    aggressive_pruning: bool = False
    filter_relations_by_question: bool = True
    threshold_object_similarity: float = 0.50
    threshold_relation_similarity: float = 0.50

    # detectors & soglie
    detectors_to_use: Tuple[str, ...] = ("owlvit", "yolov8", "detectron2")
    threshold_owl: float = 0.40
    threshold_yolo: float = 0.80
    threshold_detectron: float = 0.80

    # relazioni per oggetto
    max_relations_per_object: int = 3
    min_relations_per_object: int = 1

    # NMS / fusion
    label_nms_threshold: float = 0.50
    seg_iou_threshold: float = 0.70

    # geometria
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
    preproc_device: Optional[str] = None  # es. "cpu" o "cuda"

    # visualizzazione
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

    # maschere (morph close dei buchi)
    close_holes: bool = False
    hole_kernel: int = 7
    min_hole_area: int = 100

    # export flags
    save_image_only: bool = False
    skip_graph: bool = False
    skip_prompt: bool = False
    skip_visualization: bool = False
    export_preproc_only: bool = False  # salva PNG trasparente con solo overlay

    # cache detection
    enable_detection_cache: bool = True
    max_cache_size: int = 100

    # colori
    color_sat_boost: float = 1.30
    color_val_boost: float = 1.15


# ----------------------------- Preprocessor -----------------------------
class ImageGraphPreprocessor:
    """
    Pipeline end-to-end:
      detect → fuse(NMS/WBF) → segment → depth → relate → graph → viz/export
    """

    def __init__(self, config: PreprocessorConfig) -> None:
        self.cfg = config
        os.makedirs(self.cfg.output_folder, exist_ok=True)

        # device
        if self.cfg.preproc_device:
            self.device = self.cfg.preproc_device
        else:
            try:
                import torch

                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                self.device = "cpu"

        # detectors
        self.detectors: List[Detector] = self._init_detectors()

        # segmenter
        self.segmenter: Segmenter = self._init_segmenter()

        # depth + caption + CLIP helper (usato da relations se necessario)
        depth_config = DepthConfig(device=self.device)
        self.depth_est = DepthEstimator(config=depth_config)
        try:
            self.captioner = Captioner(device=self.device)
        except TypeError:
            # Se anche Captioner usa config, crea un config appropriato
            from igp.utils.caption import CaptionerConfig  # se esiste
            caption_config = CaptionerConfig(device=self.device)
            self.captioner = Captioner(config=caption_config)
        try:
            self.clip = CLIPWrapper(device=self.device)
        except TypeError:
            # Se anche CLIPWrapper usa config, crea un config appropriato
            from igp.utils.clip_utils import CLIPConfig
            clip_config = CLIPConfig(device=self.device)
            self.clip = CLIPWrapper(config=clip_config) 

        self.relations_inferencer = RelationInferencer(
            margin_px=config.margin,
            min_distance=config.min_distance,
            max_distance=config.max_distance
        )

        # visualizer
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

        # detection cache
        self._detection_cache: Dict[str, Dict[str, Any]] = {}
        self._det_cache: Dict[str, Dict[str, Any]] = {}

    # ----------------------------- setup helpers -----------------------------

    def _init_detectors(self) -> List[Detector]:
        """Inizializza i detector abilitati."""
        dets: List[Detector] = []
        names = set(d.strip().lower() for d in self.cfg.detectors_to_use)
        
        if "owlvit" in names:
            # ✅ CORREZIONE: Usa score_threshold invece di threshold
            dets.append(OwlViTDetector(
                device=self.device, 
                score_threshold=self.cfg.threshold_owl
            ))
        
        if "yolov8" in names:
            # ✅ CORREZIONE: Usa score_threshold invece di threshold
            dets.append(YOLOv8Detector(
                device=self.device, 
                score_threshold=self.cfg.threshold_yolo
            ))
        
        if "detectron2" in names:
            # ✅ CORREZIONE: Usa score_threshold invece di threshold
            dets.append(Detectron2Detector(
                device=self.device, 
                score_threshold=self.cfg.threshold_detectron
            ))
        
        return dets

    def _init_segmenter(self) -> Segmenter:
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
        if not self.cfg.enable_detection_cache:
            return None
        return self._detection_cache.get(key)

    def _cache_put(self, key: str, value: Dict[str, Any]) -> None:
        if not self.cfg.enable_detection_cache:
            return
        if len(self._detection_cache) >= int(self.cfg.max_cache_size):
            oldest = next(iter(self._detection_cache))
            self._detection_cache.pop(oldest, None)
        self._detection_cache[key] = value

    # ----------------------------- pipeline core -----------------------------

    def _run_detectors(self, image_pil: Image.Image) -> Dict[str, Any]:
        """Esegue tutti i detector e restituisce detections grezze (da fondere)."""
        all_dets: List[Dict[str, Any]] = []
        counts: Dict[str, int] = {}
        for det in self.detectors:
            # ✅ USA run() che gestisce RGB + threshold automaticamente
            out = det.run(image_pil)  # invece di det.detect(image_pil)
            
            src_name = det.__class__.__name__
            counts[src_name] = len(out)
            for d in out:
                all_dets.append(
                    {
                        "box": list(d.box),
                        "label": str(d.label),
                        "score": float(d.score),
                        "from": src_name.lower(),
                        "mask": d.extra.get("mask") if d.extra else None,
                    }
                )
        return {
            "detections": all_dets,
            "counts": counts,
            "boxes": [d["box"] for d in all_dets],
            "labels": [d["label"] for d in all_dets],
            "scores": [d["score"] for d in all_dets],
        }

    def _wbf_fusion(self, all_detections: List[Dict[str, Any]], image_size: Tuple[int, int]) -> Tuple[List[List[float]], List[str], List[float]]:
        """Fusione stile ensemble-boxes WBF con pesi per sorgente."""
        if not all_detections:
            return [], [], []
        W, H = image_size
        # mappa label->id stabile
        canon_labels = [base_label(d["label"]) for d in all_detections]
        uniq_labels = sorted(set(canon_labels))
        label2id = {lb: i for i, lb in enumerate(uniq_labels)}

        # ✅ CORREZIONE: Converti le detection in formato Detection per WBF
        from igp.types import Detection
        detections_obj = []
        
        for d in all_detections:
            x1, y1, x2, y2 = d["box"]
            label = base_label(d["label"])
            score = float(d["score"])
            source = d.get("from", "unknown")
            
            # Crea oggetto Detection
            try:
                det_obj = Detection(box=(x1, y1, x2, y2), label=label, score=score, source=source)
            except TypeError:
                try:
                    det_obj = Detection(box=(x1, y1, x2, y2), label=label, score=score)
                    det_obj.source = source  # Aggiungi manualmente se necessario
                except TypeError:
                    det_obj = Detection(box=(x1, y1, x2, y2), label=label)
                    det_obj.score = score
                    det_obj.source = source
            
            detections_obj.append(det_obj)

        # ✅ CORREZIONE: Usa la funzione corretta con parametri giusti
        fused_detections = weighted_boxes_fusion(
            detections_obj,
            image_size=(W, H),
            iou_thr=0.55,
            skip_box_thr=0.0,
            weights_by_source={"owlvit": 2.0, "yolov8": 1.5, "yolo": 1.5, "detectron2": 1.0},
            default_weight=1.0,
            sort_desc=True
        )
        
        # Estrai i risultati
        boxes_px = [list(d.box) for d in fused_detections]
        labels = [d.label for d in fused_detections]
        scores = [d.score for d in fused_detections]
        
        return boxes_px, labels, scores

    def _fuse_with_det2_mask(self, sam_mask: np.ndarray, det2_mask: Optional[np.ndarray]) -> np.ndarray:
        """Union con soglia IoU: se det2 presente e abbastanza compatibile, fai union."""
        if det2_mask is None:
            return sam_mask
        iou = self._mask_iou(sam_mask, det2_mask)
        if iou >= 0.5:
            return np.logical_or(sam_mask, det2_mask)
        return sam_mask

    @staticmethod
    def _mask_iou(m1: np.ndarray, m2: np.ndarray) -> float:
        inter = np.logical_and(m1, m2).sum()
        union = np.logical_or(m1, m2).sum()
        return float(inter) / float(union) if union > 0 else 0.0

    def _apply_label_nms(self, boxes: List[List[float]], labels: List[str], scores: List[float]) -> Tuple[List[List[float]], List[str], List[float], List[int]]:
        """NMS per classe (label-wise). Ritorna liste filtrate + indici mantenuti."""
        keep = labelwise_nms(boxes, labels, scores, iou_threshold=self.cfg.label_nms_threshold)
        boxes_f = [boxes[i] for i in keep]
        labels_f = [labels[i] for i in keep]
        scores_f = [scores[i] for i in keep]
        return boxes_f, labels_f, scores_f, keep

    def _parse_question(self, question: str) -> Tuple[set, set]:
        """
        Estrae (object_terms, relation_terms) dalla domanda.
        - object_terms: lemmi/noun semplici (fallback senza spaCy).
        - relation_terms: together with sinonimi basilari.
        """
        q = (question or self.cfg.question or "").strip().lower()
        if not q:
            return set(), set()

        # object terms - fallback semplice
        tokens = [t for t in q.replace("?", " ").replace(",", " ").split() if t.isalpha()]
        obj_terms = {t for t in tokens if t not in {"the", "a", "an", "is", "are", "on", "in", "of", "to"}}

        # relation terms (sinonimi basilari)
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
        """Se obj_terms non è vuoto, tiene solo gli oggetti con label base in obj_terms."""
        if not self.cfg.apply_question_filter or not obj_terms:
            return boxes, labels, scores
        idxs = [i for i, lb in enumerate(labels) if base_label(lb) in obj_terms]
        if not idxs:
            return boxes, labels, scores
        return [boxes[i] for i in idxs], [labels[i] for i in idxs], [scores[i] for i in idxs]

    # ----------------------------- single image -----------------------------

    def process_single_image(self, image_pil: Image.Image, image_name: str, custom_question: Optional[str] = None) -> None:
        t0 = time.time()
        W, H = image_pil.size
        cache_key = self._generate_cache_key(image_pil, custom_question or self.cfg.question)

        # 1) DETECTION (+ cache)
        cached = self._cache_get(cache_key)
        if cached is None:
            det_raw = self._run_detectors(image_pil)
            boxes_fused, labels_fused, scores_fused = self._wbf_fusion(det_raw["detections"], (W, H))
            # normalizza label in base form
            labels_fused = [base_label(l) for l in labels_fused]
            # salva per successive fasi
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
            pass  # usa cached

        boxes = list(cached["boxes"])
        labels = list(cached["labels"])
        scores = list(cached["scores"])
        det2_for_mask = list(cached["det2"])

        # 2) QUESTION FILTER (oggetti)
        obj_terms, rel_terms = self._parse_question(custom_question or self.cfg.question)

        # ✅ SALVA I VALORI ORIGINALI PRIMA DEL PRUNING
        original_boxes = list(cached["boxes"])
        original_labels = list(cached["labels"])
        original_scores = list(cached["scores"])

        if self.cfg.aggressive_pruning:
            # pruning "duro": tieni SOLO citati; fallback se vuoto
            bx_q, lb_q, sc_q = self._filter_by_question_terms(boxes, labels, scores, obj_terms)
            if bx_q:
                boxes, labels, scores = bx_q, lb_q, sc_q
                
                # 🆕 FALLBACK: se rimane solo 1 oggetto, ripristina tutto e filtra le relazioni
                if len(boxes) == 1:
                    print(f"[FALLBACK] Solo 1 oggetto dopo aggressive pruning, ripristino tutti gli oggetti")
                    
                    # ✅ RIPRISTINA I VALORI ORIGINALI (non quelli filtrati!)
                    boxes, labels, scores = original_boxes, original_labels, original_scores
                    
                    # Memorizza l'indice dell'oggetto target originale
                    target_obj_label = lb_q[0]  # L'unico oggetto rimasto dopo il pruning
                    target_indices = [i for i, label in enumerate(original_labels) 
                                    if base_label(label) == base_label(target_obj_label)]
                    
                    # Salva gli indici per il filtraggio delle relazioni successivo
                    self._target_object_indices = set(target_indices)
                    print(f"[FALLBACK] Oggetto target: {target_obj_label}, indici: {target_indices}")
                    print(f"[FALLBACK] Ripristinati {len(boxes)} oggetti totali")

        # 3) SEGMENTATION (SAM) + union con mask di Detectron2 se utile
        masks = self.segmenter.segment(image_pil, boxes)
        for i in range(len(masks)):
            d2m = det2_for_mask[i].get("det2_mask")
            if d2m is not None:
                masks[i]["segmentation"] = self._fuse_with_det2_mask(masks[i]["segmentation"], d2m)

        # 4) NMS per label (dopo rifinitura)
        boxes, labels, scores, keep = self._apply_label_nms(boxes, labels, scores)
        masks = [masks[i] for i in keep]

        # 5) DEPTH
        centers = [((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0) for b in boxes]
        depths = self.depth_est.relative_depth_at(image_pil, centers)

        # 6) RELATIONS
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
        
        # 🆕 FALLBACK: se abbiamo target objects da fallback, filtra le relazioni
        if hasattr(self, '_target_object_indices') and self._target_object_indices:
            rels_all = self._filter_relations_by_target_object(rels_all)
            # Reset del flag dopo l'uso
            delattr(self, '_target_object_indices')

        # 6a) filtro per domanda (relazioni)
        if self.cfg.filter_relations_by_question and rel_terms:
            rels_all = self.relations_inferencer.filter_by_question(
                rels_all,
                question_terms=rel_terms,
                threshold=self.cfg.threshold_relation_similarity
            )
        # 6b) limiti per-oggetto + rimozione duplicati inversi
        rels_all = self.relations_inferencer.limit_relationships_per_object(
            rels_all,
            boxes,
            max_relations_per_object=self.cfg.max_relations_per_object,
            min_relations_per_object=self.cfg.min_relations_per_object,
            question_rel_terms=rel_terms if rel_terms else None,
        )
        rels_all = self.relations_inferencer.drop_inverse_duplicates(rels_all)

        # 7) GRAPH + PROMPT/TRIPLES
        if not self.cfg.skip_graph or not self.cfg.skip_prompt or not self.cfg.skip_visualization:
            caption = self.captioner.caption(image_pil)
            scene_graph = build_scene_graph(
                image_size=(W, H),
                boxes=boxes,
                labels=labels,
                scores=scores,
                depths=depths,
                caption=caption,
            )
        else:
            scene_graph = None

        # salva scene graph (gpickle/json)
        if scene_graph is not None and not self.cfg.skip_graph:
            out_gpickle = os.path.join(self.cfg.output_folder, f"{image_name}_graph.gpickle")
            out_json = os.path.join(self.cfg.output_folder, f"{image_name}_graph.json")
            self._save_graph(scene_graph, out_gpickle, out_json)

        # salva triples
        if scene_graph is not None:
            triples_path = os.path.join(self.cfg.output_folder, f"{image_name}_graph_triples.txt")
            with open(triples_path, "w", encoding="utf-8") as f:
                f.write(to_triples_text(scene_graph))

        # 8) VIZ
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

        # cleanup
        self._free_memory()
        dt = time.time() - t0
        print(f"[DONE] {image_name} processed in {dt:.2f}s")

    def _filter_relations_by_target_object(self, relationships: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filtra le relazioni mantenendo solo quelle che coinvolgono gli oggetti target."""
        if not hasattr(self, '_target_object_indices') or not self._target_object_indices:
            return relationships
        
        filtered_rels = []
        for rel in relationships:
            src_idx = rel.get('src_idx', -1)
            tgt_idx = rel.get('tgt_idx', -1)
            
            # Mantieni la relazione se coinvolge almeno uno degli oggetti target
            if src_idx in self._target_object_indices or tgt_idx in self._target_object_indices:
                filtered_rels.append(rel)
        
        print(f"[FALLBACK] Relazioni filtrate: {len(filtered_rels)}/{len(relationships)} mantenute")
        return filtered_rels

    # ----------------------------- runners -----------------------------

    def run(self) -> None:
        """
        Entry-point batch:
          - json_file: lista di dict con "image_path" e opzionale "question"
          - dataset: (opzionale, richiede `datasets`) con split/colonna
          - input_path: file singolo o cartella
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
        with open(json_path, "r", encoding="utf-8") as f:
            rows = json.load(f)
        if self.cfg.num_instances > 0:
            rows = rows[: int(self.cfg.num_instances)]
        for row in rows:
            img_p = row["image_path"]
            q = row.get("question", self.cfg.question)
            try:
                img = Image.open(img_p).convert("RGB")
            except Exception as e:
                print(f"[ERROR] Loading {img_p}: {e}")
                continue
            name = Path(img_p).stem
            self.process_single_image(img, name, custom_question=q)

    def _run_from_dataset(self) -> None:
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
        """Recupera, tra le detection detectron2, la mask con IoU max rispetto al box dato."""
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
        """Applica modalità label (original/numeric/alphabetic) + eventuale suffisso indice."""
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
        """Salva gpickle + json node_link_data."""
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
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()
