# image_preprocessor.py
from __future__ import annotations

import argparse
import sys
from typing import Any, Dict, Tuple

from igp.config import default_config, PreprocessorConfig
from igp.pipeline.preprocessor import Preprocessor


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Image Graph Preprocessor (CLI orchestrator)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # I/O e dataset
    p.add_argument("--input_path", type=str, default=None, help="Percorso a immagine o cartella")
    p.add_argument("--json_file", type=str, default="", help="Batch JSON (override input_path/dataset)")
    p.add_argument("--output_folder", type=str, default="output_images")
    p.add_argument("--dataset", type=str, default=None)
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--image_column", type=str, default="image")
    p.add_argument("--num_instances", type=int, default=-1, help="Se >0, elabora solo le prime N istanze")

    # Domanda / NLP
    p.add_argument("--question", type=str, default="", help="Domanda per filtrare oggetti/relazioni")
    p.add_argument("--disable_question_filter", action="store_true", help="Disattiva i filtri basati sulla domanda")
    p.add_argument("--aggressive_pruning", action="store_true",
                   help="Tieni solo oggetti citati e relazioni richieste (pruning più duro)")
    p.add_argument("--no_filter_relations_by_question", action="store_true",
                   help="Non filtrare relazioni in base alla domanda")
    p.add_argument("--threshold_object_similarity", type=float, default=0.50)
    p.add_argument("--threshold_relation_similarity", type=float, default=0.50)

    # Detector
    p.add_argument("--detectors", type=str, default="owlvit,yolov8,detectron2",
                   help="Elenco detector separati da virgola")
    p.add_argument("--owl_threshold", type=float, default=0.40)
    p.add_argument("--yolo_threshold", type=float, default=0.80)
    p.add_argument("--detectron_threshold", type=float, default=0.80)

    # Relazioni
    p.add_argument("--max_relations_per_object", type=int, default=3)
    p.add_argument("--min_relations_per_object", type=int, default=1)
    p.add_argument("--margin", type=int, default=20, help="Margine (px) per orientamento geometrico")
    p.add_argument("--min_distance", type=float, default=50)
    p.add_argument("--max_distance", type=float, default=20000)

    # NMS e segmentazione
    p.add_argument("--label_nms_threshold", type=float, default=0.50)
    p.add_argument("--seg_iou_threshold", type=float, default=0.70)

    # SAM
    p.add_argument("--sam_version", type=str, choices=["1", "2", "hq"], default="1")
    p.add_argument("--sam_hq_model_type", type=str, choices=["vit_b", "vit_l", "vit_h"], default="vit_h")
    p.add_argument("--points_per_side", type=int, default=32)
    p.add_argument("--pred_iou_thresh", type=float, default=0.90)
    p.add_argument("--stability_score_thresh", type=float, default=0.92)
    p.add_argument("--min_mask_region_area", type=int, default=100)
    p.add_argument("--preproc_device", type=str, default=None, help="Forza device (cpu/cuda)")

    # Visualizzazione
    p.add_argument("--label_mode", type=str, choices=["original", "numeric", "alphabetic"], default="original")
    p.add_argument("--display_labels", action="store_true", help="Mostra etichette oggetti")
    p.add_argument("--display_relationships", action="store_true", help="Mostra frecce relazioni")
    p.add_argument("--display_relation_labels", action="store_true", help="Mostra testo relazioni")
    p.add_argument("--show_segmentation", action="store_true", help="Disegna maschere SAM")
    p.add_argument("--fill_segmentation", action="store_true", help="Riempie le maschere disegnate")
    p.add_argument("--no_legend", action="store_true", help="Disabilita legenda colori")
    p.add_argument("--seg_fill_alpha", type=float, default=0.2)
    p.add_argument("--bbox_linewidth", type=float, default=2.0)
    p.add_argument("--obj_fontsize_inside", type=int, default=12)
    p.add_argument("--obj_fontsize_outside", type=int, default=12)
    p.add_argument("--rel_fontsize", type=int, default=10)
    p.add_argument("--legend_fontsize", type=int, default=8)
    p.add_argument("--rel_arrow_linewidth", type=float, default=2.0)
    p.add_argument("--rel_arrow_mutation_scale", type=float, default=22.0)
    p.add_argument("--resolve_overlaps", action="store_true", help="Evita sovrapposizioni label/frecce")
    p.add_argument("--no_bboxes", action="store_true", help="Non disegnare bounding box")
    p.add_argument("--no_masks", action="store_true", help="Non disegnare maschere")
    p.add_argument("--no_instances", action="store_true", help="Nasconde sia bbox che maschere (override)")
    p.add_argument("--show_confidence", action="store_true", help="Aggiunge la confidenza nelle etichette")

    # Post-processing colori
    p.add_argument("--color_sat_boost", type=float, default=1.30)
    p.add_argument("--color_val_boost", type=float, default=1.15)

    # Hole filling
    p.add_argument("--close_holes", action="store_true", help="Chiudi buchi interni nelle maschere SAM")
    p.add_argument("--hole_kernel", type=int, default=7)
    p.add_argument("--min_hole_area", type=int, default=100)

    # Output / salvataggi
    p.add_argument("--save_image_only", action="store_true", help="Salva solo immagine elaborata")
    p.add_argument("--skip_graph", action="store_true", help="Non salvare i file di grafo")
    p.add_argument("--skip_prompt", action="store_true", help="Non salvare il file Triples/Prompt")
    p.add_argument("--skip_visualization", action="store_true", help="Non salvare la visualizzazione")
    p.add_argument("--export_preproc_only", action="store_true",
                   help="Esporta anche overlay trasparente (solo preproc)")

    # Cache
    p.add_argument("--enable_detection_cache", action="store_true", help="Abilita cache detection")
    p.add_argument("--max_cache_size", type=int, default=100)
    p.add_argument("--clear_cache", action="store_true", help="Pulisce le cache e termina se non c'è altro da fare")

    return p.parse_args()


def _build_config(args: argparse.Namespace) -> PreprocessorConfig:
    # Base config con default sensati
    cfg = default_config()

    # I/O & dataset
    cfg.input_path = args.input_path
    cfg.json_file = args.json_file
    cfg.output_folder = args.output_folder
    cfg.dataset = args.dataset
    cfg.split = args.split
    cfg.image_column = args.image_column
    cfg.num_instances = args.num_instances

    # NLP / domanda
    cfg.question = args.question
    cfg.apply_question_filter = not args.disable_question_filter
    cfg.aggressive_pruning = bool(args.aggressive_pruning)
    cfg.filter_relations_by_question = not args.no_filter_relations_by_question
    cfg.threshold_object_similarity = float(args.threshold_object_similarity)
    cfg.threshold_relation_similarity = float(args.threshold_relation_similarity)

    # Detector
    cfg.detectors_to_use = tuple(d.strip() for d in args.detectors.split(",") if d.strip())
    cfg.threshold_owl = float(args.owl_threshold)
    cfg.threshold_yolo = float(args.yolo_threshold)
    cfg.threshold_detectron = float(args.detectron_threshold)

    # Relazioni
    cfg.max_relations_per_object = int(args.max_relations_per_object)
    cfg.min_relations_per_object = int(args.min_relations_per_object)
    cfg.margin = int(args.margin)
    cfg.min_distance = float(args.min_distance)
    cfg.max_distance = float(args.max_distance)

    # NMS/seg
    cfg.label_nms_threshold = float(args.label_nms_threshold)
    cfg.seg_iou_threshold = float(args.seg_iou_threshold)

    # SAM
    cfg.sam_version = args.sam_version
    cfg.sam_hq_model_type = args.sam_hq_model_type
    cfg.points_per_side = int(args.points_per_side)
    cfg.pred_iou_thresh = float(args.pred_iou_thresh)
    cfg.stability_score_thresh = float(args.stability_score_thresh)
    cfg.min_mask_region_area = int(args.min_mask_region_area)
    cfg.preproc_device = args.preproc_device

    # Visualizzazione
    cfg.label_mode = args.label_mode
    cfg.display_labels = bool(args.display_labels)
    cfg.display_relationships = bool(args.display_relationships)
    cfg.display_relation_labels = bool(args.display_relation_labels)

    # override per no_instances: nasconde tutto
    if args.no_instances:
        cfg.show_segmentation = False
        cfg.show_bboxes = False
    else:
        # se no_masks è True, spegni segmentazione
        cfg.show_segmentation = bool(args.show_segmentation) and not args.no_masks
        # se no_bboxes è True, spegni bbox
        cfg.show_bboxes = not args.no_bboxes

    cfg.fill_segmentation = bool(args.fill_segmentation)
    cfg.display_legend = not args.no_legend
    cfg.seg_fill_alpha = float(args.seg_fill_alpha)
    cfg.bbox_linewidth = float(args.bbox_linewidth)
    cfg.obj_fontsize_inside = int(args.obj_fontsize_inside)
    cfg.obj_fontsize_outside = int(args.obj_fontsize_outside)
    cfg.rel_fontsize = int(args.rel_fontsize)
    cfg.legend_fontsize = int(args.legend_fontsize)
    cfg.rel_arrow_linewidth = float(args.rel_arrow_linewidth)
    cfg.rel_arrow_mutation_scale = float(args.rel_arrow_mutation_scale)
    cfg.resolve_overlaps = bool(args.resolve_overlaps)
    cfg.show_confidence = bool(args.show_confidence)

    # Colori
    cfg.color_sat_boost = float(args.color_sat_boost)
    cfg.color_val_boost = float(args.color_val_boost)

    # Hole filling
    cfg.close_holes = bool(args.close_holes)
    cfg.hole_kernel = int(args.hole_kernel)
    cfg.min_hole_area = int(args.min_hole_area)

    # Output / salvataggi
    cfg.save_image_only = bool(args.save_image_only)
    cfg.skip_graph = bool(args.skip_graph)
    cfg.skip_prompt = bool(args.skip_prompt)
    cfg.skip_visualization = bool(args.skip_visualization)
    cfg.export_preproc_only = bool(args.export_preproc_only)

    # Cache
    cfg.enable_detection_cache = bool(args.enable_detection_cache)
    cfg.max_cache_size = int(args.max_cache_size)

    return cfg


def main(argv: list[str] | None = None) -> int:
    args = _parse_args()
    cfg = _build_config(args)

    # Se richiesto, pulizia delle cache prima di procedere
    if args.clear_cache:
        pre = Preprocessor(cfg)
        try:
            pre.clear_caches()
            print("[INFO] Cache cleared.")
        finally:
            # Se non è stato richiesto di processare nulla, termina
            if not (args.input_path or args.json_file or args.dataset):
                print("[INFO] No processing requested. Exiting.")
                return 0

    preproc = Preprocessor(cfg)
    preproc.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
