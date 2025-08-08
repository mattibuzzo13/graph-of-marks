# vqa.py
from __future__ import annotations
import argparse
import json
import os
import torch
from huggingface_hub import login as hf_login

from igp.vqa.io import load_examples
from igp.vqa.models import VLLMWrapper, HFVLModel
from igp.vqa.runner import run_vqa, evaluate
from igp.vqa.preproc import run_preprocessing

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("VQA pipeline (modulare) + IGP preprocessor",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # I/O
    ap.add_argument("--input_file", required=True, help="JSON file of QA examples")
    ap.add_argument("--output_file", default="vqa_results.json", help="Where to save VQA results")
    ap.add_argument("--image_dir", help="Base directory for relative image paths")
    ap.add_argument("--single_question", type=str, default=None, help="Usa la stessa domanda per tutte le immagini")

    # Modello
    ap.add_argument("--model_name", default="google/gemma-3-4b-it")
    ap.add_argument("--use_vllm", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--tensor_parallel_size", type=int, default=1)

    # Preprocessing
    ap.add_argument("--preproc_folder", type=str, default="preprocessed")
    ap.add_argument("--disable_question_filter", action="store_true")
    ap.add_argument("--skip_preprocessing", action="store_true")
    ap.add_argument("--preprocess_only", action="store_true")
    ap.add_argument("--include_scene_graph", action="store_true")
    ap.add_argument("--inference_image", choices=["preprocessed", "raw"], default="preprocessed")

    # Batching/limiti
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--max_images", type=int, default=-1)
    ap.add_argument("--max_questions_per_image", type=int, default=-1)

    # Prompt
    ap.add_argument("--prompt_template", default="Question: {question}\nAnswer:")

    # ✅ AGGIUNTI: Argomenti compatibili con image_preprocessor.py
    ap.add_argument("--output_folder", type=str, help="Output folder (alias for preproc_folder)")
    
    # Detector
    ap.add_argument("--detectors", type=str, default="owlvit,yolov8,detectron2",
                   help="Comma-separated list of detectors")
    ap.add_argument("--owl_threshold", type=float, default=0.40)
    ap.add_argument("--yolo_threshold", type=float, default=0.80)
    ap.add_argument("--detectron_threshold", type=float, default=0.80)

    # SAM
    ap.add_argument("--sam_version", type=str, choices=["1", "2", "hq"], default="1")
    ap.add_argument("--sam_hq_model_type", type=str, choices=["vit_b", "vit_l", "vit_h"], default="vit_h")
    ap.add_argument("--points_per_side", type=int, default=32)
    ap.add_argument("--pred_iou_thresh", type=float, default=0.90)
    ap.add_argument("--stability_score_thresh", type=float, default=0.92)
    ap.add_argument("--min_mask_region_area", type=int, default=100)
    ap.add_argument("--preproc_device", type=str, default=None)
    
    ap.add_argument("--aggressive_pruning", action="store_true",
                   help="Tieni solo oggetti citati e relazioni richieste (pruning più duro)")

    # Relazioni
    ap.add_argument("--max_relations_per_object", type=int, default=3)
    ap.add_argument("--min_relations_per_object", type=int, default=1)
    ap.add_argument("--margin", type=int, default=20)
    ap.add_argument("--min_distance", type=float, default=50)
    ap.add_argument("--max_distance", type=float, default=20000)

    # NMS/segmentazione
    ap.add_argument("--label_nms_threshold", type=float, default=0.50)
    ap.add_argument("--seg_iou_threshold", type=float, default=0.70)

    # Visualizzazione
    ap.add_argument("--label_mode", type=str, choices=["original", "numeric", "alphabetic"], default="original")
    ap.add_argument("--display_labels", action="store_true")
    ap.add_argument("--display_relationships", action="store_true")
    ap.add_argument("--display_relation_labels", action="store_true")
    ap.add_argument("--show_segmentation", action="store_true")
    ap.add_argument("--fill_segmentation", action="store_true")
    ap.add_argument("--no_legend", action="store_true")
    ap.add_argument("--seg_fill_alpha", type=float, default=0.30)
    ap.add_argument("--bbox_linewidth", type=float, default=2.0)
    ap.add_argument("--obj_fontsize_inside", type=int, default=12)
    ap.add_argument("--obj_fontsize_outside", type=int, default=12)
    ap.add_argument("--rel_fontsize", type=int, default=10)
    ap.add_argument("--legend_fontsize", type=int, default=8)
    ap.add_argument("--rel_arrow_linewidth", type=float, default=2.5)
    ap.add_argument("--rel_arrow_mutation_scale", type=float, default=22.0)
    ap.add_argument("--resolve_overlaps", action="store_true")
    ap.add_argument("--no_bboxes", action="store_true")
    ap.add_argument("--show_confidence", action="store_true")

    # Colori e post-processing
    ap.add_argument("--color_sat_boost", type=float, default=1.30)
    ap.add_argument("--color_val_boost", type=float, default=1.15)

    # Hole filling
    ap.add_argument("--close_holes", action="store_true")
    ap.add_argument("--hole_kernel", type=int, default=7)
    ap.add_argument("--min_hole_area", type=int, default=100)

    # Output
    ap.add_argument("--save_image_only", action="store_true")
    ap.add_argument("--skip_graph", action="store_true")
    ap.add_argument("--skip_prompt", action="store_true")
    ap.add_argument("--skip_visualization", action="store_true")
    ap.add_argument("--export_preproc_only", action="store_true")

    # Cache
    ap.add_argument("--enable_detection_cache", action="store_true")
    ap.add_argument("--max_cache_size", type=int, default=100)

    # Override rapidi per la pipeline
    ap.add_argument("--preproc_override", nargs="*", default=[],
        help="Override k=v per la pipeline IGP")

    return ap.parse_args()

def _parse_overrides(pairs: list[str]) -> dict:
    out = {}
    for p in pairs:
        if "=" not in p: continue
        k, v = p.split("=", 1)
        v = v.strip()
        if v.lower() in ("true", "false"):
            v = v.lower() == "true"
        else:
            try:
                if "." in v: v = float(v)
                else: v = int(v)
            except Exception:
                pass
        out[k.strip()] = v
    return out

def main() -> int:
    args = _parse_args()

    # ✅ COMPATIBILITÀ: Gestisci output_folder vs preproc_folder
    if args.output_folder and not args.preproc_folder:
        args.preproc_folder = args.output_folder
    elif not args.output_folder and args.preproc_folder:
        args.output_folder = args.preproc_folder

    # ✅ CONVERSIONE: Converti tutti gli argomenti di image_preprocessor in preproc_override
    preproc_overrides = list(args.preproc_override) if args.preproc_override else []
    
    # Aggiungi parametri del preprocessor agli override (solo se diversi dai default)
    preprocessor_params = {
        'detectors_to_use': f"({','.join(args.detectors.split(','))})" if args.detectors != "owlvit,yolov8,detectron2" else None,
        'threshold_owl': args.owl_threshold if args.owl_threshold != 0.40 else None,
        'threshold_yolo': args.yolo_threshold if args.yolo_threshold != 0.80 else None,
        'threshold_detectron': args.detectron_threshold if args.detectron_threshold != 0.80 else None,
        'sam_version': args.sam_version if args.sam_version != "1" else None,
        'sam_hq_model_type': args.sam_hq_model_type if args.sam_hq_model_type != "vit_h" else None,
        'points_per_side': args.points_per_side if args.points_per_side != 32 else None,
        'pred_iou_thresh': args.pred_iou_thresh if args.pred_iou_thresh != 0.90 else None,
        'stability_score_thresh': args.stability_score_thresh if args.stability_score_thresh != 0.92 else None,
        'min_mask_region_area': args.min_mask_region_area if args.min_mask_region_area != 100 else None,
        'max_relations_per_object': args.max_relations_per_object if args.max_relations_per_object != 3 else None,
        'min_relations_per_object': args.min_relations_per_object if args.min_relations_per_object != 1 else None,
        'margin': args.margin if args.margin != 20 else None,
        'min_distance': args.min_distance if args.min_distance != 50 else None,
        'max_distance': args.max_distance if args.max_distance != 20000 else None,
        'label_nms_threshold': args.label_nms_threshold if args.label_nms_threshold != 0.50 else None,
        'seg_iou_threshold': args.seg_iou_threshold if args.seg_iou_threshold != 0.70 else None,
        'label_mode': args.label_mode if args.label_mode != "original" else None,
        'display_labels': True if args.display_labels else None,
        'display_relationships': True if args.display_relationships else None,
        'display_relation_labels': True if args.display_relation_labels else None,
        'show_segmentation': True if args.show_segmentation else None,
        'fill_segmentation': True if args.fill_segmentation else None,
        'display_legend': False if args.no_legend else None,
        'seg_fill_alpha': args.seg_fill_alpha if args.seg_fill_alpha != 0.30 else None,
        'bbox_linewidth': args.bbox_linewidth if args.bbox_linewidth != 2.0 else None,
        'obj_fontsize_inside': args.obj_fontsize_inside if args.obj_fontsize_inside != 12 else None,
        'obj_fontsize_outside': args.obj_fontsize_outside if args.obj_fontsize_outside != 12 else None,
        'rel_fontsize': args.rel_fontsize if args.rel_fontsize != 10 else None,
        'legend_fontsize': args.legend_fontsize if args.legend_fontsize != 8 else None,
        'rel_arrow_linewidth': args.rel_arrow_linewidth if args.rel_arrow_linewidth != 2.5 else None,
        'rel_arrow_mutation_scale': args.rel_arrow_mutation_scale if args.rel_arrow_mutation_scale != 22.0 else None,
        'resolve_overlaps': True if args.resolve_overlaps else None,
        'show_bboxes': False if args.no_bboxes else None,
        'show_confidence': True if args.show_confidence else None,
        'color_sat_boost': args.color_sat_boost if args.color_sat_boost != 1.30 else None,
        'color_val_boost': args.color_val_boost if args.color_val_boost != 1.15 else None,
        'close_holes': True if args.close_holes else None,
        'hole_kernel': args.hole_kernel if args.hole_kernel != 7 else None,
        'min_hole_area': args.min_hole_area if args.min_hole_area != 100 else None,
        'save_image_only': True if args.save_image_only else None,
        'skip_graph': True if args.skip_graph else None,
        'skip_prompt': True if args.skip_prompt else None,
        'skip_visualization': True if args.skip_visualization else None,
        'export_preproc_only': True if args.export_preproc_only else None,
        'enable_detection_cache': True if args.enable_detection_cache else None,
        'max_cache_size': args.max_cache_size if args.max_cache_size != 100 else None,
    }
    
    # Aggiungi device del preprocessor se specificato
    if args.preproc_device:
        preprocessor_params['preproc_device'] = args.preproc_device
    
    # Aggiungi solo parametri non-None ai preproc_override
    for key, value in preprocessor_params.items():
        if value is not None:
            preproc_overrides.append(f"{key}={value}")

    # Carico esempi
    examples = load_examples(args.input_file)
    if args.single_question:
        for e in examples:
            e.question = args.single_question

    # ✅ PARSING: Converti preproc_override in dict
    preproc_cfg = _parse_overrides(preproc_overrides)

    # Solo preprocessing?
    if args.preprocess_only:
        run_preprocessing(
            examples,
            preproc_folder=args.preproc_folder,
            disable_q_filter=args.disable_question_filter,
            max_imgs=args.max_images,
            max_qpi=args.max_questions_per_image,
            cfg_overrides=preproc_cfg,
            image_dir=args.image_dir,
        )
        print(f"[INFO] Preprocessing completato in: {args.preproc_folder}")
        return 0

    # Login (se serve) e modello
    tok = os.getenv("HF_TOKEN")
    if tok: 
        hf_login(token=tok)

    model = (
        VLLMWrapper(
            args.model_name,
            device=args.device, 
            max_length=args.max_length,
            temperature=args.temperature, 
            top_p=args.top_p,
            tensor_parallel_size=args.tensor_parallel_size,
        ) if args.use_vllm else
        HFVLModel(
            args.model_name,
            device=args.device, 
            max_length=args.max_length,
            temperature=args.temperature, 
            top_p=args.top_p,
        )
    )

    # VQA con tutti i parametri
    res = run_vqa(
        examples, 
        model,
        out_json=args.output_file,
        prompt_tpl=args.prompt_template,
        batch_size=args.batch_size,
        max_qpi=args.max_questions_per_image,
        max_imgs=args.max_images,
        preproc_folder=args.preproc_folder,
        disable_q_filter=args.disable_question_filter,
        preproc_cfg=preproc_cfg,
        image_dir=args.image_dir,
        skip_preproc=args.skip_preprocessing,
        include_scene_graph=args.include_scene_graph,
        inference_image=args.inference_image,
    )

    # Metriche (se gold presente)
    metrics = evaluate(res)
    if metrics:
        mfile = os.path.splitext(args.output_file)[0] + "_metrics.json"
        with open(mfile, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print("[INFO] Metrics:", metrics)
    
    print(f"[INFO] Risultati VQA salvati in: {args.output_file}")
    if metrics:
        print(f"[INFO] Metriche salvate in: {mfile}")
    
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
