# igp/vqa/runner.py
from __future__ import annotations
import gc
import glob
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import psutil
import torch

from .types import VQAExample
from .preproc import (
    preprocess_for_qa,
    get_preprocessed_path,
    get_scene_graph_path,
    load_scene_graph,
)
from .io import load_image
from .models import VLLMWrapper, HFVLModel

ModelLike = Union[VLLMWrapper, HFVLModel]

def run_vqa(
    examples: List[VQAExample],
    model: ModelLike,
    *,
    out_json: str,
    prompt_tpl: str,
    batch_size: int = 1,
    max_qpi: int = -1,
    max_imgs: int = -1,
    preproc_folder: str = "preprocessed",
    disable_q_filter: bool = False,
    preproc_cfg: Optional[Dict[str, Any]] = None,
    image_dir: Optional[str] = None,
    skip_preproc: bool = False,
    include_scene_graph: bool = False,
    inference_image: str = "preprocessed",
) -> List[Dict[str, Any]]:

    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    results: List[Dict[str, Any]] = []
    if os.path.exists(out_json):
        with open(out_json, "r", encoding="utf-8") as f:
            try:
                results = json.load(f)
            except Exception:
                results = []

    processed = {(r.get("image_path"), r.get("question")) for r in results}

    grouped: Dict[str, List[VQAExample]] = {}
    for ex in examples:
        grouped.setdefault(ex.image_path, []).append(ex)

    img_paths = list(grouped)[:max_imgs] if max_imgs > 0 else list(grouped)

    try:
        from tqdm import tqdm
    except Exception:
        tqdm = lambda x, **_: x  # type: ignore

    for img in img_paths:
        qs = grouped[img][:max_qpi] if max_qpi > 0 else grouped[img]
        for i in tqdm(range(0, len(qs), batch_size), desc=os.path.basename(img)):
            batch = qs[i : i + batch_size]
            for ex in batch:
                key = (ex.image_path, ex.question)
                if key in processed:
                    continue

                if len(processed) and len(processed) % 50 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    mem = psutil.virtual_memory()
                    print(f"[GC] RAM used {mem.percent}%")

                # 1) Immagine per l’inference
                if skip_preproc:
                    processed_img = get_preprocessed_path(ex.image_path, ex.question, preproc_folder)
                    if not os.path.exists(processed_img):
                        raw_img = ex.image_path if not image_dir or os.path.isabs(ex.image_path) else os.path.join(image_dir, ex.image_path)
                        if os.path.exists(raw_img):
                            processed_img = raw_img
                        else:
                            base = os.path.splitext(os.path.basename(ex.image_path))[0]
                            hits = []
                            for pat in (f"{base}_*.jpg", f"{base}_*.png"):
                                hits.extend(glob.glob(os.path.join(preproc_folder, pat)))
                            if not hits:
                                raise FileNotFoundError(f"No image found for {ex.image_path} (raw or preprocessed).")
                            processed_img = hits[0]
                else:
                    processed_img = preprocess_for_qa(
                        ex.image_path, ex.question,
                        output_folder=preproc_folder,
                        apply_question_filter=not disable_q_filter,
                        cfg_overrides=preproc_cfg,
                        image_dir=image_dir,  
                        aggressive_pruning=True
                    )
                    if not os.path.exists(processed_img):
                        base = os.path.splitext(os.path.basename(ex.image_path))[0]
                        qhash = __import__("hashlib").md5(ex.question.encode("utf-8")).hexdigest()[:8]
                        patterns = (f"{base}_{qhash}_output.*", f"{base}_{qhash}*.*")
                        found = None
                        for pat in patterns:
                            hits = glob.glob(os.path.join(preproc_folder, pat))
                            if hits:
                                found = hits[0]; break
                        if not found:
                            raise FileNotFoundError(f"Preprocessed image not found for {ex.image_path}")
                        processed_img = found

                raw_img = ex.image_path if not image_dir or os.path.isabs(ex.image_path) else os.path.join(image_dir, ex.image_path)
                inference_img = processed_img if inference_image == "preprocessed" else raw_img
                if not os.path.exists(inference_img):
                    raise FileNotFoundError(f"Image for inference not found: {inference_img}")

                # 2) Scene graph (opzionale)
                scene_graph_text = ""
                if include_scene_graph:
                    sg_path = get_scene_graph_path(ex.image_path, ex.question, preproc_folder)
                    if not os.path.exists(sg_path):
                        alt = get_scene_graph_path(ex.image_path, ex.question, "output_images")
                        if os.path.exists(alt):
                            sg_path = alt
                    if os.path.exists(sg_path):
                        scene_graph_text = load_scene_graph(sg_path)

                # 3) Prompt
                base_prompt = prompt_tpl.format(question=ex.question)
                prompt = f"{scene_graph_text}{base_prompt}" if scene_graph_text else base_prompt

                # 4) Generazione
                t0 = time.time()
                ans = model.generate(prompt, image_path=inference_img)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if "Answer:" in ans:
                    ans = ans.rsplit("Answer:", 1)[-1].strip().strip('"')

                out_record = {
                    **ex.to_dict(),
                    "generated_answer": ans,
                    "processing_time": time.time() - t0,
                    "used_scene_graph": bool(scene_graph_text),
                    "inference_image_type": inference_image,
                    "inference_image_path": inference_img,
                }
                results.append(out_record)
                processed.add(key)

        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    return results

def evaluate(res: List[Dict[str, Any]]) -> Dict[str, float]:
    gold = [r for r in res if r.get("answer")]
    if not gold:
        return {}
    corr = sum(r["answer"].strip().lower() == r["generated_answer"].strip().lower() for r in gold)
    return {
        "total": len(gold),
        "exact": corr,
        "exact_percent": 100 * corr / len(gold),
        "avg_time": sum(r["processing_time"] for r in gold) / len(gold),
    }
