#!/usr/bin/env python3
"""
REC Inference Script using vLLM

Run referring expression comprehension on GoM-processed images.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Optional

from vllm import LLM, SamplingParams

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

# Prompt templates from AAAI26 supplementary
SYSTEM_VISUAL = (
    "You are a multimodal assistant with spatial reasoning capabilities. "
    "Use the visual scene graph in the image to interpret spatial relations "
    "and answer questions grounded in the visual layout."
)

SYSTEM_VISUAL_TEXTUAL = (
    "You are a multimodal assistant capable of understanding both visual and "
    "textual scene graphs. Use the image and the accompanying graph description "
    "to answer the question accurately."
)

USER_VISUAL = (
    "Identify the object ID(s) for the following description(s) based on the "
    "scene graph visualization in the image.\n"
    "Target object description(s): {descriptions}\n"
    "Respond with only the ID(s)."
)

USER_VISUAL_TEXTUAL = (
    "Identify the object ID(s) for the following description(s) based on the "
    "scene graph visualization and the graph description.\n"
    "Scene Graph (Textual):\n{scene_graph}\n"
    "Target object description(s): {descriptions}\n"
    "Respond with only the ID(s)."
)


def load_dataset(path: str | Path) -> list[dict]:
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def build_prompt(
    descriptions: list[str] | str,
    mode: str,
    scene_graph_text: Optional[str] = None,
) -> tuple[str, str]:
    if isinstance(descriptions, str):
        descriptions = [descriptions]
    desc_text = json.dumps(descriptions)

    if mode == "visual":
        return SYSTEM_VISUAL, USER_VISUAL.format(descriptions=desc_text)
    else:
        if scene_graph_text is None:
            raise ValueError("scene_graph_text required for visual_textual mode")
        return SYSTEM_VISUAL_TEXTUAL, USER_VISUAL_TEXTUAL.format(
            scene_graph=scene_graph_text, descriptions=desc_text
        )


def format_messages(system: str, user: str, image_path: str) -> list[dict]:
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_path}},
                {"type": "text", "text": user},
            ],
        },
    ]


def extract_ids(response: str) -> list[str]:
    ids = []
    # Labeled IDs: chair_1, couch_3
    labeled = re.findall(r"\b([a-zA-Z]+_\d+)\b", response)
    ids.extend(labeled)
    # Numeric IDs
    numeric = re.findall(r"(?:ID\s*|is\s+|\*\*)?(\d+)(?:\*\*)?(?:\b|$)", response, re.IGNORECASE)
    for n in numeric:
        if not any(n in label for label in labeled):
            ids.append(n)
    return ids


def run_inference(
    llm: LLM,
    examples: list[dict],
    image_dir: Optional[str],
    mode: str,
    sampling_params: SamplingParams,
) -> list[dict]:
    results = []

    for i, ex in enumerate(examples):
        image_path = ex.get("gom_image_path") or ex.get("image_path")
        full_path = Path(image_dir) / image_path if image_dir else Path(image_path)

        if not full_path.exists():
            log.warning(f"Image not found: {full_path}")
            continue

        descriptions = ex.get("descriptions") or ex.get("description")
        if descriptions is None:
            continue

        try:
            system, user = build_prompt(descriptions, mode, ex.get("scene_graph_text"))
        except ValueError:
            continue

        messages = format_messages(system, user, f"file://{full_path.resolve()}")
        outputs = llm.chat(messages=[messages], sampling_params=sampling_params)
        response = outputs[0].outputs[0].text.strip()

        results.append({
            "image_path": str(image_path),
            "descriptions": descriptions if isinstance(descriptions, list) else [descriptions],
            "response": response,
            "predicted_ids": extract_ids(response),
            "target_ids": ex.get("target_ids"),
            "image_id": ex.get("image_id"),
        })

        if (i + 1) % 50 == 0:
            log.info(f"Processed {i + 1}/{len(examples)}")

    return results


def compute_metrics(results: list[dict]) -> dict:
    total = exact = tp = pred_count = target_count = 0
    for r in results:
        targets = r.get("target_ids")
        if targets is None:
            continue
        total += 1
        pred = set(str(x) for x in r["predicted_ids"])
        tgt = set(str(x) for x in targets)
        if pred == tgt:
            exact += 1
        tp += len(pred & tgt)
        pred_count += len(pred)
        target_count += len(tgt)

    return {
        "exact_match": exact / total if total else 0,
        "precision": tp / pred_count if pred_count else 0,
        "recall": tp / target_count if target_count else 0,
        "n": total,
    }


def save_results(results: list[dict], path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="REC inference with vLLM")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--image-dir", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--mode", choices=["visual", "visual_textual"], default="visual_textual")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    examples = load_dataset(args.dataset)
    if args.limit:
        examples = examples[:args.limit]

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
    )
    sampling_params = SamplingParams(max_tokens=args.max_tokens, temperature=args.temperature)

    results = run_inference(llm, examples, args.image_dir, args.mode, sampling_params)
    save_results(results, args.output)

    metrics = compute_metrics(results)
    if metrics["n"]:
        log.info(f"exact_match={metrics['exact_match']:.2%} precision={metrics['precision']:.2%} recall={metrics['recall']:.2%}")


if __name__ == "__main__":
    main()
