#!/usr/bin/env python3
"""
VQA Inference Script using vLLM

Run VQA on GoM-processed images.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

from vllm import LLM, SamplingParams

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

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
    "Answer the question based on the spatial configuration in the image.\n"
    "Question: {question}"
)

USER_VISUAL_TEXTUAL = (
    "Answer the question based on the spatial configuration in the image "
    "and the graph description.\n"
    "Scene Graph (Textual):\n{scene_graph}\n"
    "Question: {question}"
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
    question: str,
    mode: str,
    scene_graph_text: Optional[str] = None,
) -> tuple[str, str]:
    if mode == "visual":
        return SYSTEM_VISUAL, USER_VISUAL.format(question=question)
    else:
        if scene_graph_text is None:
            raise ValueError("scene_graph_text required for visual_textual mode")
        return SYSTEM_VISUAL_TEXTUAL, USER_VISUAL_TEXTUAL.format(
            scene_graph=scene_graph_text, question=question
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

        question = ex["question"]
        try:
            system, user = build_prompt(question, mode, ex.get("scene_graph_text"))
        except ValueError:
            continue

        messages = format_messages(system, user, f"file://{full_path.resolve()}")
        outputs = llm.chat(messages=[messages], sampling_params=sampling_params)
        prediction = outputs[0].outputs[0].text.strip()

        results.append({
            "image_path": str(image_path),
            "question": question,
            "prediction": prediction,
            "answer": ex.get("answer"),
            "image_id": ex.get("image_id"),
        })

        if (i + 1) % 50 == 0:
            log.info(f"Processed {i + 1}/{len(examples)}")

    return results


def save_results(results: list[dict], path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="VQA inference with vLLM")
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

    # Simple accuracy if ground truth available
    with_gt = [r for r in results if r.get("answer")]
    if with_gt:
        correct = sum(1 for r in with_gt if r["prediction"].lower() == r["answer"].lower())
        log.info(f"Accuracy: {correct}/{len(with_gt)} ({100*correct/len(with_gt):.1f}%)")


if __name__ == "__main__":
    main()
