import torch.multiprocessing as mp
mp.set_start_method("spawn", force=True)

"""
Visual Question Answering using Vision-Language Models (VLMs)

This script supports both vLLM and Hugging Face based multimodal models under
an Ubuntu 22.04 + CUDA 12.2 Docker environment.
"""

import argparse
import json
import logging
import os
import time
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict, List, Optional, Union

import requests

from huggingface_hub import login as hf_login
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("Please set HF_TOKEN environment variable with your Hugging Face access token.")
hf_login(token=HF_TOKEN)

# Monkey-patch PEFT LoraModel to expose prepare_inputs_for_generation
try:
    from peft.tuners.lora.model import LoraModel
    def _lora_prepare(self, *args, **kwargs):
        return self.model.prepare_inputs_for_generation(*args, **kwargs)
    LoraModel.prepare_inputs_for_generation = _lora_prepare
except ImportError:
    pass

import torch
from PIL import Image
from tqdm import tqdm

# Optional HF dependencies
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoProcessor,
    Blip2ForConditionalGeneration,
    BlipProcessor,
    LlavaForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    AutoTokenizer
)

# Optional vLLM dependencies
try:
    from vllm import LLM, SamplingParams
    VLLM_OK = True
except ImportError:
    VLLM_OK = False

# Qwen utils
try:
    from qwen_vl_utils import process_vision_info
    QWEN_UTILS_OK = True
except ImportError:
    QWEN_UTILS_OK = False

# Logging config
torch.backends.cudnn.benchmark = True
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
    handlers=[logging.FileHandler("vqa_generation.log"), logging.StreamHandler()],
)
logger = logging.getLogger("vqa")

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class VQAExample:
    image_path: str
    question: str
    answer: Optional[str] = None
    image_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_path": self.image_path,
            "question": self.question,
            "answer": self.answer,
            "image_id": self.image_id,
            "metadata": self.metadata or {},
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "VQAExample":
        return cls(
            image_path=d["image_path"],
            question=d["question"],
            answer=d.get("answer"),
            image_id=d.get("image_id"),
            metadata=d.get("metadata", {}),
        )

# ---------------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------------

def load_image(path_or_url: str) -> Image.Image:
    if path_or_url.lower().startswith(("http://", "https://")):
        resp = requests.get(path_or_url, stream=True, timeout=30)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content))
    else:
        img = Image.open(path_or_url)
    return img.convert("RGB")


def load_examples(json_path: str) -> List[VQAExample]:
    encodings = ["utf-8", "latin-1", "utf-16"]
    for enc in encodings:
        try:
            with open(json_path, "r", encoding=enc) as fp:
                data = json.load(fp)
            return [VQAExample.from_dict(item) for item in data]
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"Unable to decode JSON file: {json_path}")

# ---------------------------------------------------------------------------
# vLLM wrapper
# ---------------------------------------------------------------------------

class VLLMWrapper:
    def __init__(
        self,
        model_name: str,
        *,
        device: str = "cuda",
        max_length: int = 512,
        temperature: float = 0.2,
        top_p: float = 0.9,
        tensor_parallel_size: int = 1,
    ):
        if not VLLM_OK:
            raise ImportError("Please install vLLM: pip install vllm")
        self.max_length = max_length
        self.is_vl = any(tok in model_name.lower() for tok in (
            "qwen", "llava", "blip", "vision", "janus", "phi-4"
        ))
        logger.info("[vLLM] loading %s", model_name)
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            dtype="half" if device == "cuda" else "float32",
            trust_remote_code=True,
        )
        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_length,
        )

    def generate(self, prompt: str, *, image_path: Optional[str] = None) -> str:
        if self.is_vl and image_path:
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_path}},
                    {"type": "text", "text": prompt},
                ],
            }]
            out = self.llm.chat(messages, sampling_params=self.sampling_params)
        else:
            out = self.llm.generate([prompt], self.sampling_params)
        return out[0].outputs[0].text.strip() if out and out[0].outputs else ""

# ---------------------------------------------------------------------------
# Hugging Face VLM wrapper
# ---------------------------------------------------------------------------

class HFVLModel:
    def __init__(
        self,
        model_name: str,
        *,
        device: str = "cuda",
        max_length: int = 512,
        temperature: float = 0.2,
        top_p: float = 0.9,
        torch_dtype: str = "auto",
    ):
        # detect device
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.max_length = max_length
        self.temperature = temperature
        self.top_p = top_p
        self.name_lower = model_name.lower()
        self.is_qwen_vl = "qwen2.5-vl" in self.name_lower
        self.is_phi4 = "phi-4-multimodal-instruct" in self.name_lower
        self.is_pixtral = "pixtral" in self.name_lower


        logger.info("[HF] loading %s", model_name)
        # attempt to load config, fallback if unsupported
        try:
            config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        except Exception as e:
            logger.warning(
                "Could not load config for %s: %s — proceeding without explicit config",
                model_name,
                e,
            )
            config = None
        # determine dtype
        if torch_dtype == "auto":
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        else:
            dtype = getattr(torch, torch_dtype)

        # instantiate model and processor based on type
        if self.is_qwen_vl:
            self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name, config=config, torch_dtype=dtype, device_map="auto"
            )
        elif self.is_phi4:
            self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, config=config, trust_remote_code=True, torch_dtype=dtype, device_map="auto"
            )
        elif "janus-pro" in self.name_lower:
            self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name, config=config, trust_remote_code=True, torch_dtype=dtype, device_map="auto"
                )
            except ValueError:
                # fallback without explicit config
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name, trust_remote_code=True, torch_dtype=dtype, device_map="auto"
                )
        elif "llava" in self.name_lower:
            from transformers import LlavaProcessor
            self.processor = LlavaProcessor.from_pretrained(model_name, trust_remote_code=True)
            self.model = LlavaForConditionalGeneration.from_pretrained(
                model_name, config=config, torch_dtype=dtype, trust_remote_code=True, device_map="auto", low_cpu_mem_usage=True
            )
            self.model.to(self.device)
        elif "blip" in self.name_lower:
            self.processor = BlipProcessor.from_pretrained(model_name, trust_remote_code=True)
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                model_name, config=config, torch_dtype=dtype
            )
        elif self.is_pixtral:
            try:
                # First try to load LlamaTokenizer which is often used for Mistral models
                from transformers import LlamaTokenizer
                self.processor = LlamaTokenizer.from_pretrained(model_name, trust_remote_code=True)
            except Exception as e:
                logger.warning(f"Failed to load LlamaTokenizer: {e}")
                try:
                    # Fallback to MistralTokenizer if available
                    from transformers import MistralTokenizer
                    self.processor = MistralTokenizer.from_pretrained(model_name, trust_remote_code=True)
                except Exception as e:
                    logger.warning(f"Failed to load MistralTokenizer: {e}")
                    # Last resort: try to use AutoTokenizer with a similar model
                    try:
                        self.processor = AutoTokenizer.from_pretrained(
                            "mistralai/Mistral-7B-v0.1", 
                            trust_remote_code=True
                        )
                        logger.warning("Using fallback tokenizer from Mistral-7B-v0.1")
                    except Exception as e:
                        logger.error(f"All tokenizer loading attempts failed: {e}")
                        raise
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name, 
                    config=config, 
                    trust_remote_code=True, 
                    torch_dtype=dtype, 
                    device_map="auto"
                )
            except Exception as e:
                logger.error(f"Failed to load Pixtral model: {e}")
                raise

        else:
            self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, config=config, trust_remote_code=True, torch_dtype=dtype
            )

        self.model.eval()

    def generate(self, prompt: str, *, image_path: Optional[str] = None) -> str:
        if self.is_pixtral:
            if image_path:
                logger.warning("Pixtral model does not support image input. Ignoring image.")
            inputs = self.processor(prompt, return_tensors="pt").to(self.device)
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_length,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
            )
            return self.processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        if self.is_qwen_vl:
            return self._gen_qwen(prompt, image_path)
        if self.is_phi4 and image_path:
            return self._gen_phi4(prompt, image_path)
        if self.is_pixtral:
            if image_path:
                logger.warning("Pixtral model does not support image input. Ignoring image.")
            # Use tokenizer-style encoding for Pixtral
            inputs = self.processor(prompt, return_tensors="pt").to(self.device)
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_length,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
            )
            return self.processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        if image_path:
            img = load_image(image_path)
            multimodal_prompt = "<image> " + prompt
            inputs = self.processor(text=multimodal_prompt, images=img, return_tensors="pt").to(self.device)
        else:
            inputs = self.processor(text=prompt, return_tensors="pt").to(self.device)
        out = self.model.generate(
            **inputs,
            max_new_tokens=self.max_length,
            do_sample=True,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        return self.processor.decode(out[0], skip_special_tokens=True).strip()

    def _gen_qwen(self, prompt: str, image_path: Optional[str]) -> str:
        if not image_path:
            inp = self.processor(prompt, return_tensors="pt").to(self.device)
            ids = self.model.generate(
                **inp,
                max_new_tokens=self.max_length,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
            )
            return self.processor.decode(ids[0], skip_special_tokens=True).strip()
        if not QWEN_UTILS_OK:
            raise RuntimeError("Install qwen_vl_utils for Qwen2.5-VL image inference")
        msgs = [{"role":"user","content":[{"type":"image","image":f"file://{image_path}"},{"type":"text","text":prompt}]}]
        text = self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        img_inputs, vid_inputs = process_vision_info(msgs)
        inp = self.processor(text=[text], images=img_inputs, videos=vid_inputs, padding=True, return_tensors="pt").to(self.device)
        gen = self.model.generate(
            **inp,
            max_new_tokens=self.max_length,
            do_sample=True,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        trimmed = gen[:, inp.input_ids.shape[1]:]
        return self.processor.decode(trimmed[0], skip_special_tokens=True).strip()

    def _gen_phi4(self, prompt: str, image_path: str) -> str:
        placeholder = "<|image_1|>"
        c = f"{placeholder}{prompt}"
        template = self.processor.tokenizer.apply_chat_template([{"role":"user","content":c}], tokenize=False, add_generation_prompt=True)
        img = load_image(image_path)
        inputs = self.processor(template, [img], return_tensors="pt").to(self.device)
        prefix_len = inputs["input_ids"].shape[1]
        gen = self.model.generate(
            **inputs,
            max_new_tokens=self.max_length,
            do_sample=True,
            temperature=self.temperature,
            top_p=self.top_p,
            num_logits_to_keep=prefix_len,
        )
        out = gen[:, prefix_len:]
        return self.processor.batch_decode(out, skip_special_tokens=True)[0].strip()

# ---------------------------------------------------------------------------
# Runner & evaluation
# ---------------------------------------------------------------------------

def run_vqa(
    examples: List[VQAExample],
    model: Union[VLLMWrapper, HFVLModel],
    out_json: str,
    prompt_tpl: str,
    batch_size: int,
    max_qpi: int,
    max_imgs: int,
) -> List[Dict[str, Any]]:
    os.makedirs(os.path.dirname(out_json) or '.', exist_ok=True)
    grouped: Dict[str, List[VQAExample]] = {}
    for ex in examples:
        grouped.setdefault(ex.image_path, []).append(ex)
    paths = list(grouped)
    if max_imgs > 0:
        paths = paths[:max_imgs]
    results: List[Dict[str, Any]] = []
    for img_path in paths:
        group = grouped[img_path]
        if max_qpi > 0:
            group = group[:max_qpi]
        for i in tqdm(range(0, len(group), batch_size), desc='batches'):
            batch = group[i:i+batch_size]
            for ex in batch:
                prompt = prompt_tpl.format(question=ex.question)
                start_time = time.time()
                ans = model.generate(prompt, image_path=ex.image_path)
                # after you call model.generate…
                # if full looks like "Answer with only one word.\nQuestion: Is this a horse?\nAnswer: No"
                # this will grab just "No"
                if "Answer:" in ans:
                    ans = ans.rsplit("Answer:", 1)[-1].strip()
                    ans = ans.strip().strip("'\"")
                else:
                    ans = ans.strip()

                elapsed = time.time() - start_time
                logger.info("%s | %.2fs | %s", os.path.basename(ex.image_path), elapsed, ans[:80])
                results.append({**ex.to_dict(), "generated_answer": ans, "processing_time": elapsed})
            with open(out_json, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
    return results


def evaluate(results: List[Dict[str, Any]]) -> Dict[str, float]:
    gold = [r for r in results if r.get("answer")]
    if not gold:
        return {}
    correct = sum(r["answer"].strip().lower() == r["generated_answer"].strip().lower() for r in gold)
    avg_time = sum(r["processing_time"] for r in gold) / len(gold)
    return {"total": len(gold), "exact": correct, "exact_percent": 100 * correct / len(gold), "avg_time": avg_time}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("VQA pipeline")
    ap.add_argument("--input_file", required=True)
    ap.add_argument("--output_file", default="vqa_results.json")
    ap.add_argument("--image_dir")
    ap.add_argument("--model_name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--use_vllm", action="store_true")
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--prompt_template", default="Question: {question}\nAnswer:")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    ap.add_argument("--max_images", type=int, default=-1)
    ap.add_argument("--max_questions_per_image", type=int, default=-1)
    return ap.parse_args()


def main():
    args = parse_args()
    examples = load_examples(args.input_file)
    if args.image_dir:
        for ex in examples:
            if not os.path.isabs(ex.image_path):
                ex.image_path = os.path.join(args.image_dir, ex.image_path)
    if args.use_vllm:
        model = VLLMWrapper(
            args.model_name,
            device=args.device,
            max_length=args.max_length,
            temperature=args.temperature,
            top_p=args.top_p,
            tensor_parallel_size=args.tensor_parallel_size,
        )
    else:
        model = HFVLModel(
            args.model_name,
            device=args.device,
            max_length=args.max_length,
            temperature=args.temperature,
            top_p=args.top_p,
        )
    results = run_vqa(
        examples,
        model,
        out_json=args.output_file,
        prompt_tpl=args.prompt_template,
        batch_size=args.batch_size,
        max_qpi=args.max_questions_per_image,
        max_imgs=args.max_images,
    )
    metrics = evaluate(results)
    if metrics:
        metrics_path = os.path.splitext(args.output_file)[0] + "_metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as fp:
            json.dump(metrics, fp, indent=2, ensure_ascii=False)
        logger.info("Metrics saved to %s: %s", metrics_path, metrics)

if __name__ == "__main__":
    main()
