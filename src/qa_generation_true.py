
from __future__ import annotations
import torch.multiprocessing as mp
mp.set_start_method("spawn", force=True)

# stdlib
import argparse, json, logging, os, time
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict, List, Optional, Union

# third‑party
import requests, torch
from PIL import Image
from tqdm import tqdm
from huggingface_hub import login as hf_login

# vllm (optional)
try:
    from vllm import LLM, SamplingParams  # type: ignore
    VLLM_OK = True
except ImportError:
    VLLM_OK = False

# transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
    Blip2ForConditionalGeneration,
    BlipProcessor,
    LlavaForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Gemma3ForConditionalGeneration,
)

# qwen helper (optional)
try:
    from qwen_vl_utils import process_vision_info  # type: ignore
    QWEN_UTILS_OK = True
except ImportError:
    QWEN_UTILS_OK = False

# ---------------------------------------------------------------------------
# Setup & logging
# ---------------------------------------------------------------------------
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("Please set HF_TOKEN environment variable with your Hugging Face access token (HF_TOKEN).")
hf_login(token=HF_TOKEN)

torch.backends.cudnn.benchmark = True
logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
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

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "VQAExample":
        return cls(
            image_path=d["image_path"],
            question=d["question"],
            answer=d.get("answer"),
            image_id=d.get("image_id"),
            metadata=d.get("metadata", {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_path": self.image_path,
            "question": self.question,
            "answer": self.answer,
            "image_id": self.image_id,
            "metadata": self.metadata or {},
        }

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_image(path_or_url: str) -> Image.Image:
    if path_or_url.lower().startswith(("http://", "https://")):
        data = requests.get(path_or_url, timeout=30).content
        img = Image.open(BytesIO(data))
    else:
        img = Image.open(path_or_url)
    return img.convert("RGB")


def load_examples(fp: str) -> List[VQAExample]:
    for enc in ("utf‑8", "latin‑1", "utf‑16"):
        try:
            with open(fp, "r", encoding=enc) as f:
                js = json.load(f)
            return [VQAExample.from_dict(d) for d in js]
        except UnicodeDecodeError:
            continue
    raise RuntimeError("Cannot decode JSON file: " + fp)

# ---------------------------------------------------------------------------
# vLLM wrapper
# ---------------------------------------------------------------------------
class VLLMWrapper:
    def __init__(self, model_name: str, *, device="cuda", max_length=512, temperature=0.2, top_p=0.9, tensor_parallel_size=1):
        if not VLLM_OK:
            raise ImportError("Install vLLM to use this mode.")
        self.is_vl = any(t in model_name.lower() for t in ("gemma", "smolvlm2", "qwen", "llava", "blip", "phi-4"))
        self.llm = LLM(model=model_name, tensor_parallel_size=tensor_parallel_size, dtype="half" if device=="cuda" else "float32", trust_remote_code=True)
        self.sampling = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_length)

    def generate(self, prompt: str, *, image_path: Optional[str]=None) -> str:
        if self.is_vl and image_path:
            messages=[{"role":"user","content":[{"type":"image_url","image_url":{"url":image_path}},{"type":"text","text":prompt}]}]
            out=self.llm.chat(messages, sampling_params=self.sampling)
        else:
            out=self.llm.generate([prompt], self.sampling)
        return out[0].outputs[0].text.strip() if out and out[0].outputs else ""

# ---------------------------------------------------------------------------
# Hugging Face wrapper
# ---------------------------------------------------------------------------
class HFVLModel:
    def __init__(self, model_name: str, *, device="cuda", max_length=512, temperature=0.2, top_p=0.9, torch_dtype="auto"):
        self.device = torch.device(device if torch.cuda.is_available() and device=="cuda" else "cpu")
        self.max_length, self.temperature, self.top_p = max_length, temperature, top_p
        name = model_name.lower()
        self.is_gemma = "gemma" in name
        self.is_smol = "smolvlm2" in name
        self.is_qwen = "qwen2.5-vl" in name
        self.is_phi4 = "phi-4-multimodal-instruct" in name
        self.is_pixtral = "pixtral" in name

        dtype = torch.bfloat16 if torch_dtype=="auto" and torch.cuda.is_available() else getattr(torch, torch_dtype, torch.float32)
        cfg = None
        try:
            cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        except Exception:
            pass

        # Gemma 3 multimodal
        if self.is_gemma:
            self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            self.model = Gemma3ForConditionalGeneration.from_pretrained(model_name, device_map="auto", torch_dtype=dtype)

        # SmolVLM2 multimodal
        elif self.is_smol:
            self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModelForImageTextToText.from_pretrained(model_name, device_map="auto", torch_dtype=dtype, trust_remote_code=True)

        # Qwen‑VL
        elif self.is_qwen:
            self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, config=cfg, device_map="auto", torch_dtype=dtype)

        # Phi‑4 (text + image)
        elif self.is_phi4:
            self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(model_name, config=cfg, device_map="auto", torch_dtype=dtype, trust_remote_code=True)

        # LLaVA
        elif "llava" in name:
            from transformers import LlavaProcessor  # type: ignore
            self.processor = LlavaProcessor.from_pretrained(model_name, trust_remote_code=True)
            self.model = LlavaForConditionalGeneration.from_pretrained(model_name, config=cfg, device_map="auto", torch_dtype=dtype, low_cpu_mem_usage=True)
            self.model.to(self.device)

        # BLIP2
        elif "blip" in name:
            self.processor = BlipProcessor.from_pretrained(model_name, trust_remote_code=True)
            self.model = Blip2ForConditionalGeneration.from_pretrained(model_name, config=cfg, torch_dtype=dtype)

        # Pixtral (text‑only)
        elif self.is_pixtral:
            try:
                from transformers import LlamaTokenizer  # type: ignore
                self.processor = LlamaTokenizer.from_pretrained(model_name, trust_remote_code=True)
            except Exception:
                from transformers import MistralTokenizer  # type: ignore
                self.processor = MistralTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(model_name, config=cfg, device_map="auto", torch_dtype=dtype, trust_remote_code=True)
        else:
            self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(model_name, config=cfg, torch_dtype=dtype)

        self.model.eval()

    # ------------------------ generation helpers ------------------------
    def _chat(self, messages: List[Dict[str, Any]]) -> str:
        inp = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(self.model.device, dtype=self.model.dtype if hasattr(self.model, "dtype") else torch.float32)
        out = self.model.generate(**inp, max_new_tokens=self.max_length, do_sample=True, temperature=self.temperature, top_p=self.top_p)
        return self.processor.batch_decode(out, skip_special_tokens=True)[0].strip()

    def generate(self, prompt: str, *, image_path: Optional[str]=None) -> str:
        if self.is_gemma or self.is_smol:
            msgs=[{"role":"user","content":[{"type":"image" if image_path else "text","url" if image_path else "text": image_path if image_path else prompt},{"type":"text","text":prompt}] if image_path else [{"type":"text","text":prompt}]}]
            return self._chat(msgs)

        if self.is_qwen:
            return self._gen_qwen(prompt, image_path)

        if self.is_phi4 and image_path:
            return self._gen_phi4(prompt, image_path)

        # Pixtral & generic text models
        if self.is_pixtral or (not image_path):
            toks = self.processor(prompt, return_tensors="pt").to(self.device)
            gen = self.model.generate(**toks, max_new_tokens=self.max_length, do_sample=True, temperature=self.temperature, top_p=self.top_p)
            return self.processor.batch_decode(gen, skip_special_tokens=True)[0].strip()

        # Generic multimodal (BLIP / LLaVA)
        img = load_image(image_path)
        multimodal_prompt = "<image> " + prompt
        inputs = self.processor(text=multimodal_prompt, images=img, return_tensors="pt").to(self.device)
        gen = self.model.generate(**inputs, max_new_tokens=self.max_length, do_sample=True, temperature=self.temperature, top_p=self.top_p)
        return self.processor.decode(gen[0], skip_special_tokens=True).strip()

    # Qwen helper
    def _gen_qwen(self, prompt: str, image_path: Optional[str]):
        if not image_path:
            toks = self.processor(prompt, return_tensors="pt").to(self.device)
            gen = self.model.generate(**toks, max_new_tokens=self.max_length, do_sample=True, temperature=self.temperature, top_p=self.top_p)
            return self.processor.decode(gen[0], skip_special_tokens=True).strip()
        if not QWEN_UTILS_OK:
            raise RuntimeError("Install qwen_vl_utils for Qwen image support.")
        msgs=[{"role":"user","content":[{"type":"image","image":f"file://{image_path}"},{"type":"text","text":prompt}]}]
        txt=self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        img_inputs,vid_inputs = process_vision_info(msgs)
        inp=self.processor(text=[txt], images=img_inputs, videos=vid_inputs, padding=True, return_tensors="pt").to(self.device)
        gen=self.model.generate(**inp, max_new_tokens=self.max_length, do_sample=True, temperature=self.temperature, top_p=self.top_p)
        trimmed=gen[:, inp.input_ids.shape[1]:]
        return self.processor.decode(trimmed[0], skip_special_tokens=True).strip()

    # Phi‑4 helper
    def _gen_phi4(self, prompt: str, image_path: str):
        placeholder="<|image_1|>"
        content=f"{placeholder}{prompt}"
        tmpl=self.processor.tokenizer.apply_chat_template([{"role":"user","content":content}], tokenize=False, add_generation_prompt=True)
        img=load_image(image_path)
        inp=self.processor(tmpl, [img], return_tensors="pt").to(self.device)
        prefix=inp["input_ids"].shape[1]
        gen=self.model.generate(**inp, max_new_tokens=self.max_length, do_sample=True, temperature=self.temperature, top_p=self.top_p, num_logits_to_keep=prefix)
        return self.processor.batch_decode(gen[:, prefix:], skip_special_tokens=True)[0].strip()

# ---------------------------------------------------------------------------
# VQA runner
# ---------------------------------------------------------------------------

def run_vqa(examples: List[VQAExample], model: Union[VLLMWrapper, HFVLModel], *, out_json: str, prompt_tpl: str, batch_size: int, max_qpi: int, max_imgs: int) -> List[Dict[str, Any]]:
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    grouped: Dict[str, List[VQAExample]] = {}
    for ex in examples:
        grouped.setdefault(ex.image_path, []).append(ex)
    img_paths=list(grouped)[:max_imgs] if max_imgs>0 else list(grouped)
    results: List[Dict[str, Any]] = []
    for img in img_paths:
        qs=grouped[img][:max_qpi] if max_qpi>0 else grouped[img]
        for i in tqdm(range(0, len(qs), batch_size), desc=os.path.basename(img)):
            batch=qs[i:i+batch_size]
            for ex in batch:
                prompt=prompt_tpl.format(question=ex.question)
                t0=time.time()
                ans=model.generate(prompt, image_path=ex.image_path)
                if "Answer:" in ans:
                    ans=ans.rsplit("Answer:",1)[-1].strip().strip("'\"")
                results.append({**ex.to_dict(), "generated_answer": ans, "processing_time": time.time()-t0})
        json.dump(results, open(out_json,"w",encoding="utf‑8"), indent=2, ensure_ascii=False)
    return results

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def evaluate(res: List[Dict[str, Any]]) -> Dict[str, float]:
    gold=[r for r in res if r.get("answer")]
    if not gold:
        return {}
    corr=sum(r["answer"].strip().lower()==r["generated_answer"].strip().lower() for r in gold)
    return {"total":len(gold),"exact":corr,"exact_percent":100*corr/len(gold),"avg_time":sum(r["processing_time"] for r in gold)/len(gold)}

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    ap=argparse.ArgumentParser("VQA pipeline (Gemma & SmolVLM2 ready)")
    ap.add_argument("--input_file", required=True)
    ap.add_argument("--output_file", default="vqa_results.json")
    ap.add_argument("--image_dir")
    ap.add_argument("--model_name", default="google/gemma-3-4b-it")
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

# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main():
    args=parse_args()
    ex=load_examples(args.input_file)
    if args.image_dir:
        for e in ex:
            if not os.path.isabs(e.image_path):
                e.image_path=os.path.join(args.image_dir, e.image_path)
    mdl= VLLMWrapper(args.model_name, device=args.device, max_length=args.max_length, temperature=args.temperature, top_p=args.top_p, tensor_parallel_size=args.tensor_parallel_size) if args.use_vllm else HFVLModel(args.model_name, device=args.device, max_length=args.max_length, temperature=args.temperature, top_p=args.top_p)
    res=run_vqa(ex, mdl, out_json=args.output_file, prompt_tpl=args.prompt_template, batch_size=args.batch_size, max_qpi=args.max_questions_per_image, max_imgs=args.max_images)
    met=evaluate(res)
    if met:
        p=os.path.splitext(args.output_file)[0]+"_metrics.json"
        json.dump(met, open(p,"w",encoding="utf‑8"), indent=2, ensure_ascii=False)
        logger.info("Metrics: %s", met)

if __name__=="__main__":
    main()
