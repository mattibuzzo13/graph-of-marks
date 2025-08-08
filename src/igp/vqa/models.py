# igp/vqa/models.py
from __future__ import annotations
import os
import time
import torch
from typing import Any, Dict, List, Optional

from torch.amp import autocast
from transformers import (
    AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText,
    AutoProcessor, BitsAndBytesConfig,
    Blip2ForConditionalGeneration, BlipProcessor,
    LlavaForConditionalGeneration,
)
from .io import load_image

# vLLM (opzionale)
try:
    from vllm import LLM, SamplingParams  # type: ignore
    VLLM_OK = True
except Exception:
    VLLM_OK = False

# Qwen (opzionale)
try:
    from transformers import Qwen2_5_VLForConditionalGeneration  # type: ignore
    from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig  # type: ignore
    from qwen_vl_utils import process_vision_info  # type: ignore
    QWEN_OK = True
except Exception:
    QWEN_OK = False

# ---------------------------------------------------------------------
# Helper download (progress bar) — compatibile Hugging Face hub
# ---------------------------------------------------------------------
from pathlib import Path
from packaging import version
from huggingface_hub import __version__ as HF_VER
from huggingface_hub import HfApi, hf_hub_download

def download_repo_with_bar(repo_id: str, cache_dir: str) -> str:
    api = HfApi()
    try:
        info = api.model_info(repo_id, repo_type="model")
    except TypeError:
        info = api.model_info(repo_id)

    siblings = getattr(info, "siblings", getattr(info, "files", []))
    files    = [s.rfilename for s in siblings]
    total    = sum((s.size or 0) for s in siblings)

    try:
        from tqdm.auto import tqdm
        pbar = tqdm(total=total, unit="B", unit_scale=True, desc=f"Downloading {repo_id}")
    except Exception:
        pbar = None

    local_snapshot: Optional[Path] = None
    for f in files:
        local_file = Path(hf_hub_download(repo_id, filename=f, cache_dir=cache_dir, resume_download=True))
        if local_snapshot is None:
            local_snapshot = local_file.parent
        if pbar:
            pbar.update(local_file.stat().st_size)
    if pbar:
        pbar.close()
    return str(local_snapshot)

# ---------------------------------------------------------------------
# vLLM wrapper
# ---------------------------------------------------------------------
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
            raise ImportError("Install vLLM to use VLLMWrapper.")
        self.is_vl = any(t in model_name.lower() for t in ("gemma", "smolvlm2", "qwen", "llava", "blip", "phi-4", "bagel", "llamav"))
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            dtype="half" if device == "cuda" else "float32",
            trust_remote_code=True,
        )
        self.sampling = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_length)

    def generate(self, prompt: str, *, image_path: Optional[str] = None) -> str:
        with torch.inference_mode(), autocast(device_type="cuda"):
            if self.is_vl and image_path:
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_path}},
                        {"type": "text", "text": prompt},
                    ],
                }]
                out = self.llm.chat(messages, sampling_params=self.sampling)
            else:
                out = self.llm.generate([prompt], self.sampling)
        return out[0].outputs[0].text.strip() if out and out[0].outputs else ""

# ---------------------------------------------------------------------
# HF Transformers wrapper (multi–modello, robusto)
# ---------------------------------------------------------------------
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
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.max_length, self.temperature, self.top_p = max_length, temperature, top_p
        name = model_name.lower()
        self.is_gemma  = "gemma" in name
        self.is_smol   = "smolvlm2" in name
        self.is_qwen   = "qwen2.5-vl" in name
        self.is_phi4   = "phi-4-multimodal-instruct" in name
        self.is_pixtral= "pixtral" in name
        self.is_bagel  = "bagel" in name
        self.is_llamav = ("llamav" in name) or ("llamav-o1" in name) or ("llamav_o1" in name)

        if torch_dtype == "auto" and torch.cuda.is_available():
            dtype = torch.bfloat16
        else:
            dtype = getattr(torch, torch_dtype, torch.float32)

        cfg = None
        try:
            cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        except Exception:
            pass

        def _load_auto():
            self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(model_name, config=cfg, torch_dtype=dtype, trust_remote_code=True)

        # — Gemma, Qwen, Bagel, LLaMA-V: percorsi dedicati con offload/quant —
        if self.is_gemma:
            cache_dir  = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
            local_repo = download_repo_with_bar(model_name, cache_dir)
            self.processor = AutoProcessor.from_pretrained(local_repo, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                local_repo, device_map="auto", torch_dtype=dtype, trust_remote_code=True,
                low_cpu_mem_usage=True, max_memory={0: "20GiB", "cpu": "8GiB"},
                offload_folder="./offload", offload_buffers=True,
            )
        elif self.is_smol:
            self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_name, device_map="auto", torch_dtype=dtype, trust_remote_code=True
            )
        elif self.is_qwen and QWEN_OK:
            bnb_cfg = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0)
            cache_dir  = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
            local_repo = download_repo_with_bar(model_name, cache_dir)

            # Patch TF >= 4.52 parallel_attn (robusta su cfg dict)
            import transformers.modeling_utils as _mu
            if not getattr(_mu, "ALL_PARALLEL_STYLES", None):
                _mu.ALL_PARALLEL_STYLES = ["none", "tp", "rowwise", "colwise"]

            raw_cfg_dict = AutoConfig.from_pretrained(local_repo, trust_remote_code=True).to_dict()
            def _as_list(v):
                if v is None: return ["none"]
                return v if isinstance(v, list) else [v]
            raw_cfg_dict["parallel_attn"] = _as_list(raw_cfg_dict.get("parallel_attn"))
            if "text_config" in raw_cfg_dict:
                raw_cfg_dict["text_config"]["parallel_attn"] = _as_list(raw_cfg_dict["text_config"].get("parallel_attn"))
            if "vision_config" in raw_cfg_dict:
                raw_cfg_dict["vision_config"]["parallel_attn"] = _as_list(raw_cfg_dict["vision_config"].get("parallel_attn"))
            cfg = Qwen2_5_VLConfig(**raw_cfg_dict)

            self.processor = AutoProcessor.from_pretrained(local_repo, trust_remote_code=True)
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                local_repo, config=cfg, device_map="auto", torch_dtype=torch.float16,
                quantization_config=bnb_cfg, low_cpu_mem_usage=True,
                max_memory={0: "15GiB", "cpu": "4GiB"},
                offload_folder="./offload", trust_remote_code=True,
            )
            self.model.eval()
        elif self.is_bagel:
            bnb_cfg = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0)
            cache_dir  = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
            local_repo = download_repo_with_bar(model_name, cache_dir)
            self.processor = AutoProcessor.from_pretrained(local_repo, trust_remote_code=True)
            try:
                self.model = AutoModelForImageTextToText.from_pretrained(
                    local_repo, device_map="auto", torch_dtype=torch.float16,
                    quantization_config=bnb_cfg, trust_remote_code=True,
                    low_cpu_mem_usage=True, max_memory={0: "20GiB", "cpu": "8GiB"},
                    offload_folder="./offload", offload_buffers=True,
                )
            except Exception:
                self.model = AutoModelForCausalLM.from_pretrained(
                    local_repo, device_map="auto", torch_dtype=torch.float16,
                    quantization_config=bnb_cfg, trust_remote_code=True,
                    low_cpu_mem_usage=True, max_memory={0: "20GiB", "cpu": "8GiB"},
                    offload_folder="./offload", offload_buffers=True,
                )
            self.model.eval()
        elif self.is_llamav:
            bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
            cache_dir  = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
            local_repo = download_repo_with_bar(model_name, cache_dir)
            self.processor = AutoProcessor.from_pretrained(local_repo, trust_remote_code=True)
            try:
                self.model = AutoModelForImageTextToText.from_pretrained(
                    local_repo, device_map="auto", torch_dtype=torch.float16,
                    quantization_config=bnb_cfg, trust_remote_code=True,
                    low_cpu_mem_usage=True, max_memory={0: "20GiB", "cpu": "8GiB"},
                    offload_folder="./offload", offload_buffers=True,
                )
            except Exception:
                self.model = AutoModelForCausalLM.from_pretrained(
                    local_repo, config=cfg, device_map="auto", torch_dtype=torch.float16,
                    quantization_config=bnb_cfg, trust_remote_code=True,
                    low_cpu_mem_usage=True, max_memory={0: "20GiB", "cpu": "8GiB"},
                    offload_folder="./offload", offload_buffers=True,
                )
            self.model.eval()
        elif self.is_phi4:
            self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, config=cfg, device_map="auto", torch_dtype=dtype, trust_remote_code=True
            )
        elif "llava" in name:
            from transformers import LlavaProcessor  # type: ignore
            self.processor = LlavaProcessor.from_pretrained(model_name, trust_remote_code=True)
            self.model = LlavaForConditionalGeneration.from_pretrained(
                model_name, config=cfg, device_map="auto", torch_dtype=dtype
            )
        elif "blip" in name:
            self.processor = BlipProcessor.from_pretrained(model_name, trust_remote_code=True)
            self.model = Blip2ForConditionalGeneration.from_pretrained(model_name, config=cfg, torch_dtype=dtype)
        else:
            _load_auto()

        if not getattr(self.model, "hf_device_map", None):
            self.model.to(self.device)
        self.model.eval()

    # — Implementazioni per modello (solo quelle che necessitano handler custom) —
    def _gen_qwen(self, prompt: str, image_path: Optional[str]):
        assert QWEN_OK, "Install qwen_vl_utils to use Qwen2.5-VL"
        if not image_path:
            toks = self.processor(prompt, return_tensors="pt").to(self.device)
            gen = self.model.generate(**toks, max_new_tokens=self.max_length, do_sample=True, temperature=self.temperature, top_p=self.top_p)
            return self.processor.decode(gen[0], skip_special_tokens=True).strip()
        msgs=[{"role":"user","content":[{"type":"image","image":f"file://{image_path}"},{"type":"text","text":prompt}]}]
        txt=self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        img_inputs,vid_inputs = process_vision_info(msgs)
        inp=self.processor(text=[txt], images=img_inputs, videos=vid_inputs, padding=True, return_tensors="pt").to(self.device)
        gen=self.model.generate(**inp, max_new_tokens=self.max_length, do_sample=True, temperature=self.temperature, top_p=self.top_p)
        trimmed=gen[:, inp.input_ids.shape[1]:]
        return self.processor.decode(trimmed[0], skip_special_tokens=True).strip()

    def _gen_llamav(self, prompt: str, image_path: Optional[str]):
        if not image_path:
            toks = self.processor(prompt, return_tensors="pt").to(self.device)
            gen = self.model.generate(**toks, max_new_tokens=self.max_length, do_sample=True, temperature=self.temperature, top_p=self.top_p)
            return self.processor.decode(gen[0], skip_special_tokens=True).strip()
        img = load_image(image_path)
        messages = [{"role":"user","content":[{"type":"image","image":f"file://{image_path}"},{"type":"text","text":prompt}]}]
        chat_txt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[chat_txt], images=[img], return_tensors="pt", padding=True).to(self.device)
        prefix = inputs["input_ids"].shape[1]
        gen = self.model.generate(**inputs, max_new_tokens=self.max_length, do_sample=True, temperature=self.temperature, top_p=self.top_p)
        generated = gen[:, prefix:]
        return self.processor.batch_decode(generated, skip_special_tokens=True)[0].strip()

    def generate(self, prompt: str, *, image_path: Optional[str]=None) -> str:
        with torch.inference_mode(), autocast(device_type="cuda"):
            # Gemma (placeholder)
            if self.is_gemma and image_path:
                img = load_image(image_path)
                if "<start_of_image>" not in prompt:
                    prompt = f"<start_of_image> {prompt}"
                inputs = self.processor(text=prompt, images=img, return_tensors="pt").to(self.device)
                gen = self.model.generate(**inputs, max_new_tokens=self.max_length, do_sample=True, temperature=self.temperature, top_p=self.top_p, pad_token_id=self.processor.tokenizer.eos_token_id)
                generated = gen[:, inputs["input_ids"].shape[1]:]
                return self.processor.decode(generated[0], skip_special_tokens=True).strip()

            if self.is_qwen:   return self._gen_qwen(prompt, image_path)
            if self.is_llamav: return self._gen_llamav(prompt, image_path)

            # Generico multimodale (BLIP/LLaVA/Auto ImageTextToText)
            if image_path:
                img = load_image(image_path)
                multimodal_prompt = "<image> " + prompt
                inputs = self.processor(text=multimodal_prompt, images=img, return_tensors="pt").to(self.device)
                gen = self.model.generate(**inputs, max_new_tokens=self.max_length, do_sample=True, temperature=self.temperature, top_p=self.top_p)
                return self.processor.decode(gen[0], skip_special_tokens=True).strip()

            # Solo testo
            toks = self.processor(prompt, return_tensors="pt").to(self.device)
            gen = self.model.generate(**toks, max_new_tokens=self.max_length, do_sample=True, temperature=self.temperature, top_p=self.top_p)
            return self.processor.decode(gen[0], skip_special_tokens=True).strip()
