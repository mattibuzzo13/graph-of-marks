

from __future__ import annotations
import torch.multiprocessing as mp
# Imposta solo una volta lo start method "spawn" per evitare leak di semafori e segfault
try:
    if mp.get_start_method(allow_none=True) != "spawn":
        mp.set_start_method("spawn")
except RuntimeError:
    pass

# stdlib
import argparse
import json
import logging
import os
import time
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict, List, Optional, Union
import gc
import psutil
import glob

# third-party
import requests
import torch
from torch.amp import autocast
from PIL import Image
from tqdm import tqdm
from huggingface_hub import login as hf_login
from huggingface_hub import snapshot_download
from transformers import BitsAndBytesConfig
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"


# import the image preprocessor
try:
    from image_preprocessor import ImageGraphPreprocessor, parse_preproc_args
except ImportError:
    # se ti interessa solo l'inference non ti serve la libreria
    ImageGraphPreprocessor = None
    def parse_preproc_args():
        return argparse.Namespace()

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

from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig


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

torch.backends.cudnn.benchmark = True
logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger("vqa")




from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download
from tqdm.auto import tqdm

from packaging import version
from huggingface_hub import __version__ as HF_VER
from huggingface_hub import HfApi, hf_hub_download
from tqdm.auto import tqdm
from pathlib import Path

def download_repo_with_bar(repo_id: str, cache_dir: str) -> str:
    api = HfApi()

    # -- chiamata compatibile con tutte le versioni ------------------------
    try:
        info = api.model_info(repo_id, repo_type="model")   # new API (>=0.20)
    except TypeError:
        info = api.model_info(repo_id)                      # old API

    # nelle release <0.20 il campo si chiama .siblings anziché .files
    siblings = getattr(info, "siblings", getattr(info, "files", []))
    files    = [s.rfilename for s in siblings]
    total    = sum((s.size or 0) for s in siblings)

    pbar = tqdm(total=total, unit="B", unit_scale=True,
                desc=f"Downloading {repo_id}")

    local_snapshot: Path | None = None
    for f in files:
        local_file = Path(
            hf_hub_download(
                repo_id,
                filename=f,
                cache_dir=cache_dir,
                resume_download=True,
            )
        )
        if local_snapshot is None:
            # ~/.cache/.../models--repo--name/snapshots/<commit>
            local_snapshot = local_file.parent
        pbar.update(local_file.stat().st_size)

    pbar.close()
    return str(local_snapshot)





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
            question=d.get("question", ""),
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
    for enc in ("utf-8", "latin-1", "utf-16"):
        try:
            with open(fp, "r", encoding=enc) as f:
                js = json.load(f)
            return [VQAExample.from_dict(d) for d in js]
        except UnicodeDecodeError:
            continue
    raise RuntimeError("Cannot decode JSON file: " + fp)

# ---------------------------------------------------------------------------
# Image preprocessing per QA pair
# ---------------------------------------------------------------------------
def preprocess_for_qa(
    image_path: str,
    question: str,
    *,
    output_folder: str = "preprocessed",
    apply_question_filter: bool = True,
    preproc_obj: Optional[ImageGraphPreprocessor] = None,
    preproc_cli_args: Optional[Dict[str, Any]] = None,
    base_config: Optional[Dict[str, Any]] = None,
    force_reprocess: bool = False,
) -> str:
    """
    Restituisce il path dell'immagine annotata.
    SEMPRE usa nome originale + hash domanda per il matching.
    Ritorna il path effettivo scritto dal pre-processor se disponibile,
    altrimenti effettua un fallback di ricerca.
    """
    import hashlib, os, glob

    base  = os.path.splitext(os.path.basename(image_path))[0]
    qhash = hashlib.md5(question.encode("utf-8")).hexdigest()[:8]
    out_fn = f"{base}_{qhash}_output.jpg"
    os.makedirs(output_folder, exist_ok=True)
    out_path = os.path.join(output_folder, out_fn)

    # 1) Cache: se già esiste e non forzo, riusa
    if os.path.exists(out_path) and not force_reprocess:
        logger.info(f"Immagine preprocessata già esistente: {out_path}")
        return out_path

    # 2) Costruisci config per il pre-processor
    if base_config:
        cfg = base_config.copy()
    else:
        cfg = preproc_cli_args.__dict__.copy() if preproc_cli_args else {}

    cfg.update({
        "input_path": None,
        "output_folder": output_folder,
        "preproc_device": cfg.get("preproc_device", "cuda"),
        "apply_question_filter": apply_question_filter,
        "question": question,
    })

    # 3) Istanzia/aggiorna pre-processor (nuovo per domanda, o riusa con stato aggiornato)
    if preproc_obj is None:
        fresh_preproc = ImageGraphPreprocessor(cfg)
    else:
        fresh_preproc = preproc_obj
        fresh_preproc.config.update(cfg)
        fresh_preproc.question = question
        fresh_preproc._build_question_semantics()

    # 4) Esegui preprocessing
    img_pil = load_image(image_path)
    ret = fresh_preproc.process_single_image(
        img_pil, f"{base}_{qhash}", det_cache_key=f"{base}_{qhash}"
    )

    # 5) Se il pre-processor ritorna un path/dict con path valido, usalo
    if isinstance(ret, str) and os.path.exists(ret):
        return ret
    if isinstance(ret, dict):
        for k in ("output_path", "annotated_path", "output", "annotated"):
            p = ret.get(k)
            if p and os.path.exists(p):
                return p

    # 6) Se il path "atteso" esiste, ritorna quello
    if os.path.exists(out_path):
        return out_path

    # 7) Fallback di ricerca (matching su nome originale + hash)
    candidate_dirs = list({
        output_folder,
        fresh_preproc.config.get("output_folder", ""),
        "output_images"  # molti pre-processor usano questo default
    })
    candidate_dirs = [d for d in candidate_dirs if d]

    # cerca varianti di nome/estensione
    patterns = [
        f"{base}_{qhash}_output.*",
        f"{base}_{qhash}*annotat*.*",   # in caso cambi suffisso
        f"{base}_{qhash}*.*",           # ultima spiaggia
    ]
    for d in candidate_dirs:
        for pat in patterns:
            matches = glob.glob(os.path.join(d, pat))
            if matches:
                logger.info(f"Usato fallback: {matches[0]}")
                return matches[0]

    logger.warning(f"File annotato non trovato: atteso {out_path}. Ritorno il path atteso comunque.")
    return out_path



# ---------------------------------------------------------------------------
# vLLM wrapper
# ---------------------------------------------------------------------------
class VLLMWrapper:
    def __init__(
        self,
        model_name: str,
        *,
        device="cuda",
        max_length=512,
        temperature=0.2,
        top_p=0.9,
        tensor_parallel_size=1
    ):
        if not VLLM_OK:
            raise ImportError("Install vLLM to use this mode.")
        self.is_vl = any(
            t in model_name.lower() for t in
            ("gemma", "smolvlm2", "qwen", "llava", "blip", "phi-4", "bagel")
        )
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            dtype="half" if device == "cuda" else "float32",
            trust_remote_code=True
        )
        self.sampling = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_length
        )

    def generate(self, prompt: str, *, image_path: Optional[str] = None) -> str:
        # Mixed-precision + no-grad for vLLM
        with torch.inference_mode(), autocast(device_type='cuda'):
            if self.is_vl and image_path:
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_path}},
                        {"type": "text", "text": prompt}
                    ]
                }]
                out = self.llm.chat(messages, sampling_params=self.sampling)
            else:
                out = self.llm.generate([prompt], self.sampling)
        return out[0].outputs[0].text.strip() if out and out[0].outputs else ""

# ---------------------------------------------------------------------------
# Hugging Face wrapper
# ---------------------------------------------------------------------------
class HFVLModel:
    def __init__(
        self,
        model_name: str,
        *,
        device="cuda",
        max_length=512,
        temperature=0.2,
        top_p=0.9,
        torch_dtype="auto"
    ):
        self.device = torch.device(
            device if torch.cuda.is_available() and device == "cuda" else "cpu"
        )
        self.max_length, self.temperature, self.top_p = max_length, temperature, top_p
        name = model_name.lower()
        self.is_gemma = "gemma" in name
        self.is_smol = "smolvlm2" in name
        self.is_qwen = "qwen2.5-vl" in name
        self.is_phi4 = "phi-4-multimodal-instruct" in name
        self.is_pixtral = "pixtral" in name
        self.is_bagel   = "bagel" in name
        self.is_llamav = ("llamav" in name) or ("llamav-o1" in name) or ("llamav_o1" in name)


        dtype = (
            torch.bfloat16 if torch_dtype == "auto" and torch.cuda.is_available()
            else getattr(torch, torch_dtype, torch.float32)
        )
        cfg = None
        try:
            cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        except Exception:
            pass

        # Model-specific loading with device offload
        # ────────────────────────────────────────────────────────
        # Gemma 3‑4B‑IT  ➜  caricamento “vanilla” da Hugging Face
        # ────────────────────────────────────────────────────────
        # Model-specific loading with device offload
        if self.is_gemma:
            # Usa la stessa strategia di Qwen per Gemma
            cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
            local_repo = download_repo_with_bar(model_name, cache_dir)

            self.processor = AutoProcessor.from_pretrained(local_repo, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                local_repo,
                device_map="auto",
                torch_dtype=dtype,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                max_memory={0: "20GiB", "cpu": "8GiB"},  # Gemma 3-4B richiede meno memoria
                offload_folder="./offload",
                offload_buffers=True,
            )
        elif self.is_smol:
            self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_name, device_map="auto", torch_dtype=dtype, trust_remote_code=True
            )
        elif self.is_qwen:
            # ①  quantizzazione (8-bit) facoltativa
            bnb_cfg = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0)

            # ②  scarica lo snapshot una sola volta con barra di progresso
            cache_dir  = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
            local_repo = download_repo_with_bar(model_name, cache_dir)

            # ③  patch temporanea per il bug di TF 4.52 → ALL_PARALLEL_STYLES None
            import transformers.modeling_utils as _mu
            if not getattr(_mu, "ALL_PARALLEL_STYLES", None):
                _mu.ALL_PARALLEL_STYLES = ["none", "tp", "rowwise", "colwise"]

            # ④  carica processor e config "grezzo"
            self.processor = AutoProcessor.from_pretrained(local_repo, trust_remote_code=True)
            raw_cfg_dict   = AutoConfig.from_pretrained(local_repo,
                                                        trust_remote_code=True).to_dict()

            # helper
            def _as_list(v):
                if v is None:
                    return ["none"]          # valore esplicito valido
                if isinstance(v, str):
                    return [v]
                return v

            # ⑤  correzione nel dict (root / text / vision)
            raw_cfg_dict["parallel_attn"] = _as_list(raw_cfg_dict.get("parallel_attn"))
            if "text_config" in raw_cfg_dict:
                raw_cfg_dict["text_config"]["parallel_attn"] = \
                    _as_list(raw_cfg_dict["text_config"].get("parallel_attn"))
            if "vision_config" in raw_cfg_dict:
                raw_cfg_dict["vision_config"]["parallel_attn"] = \
                    _as_list(raw_cfg_dict["vision_config"].get("parallel_attn"))

            # ⑥  ricrea il config patchato
            cfg = Qwen2_5_VLConfig(**raw_cfg_dict)

            # ⑦  doppio-check dopo la costruzione (property text_config restituisce copia)
            if getattr(cfg, "parallel_attn", None) is None:
                cfg.parallel_attn = ["none"]
            if getattr(cfg, "text_config", None) is not None and \
               getattr(cfg.text_config, "parallel_attn", None) is None:
                cfg.text_config.parallel_attn = ["none"]
            if getattr(cfg, "vision_config", None) is not None and \
               getattr(cfg.vision_config, "parallel_attn", None) is None:
                cfg.vision_config.parallel_attn = ["none"]

            # ⑧  carica finalmente il modello senza lanciare TypeError
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                local_repo,
                config              = cfg,
                device_map          = "auto",
                torch_dtype         = torch.float16,
                quantization_config = bnb_cfg,
                low_cpu_mem_usage   = True,
                max_memory          = {0: "15GiB", "cpu": "4GiB"},
                offload_folder      = "./offload",
                trust_remote_code   = True,
            )

        elif self.is_bagel:
            # ───────────────────────────────────────────────────────────────
            # BAGEL 7B MoT INT8 (multimodale, derivato da Qwen2.5)
            # ───────────────────────────────────────────────────────────────
            bnb_cfg = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0)

            cache_dir  = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
            local_repo = download_repo_with_bar(model_name, cache_dir)

            # Processor + modello (può essere ImageTextToText oppure CausalLM con trust_remote_code)
            self.processor = AutoProcessor.from_pretrained(local_repo, trust_remote_code=True)
            try:
                self.model = AutoModelForImageTextToText.from_pretrained(
                    local_repo,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    quantization_config=bnb_cfg,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    max_memory={0: "20GiB", "cpu": "8GiB"},
                    offload_folder="./offload",
                    offload_buffers=True,
                )
            except Exception:
                # fallback se la classe specifica non è disponibile
                self.model = AutoModelForCausalLM.from_pretrained(
                    local_repo,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    quantization_config=bnb_cfg,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    max_memory={0: "20GiB", "cpu": "8GiB"},
                    offload_folder="./offload",
                    offload_buffers=True,
                )
            # assicura eval mode
            self.model.eval()

        elif self.is_llamav:
            # Quantizzazione leggera (facoltativa)
            bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)

            # Scarico una sola volta con barra di progresso
            cache_dir  = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
            local_repo = download_repo_with_bar(model_name, cache_dir)

            # Processor + modello multimodale
            self.processor = AutoProcessor.from_pretrained(local_repo, trust_remote_code=True)

            try:
                # Molti port LLaMA-V espongono ImageTextToText
                self.model = AutoModelForImageTextToText.from_pretrained(
                    local_repo,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    quantization_config=bnb_cfg,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    max_memory={0: "20GiB", "cpu": "8GiB"},
                    offload_folder="./offload",
                    offload_buffers=True,
                )
            except Exception:
                # Fallback CausalLM
                self.model = AutoModelForCausalLM.from_pretrained(
                    local_repo,
                    config=cfg,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    quantization_config=bnb_cfg,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    max_memory={0: "20GiB", "cpu": "8GiB"},
                    offload_folder="./offload",
                    offload_buffers=True,
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
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                model_name, config=cfg, torch_dtype=dtype
            )
        elif self.is_pixtral:
            try:
                from transformers import LlamaTokenizer  # type: ignore
                self.processor = LlamaTokenizer.from_pretrained(model_name, trust_remote_code=True)
            except Exception:
                from transformers import MistralTokenizer  # type: ignore
                self.processor = MistralTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, config=cfg, device_map="auto", torch_dtype=dtype, trust_remote_code=True
            )
        else:
            self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, config=cfg, torch_dtype=dtype
            )
        if not getattr(self.model, "hf_device_map", None):
            self.model.to(self.device)
        self.model.eval()

    def _chat(self, messages: List[Dict[str, Any]]) -> str:
        inp = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt"
        ).to(self.model.device, dtype=getattr(self.model, "dtype", torch.float32))
        out = self.model.generate(
            **inp, max_new_tokens=self.max_length,
            do_sample=True, temperature=self.temperature, top_p=self.top_p
        )
        return self.processor.batch_decode(out, skip_special_tokens=True)[0].strip()

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
    def _gen_bagel(self, prompt: str, image_path: Optional[str]):
        """
        Generazione per BAGEL: usa il chat_template del processor e passa immagine+testo.
        """
        if image_path is None:
            # solo testo
            toks = self.processor(prompt, return_tensors="pt").to(self.device)
            gen = self.model.generate(
                **toks,
                max_new_tokens=self.max_length,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p
            )
            return self.processor.decode(gen[0], skip_special_tokens=True).strip()

        # multimodale
        img = load_image(image_path)
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{image_path}"},
                {"type": "text",  "text": prompt}
            ]
        }]

        # Crea il testo con il template del modello (tokenize=False per gestire noi le immagini)
        chat_txt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(
            text=[chat_txt],
            images=[img],
            return_tensors="pt",
            padding=True
        ).to(self.device)

        prefix = inputs["input_ids"].shape[1]
        gen = self.model.generate(
            **inputs,
            max_new_tokens=self.max_length,
            do_sample=True, temperature=self.temperature, top_p=self.top_p,
            num_logits_to_keep=prefix
        )
        return self.processor.batch_decode(gen[:, prefix:], skip_special_tokens=True)[0].strip()
        
    def _gen_llamav(self, prompt: str, image_path: Optional[str]):
        """Generazione per LlamaV-o1."""
        if image_path is None:
            toks = self.processor(prompt, return_tensors="pt").to(self.device)
            gen = self.model.generate(
                **toks,
                max_new_tokens=self.max_length,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
            )
            return self.processor.decode(gen[0], skip_special_tokens=True).strip()

        img = load_image(image_path)
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{image_path}"},
                {"type": "text",  "text": prompt}
            ]
        }]

        chat_txt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(
            text=[chat_txt],
            images=[img],
            return_tensors="pt",
            padding=True
        ).to(self.device)

        prefix = inputs["input_ids"].shape[1]
        gen = self.model.generate(
            **inputs,
            max_new_tokens=self.max_length,
            do_sample=True,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        generated = gen[:, prefix:]
        return self.processor.batch_decode(generated, skip_special_tokens=True)[0].strip()



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

    def generate(self, prompt: str, *, image_path: Optional[str]=None) -> str:
        # Mixed-precision + no-grad for HF transformers
        with torch.inference_mode(), autocast(device_type='cuda'):
            if self.is_gemma:
                if image_path:
                    # ① carica l’immagine
                    img = load_image(image_path)

                    # ② assicura il placeholder nel testo
                    if "<start_of_image>" not in prompt:
                        prompt = f"<start_of_image> {prompt}"

                    # ③ tokenizza testo + immagine
                    inputs = self.processor(
                        text=prompt,
                        images=img,
                        return_tensors="pt"
                    ).to(self.device)

                    # ④ genera
                    gen = self.model.generate(
                        **inputs,
                        max_new_tokens=self.max_length,
                        do_sample=True,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        pad_token_id=self.processor.tokenizer.eos_token_id,
                    )

                    # ⑤ estrai solo la parte generata
                    generated = gen[:, inputs["input_ids"].shape[1]:]
                    return self.processor.decode(generated[0],
                                                  skip_special_tokens=True).strip()

            # Qwen2.5
            if self.is_qwen:
                return self._gen_qwen(prompt, image_path)
            if self.is_bagel:
                return self._gen_bagel(prompt, image_path)
            if self.is_llamav:
                return self._gen_llamav(prompt, image_path)


            # Generic multimodal (BLIP / LLaVA)
            img = load_image(image_path)
            multimodal_prompt = "<image> " + prompt
            inputs = self.processor(text=multimodal_prompt, images=img, return_tensors="pt").to(self.device)
            gen = self.model.generate(
                **inputs,
                max_new_tokens=self.max_length,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p
            )
            return self.processor.decode(gen[0], skip_special_tokens=True).strip()

# ---------------------------------------------------------------------------
# Pre-processing batch runner (solo annotazione immagini)
# ---------------------------------------------------------------------------
def run_preprocessing(
    examples: List[VQAExample],
    *,
    preproc_folder: str = "preprocessed",
    disable_q_filter: bool = False,
    max_imgs: int = -1,
    max_qpi: int = -1,
    preproc_args: Optional[argparse.Namespace] = None,
    preproc_obj: Optional[ImageGraphPreprocessor] = None,
    base_config: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Esegue SOLO il preprocessing (segmentazione, detezione, prompt-guided masking…)
    salvando l'immagine annotata su disco.

    Non carica alcun modello linguistico.

    Parametri:
        examples        – lista di VQAExample letti dal JSON
        preproc_folder  – directory dove salvare gli output
        disable_q_filter– disattiva il filtro basato sulla domanda
        max_imgs / max_qpi – limiti facoltativi
        preproc_args    – Namespace ritornato da parse_preproc_args()
        preproc_obj     – eventuale ImageGraphPreprocessor già istanziato
    """
    grouped: Dict[str, List[VQAExample]] = {}
    for ex in examples:
        grouped.setdefault(ex.image_path, []).append(ex)

    img_paths = list(grouped)[:max_imgs] if max_imgs > 0 else list(grouped)

    for img in tqdm(img_paths, desc="Preprocessing"):
        qs = grouped[img][:max_qpi] if max_qpi > 0 else grouped[img]
        for ex in qs:
            preprocess_for_qa(
                ex.image_path,
                ex.question,
                output_folder=preproc_folder,
                apply_question_filter=not disable_q_filter,
                preproc_obj=preproc_obj,
                preproc_cli_args=preproc_args,
                base_config=base_config,
            )


# ---------------------------------------------------------------------------
# Build the path of an already–preprocessed image for a (image, question) pair
# ---------------------------------------------------------------------------
def get_preprocessed_path(image_path: str,
                          question: str,
                          output_folder: str = "preprocessed") -> str:
    """
    Riproduce la stessa logica di `preprocess_for_qa` per ottenere il nome
    del file annotato SENZA rieseguire il preprocessing.
    """
    import hashlib, os

    base  = os.path.splitext(os.path.basename(image_path))[0]
    qhash = hashlib.md5(question.encode("utf-8")).hexdigest()[:8]
    out_fn = f"{base}_{qhash}_output.jpg"
    return os.path.join(output_folder, out_fn)

def get_scene_graph_path(image_path: str, question: str, preproc_folder: str) -> str:
    """
    Costruisce il path al file scene_graph.json per una coppia (immagine, domanda).
    """
    import hashlib, os
    
    base = os.path.splitext(os.path.basename(image_path))[0]
    qhash = hashlib.md5(question.encode("utf-8")).hexdigest()[:8]
    sg_filename = f"{base}_{qhash}_graph.json"   # <— invece di _scene_graph.json
    return os.path.join(preproc_folder, sg_filename)


def load_scene_graph(scene_graph_path: str) -> str:
    with open(scene_graph_path, "r", encoding="utf-8") as f:
        sg = json.load(f)
    nodes = sg.get("nodes", [])
    links = sg.get("links", sg.get("edges", []))
    id2lab = {i: n.get("label", "unknown") for i, n in enumerate(nodes)}
    # escludi eventuale nodo 'scene'
    scene_ids = {i for i, n in enumerate(nodes) if n.get("label") == "scene"}
    objs = [n.get("label","unknown") for i,n in enumerate(nodes) if i not in scene_ids]
    txt = "Scene Graph Information:\n"
    if objs: txt += "Objects: " + ", ".join(objs) + "\n"
    if links:
        txt += "Spatial Relationships:\n"
        for e in links:
            u, v = e.get("source"), e.get("target")
            if u in scene_ids: continue
            # puoi derivare relazioni grezze da attributi, se presenti
            rel = e.get("relation", "near")
            txt += f"- {id2lab.get(u,'?')} {rel} {id2lab.get(v,'?')}\n"
    return txt + "\n"

    
    try:
        with open(scene_graph_path, "r", encoding="utf-8") as f:
            sg_data = json.load(f)
        
        # Converte il scene graph in formato testuale
        sg_text = "Scene Graph Information:\n"
        
        # Oggetti rilevati
        if "objects" in sg_data:
            sg_text += "Objects: "
            objects = [obj.get("label", "unknown") for obj in sg_data["objects"]]
            sg_text += ", ".join(objects) + "\n"
        
        # Relazioni spaziali
        if "relationships" in sg_data:
            sg_text += "Spatial Relationships:\n"
            for rel in sg_data["relationships"]:
                subj = rel.get("subject", "unknown")
                pred = rel.get("predicate", "unknown")
                obj = rel.get("object", "unknown")
                sg_text += f"- {subj} {pred} {obj}\n"
        
        return sg_text + "\n"
        
    except Exception as e:
        logger.warning(f"Failed to load scene graph from {scene_graph_path}: {e}")
        return ""

# ---------------------------------------------------------------------------
# VQA runner with QA preprocessing
# ---------------------------------------------------------------------------
def run_vqa(
    examples: List[VQAExample],
    model: Union[VLLMWrapper, HFVLModel],
    *,
    out_json: str,
    prompt_tpl: str,
    batch_size: int,
    max_qpi: int,
    max_imgs: int,
    preproc_folder: str = "preprocessed",
    disable_q_filter: bool = False,
    preproc_args: Optional[Dict[str, Any]] = None,
    image_dir: Optional[str] = None,
    skip_preproc: bool = False,
    preproc_obj: Optional[ImageGraphPreprocessor] = None,
    include_scene_graph: bool = False
) -> List[Dict[str, Any]]:
    import os, glob, hashlib

    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)

    # 1) Carico eventuali risultati già salvati
    if os.path.exists(out_json):
        with open(out_json, "r", encoding="utf-8") as f:
            results = json.load(f)
    else:
        results = []

    # 2) Costruisco un set di chiavi già elaborate (image_path + question)
    processed = {(r["image_path"], r["question"]) for r in results}

    # 3) Raggruppo gli esempi per immagine
    grouped: Dict[str, List[VQAExample]] = {}
    for ex in examples:
        grouped.setdefault(ex.image_path, []).append(ex)

    img_paths = list(grouped)[:max_imgs] if max_imgs > 0 else list(grouped)

    for img in img_paths:
        qs = grouped[img][:max_qpi] if max_qpi > 0 else grouped[img]
        for i in tqdm(range(0, len(qs), batch_size), desc=os.path.basename(img)):
            batch = qs[i : i + batch_size]
            for ex in batch:
                # Se questa coppia è già processata, skip
                key = (ex.image_path, ex.question)
                if key in processed:
                    continue

                # ---- memory cleanup periodico ----
                if len(processed) > 0 and len(processed) % 50 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()
                    mem = psutil.virtual_memory()
                    logger.info(f"Memory cleanup: RAM used {mem.percent}%")
                # -----------------------------------

                # Precalcolo base+hash per i fallback
                base   = os.path.splitext(os.path.basename(ex.image_path))[0]
                qhash  = hashlib.md5(ex.question.encode("utf-8")).hexdigest()[:8]

                # 4) Preprocessing (o ricerca immagine) ----------------------
                if skip_preproc:
                    # ① file convenzionale ..._output.jpg in preproc_folder
                    processed_img = get_preprocessed_path(
                        ex.image_path, ex.question, preproc_folder
                    )

                    if not os.path.exists(processed_img):
                        # ② prova RAW esplicito (dal JSON)
                        raw_img = ex.image_path
                        if image_dir and not os.path.isabs(raw_img):
                            raw_img = os.path.join(image_dir, raw_img)

                        if os.path.exists(raw_img):
                            processed_img = raw_img
                        else:
                            # ③ qualunque <basename>_*.jpg|png in orig/preproc_folder
                            parent = os.path.dirname(raw_img) or "."
                            cand_patterns = [
                                os.path.join(parent,         f"{base}_*.jpg"),
                                os.path.join(parent,         f"{base}_*.png"),
                                os.path.join(preproc_folder, f"{base}_*.jpg"),
                                os.path.join(preproc_folder, f"{base}_*.png"),
                            ]
                            matches = []
                            for p in cand_patterns:
                                matches.extend(glob.glob(p))
                            if matches:
                                processed_img = matches[0]
                            else:
                                raise FileNotFoundError(
                                    f"Could not find an image for {ex.image_path} "
                                    f"(looked for *_output.jpg, raw path, or <base>_*.jpg|png)."
                                )
                else:
                    processed_img = preprocess_for_qa(
                        ex.image_path, ex.question,
                        output_folder=preproc_folder,
                        apply_question_filter=not disable_q_filter,
                        preproc_obj=preproc_obj,
                        preproc_cli_args=preproc_args,
                        base_config=preproc_args.__dict__ if preproc_args else None
                    )

                # 4b) Fallback robusto se il file non esiste dove atteso -----
                if not os.path.exists(processed_img):
                    patterns = [f"{base}_{qhash}_output.*", f"{base}_{qhash}*.*"]
                    candidate_dirs = [
                        preproc_folder,
                        "output_images",                        # default frequente dei pre-processor
                        os.path.dirname(processed_img) or ".",  # dove ha provato a scrivere
                    ]
                    found = None
                    for d in candidate_dirs:
                        for pat in patterns:
                            matches = glob.glob(os.path.join(d, pat))
                            if matches:
                                found = matches[0]
                                break
                        if found:
                            break
                    if found:
                        logger.info(f"Usato fallback immagine: {found}")
                        processed_img = found
                    else:
                        raise FileNotFoundError(
                            f"Preprocessed image not found for {ex.image_path} ({base}_{qhash})."
                        )

                # 5) Caricamento dello scene graph (se richiesto) ------------
                scene_graph_text = ""
                if include_scene_graph:
                    # path atteso
                    sg_path = get_scene_graph_path(ex.image_path, ex.question, preproc_folder)

                    # fallback: cerca anche in output_images
                    if not os.path.exists(sg_path):
                        alt = get_scene_graph_path(ex.image_path, ex.question, "output_images")
                        if os.path.exists(alt):
                            sg_path = alt

                    if os.path.exists(sg_path):
                        scene_graph_text = load_scene_graph(sg_path)
                    else:
                        logger.info(f"Scene graph non trovato per {base}_{qhash} (ok, continuo senza).")

                # 6) Costruzione prompt (non sovrascrivere!) ------------------
                base_prompt = prompt_tpl.format(question=ex.question)
                prompt = f"{scene_graph_text}{base_prompt}" if scene_graph_text else base_prompt

                # 7) Generazione ----------------------------------------------
                t0 = time.time()
                ans = model.generate(prompt, image_path=processed_img)
                torch.cuda.empty_cache()

                if "Answer:" in ans:
                    ans = ans.rsplit("Answer:", 1)[-1].strip().strip('"')

                # 8) Salvataggio record ---------------------------------------
                out_record = {
                    **ex.to_dict(),
                    "generated_answer": ans,
                    "processing_time": time.time() - t0,
                    "used_scene_graph": bool(scene_graph_text)
                }
                results.append(out_record)
                processed.add(key)

        # Salvo i risultati intermedi
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    return results



# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def evaluate(res: List[Dict[str, Any]]) -> Dict[str, float]:
    gold = [r for r in res if r.get("answer")]
    if not gold:
        return {}
    corr = sum(
        r["answer"].strip().lower() == r["generated_answer"].strip().lower()
        for r in gold
    )
    return {
        "total": len(gold),
        "exact": corr,
        "exact_percent": 100 * corr / len(gold),
        "avg_time": sum(r["processing_time"] for r in gold) / len(gold)
    }
# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser("VQA pipeline with QA preprocessor")
    ap.add_argument("--input_file", required=True, help="JSON file of QA examples")
    ap.add_argument("--output_file", default="vqa_results.json", help="Where to save VQA results")
    ap.add_argument("--image_dir", help="Base directory for relative image paths")
    ap.add_argument(
        "--single_question",
        type=str,
        default=None,
        help="Se fornita, questa domanda verrà usata per tutte le immagini, ignorando quelle nel file JSON."
    )
    ap.add_argument("--model_name", default="google/gemma-3-4b-it", help="Hugging Face model name")
    ap.add_argument("--use_vllm", action="store_true", help="Use vLLM instead of HF transformers")
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--prompt_template", default="Question: {question}\nAnswer:")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    ap.add_argument("--max_images", type=int, default=-1)
    ap.add_argument("--max_questions_per_image", type=int, default=-1)
    ap.add_argument(
        "--disable_question_filter",
        action="store_true",
        help="Run preprocessing without question-based filtering"
    )
    ap.add_argument(
        "--preproc_folder",
        type=str,
        default="preprocessed",
        help="Directory to store preprocessed images"
    )
    ap.add_argument(
        "--skip-preprocessing",
        action="store_true",
        help="If set, do not run preprocess_for_qa on the images; pass raw images to the model"
    )

    ap.add_argument(
        "--preprocess_only",
        action="store_true",
        help="Esegui solo il preprocessing, senza caricare il modello e senza inference"
    )
    ap.add_argument(
        "--include_scene_graph",
        action="store_true",
        help="Include scene graph information in the model input during inference"
    )

    args, _ = ap.parse_known_args()

    return args

    
def main():
    args = parse_args()

    # ------------------------------------------------------------
    # 1️⃣  CONFIGURAZIONE PREPROCESSING (solo se serve davvero)
    # ------------------------------------------------------------
    if args.skip_preprocessing:
        # ◼️ Modalità inference-only  →  nessuna dipendenza sul pre-processor
        preproc_args  = None
        preproc_cfg   = {}
        GLOBAL_PREPROC = None
    else:
        # ◼️ Pipeline completa  →  parsiamo i flag extra e creiamo il pre-processor
        preproc_args = parse_preproc_args()

        preproc_cfg = preproc_args.__dict__.copy()
        preproc_cfg.update({
            "input_path": None,               
            "preproc_device": args.device,    
            "apply_question_filter": not getattr(preproc_args, 'disable_question_filter', args.disable_question_filter),
            "display_legend":     not getattr(preproc_args, 'no_legend', False),
            "aggressive_pruning":  getattr(preproc_args, 'aggressive_pruning', False),
        })

        GLOBAL_PREPROC = ImageGraphPreprocessor(preproc_cfg)

    # ------------------------------------------------------------
    # 2️⃣  CARICAMENTO DEGLI ESEMPI
    # ------------------------------------------------------------
    examples = load_examples(args.input_file)

    # Se è fornita una domanda unica, la imposta per tutti gli esempi
    if args.single_question:
        for ex in examples:
            ex.question = args.single_question

    if args.image_dir:
        if args.skip_preprocessing:
            logger.warning(
                "--image_dir è stato ignorato perché hai attivato --skip-preprocessing"
            )
        else:
            # Prefissa i path relativi con la directory indicata
            for e in examples:
                if not os.path.isabs(e.image_path):
                    e.image_path = os.path.join(args.image_dir, e.image_path)


    # ------------------------------------------------------------
    # 3️⃣  SOLO PREPROCESSING (se richiesto esplicitamente)
    # ------------------------------------------------------------
    if args.preprocess_only:
        run_preprocessing(
            examples,
            preproc_folder=args.preproc_folder,
            disable_q_filter=args.disable_question_filter,
            max_imgs=args.max_images,
            max_qpi=args.max_questions_per_image,
            preproc_args=preproc_args,
            preproc_obj=GLOBAL_PREPROC,
            base_config=preproc_cfg,
        )
        logger.info("Preprocessing completato: immagini in «%s»", args.preproc_folder)
        return

    # ------------------------------------------------------------
    # 4️⃣  CARICAMENTO DEL MODELLO (sempre necessario)
    # ------------------------------------------------------------

    if not args.preprocess_only:
        HF_TOKEN = os.getenv("HF_TOKEN")
        if HF_TOKEN:
            hf_login(token=HF_TOKEN)
        else:
            logger.warning("HF_TOKEN non impostato: il modello deve essere pubblico o già in cache.")

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


    import gc, torch
    gc.collect(); torch.cuda.empty_cache()

    # ------------------------------------------------------------
    # 5️⃣  INFERENCE + (eventuale) PRE-PROCESSING
    # ------------------------------------------------------------
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
        preproc_args=preproc_args,       # None se skip-preprocessing
        skip_preproc=args.skip_preprocessing,
        preproc_obj=GLOBAL_PREPROC,      # None se skip-preprocessing
        image_dir=args.image_dir,
        include_scene_graph=args.include_scene_graph
    )

    # ------------------------------------------------------------
    # 6️⃣  METRICHE (opzionali)
    # ------------------------------------------------------------
    metrics = evaluate(res)
    if metrics:
        mfile = os.path.splitext(args.output_file)[0] + "_metrics.json"
        with open(mfile, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        logger.info("Metrics: %s", metrics)

if __name__ == "__main__":
    main()
