# igp/vqa/preproc.py
from __future__ import annotations
import glob
import hashlib
import os
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image
from ..pipeline.preprocessor import ImageGraphPreprocessor as Preprocessor
from ..config import PreprocessorConfig, default_config
from .io import load_image

# -----------------------------
# Path helpers (naming coerente)
# -----------------------------
def _qa_key(image_path: str, question: str) -> Tuple[str, str]:
    base = os.path.splitext(os.path.basename(image_path))[0]
    qhash = hashlib.md5(question.encode("utf-8")).hexdigest()[:8]
    return base, qhash

def get_preprocessed_path(image_path: str, question: str, output_folder: str = "preprocessed") -> str:
    base, qhash = _qa_key(image_path, question)
    return os.path.join(output_folder, f"{base}_{qhash}_output.jpg")

def get_scene_graph_path(image_path: str, question: str, output_folder: str) -> str:
    base, qhash = _qa_key(image_path, question)
    return os.path.join(output_folder, f"{base}_{qhash}_graph_triples.txt")

# -----------------------------
# Scene graph text loader
# -----------------------------
def load_scene_graph(scene_graph_path: str) -> str:
    # 1) Se è già un txt, torna con intestazione
    if scene_graph_path.lower().endswith(".txt"):
        try:
            with open(scene_graph_path, "r", encoding="utf-8") as f:
                txt = f.read()
            return txt if txt.lstrip().startswith("Triples:") else ("Triples:\n" + txt)
        except Exception:
            return ""
    # 2) Cerca txt “gemello”
    base, _ = os.path.splitext(scene_graph_path)
    for cand in (base + ".txt", base + "_triples.txt"):
        if os.path.exists(cand):
            with open(cand, "r", encoding="utf-8") as f:
                txt = f.read()
            return txt if txt.lstrip().startswith("Triples:") else ("Triples:\n" + txt)
    # 3) Fallback JSON (evita dipendenze forti; assume pre-processore abbia già scritto *txt*)
    try:
        with open(scene_graph_path, "r", encoding="utf-8") as f:
            raw = f.read()
        if "---->" in raw:
            return raw if raw.lstrip().startswith("Triples:") else ("Triples:\n" + raw)
    except Exception:
        pass
    return ""

# -----------------------------
# Bridge verso la nostra pipeline
# -----------------------------
def _ensure_preproc(cfg_updates: Dict[str, Any] | None, preproc_obj: Optional[Preprocessor]) -> Preprocessor:
    if preproc_obj is not None:
        if cfg_updates:
            # aggiorna i campi esistenti in modo “best effort”
            for k, v in cfg_updates.items():
                if hasattr(preproc_obj.config, k):
                    setattr(preproc_obj.config, k, v)
        return preproc_obj

    cfg = default_config()
    if cfg_updates:
        for k, v in cfg_updates.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
    return Preprocessor(cfg)

def preprocess_for_qa(
    image_path: str,
    question: str,
    *,
    output_folder: str = "preprocessed",
    apply_question_filter: bool = True,
    preproc_obj: Optional[Preprocessor] = None,
    cfg_overrides: Optional[Dict[str, Any]] = None,
    force_reprocess: bool = False,
    image_dir: Optional[str] = None,
) -> str:
    """
    Esegue la pipeline IGP per una singola coppia (immagine, domanda) e
    restituisce il path dell'immagine annotata.
    """
    base, qhash = _qa_key(image_path, question)
    os.makedirs(output_folder, exist_ok=True)
    expected = os.path.join(output_folder, f"{base}_{qhash}_output.jpg")
    if os.path.exists(expected) and not force_reprocess:
        return expected

    # prepara il pre-processor
    cfg_updates = dict(cfg_overrides or {})
    cfg_updates.update({
        "input_path": None,                  # elaborazione in-memory
        "output_folder": output_folder,
        "apply_question_filter": bool(apply_question_filter),
        "question": question,
    })
    pre = _ensure_preproc(cfg_updates, preproc_obj)

    # prova a usare un metodo “single image”; altrimenti fallback a run()
    img_pil = load_image(image_path, image_dir=image_dir)
    if hasattr(pre, "process_single_image"):

        out = pre.process_single_image(img_pil, f"{base}_{qhash}")
        # se la pipeline restituisce direttamente un path valido
        if isinstance(out, str) and os.path.exists(out):
            return out
        if isinstance(out, dict):
            for k in ("output_path", "annotated_path", "output", "annotated"):
                p = out.get(k)
                if p and os.path.exists(p):
                    return p

    else:
        # Fallback: imposta input_path e invoca run() (solo quell’immagine)
        old_in = pre.config.input_path
        pre.config.input_path = image_path
        try:
            pre.run()
        finally:
            pre.config.input_path = old_in

    # se non è stato scritto il nome “atteso”, prova pattern di fallback
    if os.path.exists(expected):
        return expected

    patterns = [f"{base}_{qhash}_output.*", f"{base}_{qhash}*annotat*.*", f"{base}_{qhash}*.*"]
    for pat in patterns:
        hits = glob.glob(os.path.join(output_folder, pat))
        if hits:
            return hits[0]
    # ultima spiaggia: torna il path “atteso”
    return expected

def run_preprocessing(
    examples: List["VQAExample"],
    *,
    preproc_folder: str = "preprocessed",
    disable_q_filter: bool = False,
    max_imgs: int = -1,
    max_qpi: int = -1,
    preproc_obj: Optional[Preprocessor] = None,
    cfg_overrides: Optional[Dict[str, Any]] = None,
    image_dir: Optional[str] = None,
) -> None:
    # group by image
    by_img: Dict[str, List["VQAExample"]] = {}
    for ex in examples:
        by_img.setdefault(ex.image_path, []).append(ex)

    img_list = list(by_img)[:max_imgs] if max_imgs > 0 else list(by_img)
    try:
        from tqdm import tqdm
    except Exception:
        tqdm = lambda x, **_: x  # type: ignore

    for img in tqdm(img_list, desc="Preprocessing"):
        qs = by_img[img][:max_qpi] if max_qpi > 0 else by_img[img]
        for ex in qs:
            preprocess_for_qa(
                ex.image_path, ex.question,
                output_folder=preproc_folder,
                apply_question_filter=not disable_q_filter,
                preproc_obj=preproc_obj,
                cfg_overrides=cfg_overrides,
                image_dir=image_dir,
            )
