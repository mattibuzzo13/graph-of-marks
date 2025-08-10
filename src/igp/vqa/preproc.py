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
# Path helpers (consistent naming and deterministic caching)
# -----------------------------
# These helpers centralize how we derive output filenames from an (image, question) pair.
# We hash the question text to keep paths short and filesystem-safe while still being unique
# per question. This makes all downstream artifacts easy to discover and re-use across runs.

def _qa_key(image_path: str, question: str) -> Tuple[str, str]:
    """
    Build a stable key for an (image, question) pair.

    Returns:
        (base, qhash)
        - base: the image filename without extension (e.g., "img_001").
        - qhash: an 8-char MD5 prefix of the question text, used to disambiguate outputs
                 for the same image under different questions.
    """
    base = os.path.splitext(os.path.basename(image_path))[0]
    qhash = hashlib.md5(question.encode("utf-8")).hexdigest()[:8]
    return base, qhash

def get_preprocessed_path(image_path: str, question: str, output_folder: str = "preprocessed") -> str:
    """
    Compute the expected path of the annotated/preprocessed image for this (image, question).

    The naming convention is:
        {output_folder}/{image_base}_{md5(question)[:8]}_output.jpg
    """
    base, qhash = _qa_key(image_path, question)
    return os.path.join(output_folder, f"{base}_{qhash}_output.jpg")

def get_scene_graph_path(image_path: str, question: str, output_folder: str) -> str:
    """
    Compute the expected path of the scene-graph triples text file emitted by the pipeline.

    The naming convention is:
        {output_folder}/{image_base}_{md5(question)[:8]}_graph_triples.txt
    """
    base, qhash = _qa_key(image_path, question)
    return os.path.join(output_folder, f"{base}_{qhash}_graph_triples.txt")

# -----------------------------
# Scene graph text loader
# -----------------------------
# The preprocessor typically writes a human-readable triples file alongside the image.
# This loader returns the triples as a single string, trying several reasonable locations
# and formats for robustness (TXT primary, JSON/other as last resort).

def load_scene_graph(scene_graph_path: str) -> str:
    """
    Load scene-graph triples text with light heuristics:

    Strategy (in priority order):
      1) If `scene_graph_path` is already a .txt file, read and return it, ensuring it has the
         "Triples:\n" header for consistency in downstream rendering.
      2) Otherwise, look for sibling TXT files that share the same base name:
           - {base}.txt
           - {base}_triples.txt
      3) As a last resort, read the given file verbatim and return it if it contains a recognizable
         triples marker (e.g., "---->"). This avoids pulling in heavy JSON dependencies here.

    Returns:
        A triples string (possibly empty if nothing is found).
    """
    # 1) If it is already a TXT, return its content (normalize header)
    if scene_graph_path.lower().endswith(".txt"):
        try:
            with open(scene_graph_path, "r", encoding="utf-8") as f:
                txt = f.read()
            return txt if txt.lstrip().startswith("Triples:") else ("Triples:\n" + txt)
        except Exception:
            return ""
    # 2) Search for a "sibling" TXT
    base, _ = os.path.splitext(scene_graph_path)
    for cand in (base + ".txt", base + "_triples.txt"):
        if os.path.exists(cand):
            with open(cand, "r", encoding="utf-8") as f:
                txt = f.read()
            return txt if txt.lstrip().startswith("Triples:") else ("Triples:\n" + txt)
    # 3) Fallback: accept raw content if it looks like triples (we assume the preprocessor wrote TXT)
    try:
        with open(scene_graph_path, "r", encoding="utf-8") as f:
            raw = f.read()
        if "---->" in raw:
            return raw if raw.lstrip().startswith("Triples:") else ("Triples:\n" + raw)
    except Exception:
        pass
    return ""

# -----------------------------
# Bridge into our preprocessing pipeline
# -----------------------------
# We expose a lightweight "ensure" function that either reuses an existing Preprocessor instance
# (optionally updating its config) or creates a fresh one from the project defaults.

def _ensure_preproc(cfg_updates: Dict[str, Any] | None, preproc_obj: Optional[Preprocessor]) -> Preprocessor:
    """
    Ensure we have a configured Preprocessor instance.

    Behavior:
      - If an existing `preproc_obj` is passed, update its *known* config attributes
        using `cfg_updates` (best-effort) and return it.
      - Otherwise, create a new config via `default_config()`, apply `cfg_updates` to
        matching fields, and instantiate a new `Preprocessor`.

    Notes:
      - Unknown keys in `cfg_updates` are ignored to keep this robust across versions.
      - This avoids re-loading heavy models when reusing an existing pipeline object.
    """
    if preproc_obj is not None:
        if cfg_updates:
            # Update existing fields in a best-effort manner (ignore unknown keys)
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
    aggressive_pruning: bool = True
) -> str:
    """
    Run the IGP pipeline on a single (image, question) pair and return the path to the
    annotated output image.

    Parameters:
        image_path: str
            Path or URL of the input image (relative paths can be resolved with `image_dir`).
        question: str
            Natural-language question used to guide filtering/pruning and relation selection.
        output_folder: str
            Destination directory for all emitted artifacts (image overlays, triples, graphs).
        apply_question_filter: bool
            If True, attempt to filter detections/relations based on question terms.
        preproc_obj: Optional[Preprocessor]
            Optional pre-constructed pipeline (reused to avoid reloading models).
        cfg_overrides: Optional[Dict[str, Any]]
            Extra configuration values applied on top of the default preprocessor config.
        force_reprocess: bool
            If True, ignore cached/previous outputs and recompute overlays for this pair.
        image_dir: Optional[str]
            Optional base directory for resolving relative `image_path` values.
        aggressive_pruning: bool
            If True, perform stricter object pruning based on question cues (with built-in fallback).

    Returns:
        The path to the annotated image (best effort). If the exact expected filename is not
        found, a set of fallback glob patterns is tried before returning the "expected" path.
    """
    base, qhash = _qa_key(image_path, question)
    os.makedirs(output_folder, exist_ok=True)
    expected = os.path.join(output_folder, f"{base}_{qhash}_output.jpg")
    # Fast path: reuse previous output unless forced to recompute
    if os.path.exists(expected) and not force_reprocess:
        return expected

    # Prepare/merge configuration for this run. We overwrite a few fields to ensure
    # in-memory processing and consistent output placement, while honoring caller overrides.
    cfg_updates = dict(cfg_overrides or {})
    cfg_updates.update({
        "input_path": None,                  # in-memory processing instead of directory traversal
        "output_folder": output_folder,
        "apply_question_filter": bool(apply_question_filter),
        "aggressive_pruning": bool(aggressive_pruning),
        "question": question,
    })
    pre = _ensure_preproc(cfg_updates, preproc_obj)

    # Load the image (local path, URL, or resolved via `image_dir`) and prefer the
    # single-image API if available. This avoids setting global input paths on the pipeline.
    img_pil = load_image(image_path, image_dir=image_dir)
    if hasattr(pre, "process_single_image"):

        out = pre.process_single_image(img_pil, f"{base}_{qhash}")
        # Some pipeline versions return a concrete output path directly.
        if isinstance(out, str) and os.path.exists(out):
            return out
        # Others may return a dict of potential keys; we check common names in priority order.
        if isinstance(out, dict):
            for k in ("output_path", "annotated_path", "output", "annotated"):
                p = out.get(k)
                if p and os.path.exists(p):
                    return p

    else:
        # Fallback path: temporarily set `input_path` and invoke the batch runner on this single image.
        old_in = pre.config.input_path
        pre.config.input_path = image_path
        try:
            pre.run()
        finally:
            pre.config.input_path = old_in

    # If the exact "expected" filename was produced, return it immediately.
    if os.path.exists(expected):
        return expected

    # Otherwise, try a few best-effort glob patterns to locate the annotated image.
    patterns = [f"{base}_{qhash}_output.*", f"{base}_{qhash}*annotat*.*", f"{base}_{qhash}*.*"]
    for pat in patterns:
        hits = glob.glob(os.path.join(output_folder, pat))
        if hits:
            return hits[0]
    # As a last resort, return the canonical expected path (may not exist if something failed silently).
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
    """
    Batch preprocessing driver for a list of VQA examples.

    This groups questions by image to avoid redundant image loading and then
    runs the single-pair preprocessor for each (image, question), respecting
    optional limits for the number of images and the number of questions per image.

    Parameters:
        examples:
            A list of VQAExample instances (each with .image_path and .question).
        preproc_folder:
            Output directory for all generated overlays and artifacts.
        disable_q_filter:
            If True, disables question-guided filtering when invoking the pipeline.
        max_imgs:
            Upper bound on the number of distinct images to process (-1 = no limit).
        max_qpi:
            Upper bound on how many questions per image to process (-1 = no limit).
        preproc_obj:
            Optional shared Preprocessor instance to reuse across all calls.
        cfg_overrides:
            Optional configuration dictionary applied on top of pipeline defaults.
        image_dir:
            Base directory to resolve relative example image paths.
    """
    # Group examples by image to minimize repeated I/O and model inference on the same image.
    by_img: Dict[str, List["VQAExample"]] = {}
    for ex in examples:
        by_img.setdefault(ex.image_path, []).append(ex)

    # Respect the optional image cap.
    img_list = list(by_img)[:max_imgs] if max_imgs > 0 else list(by_img)
    try:
        from tqdm import tqdm
    except Exception:
        # If tqdm is not installed, fall back to a no-op wrapper.
        tqdm = lambda x, **_: x  # type: ignore

    for img in tqdm(img_list, desc="Preprocessing"):
        # Respect the optional per-image question cap.
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
