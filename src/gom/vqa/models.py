# gom/vqa/models.py
"""
VQA Model Utilities

This module provides utility functions for VQA model loading.
For inference, users should use their preferred libraries directly
(vLLM, Transformers, Ollama, etc.).

The graph-of-marks library focuses on scene graph extraction and
preprocessing. VQA inference is left to the user's choice of VLM library.

Example Usage with External Libraries:

    # With vLLM
    from vllm import LLM
    llm = LLM(model="Qwen/Qwen2.5-VL-7B-Instruct")
    
    # With Transformers
    from transformers import AutoProcessor, AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained("llava-hf/llava-1.5-7b-hf")
    
    # With Ollama
    import ollama
    response = ollama.chat(model='llava', messages=[...])
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from huggingface_hub import HfApi, hf_hub_download


def download_repo_with_bar(repo_id: str, cache_dir: Optional[str] = None) -> str:
    """
    Download a HuggingFace model repository with progress tracking.
    
    Args:
        repo_id: HuggingFace model repository ID (e.g., "Qwen/Qwen2.5-VL-7B-Instruct")
        cache_dir: Optional cache directory, defaults to HF cache
        
    Returns:
        Path to the local snapshot directory
        
    Example:
        >>> local_path = download_repo_with_bar("meta-llama/Llama-2-7b-hf")
        >>> print(f"Model downloaded to: {local_path}")
    """
    if cache_dir is None:
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
    
    api = HfApi()
    try:
        info = api.model_info(repo_id, repo_type="model")
    except TypeError:
        info = api.model_info(repo_id)

    siblings = getattr(info, "siblings", getattr(info, "files", []))
    files = [s.rfilename for s in siblings]
    total = sum((s.size or 0) for s in siblings)

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
    
    return str(local_snapshot) if local_snapshot else cache_dir
