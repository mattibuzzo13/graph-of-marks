# igp/vqa/io.py
from __future__ import annotations
import json
from io import BytesIO
from typing import List
import requests
import os
from PIL import Image
from .types import VQAExample

def load_image(path_or_url: str, image_dir: str = None) -> Image.Image:
    """
    Carica un'immagine da path locale, URL o combinazione path + image_dir.
    """
    # Debug info
    print(f"[DEBUG] load_image called with:")
    print(f"  path_or_url: '{path_or_url}'")
    print(f"  image_dir: '{image_dir}'")
    
    if path_or_url.lower().startswith(("http://", "https://")):
        data = requests.get(path_or_url, timeout=30).content
        img = Image.open(BytesIO(data))
    else:
        # ✅ GESTIONE PATH: Prova prima il path assoluto, poi relativo a image_dir
        full_path = None
        
        # Lista tutti i candidati
        candidates = [path_or_url]
        if image_dir:
            candidates.extend([
                os.path.join(image_dir, path_or_url),
                os.path.join(image_dir, os.path.basename(path_or_url))
            ])
        
        print(f"[DEBUG] Trying candidates:")
        for candidate in candidates:
            exists = os.path.isfile(candidate)
            print(f"  {candidate} -> {'✓' if exists else '✗'}")
            if exists and full_path is None:
                full_path = candidate
        
        if full_path is None:
            raise FileNotFoundError(
                f"Image not found. Tried {len(candidates)} paths:\n" + 
                "\n".join(f"  - {p}" for p in candidates)
            )
        
        print(f"[DEBUG] Using: {full_path}")
        img = Image.open(full_path)
    
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
