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
    if path_or_url.lower().startswith(("http://", "https://")):
        data = requests.get(path_or_url, timeout=30).content
        img = Image.open(BytesIO(data))
    else:
        # ✅ GESTIONE PATH: Prova prima il path assoluto, poi relativo a image_dir
        if os.path.isfile(path_or_url):
            # Path assoluto esistente
            img = Image.open(path_or_url)
        elif image_dir and os.path.isfile(os.path.join(image_dir, path_or_url)):
            # Path relativo a image_dir
            full_path = os.path.join(image_dir, path_or_url)
            img = Image.open(full_path)
        elif image_dir and os.path.isfile(os.path.join(image_dir, os.path.basename(path_or_url))):
            # Solo il nome del file in image_dir
            full_path = os.path.join(image_dir, os.path.basename(path_or_url))
            img = Image.open(full_path)
        else:
            # Ultimo tentativo: path originale (per errori informativi)
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
