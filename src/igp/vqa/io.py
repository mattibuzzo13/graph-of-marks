# igp/vqa/io.py
from __future__ import annotations
import json
from io import BytesIO
from typing import Iterable, List, Union, Optional
import logging
import os
from pathlib import Path
from urllib.parse import urlparse

import requests
from PIL import Image, ImageOps, UnidentifiedImageError

from .types import VQAExample

log = logging.getLogger(__name__)

# ---- internals --------------------------------------------------------------

def _is_http_url(s: str) -> bool:
    try:
        p = urlparse(s)
        return p.scheme in ("http", "https")
    except Exception:
        return False

def _is_file_url(s: str) -> bool:
    try:
        p = urlparse(s)
        return p.scheme == "file" and p.path != ""
    except Exception:
        return False

def _candidate_paths(path_or_url: str, image_dir: Optional[Union[str, Path]]) -> List[Path]:
    """Build a small set of plausible file paths to try, de-duplicated."""
    candidates: List[Path] = []
    p = Path(path_or_url).expanduser()
    candidates.append(p)

    if image_dir:
        base = Path(image_dir).expanduser()
        candidates.append(base / path_or_url)
        candidates.append(base / Path(path_or_url).name)

    # Normalize & dedupe
    uniq: List[Path] = []
    seen = set()
    for c in candidates:
        try:
            rc = c if c.is_absolute() else (Path.cwd() / c)
            key = rc.resolve(strict=False)
        except Exception:
            key = c
        if key not in seen:
            seen.add(key)
            uniq.append(c)
    return uniq

def _open_pil_rgb(fp: BytesIO | str | os.PathLike) -> Image.Image:
    """Open with PIL, verify, fix EXIF orientation, return RGB."""
    try:
        img = Image.open(fp)
        img.verify()              # verify header
    except UnidentifiedImageError as e:
        raise UnidentifiedImageError(f"Not an image or corrupted: {fp}") from e
    except Exception:
        # re-raise; caller will annotate
        raise
    # reopen for actual load after verify
    img = Image.open(fp) if isinstance(fp, BytesIO) else Image.open(fp)
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")

# ---- public API -------------------------------------------------------------

def load_image(path_or_url: str, image_dir: str | None = None, *, timeout: float = 30.0, debug: bool = False) -> Image.Image:
    """
    Load an image from:
      • http(s) URL
      • file:// URL
      • local path, optionally relative to `image_dir`

    Returns RGB PIL.Image with EXIF orientation applied.

    Raises:
      FileNotFoundError, requests.HTTPError, PIL.UnidentifiedImageError, ValueError
    """
    if debug:
        log.setLevel(logging.DEBUG)
        log.debug("load_image(path_or_url=%r, image_dir=%r)", path_or_url, image_dir)

    # Remote (http/https)
    if _is_http_url(path_or_url):
        try:
            with requests.get(path_or_url, timeout=timeout, stream=True) as r:
                r.raise_for_status()
                ctype = r.headers.get("Content-Type", "")
                if "image" not in ctype.lower():
                    log.debug("Content-Type %r does not look like an image", ctype)
                data = BytesIO(r.content)  # small/medium images; for huge, iter_content
            return _open_pil_rgb(data)
        except Exception as e:
            raise

    # file:// URL
    if _is_file_url(path_or_url):
        local_path = Path(urlparse(path_or_url).path)
        if not local_path.is_file():
            raise FileNotFoundError(f"File URL not found: {local_path}")
        return _open_pil_rgb(str(local_path))

    # Local paths (try a few candidates)
    tried: List[str] = []
    for cand in _candidate_paths(path_or_url, image_dir):
        tried.append(str(cand))
        if cand.is_file():
            try:
                return _open_pil_rgb(str(cand))
            except Exception as e:
                raise UnidentifiedImageError(f"Failed to open image '{cand}': {e}") from e

    raise FileNotFoundError(
        "Image not found. Tried the following paths:\n" + "\n".join(f"  - {p}" for p in tried)
    )

def _read_json(fp: Path) -> list:
    """Try a few encodings; prefer utf-8 with BOM handling. Return a Python object (expects list)."""
    # Try JSON Lines first (common for datasets)
    try:
        with fp.open("r", encoding="utf-8") as f:
            first = f.read(2048)
            if "\n" in first and first.strip().startswith("{"):
                # JSONL heuristic: parse line by line
                f.seek(0)
                items = [json.loads(line) for line in f if line.strip()]
                if items:
                    return items
    except UnicodeDecodeError:
        pass
    except json.JSONDecodeError:
        pass

    # Regular JSON; handle BOM
    for enc in ("utf-8-sig", "utf-8", "latin-1", "utf-16"):
        try:
            with fp.open("r", encoding=enc) as f:
                return json.load(f)
        except UnicodeDecodeError:
            continue
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"Invalid JSON in '{fp}': {e}", e.doc, e.pos)
    raise RuntimeError(f"Cannot decode JSON file: {fp}")

def load_examples(fp: str | os.PathLike) -> List[VQAExample]:
    """
    Load VQA examples from a JSON (list) or JSONL file.
    """
    p = Path(fp)
    if not p.is_file():
        raise FileNotFoundError(f"No such file: {p}")

    js = _read_json(p)
    if not isinstance(js, list):
        raise ValueError(f"Expected a list of examples in '{p}', got {type(js).__name__}")
    return [VQAExample.from_dict(d) for d in js]
