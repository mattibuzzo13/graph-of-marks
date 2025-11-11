"""Small profiling helper to run the preprocessor on one image and save cProfile output.

Usage (from repo root):
    python -m scripts.profile_preprocessor --image path/to/img.jpg --out profile.prof

This script runs the full pipeline on a single image under cProfile and
prints the top hot functions. It also saves the raw profiler stats for
deeper inspection (snakeviz, pstats, etc.).
"""
from __future__ import annotations

import argparse
import cProfile
import pstats
import sys
from pathlib import Path

from PIL import Image

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to an image to profile")
    parser.add_argument("--out", default="preproc_profile.prof", help="Where to save the .prof file")
    args = parser.parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        print(f"Image not found: {img_path}")
        return 2

    # Lazy import heavy modules only when required
    try:
        from gom.pipeline.preprocessor import ImageGraphPreprocessor, PreprocessorConfig
    except Exception as e:
        print(f"Failed to import preprocessor: {e}")
        return 3

    cfg = PreprocessorConfig()
    # Use CPU by default for profiling unless user has GPUs configured
    cfg.preproc_device = "cpu"
    preproc = ImageGraphPreprocessor(cfg)

    img = Image.open(str(img_path)).convert("RGB")

    prof_file = args.out
    print(f"Profiling preprocessor on {img_path} -> {prof_file}")

    # Run under cProfile
    pr = cProfile.Profile()
    pr.enable()
    try:
        preproc.process_single_image(img, img_path.name)
    finally:
        pr.disable()
        pr.dump_stats(prof_file)

    ps = pstats.Stats(pr).sort_stats("cumtime")
    ps.print_stats(50)
    print(f"Saved profiling output to {prof_file}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
