###############################################################################
# PyTorch 2.5.0 • CUDA 12.1 • cuDNN 9 • Ubuntu 22.04
###############################################################################
FROM pytorch/pytorch:2.5.0-cuda12.1-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 \
    TRANSFORMERS_NO_ADVISORY_WARNINGS=1 \
    PIP_NO_BUILD_ISOLATION=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ---------- system ----------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        git cmake wget curl nano \
        libgl1-mesa-glx libglib2.0-0 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# ---------- pip base --------------------------------------------------------
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# ---------- pure-python requirements ---------------------------------------
COPY /build/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir --upgrade numba

# ---------- Torch / CUDA extensions ----------------------------------------
RUN TORCH_CUDA_ARCH_LIST="8.6;8.9+PTX" \
    # Detectron2 (usa torch già installato)
    pip install --no-cache-dir --no-build-isolation --no-deps \
        git+https://github.com/facebookresearch/detectron2.git@main && \
    # FlashAttention (no deps, no build isolation)
    pip install --no-cache-dir --no-deps --no-build-isolation \
        flash-attn==2.7.4.post1 && \
    # Extra libs
    pip install --no-cache-dir \
        git+https://github.com/openai/CLIP.git@main \
        git+https://github.com/facebookresearch/segment-anything.git@main && \
    # SAM-HQ
    pip install --no-cache-dir \
        segment-anything-hq && \
    # SAM-2 (no deps, no build isolation → evita doppio Torch)
    pip install --no-cache-dir --no-deps --no-build-isolation \
        git+https://github.com/facebookresearch/sam2.git@main && \
    # PyG + Ultralytics
    pip install --no-cache-dir \
        torch-scatter -f https://data.pyg.org/whl/torch-2.5.0+cu121.html \
        torch-sparse  -f https://data.pyg.org/whl/torch-2.5.0+cu121.html \
        torch-geometric==2.5.0 \
        ultralytics==8.3.99

# ---------- Utility libs per Detectron2 ------------------------------------
RUN pip install --no-cache-dir pyyaml opencv-python fvcore iopath>=0.1.10

# ---------- SpaCy + NLTK ---------------------------------------------------
# Fix per conflitti NumPy/SciPy e installa blis richiesto da thinc
RUN pip install --no-cache-dir \
        scipy>=1.11.4 \
        blis && \
    pip install --no-cache-dir --force-reinstall spacy && \
    python -m spacy download en_core_web_md && \
    python -m nltk.downloader wordnet

# ---------- Extra runtime tools -------------------------------------------
RUN pip install --no-cache-dir numpy==2.2.0 numba && \
    pip install --no-cache-dir --upgrade bitsandbytes

# ---------- project sources ------------------------------------------------
WORKDIR /workdir
COPY . .

ENV PYTHONPATH=/workdir/src:/workdir:${PYTHONPATH}

ENTRYPOINT ["make", "-f", "/workdir/Makefile"]
CMD ["help"]
