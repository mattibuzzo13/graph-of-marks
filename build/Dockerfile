###############################################################################
# PyTorch 2.5.0 • CUDA 12.1 • cuDNN 9 • Ubuntu 22.04
###############################################################################
FROM pytorch/pytorch:2.5.0-cuda12.1-cudnn9-devel
LABEL maintainer="UniboNLP"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 \
    TRANSFORMERS_NO_ADVISORY_WARNINGS=1 \
    PIP_NO_BUILD_ISOLATION=1          

# ---------- system ----------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        git cmake wget curl nano \
        libgl1-mesa-glx libglib2.0-0 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# ---------- pip base --------------------------------------------------------
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# ---------- pure-python requirements ---------------------------------------
COPY /build/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# ---------- Torch / CUDA extensions ----------------------------------------
RUN TORCH_CUDA_ARCH_LIST="8.6;8.9+PTX" \
    # 1️⃣ detectron2 (porta iopath 0.1.9)
    pip install --no-cache-dir \
        git+https://github.com/facebookresearch/detectron2.git@main && \
    # 2️⃣ extra libs
    pip install --no-cache-dir \
        flash-attn==2.7.4.post1 \
        git+https://github.com/openai/CLIP.git@main \
        git+https://github.com/facebookresearch/segment-anything.git@main && \
    # 3️⃣ sam-2  (no deps, no build isolation → niente Torch duplicato)
    pip install --no-cache-dir --no-deps --no-build-isolation \
        git+https://github.com/facebookresearch/sam2.git@main && \
    # 4️⃣ PyG + Ultralytics
    pip install --no-cache-dir \
        torch-scatter -f https://data.pyg.org/whl/torch-2.5.0+cu121.html \
        torch-sparse  -f https://data.pyg.org/whl/torch-2.5.0+cu121.html \
        torch-geometric==2.5.0 \
        ultralytics==8.3.99

# ---------- SpaCy model -----------------------------------------------------
# Fix per conflitti NumPy/SciPy e installa blis richiesto da thinc
RUN pip install --no-cache-dir \
        numpy>=1.25.0 \
        scipy>=1.11.4 \
        blis \
        iopath>=0.1.10 && \
    pip install --no-cache-dir --force-reinstall spacy && \
    python -m spacy download en_core_web_md

# ---------- project sources -------------------------------------------------
WORKDIR /workdir
COPY . .

ENTRYPOINT ["make"]
CMD ["-f", "/workdir/Makefile"]
