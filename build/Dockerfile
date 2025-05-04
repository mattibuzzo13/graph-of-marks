# Base image with CUDA 12.2 and Ubuntu 22.04
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04
LABEL maintainer="UniboNLP"

# Non-interactive APT installs
ENV DEBIAN_FRONTEND=noninteractive

# Working directory setup
WORKDIR /workdir
ENV APP_PATH=/workdir

# Install system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3.11-venv python3-pip \
    build-essential cmake \
    curl git wget unzip zip nano bash \
    libopenblas-dev liblapack-dev \
    libavcodec-dev libavformat-dev libavdevice-dev libavfilter-dev \
    libgl1-mesa-glx libglib2.0-0 \
 && apt-get autoremove -y && apt-get clean && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install core utils
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
 && pip install --no-cache-dir gdown wrapt

# Copy in project files
COPY /build/requirements.txt .
COPY Makefile .
COPY . .

# Install core project requirements (should not include torch)
RUN pip install --no-cache-dir -r requirements.txt

# Install PyTorch & compatible packages early
RUN pip install --no-cache-dir torch==2.6.0 torchvision==0.21.0 timm

# Install sentence-transformers and huggingface utilities
RUN pip install --no-cache-dir sentence-transformers==3.4.1 huggingface_hub

# GitHub packages (installed in order, with torch already present)
RUN pip install --no-cache-dir git+https://github.com/huggingface/transformers.git@main
RUN pip install --no-cache-dir git+https://github.com/deepseek-ai/Janus.git
RUN pip install --no-cache-dir --no-build-isolation git+https://github.com/Dao-AILab/flash-attention.git
RUN pip install --no-cache-dir flash-attn==2.7.4.post1
RUN pip install --no-cache-dir git+https://github.com/facebookresearch/segment-anything.git
RUN pip install --no-cache-dir git+https://github.com/facebookresearch/fvcore
RUN pip install --no-cache-dir git+https://github.com/facebookresearch/detectron2.git

RUN pip install --upgrade transformers huggingface_hub

# NLP model downloads
RUN python -m spacy download en_core_web_md && \
    python -m nltk.downloader wordnet

# Download pretrained SAM weights
# RUN mkdir -p ${APP_PATH}/src && \
#    wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth \
#        -O ${APP_PATH}/src/sam_vit_h_4b8939.pth

# Default entry
ENTRYPOINT ["make"]
CMD ["-f", "/workdir/Makefile"]

# Reset APT frontend mode
ENV DEBIAN_FRONTEND=dialog
