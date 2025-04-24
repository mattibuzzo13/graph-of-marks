FROM nvidia/cuda:12.2.0-devel-ubuntu22.04
LABEL maintainer="UniboNLP"

# Zero interaction
ENV DEBIAN_FRONTEND=noninteractive

# Set work directory
WORKDIR /workdir
ENV APP_PATH=/workdir

# Install system deps
RUN apt-get update -y && \
    apt-get install -y \
      curl git bash wget nano zip unzip \
      python3.11 python3.11-dev python3-pip \
      libopenblas-dev liblapack-dev \
      libavcodec-dev libavformat-dev libavdevice-dev libavfilter-dev \
      libgl1-mesa-glx libglib2.0-0 && \
    apt-get autoremove -y && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --upgrade pip wrapt --upgrade --ignore-installed gdown

# Copy over requirements and code
COPY /build/requirements.txt /workdir/
COPY Makefile              /workdir/
COPY .                     /workdir/

# Install all other Python deps
RUN pip3 install --no-cache-dir -r requirements.txt

# Install the bleeding‑edge Transformers directly from GitHub
RUN python3 -m pip install --upgrade \
        sentence-transformers huggingface_hub torch torchvision timm
        
RUN pip install --no-cache-dir \
        git+https://github.com/huggingface/transformers.git@main

# Other git‑based installs
RUN python3 -m pip install git+https://github.com/deepseek-ai/Janus.git
RUN python3 -m pip install git+https://github.com/Dao-AILab/flash-attention.git
RUN pip3 install flash-attn==2.7.4.post1 --no-build-isolation

RUN python3 -m pip install --upgrade -q git+https://github.com/huggingface/transformers
RUN python3 -m pip install --upgrade transformers sentence-transformers huggingface_hub torch torchvision timm

# Preprocessing utilities
RUN python3 -m spacy download en_core_web_md
RUN python3 -m nltk.downloader wordnet

# Facebook research repos
RUN pip3 install 'git+https://github.com/facebookresearch/segment-anything.git'
RUN pip3 install 'git+https://github.com/facebookresearch/fvcore'
RUN pip3 install 'git+https://github.com/facebookresearch/detectron2.git'

# Download SAM weights
RUN wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth \
     -O $APP_PATH/sam_vit_h_4b8939.pth

ENTRYPOINT ["make", "-f", "/workdir/Makefile"]

# Restore interactive frontend
ENV DEBIAN_FRONTEND=dialog
