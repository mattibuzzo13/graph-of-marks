"""
Graph of Marks (GoM) - Visual Scene Understanding Pipeline
Setup configuration for pip installation
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Core dependencies that should always be installed
core_deps = [
    "torch>=2.4.0",
    "torchvision>=0.19.0",
    "pillow>=9.5.0",  # Support Python 3.9+, was >=10.2.0
    "numpy>=1.24.0,<=2.2.0",  # Notebook uses 2.1.1
    "opencv-python>=4.11.0",
    "transformers>=4.50.0",
    "networkx>=3.1",
    "matplotlib>=3.7.0",  # Changed from >=3.10.0 to support Python 3.9
    "tqdm>=4.65.0",
    "datasets>=3.3.1",
    "sentence-transformers>=3.4.1",
    "ensemble-boxes>=1.0.7",
    "huggingface_hub>=0.31.1",
    "psutil>=5.9.5",
    "omegaconf",
    "pycocotools",
    "scipy>=1.11.4",
    "pyyaml",
    "fvcore>=0.1.5",
    "iopath>=0.1.7",
    "hydra-core>=1.3.2",
    "einops",
    "timm==0.9.12",
    "spacy==3.8.4",
    "nltk==3.9.1",
    "blis",
    "colorlog>=6.9.0",
    "pretty-errors==1.2.25",
    "sentencepiece==0.2.0",
    "num2words==0.5.13",
]

# Optional dependencies for specific features
extras_require = {
    "segmentation": [
        "segment-anything-hq",
        # Note: fastsam, segment-anything, sam2 are git deps in notebook
    ],
    "detection": [
        "ultralytics==8.3.99",
        # detectron2 requires separate install: pip install git+https://github.com/facebookresearch/detectron2.git
    ],
    "vqa": [
        "accelerate==1.4.0",
        "peft>=0.9.0",
        # vllm and ollama removed - users should install their preferred VLM library
    ],
    "dev": [
        "pytest>=7.0.0",
        "pytest-cov>=4.0.0",
        "black>=23.0.0",
        "isort>=5.12.0",
        "mypy>=1.0.0",
        "flake8>=6.0.0",
    ],
    "visualization": [
        "adjustText==0.8",
    ],
    "logging": [
        "wandb>=0.19.0",
    ],
}

# All optional dependencies combined
extras_require["all"] = list(set(sum(extras_require.values(), [])))

# All optional dependencies combined (MPS/CPU friendly - no vllm)
all_mps = []
for k, v in extras_require.items():
    if k == "all": continue
    for dep in v:
        if not dep.startswith("vllm"):
            all_mps.append(dep)
extras_require["all-mps"] = list(set(all_mps))

setup(
    name="graph-of-mark",
    version="1.1.0",
    author="DISI-UNIBO-NLP",
    author_email="",
    description="Visual scene understanding pipeline with multi-model detection, segmentation, and scene graph generation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/disi-unibo-nlp/graph-of-mark",
    project_urls={
        "Bug Reports": "https://github.com/disi-unibo-nlp/graph-of-mark/issues",
        "Source": "https://github.com/disi-unibo-nlp/graph-of-mark",
    },
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS :: MacOS X",
        "Environment :: GPU :: NVIDIA CUDA",
    ],
    python_requires=">=3.8",
    install_requires=core_deps,
    extras_require=extras_require,
    entry_points={
        "console_scripts": [
            "gom-preprocess=gom.cli.preprocess:main",
            "gom-vqa=gom.cli.vqa:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords=[
        "computer-vision",
        "scene-graph",
        "object-detection",
        "segmentation",
        "visual-question-answering",
        "sam",
        "yolo",
        "clip",
        "deep-learning",
        "pytorch",
    ],
)
