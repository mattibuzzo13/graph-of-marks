"""
Graph of Marks (GoM) - Visual Scene Understanding Pipeline
Setup configuration for pip installation
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Read requirements
requirements = []
with open(this_directory / "build" / "requirements.txt", encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

# Core dependencies that should always be installed
core_deps = [
    "torch>=2.0.0",
    "torchvision>=0.15.0",
    "pillow>=11.0.0",
    "numpy>=1.24.0,<=2.2.0",
    "opencv-python>=4.8.0",
    "transformers>=4.50.0",
    "networkx>=3.1",
    "matplotlib>=3.8.0",
    "tqdm>=4.65.0",
    "datasets>=3.0.0",
    "sentence-transformers>=3.0.0",
    "ensemble-boxes>=1.0.7",
    "huggingface_hub>=0.31.0",
    "psutil>=5.9.0",
    "omegaconf",
    "pycocotools",
]

# Optional dependencies for specific features
extras_require = {
    "segmentation": [
        "segment-anything",
        "segment-anything-hq",
        "fastsam",
    ],
    "sam2": [
        "sam2",
    ],
    "detection": [
        "ultralytics",
    ],
    "depth": [
        "depth-anything-v2",
    ],
    "vqa": [
        "accelerate>=1.4.0",
        "vllm>=0.8.0",
        "qwen-vl-utils>=0.0.10",
        "peft>=0.9.0",
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
        "adjustText>=0.8",
    ],
    "logging": [
        "wandb>=0.19.0",
        "colorlog>=6.8.0",
    ],
}

# All optional dependencies combined
extras_require["all"] = list(set(sum(extras_require.values(), [])))

setup(
    name="graph-of-marks",
    version="0.1.0",
    author="DISI-UNIBO-NLP",
    author_email="",
    description="Visual scene understanding pipeline with multi-model detection, segmentation, and scene graph generation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/disi-unibo-nlp/graph-of-marks",
    project_urls={
        "Bug Reports": "https://github.com/disi-unibo-nlp/graph-of-marks/issues",
        "Source": "https://github.com/disi-unibo-nlp/graph-of-marks",
    },
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "License :: OSI Approved :: MIT License",
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
