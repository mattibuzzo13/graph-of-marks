import sys
from pathlib import Path

# Add src to path to import setup (if needed, but we can just parse it or import setuptools mock)
# Actually, let's just read the file and check for strings, or try to run python setup.py egg_info if possible.
# But running egg_info might fail if dependencies are missing.
# Let's parse setup.py using ast or just simple string checks for critical dependencies.

def check_dependencies():
    setup_content = Path("setup.py").read_text()
    
    required_strings = [
        'torch>=2.6.0',
        'torchvision>=0.21.0',
        'pillow>=10.2.0',
        'numpy>=1.24.0,<=2.2.0',
        'opencv-python>=4.11.0',
        'transformers>=4.50.0',
        'networkx>=3.1',
        'matplotlib>=3.10.0',
        'tqdm>=4.65.0',
        'datasets>=3.3.1',
        'sentence-transformers>=3.4.1',
        'ensemble-boxes>=1.0.7',
        'huggingface_hub>=0.31.1',
        'psutil>=5.9.5',
        'omegaconf',
        'pycocotools',
        'scipy>=1.11.4',
        'pyyaml',
        'fvcore>=0.1.5',
        'iopath>=0.1.10',
        'hydra-core>=1.3.2',
        'einops',
        'timm==0.9.12',
        'spacy==3.8.4',
        'nltk==3.9.1',
        'blis',
        'colorlog>=6.9.0',
        'pretty-errors==1.2.25',
        'sentencepiece==0.2.0',
        'num2words==0.5.13',
    ]
    
    missing = []
    for req in required_strings:
        if req not in setup_content:
            missing.append(req)
            
    if missing:
        print("❌ Missing dependencies in setup.py:")
        for m in missing:
            print(f"  - {m}")
        sys.exit(1)
    else:
        print("✅ All core dependencies found in setup.py")

    # Check pyproject.toml
    toml_content = Path("pyproject.toml").read_text()
    missing_toml = []
    for req in required_strings:
        # TOML might use slightly different formatting but usually strings are same
        if req not in toml_content:
            missing_toml.append(req)
            
    if missing_toml:
        print("❌ Missing dependencies in pyproject.toml:")
        for m in missing_toml:
            print(f"  - {m}")
        sys.exit(1)
    else:
        print("✅ All core dependencies found in pyproject.toml")

if __name__ == "__main__":
    check_dependencies()
