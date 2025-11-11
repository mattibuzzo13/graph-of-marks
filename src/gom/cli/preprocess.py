"""
CLI entry point for image preprocessing
Wraps the existing image_preprocessor.py functionality
"""
import sys
from pathlib import Path

# Add parent directory to path to import image_preprocessor
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def main():
    """Main entry point for gom-preprocess command"""
    from image_preprocessor import main as preprocess_main
    return preprocess_main()

if __name__ == "__main__":
    main()
