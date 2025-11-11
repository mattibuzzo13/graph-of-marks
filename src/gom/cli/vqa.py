"""
CLI entry point for VQA
Wraps the existing vqa.py functionality
"""
import sys
from pathlib import Path

# Add parent directory to path to import vqa
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def main():
    """Main entry point for gom-vqa command"""
    from vqa import main as vqa_main
    return vqa_main()

if __name__ == "__main__":
    main()
