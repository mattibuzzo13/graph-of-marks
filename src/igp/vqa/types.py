# igp/vqa/types.py
# -----------------------------------------------------------------------------
# Minimal, serialization-friendly data container for a VQA sample.
# Designed for dataset IO, preprocessing, and evaluation pipelines.
# -----------------------------------------------------------------------------
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, List

@dataclass
class VQAExample:
    # Path or URL to the image associated with this QA pair.
    image_path: str
    # Natural-language question about the image.
    question: str
    # Optional ground-truth answer (for evaluation or supervised training).
    answer: Optional[str] = None
    # Optional unique identifier for the image (dataset-specific).
    image_id: Optional[str] = None
    # Optional arbitrary metadata (e.g., split name, source dataset, tags).
    metadata: Optional[Dict[str, Any]] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "VQAExample":
        # Robust construction from a dict-like record.
        # - image_path is required.
        # - question defaults to "" if missing.
        # - metadata normalized to {} (never None) for safer downstream use.
        return cls(
            image_path=d["image_path"],
            question=d.get("question", ""),
            answer=d.get("answer"),
            image_id=d.get("image_id"),
            metadata=d.get("metadata", {}) or {},
        )

    def to_dict(self) -> Dict[str, Any]:
        # Convert to a JSON-serializable dict (stable field names).
        # Ensures metadata is a dict (not None) to simplify consumers.
        return {
            "image_path": self.image_path,
            "question": self.question,
            "answer": self.answer,
            "image_id": self.image_id,
            "metadata": self.metadata or {},
        }
