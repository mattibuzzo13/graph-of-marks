# igp/vqa/types.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, List

@dataclass
class VQAExample:
    image_path: str
    question: str
    answer: Optional[str] = None
    image_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "VQAExample":
        return cls(
            image_path=d["image_path"],
            question=d.get("question", ""),
            answer=d.get("answer"),
            image_id=d.get("image_id"),
            metadata=d.get("metadata", {}) or {},
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_path": self.image_path,
            "question": self.question,
            "answer": self.answer,
            "image_id": self.image_id,
            "metadata": self.metadata or {},
        }
