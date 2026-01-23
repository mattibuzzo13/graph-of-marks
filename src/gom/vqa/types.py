# igp/vqa/types.py
"""
Visual Question Answering Data Types

Lightweight, serialization-friendly data structures for VQA examples.
Designed for dataset I/O, preprocessing pipelines, and evaluation workflows.

Key Features:
    - JSON-serializable dataclass
    - Optional metadata support
    - Robust from_dict() parsing
    - Consistent to_dict() serialization
    - Dataset-agnostic design

Data Model:
    VQAExample:
        - image_path: Path or URL to image
        - question: Natural language question
        - answer: Ground truth answer (optional)
        - image_id: Unique identifier (optional)
        - metadata: Arbitrary metadata dict (optional)

Usage:
    >>> from gom.vqa.types import VQAExample
    >>> 
    >>> # Create from dict (e.g., JSON loaded)
    >>> example = VQAExample.from_dict({
    ...     "image_path": "images/img123.jpg",
    ...     "question": "What color is the car?",
    ...     "answer": "red",
    ...     "image_id": "img123",
    ...     "metadata": {"split": "val", "dataset": "vqav2"}
    ... })
    >>> 
    >>> # Serialize to dict for JSON export
    >>> data = example.to_dict()
    >>> json.dump(data, f)
    >>> 
    >>> # Access fields
    >>> print(example.question)
    "What color is the car?"

See Also:
    - gom.vqa.io: Dataset loading utilities
    - gom.vqa.runner: VQA inference pipeline
    - gom.vqa.preproc: Preprocessing functions
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class VQAExample:
    """
    Single visual question answering example.
    
    Represents one image-question pair with optional answer and metadata.
    Designed for compatibility across VQA datasets (VQAv2, GQA, TextVQA, etc.).
    
    Attributes:
        image_path: Path or URL to image file (str, required)
        question: Natural language question text (str, required)
        answer: Ground truth answer string (Optional[str], default None)
        image_id: Unique image identifier (Optional[str], default None)
        metadata: Additional metadata dict (Optional[Dict], default None)
            Common keys: "split", "dataset", "question_id", "confidence"
    
    Examples:
        >>> # Minimal example
        >>> example = VQAExample(
        ...     image_path="img.jpg",
        ...     question="How many dogs?"
        ... )
        
        >>> # Full example with metadata
        >>> example = VQAExample(
        ...     image_path="val2014/COCO_val2014_000000123.jpg",
        ...     question="What color is the ball?",
        ...     answer="red",
        ...     image_id="123",
        ...     metadata={
        ...         "split": "val",
        ...         "dataset": "vqav2",
        ...         "question_id": "123456",
        ...         "confidence": 3
        ...     }
        ... )
    
    Notes:
        - image_path can be relative, absolute, or URL
        - metadata should be JSON-serializable
        - answer is None for test sets without labels
        - Robust to missing fields via from_dict()
    """
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
        """
        Construct VQAExample from dictionary with robust parsing.
        
        Args:
            d: Dictionary with keys matching VQAExample fields
               Required: "image_path"
               Optional: "question", "answer", "image_id", "metadata"
        
        Returns:
            VQAExample instance with normalized fields
        
        Behavior:
            - image_path: Required (raises KeyError if missing)
            - question: Defaults to "" if missing
            - answer: None if missing
            - image_id: None if missing
            - metadata: Normalized to {} (never None)
        
        Examples:
            >>> # Minimal dict
            >>> ex = VQAExample.from_dict({"image_path": "img.jpg"})
            >>> ex.question
            ""
            
            >>> # Full dict
            >>> ex = VQAExample.from_dict({
            ...     "image_path": "img.jpg",
            ...     "question": "What is this?",
            ...     "answer": "dog",
            ...     "image_id": "123",
            ...     "metadata": {"split": "train"}
            ... })
            
            >>> # Handles missing metadata gracefully
            >>> ex = VQAExample.from_dict({
            ...     "image_path": "img.jpg",
            ...     "question": "Color?",
            ...     "metadata": None
            ... })
            >>> ex.metadata
            {}
        
        Notes:
            - Designed for JSON deserialization
            - Safe for untrusted input (validates types implicitly)
            - Metadata always dict (never None) for easier downstream use
        """
        return cls(
            image_path=d["image_path"],
            question=d.get("question", ""),
            answer=d.get("answer"),
            image_id=d.get("image_id"),
            metadata=d.get("metadata", {}) or {},
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to JSON-serializable dictionary.
        
        Returns:
            Dictionary with stable field names:
                - image_path: str
                - question: str
                - answer: str | None
                - image_id: str | None
                - metadata: dict (never None)
        
        Examples:
            >>> example = VQAExample(
            ...     image_path="img.jpg",
            ...     question="What?",
            ...     answer="dog"
            ... )
            >>> data = example.to_dict()
            >>> data
            {
                'image_path': 'img.jpg',
                'question': 'What?',
                'answer': 'dog',
                'image_id': None,
                'metadata': {}
            }
            
            >>> # Round-trip serialization
            >>> json_str = json.dumps(example.to_dict())
            >>> loaded = VQAExample.from_dict(json.loads(json_str))
            >>> loaded == example
            True
        
        Notes:
            - metadata always dict (never None)
            - Stable field order for consistent JSON
            - Safe for json.dump() without custom encoder
        """
        return {
            "image_path": self.image_path,
            "question": self.question,
            "answer": self.answer,
            "image_id": self.image_id,
            "metadata": self.metadata or {},
        }
