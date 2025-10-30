# igp/relations/llm_guided.py
# LLM-Guided Spatial Relation Inference (SOTA)
# Uses Large Language Models (GPT-4V, LLaVA, etc.) to enhance relation detection
# Paper references: "Visual Programming" (CVPR 2023), "VisProg" (NeurIPS 2022)

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
import json
import numpy as np
from PIL import Image
import base64
from io import BytesIO


@dataclass
class LLMRelationsConfig:
    """Configuration for LLM-guided relation inference."""
    
    # LLM backend
    backend: str = "gpt4v"  # "gpt4v" | "llava" | "mock"
    api_key: Optional[str] = None
    
    # Model settings
    model: str = "gpt-4-vision-preview"  # GPT-4V model name
    temperature: float = 0.2  # Low for consistency
    max_tokens: int = 500
    
    # Relation inference
    confidence_threshold: float = 0.6
    max_relations_per_query: int = 10
    use_visual_hints: bool = True  # Draw boxes on image
    
    # Prompting
    prompt_template: str = "default"  # "default" | "visprog" | "custom"
    custom_prompt: Optional[str] = None
    
    # Caching
    enable_cache: bool = True
    cache_ttl: int = 3600  # 1 hour


class LLMRelationInferencer:
    """
    Use Large Language Models to infer spatial relations.
    
    Benefits over pure geometric/CLIP:
    - Understands context and common sense
    - Handles occlusion and ambiguity better
    - Can reason about functional relations (e.g., "sits on")
    - Provides natural language explanations
    
    Supports:
    - GPT-4V (OpenAI) - best quality, requires API key
    - LLaVA (open-source) - free, local inference
    - Mock mode - for testing without API
    """
    
    def __init__(self, config: Optional[LLMRelationsConfig] = None):
        self.config = config or LLMRelationsConfig()
        self._cache: Dict[str, List[dict]] = {}
        
        # Initialize backend
        if self.config.backend == "gpt4v":
            self._init_gpt4v()
        elif self.config.backend == "llava":
            self._init_llava()
        else:
            print(f"[LLMRelations] Using mock backend (no actual LLM calls)")
    
    def _init_gpt4v(self) -> None:
        """Initialize GPT-4V (OpenAI API)."""
        try:
            import openai  # type: ignore
            
            if not self.config.api_key:
                print("[LLMRelations] WARNING: No OpenAI API key provided.")
                print("              Set config.api_key or env var OPENAI_API_KEY")
                self.config.backend = "mock"
                return
            
            self._client = openai.OpenAI(api_key=self.config.api_key)
            print(f"[LLMRelations] GPT-4V initialized: {self.config.model}")
            
        except ImportError:
            print("[LLMRelations] WARNING: openai package not installed.")
            print("              Install with: pip install openai")
            self.config.backend = "mock"
    
    def _init_llava(self) -> None:
        """Initialize LLaVA (local inference)."""
        try:
            # Check if LLaVA is available
            import torch  # type: ignore
            from transformers import AutoProcessor, LlavaForConditionalGeneration  # type: ignore
            
            print("[LLMRelations] Loading LLaVA model (this may take a while)...")
            
            model_name = "llava-hf/llava-1.5-7b-hf"
            self._llava_processor = AutoProcessor.from_pretrained(model_name)
            self._llava_model = LlavaForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            
            print(f"[LLMRelations] LLaVA initialized: {model_name}")
            
        except ImportError as e:
            print(f"[LLMRelations] WARNING: LLaVA dependencies not installed: {e}")
            print("              Install with: pip install transformers torch")
            self.config.backend = "mock"
    
    def infer_relations(
        self,
        image: Image.Image,
        boxes: Sequence[Sequence[float]],
        labels: Sequence[str],
        *,
        existing_relations: Optional[List[dict]] = None,
        question: Optional[str] = None,
    ) -> List[dict]:
        """
        Infer spatial relations using LLM vision-language understanding.
        
        Args:
            image: PIL Image
            boxes: List of [x1, y1, x2, y2] bounding boxes
            labels: Object labels for each box
            existing_relations: Optional existing relations (from geometry/CLIP)
            question: Optional context question
            
        Returns:
            List of relation dicts with keys:
              - src_idx: source object index
              - tgt_idx: target object index
              - relation: relation type (e.g., "on", "next_to")
              - confidence: LLM confidence score (0-1)
              - explanation: Optional natural language explanation
        """
        if len(boxes) <= 1:
            return []
        
        # Check cache
        cache_key = self._get_cache_key(image, boxes, labels, question)
        if self.config.enable_cache and cache_key in self._cache:
            return self._cache[cache_key]
        
        # Prepare image with visual hints (draw boxes)
        if self.config.use_visual_hints:
            annotated_image = self._draw_boxes(image, boxes, labels)
        else:
            annotated_image = image
        
        # Build prompt
        prompt = self._build_prompt(labels, question, existing_relations)
        
        # Call LLM
        if self.config.backend == "gpt4v":
            relations = self._infer_gpt4v(annotated_image, prompt, labels)
        elif self.config.backend == "llava":
            relations = self._infer_llava(annotated_image, prompt, labels)
        else:
            relations = self._infer_mock(boxes, labels)
        
        # Filter by confidence
        relations = [
            r for r in relations
            if r.get("confidence", 1.0) >= self.config.confidence_threshold
        ]
        
        # Cache result
        if self.config.enable_cache:
            self._cache[cache_key] = relations
        
        return relations
    
    def _build_prompt(
        self,
        labels: Sequence[str],
        question: Optional[str] = None,
        existing_relations: Optional[List[dict]] = None,
    ) -> str:
        """Build the prompt for the LLM."""
        
        if self.config.prompt_template == "visprog":
            return self._build_visprog_prompt(labels, question)
        elif self.config.custom_prompt:
            return self.config.custom_prompt
        
        # Default prompt
        prompt = "You are an expert at understanding spatial relationships in images.\n\n"
        prompt += f"Objects in the image:\n"
        for i, label in enumerate(labels):
            prompt += f"  {i}: {label}\n"
        prompt += "\n"
        
        if question:
            prompt += f"Context question: {question}\n\n"
        
        if existing_relations:
            prompt += "Geometric analysis suggests these relations:\n"
            for rel in existing_relations[:5]:  # Show top 5
                src_label = labels[rel["src_idx"]]
                tgt_label = labels[rel["tgt_idx"]]
                rel_type = rel["relation"]
                prompt += f"  - {src_label} {rel_type} {tgt_label}\n"
            prompt += "\n"
        
        prompt += "Please identify spatial relationships between these objects.\n"
        prompt += "Focus on: positional relations (on, under, left, right, etc.), "
        prompt += "functional relations (sitting on, leaning against, etc.), "
        prompt += "and contextual relations (part of, contains, etc.).\n\n"
        prompt += "Format your response as a JSON list of relations:\n"
        prompt += '[\n  {"src": <index>, "tgt": <index>, "relation": "<type>", "confidence": <0-1>, "explanation": "<why>"}\n]\n\n'
        prompt += "Provide up to 10 most important relations."
        
        return prompt
    
    def _build_visprog_prompt(
        self,
        labels: Sequence[str],
        question: Optional[str] = None,
    ) -> str:
        """Build VisProg-style prompt (visual programming)."""
        prompt = "# Visual Relationship Detection\n\n"
        prompt += "## Objects\n"
        for i, label in enumerate(labels):
            prompt += f"OBJ[{i}] = {label}\n"
        prompt += "\n## Task\n"
        prompt += "Detect spatial relationships using these predicates:\n"
        prompt += "- on(a, b): a is physically on top of b\n"
        prompt += "- next_to(a, b): a is beside b\n"
        prompt += "- above(a, b): a is higher than b (not necessarily on)\n"
        prompt += "- contains(a, b): a contains or surrounds b\n"
        prompt += "- in_front_of(a, b): a is in front of b (depth)\n\n"
        
        if question:
            prompt += f"## Context\nQuestion: {question}\n\n"
        
        prompt += "## Output\nReturn JSON list of detected relationships."
        
        return prompt
    
    def _infer_gpt4v(
        self,
        image: Image.Image,
        prompt: str,
        labels: Sequence[str],
    ) -> List[dict]:
        """Call GPT-4V API."""
        try:
            # Convert image to base64
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            # Call API
            response = self._client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{img_base64}"
                                },
                            },
                        ],
                    }
                ],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            )
            
            # Parse response
            content = response.choices[0].message.content
            relations = self._parse_llm_response(content, labels)
            
            return relations
            
        except Exception as e:
            print(f"[LLMRelations] GPT-4V error: {e}")
            return []
    
    def _infer_llava(
        self,
        image: Image.Image,
        prompt: str,
        labels: Sequence[str],
    ) -> List[dict]:
        """Call LLaVA (local inference)."""
        try:
            import torch
            
            # Prepare inputs
            inputs = self._llava_processor(
                text=prompt,
                images=image,
                return_tensors="pt",
            ).to(self._llava_model.device)
            
            # Generate
            with torch.inference_mode():
                output_ids = self._llava_model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    do_sample=True,
                )
            
            # Decode
            output_text = self._llava_processor.decode(
                output_ids[0],
                skip_special_tokens=True,
            )
            
            # Parse response
            relations = self._parse_llm_response(output_text, labels)
            
            return relations
            
        except Exception as e:
            print(f"[LLMRelations] LLaVA error: {e}")
            return []
    
    def _infer_mock(
        self,
        boxes: Sequence[Sequence[float]],
        labels: Sequence[str],
    ) -> List[dict]:
        """Mock inference for testing (heuristic-based)."""
        relations = []
        
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                box_i = boxes[i]
                box_j = boxes[j]
                
                # Simple heuristic: vertical relation
                cy_i = (box_i[1] + box_i[3]) / 2
                cy_j = (box_j[1] + box_j[3]) / 2
                
                if abs(cy_i - cy_j) > 50:
                    if cy_i < cy_j:
                        relation = "above"
                    else:
                        relation = "below"
                    
                    relations.append({
                        "src_idx": i,
                        "tgt_idx": j,
                        "relation": relation,
                        "confidence": 0.7,
                        "explanation": "Mock inference (vertical)",
                    })
        
        return relations[:self.config.max_relations_per_query]
    
    def _parse_llm_response(
        self,
        response_text: str,
        labels: Sequence[str],
    ) -> List[dict]:
        """Parse LLM response into structured relations."""
        try:
            # Try to extract JSON
            start = response_text.find("[")
            end = response_text.rfind("]") + 1
            
            if start == -1 or end == 0:
                print("[LLMRelations] No JSON found in response")
                return []
            
            json_str = response_text[start:end]
            data = json.loads(json_str)
            
            # Convert to standard format
            relations = []
            for item in data:
                src_idx = item.get("src", item.get("src_idx"))
                tgt_idx = item.get("tgt", item.get("tgt_idx"))
                relation = item.get("relation", "")
                confidence = item.get("confidence", 0.8)
                explanation = item.get("explanation", "")
                
                # Validate indices
                if (
                    src_idx is not None
                    and tgt_idx is not None
                    and 0 <= src_idx < len(labels)
                    and 0 <= tgt_idx < len(labels)
                    and src_idx != tgt_idx
                ):
                    relations.append({
                        "src_idx": src_idx,
                        "tgt_idx": tgt_idx,
                        "relation": relation,
                        "confidence": float(confidence),
                        "explanation": explanation,
                    })
            
            return relations
            
        except Exception as e:
            print(f"[LLMRelations] Failed to parse response: {e}")
            return []
    
    def _draw_boxes(
        self,
        image: Image.Image,
        boxes: Sequence[Sequence[float]],
        labels: Sequence[str],
    ) -> Image.Image:
        """Draw labeled boxes on image for visual context."""
        from PIL import ImageDraw, ImageFont
        
        img = image.copy()
        draw = ImageDraw.Draw(img)
        
        # Try to load a font
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        # Draw boxes with labels
        colors = self._get_colors(len(boxes))
        
        for i, (box, label) in enumerate(zip(boxes, labels)):
            x1, y1, x2, y2 = box
            color = colors[i]
            
            # Draw box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            
            # Draw label
            label_text = f"{i}: {label}"
            bbox = draw.textbbox((x1, y1 - 20), label_text, font=font)
            draw.rectangle(bbox, fill=color)
            draw.text((x1, y1 - 20), label_text, fill="white", font=font)
        
        return img
    
    @staticmethod
    def _get_colors(n: int) -> List[Tuple[int, int, int]]:
        """Generate n distinct colors."""
        import colorsys
        colors = []
        for i in range(n):
            hue = i / n
            r, g, b = colorsys.hsv_to_rgb(hue, 0.9, 0.9)
            colors.append((int(r * 255), int(g * 255), int(b * 255)))
        return colors
    
    def _get_cache_key(
        self,
        image: Image.Image,
        boxes: Sequence[Sequence[float]],
        labels: Sequence[str],
        question: Optional[str],
    ) -> str:
        """Generate cache key."""
        import hashlib
        
        # Hash image
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_hash = hashlib.md5(buffered.getvalue()).hexdigest()[:16]
        
        # Hash boxes and labels
        boxes_str = str(boxes)
        labels_str = str(labels)
        question_str = question or ""
        
        key = f"{img_hash}_{boxes_str}_{labels_str}_{question_str}"
        return hashlib.md5(key.encode()).hexdigest()
