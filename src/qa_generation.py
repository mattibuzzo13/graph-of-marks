"""
Visual Question Answering using VLLM (Vision-Language Large Language Models)

This script handles:
1. Loading images and extracting visual features
2. Processing questions/prompts related to visual content
3. Generating answers using vision-language models
4. Evaluating and saving results

Supports multiple VLM architectures including LLaVA, MiniGPT4, and others via VLLM.
"""

import os
import json
import argparse
import torch
import logging
from typing import List, Dict, Any, Tuple, Optional, Union
from PIL import Image
import requests
from io import BytesIO
from tqdm import tqdm
from dataclasses import dataclass
import time
import numpy as np

# Import VLLM
from vllm import LLM, SamplingParams
from vllm.model_executor.guided_decoding import get_guided_decoding_logits_processor

# Import various vision-language models
try:
    from transformers import (
        AutoProcessor, 
        AutoModelForCausalLM,
        LlavaForConditionalGeneration, 
        BlipProcessor, 
        Blip2ForConditionalGeneration
    )
    HF_TRANSFORMERS_AVAILABLE = True
except ImportError:
    HF_TRANSFORMERS_AVAILABLE = False
    print("Warning: Hugging Face Transformers not available. Install with: pip install transformers")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("vqa_generation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Data structures
@dataclass
class VQAExample:
    """Data structure for a visual question answering example."""
    image_path: str
    question: str
    answer: Optional[str] = None
    image_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "image_path": self.image_path,
            "question": self.question,
            "answer": self.answer,
            "image_id": self.image_id,
            "metadata": self.metadata or {}
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VQAExample':
        """Create from dictionary."""
        return cls(
            image_path=data["image_path"],
            question=data["question"],
            answer=data.get("answer"),
            image_id=data.get("image_id"),
            metadata=data.get("metadata", {})
        )

# -------------------------------------------------------------------------
# Image Loading and Processing Functions
# -------------------------------------------------------------------------
def load_image(image_path: str) -> Image.Image:
    """
    Load an image from a file path or URL.
    
    Args:
        image_path: Path to the image file or URL
        
    Returns:
        PIL Image object
    """
    if image_path.startswith(('http://', 'https://')):
        response = requests.get(image_path, stream=True)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
    else:
        image = Image.open(image_path)
    
    # Convert to RGB if needed
    if image.mode != "RGB":
        image = image.convert("RGB")
        
    return image

def load_examples_from_json(json_file: str) -> List[VQAExample]:
    """
    Load VQA examples from a JSON file.
    
    Expected format:
    [
        {
            "image_path": "/path/to/image.jpg",
            "question": "What color is the car?",
            "answer": "red"  # Optional
        },
        ...
    ]
    
    Args:
        json_file: Path to the JSON file
        
    Returns:
        List of VQAExample objects
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    examples = []
    for item in data:
        examples.append(VQAExample.from_dict(item))
    
    return examples

# -------------------------------------------------------------------------
# VLLM Integration
# -------------------------------------------------------------------------
class VLLMWrapper:
    """Wrapper for VLLM to handle vision-language models."""
    
    def __init__(
        self, 
        model_name: str, 
        device: str = "cuda", 
        max_length: int = 512,
        temperature: float = 0.2,
        top_p: float = 0.9,
        quantization: Optional[str] = None,
        tensor_parallel_size: int = 1
    ):
        """
        Initialize the VLLM wrapper.
        
        Args:
            model_name: Name or path of the model
            device: Device to run on ("cuda" or "cpu")
            max_length: Maximum length of generated text
            temperature: Temperature for sampling
            top_p: Top-p probability threshold for sampling
            quantization: Quantization method (None, "int8", "int4")
            tensor_parallel_size: Number of GPUs for tensor parallelism
        """
        
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.temperature = temperature
        self.top_p = top_p
        
        # Initialize VLLM
        logger.info(f"Loading model {model_name} with VLLM...")
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            quantization=quantization,
            dtype="half" if device == "cuda" else "float32",
            trust_remote_code=True,
        )
        
        # Set up sampling parameters
        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_length,
        )
        
        # Check if this is a vision-language model
        self.is_vision_model = "llava" in model_name.lower() or "mpt-7b-instruct" in model_name.lower()
        logger.info(f"Model loaded: {model_name}")
        
    def generate(self, 
                 prompt: str, 
                 image_path: Optional[str] = None) -> str:
        """
        Generate a response for a given prompt and optional image.
        
        Args:
            prompt: Text prompt
            image_path: Optional path to an image file
            
        Returns:
            Generated text response
        """
        if self.is_vision_model and image_path:
            try:
                # Format input as messages with text and image content
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image_path
                                }
                            }
                        ]
                    }
                ]
                
                # Use the chat API instead of generate for better image handling
                outputs = self.llm.chat(messages, sampling_params=self.sampling_params)
                
            except Exception as e:
                logger.error(f"Error processing image: {e}")
                # Fallback to text-only generation
                outputs = self.llm.generate([prompt], self.sampling_params)
                
                # Extract the generated text from generate API
                if outputs and len(outputs) > 0:
                    generated_text = outputs[0].outputs[0].text.strip()
                    return generated_text
                else:
                    logger.warning("No output generated")
                    return ""
        else:
            # For text-only models or when no image is provided
            outputs = self.llm.generate([prompt], self.sampling_params)
            
            # Extract the generated text from generate API
            if outputs and len(outputs) > 0:
                generated_text = outputs[0].outputs[0].text.strip()
                return generated_text
            else:
                logger.warning("No output generated")
                return ""
        
        # Extract the generated text from chat API
        if outputs and len(outputs) > 0:
            generated_text = outputs[0].outputs[0].text.strip()
            return generated_text
        else:
            logger.warning("No output generated")
            return ""

# -------------------------------------------------------------------------
# Hugging Face VL Models Integration 
# -------------------------------------------------------------------------
class HFVisionLanguageModel:
    """Wrapper for Hugging Face vision-language models."""
    
    def __init__(
        self, 
        model_name: str, 
        device: str = "cuda", 
        max_length: int = 512,
        temperature: float = 0.2,
        top_p: float = 0.9,
    ):
        """
        Initialize the Hugging Face vision-language model.
        
        Args:
            model_name: Name or path of the model
            device: Device to run on ("cuda" or "cpu")
            max_length: Maximum length of generated text
            temperature: Temperature for sampling
            top_p: Top-p probability threshold for sampling
        """
        if not HF_TRANSFORMERS_AVAILABLE:
            raise ImportError("Hugging Face Transformers is not installed")
        
        self.model_name = model_name
        self.device = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.temperature = temperature
        self.top_p = top_p
        
        # Select the right model class based on the model name
        if "llava" in model_name.lower():
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = LlavaForConditionalGeneration.from_pretrained(
                model_name, torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32
            )
        elif "blip" in model_name.lower():
            self.processor = BlipProcessor.from_pretrained(model_name)
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                model_name, torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32
            )
        else:
            # Generic model loading
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32
            )
        
        self.model.to(self.device)
        self.model.eval()
        
    def generate(self, prompt: str, image_path: Optional[str] = None) -> str:
        """
        Generate a response for a given prompt and optional image.
        
        Args:
            prompt: Text prompt
            image_path: Optional path to an image file
            
        Returns:
            Generated text response
        """
        with torch.no_grad():
            try:
                if image_path:
                    # Process image and text for vision-language model
                    image = load_image(image_path)
                    
                    # Different models have different input formats
                    if "llava" in self.model_name.lower():
                        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)
                    elif "blip" in self.model_name.lower():
                        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)
                    else:
                        # Generic processing, may not work for all models
                        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)
                else:
                    # Text-only input
                    inputs = self.processor(text=prompt, return_tensors="pt").to(self.device)
                
                # Generate response
                outputs = self.model.generate(
                    **inputs,
                    max_length=self.max_length,
                    do_sample=True,
                    temperature=self.temperature,
                    top_p=self.top_p
                )
                
                # Process output
                if "llava" in self.model_name.lower():
                    generated_text = self.processor.decode(outputs[0], skip_special_tokens=True)
                elif "blip" in self.model_name.lower():
                    generated_text = self.processor.decode(outputs[0], skip_special_tokens=True)
                else:
                    generated_text = self.processor.decode(outputs[0], skip_special_tokens=True)
                
                return generated_text.strip()
                
            except Exception as e:
                logger.error(f"Error generating response: {e}")
                return ""

# -------------------------------------------------------------------------
# Main VQA Functions
# -------------------------------------------------------------------------
def run_vqa_batch(
    examples: List[VQAExample],
    model_wrapper: Union[VLLMWrapper, HFVisionLanguageModel],
    output_file: str,
    prompt_template: str = "Question: {question}\nAnswer:",
    batch_size: int = 1,
):
    """
    Run visual question answering on a batch of examples.
    
    Args:
        examples: List of VQAExample objects
        model_wrapper: Model wrapper (VLLM or HF)
        output_file: Path to save results
        prompt_template: Template for formatting the prompt
        batch_size: Batch size for processing
    """
    results = []
    
    # Process examples in batches
    for i in tqdm(range(0, len(examples), batch_size)):
        batch = examples[i:i+batch_size]
        
        for example in batch:
            try:
                # Format the prompt
                prompt = prompt_template.format(question=example.question)
                
                # Generate answer
                start_time = time.time()
                answer = model_wrapper.generate(prompt, image_path=example.image_path)
                end_time = time.time()
                
                # Log results
                logger.info(f"Processed: {example.image_path}")
                logger.info(f"Question: {example.question}")
                logger.info(f"Answer: {answer}")
                logger.info(f"Processing time: {end_time - start_time:.2f} seconds")
                
                # Store results
                result = example.to_dict()
                result["generated_answer"] = answer
                result["processing_time"] = end_time - start_time
                results.append(result)
                
                # Save intermediate results
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent=2)
                    
            except Exception as e:
                logger.error(f"Error processing example {example.image_path}: {e}")
    
    # Final save
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Saved results to {output_file}")
    return results

# -------------------------------------------------------------------------
# Evaluation Functions
# -------------------------------------------------------------------------
def evaluate_vqa_results(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Evaluate VQA results if ground truth answers are available.
    
    Args:
        results: List of result dictionaries
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Filter results with both ground truth and generated answers
    valid_results = [r for r in results if r.get("answer") and r.get("generated_answer")]
    
    if not valid_results:
        logger.warning("No valid results with ground truth answers for evaluation")
        return {}
    
    # Initialize metrics
    metrics = {
        "exact_match": 0,
        "total": len(valid_results),
        "avg_processing_time": 0,
    }
    
    # Calculate metrics
    for result in valid_results:
        # Check for exact match (can be extended with more sophisticated metrics)
        if result["answer"].lower() == result["generated_answer"].lower():
            metrics["exact_match"] += 1
        
        # Accumulate processing time
        metrics["avg_processing_time"] += result.get("processing_time", 0)
    
    # Calculate percentages and averages
    if metrics["total"] > 0:
        metrics["exact_match_percent"] = 100 * metrics["exact_match"] / metrics["total"]
        metrics["avg_processing_time"] = metrics["avg_processing_time"] / metrics["total"]
    
    return metrics

# -------------------------------------------------------------------------
# Main function
# -------------------------------------------------------------------------
def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Visual Question Answering using VLLM")
    
    # Input/output arguments
    parser.add_argument("--input_file", type=str, help="Path to input JSON file with VQA examples")
    parser.add_argument("--output_file", type=str, default="vqa_results.json", help="Path to output JSON file")
    parser.add_argument("--image_dir", type=str, help="Path to directory containing images (optional)")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="llava-hf/llava-1.5-7b-hf", help="Name or path of the model")
    parser.add_argument("--use_vllm", action="store_true", help="Use VLLM for generation")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on (cuda or cpu)")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Number of GPUs for tensor parallelism")
    parser.add_argument("--quantization", type=str, choices=[None, "int8", "int4"], help="Quantization method")
    
    # Generation arguments
    parser.add_argument("--max_length", type=int, default=512, help="Maximum length of generated text")
    parser.add_argument("--temperature", type=float, default=0.2, help="Temperature for sampling")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p probability threshold for sampling")
    parser.add_argument("--prompt_template", type=str, default="Question: {question}\nAnswer:", 
                        help="Template for formatting the prompt")
    
    # Other arguments
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for processing")
    
    args = parser.parse_args()
    
    # Load examples
    if args.input_file:
        logger.info(f"Loading examples from {args.input_file}")
        examples = load_examples_from_json(args.input_file)
    else:
        logger.error("No input file specified")
        return
    
    # Update image paths if image_dir is provided
    if args.image_dir:
        for example in examples:
            if not os.path.isabs(example.image_path):
                example.image_path = os.path.join(args.image_dir, example.image_path)
    
    # Initialize model
    model_wrapper = VLLMWrapper(
        model_name=args.model_name,
        device=args.device,
        max_length=args.max_length,
        temperature=args.temperature,
        top_p=args.top_p,
        quantization=args.quantization,
        tensor_parallel_size=args.tensor_parallel_size
    )
    
    # Run VQA
    results = run_vqa_batch(
        examples=examples,
        model_wrapper=model_wrapper,
        output_file=args.output_file,
        prompt_template=args.prompt_template,
        batch_size=args.batch_size
    )
    
    # Evaluate results
    metrics = evaluate_vqa_results(results)
    logger.info(f"Evaluation metrics: {metrics}")
    
    # Save metrics
    metrics_file = os.path.splitext(args.output_file)[0] + "_metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Saved metrics to {metrics_file}")

if __name__ == "__main__":
    main()
