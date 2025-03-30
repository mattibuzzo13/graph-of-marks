#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TextVQA dataset loader and parser

This script provides utilities for loading and parsing the TextVQA dataset,
which consists of questions about text in images.
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple


class TextVQADataset:
    """Class for loading and processing TextVQA data."""
    
    def __init__(self, data_dir: str = "textvqa"):
        """
        Initialize the TextVQA dataset loader.
        
        Args:
            data_dir: Path to the directory containing TextVQA data
        """
        self.data_dir = Path(data_dir)
        self.annotation_dir = self.data_dir / "annotations"
        self.image_dir = self.data_dir / "images"
        
        # Verify data directory structure
        if not self.annotation_dir.exists():
            raise FileNotFoundError(f"Annotation directory not found: {self.annotation_dir}")
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
    
    def load_split(self, split: str = "train") -> Dict[str, Any]:
        """
        Load a specific dataset split.
        
        Args:
            split: Dataset split to load ('train', 'val', or 'test')
            
        Returns:
            Dictionary containing the dataset
        """
        # Validate split
        if split not in ["train", "val", "test"]:
            raise ValueError(f"Invalid split: {split}. Must be one of 'train', 'val', or 'test'")
        
        # Load annotations for the split
        anno_path = self.annotation_dir / f"TextVQA_0.5.1_{split}.json"
        if not anno_path.exists():
            raise FileNotFoundError(f"Annotation file not found: {anno_path}")
        
        with open(anno_path, 'r') as f:
            dataset = json.load(f)
        
        # Load OCR tokens for the split
        ocr_path = self.annotation_dir / f"TextVQA_0.5.1_{split}_ocr_tokens.json"
        if not ocr_path.exists():
            print(f"Warning: OCR token file not found: {ocr_path}")
            ocr_tokens = {}
        else:
            with open(ocr_path, 'r') as f:
                ocr_tokens = json.load(f)
        
        # Merge OCR tokens with the main dataset
        for item in dataset['data']:
            img_id = item['image_id']
            if img_id in ocr_tokens:
                item['ocr_tokens'] = ocr_tokens[img_id]
            else:
                item['ocr_tokens'] = []
        
        # Add absolute image paths
        for item in dataset['data']:
            img_id = item['image_id']
            item['image_path_abs'] = str(self.image_dir / f"{img_id}.jpg")
        
        return dataset
    
    def load_test_annotations(self) -> Dict[str, Any]:
        """
        Load test split annotations (if available).
        
        Returns:
            Dictionary containing the test annotations
        """
        anno_path = self.annotation_dir / "TextVQA_0.5.1_test_annotations.json"
        if not anno_path.exists():
            raise FileNotFoundError(f"Test annotation file not found: {anno_path}")
        
        with open(anno_path, 'r') as f:
            annotations = json.load(f)
        
        return annotations
    
    def get_question_answer_pairs(self, split: str = "train") -> List[Dict[str, Any]]:
        """
        Extract question-answer pairs from the dataset.
        
        Args:
            split: Dataset split to process
            
        Returns:
            List of dictionaries with question and answer information
        """
        dataset = self.load_split(split)
        
        qa_pairs = []
        for item in dataset['data']:
            img_id = item['image_id']
            img_path = item['image_path_abs']
            
            for qa in item['questions']:
                question_id = qa['question_id']
                question_text = qa['question']
                
                # Find answers for this question
                answers = []
                for ans in item.get('answers', []):
                    if ans['question_id'] == question_id:
                        answers.append(ans['answer'])
                
                qa_pairs.append({
                    'question_id': question_id,
                    'image_id': img_id,
                    'image_path': img_path,
                    'question': question_text,
                    'answers': answers,
                    'ocr_tokens': item.get('ocr_tokens', [])
                })
        
        return qa_pairs
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get dataset statistics.
        
        Returns:
            Dictionary with dataset statistics
        """
        stats = {}
        
        for split in ['train', 'val', 'test']:
            try:
                dataset = self.load_split(split)
                
                # Count images, questions, and answers
                num_images = len({item['image_id'] for item in dataset['data']})
                num_questions = sum(len(item['questions']) for item in dataset['data'])
                
                # Count OCR tokens
                total_ocr_tokens = sum(len(item.get('ocr_tokens', [])) for item in dataset['data'])
                avg_ocr_tokens = total_ocr_tokens / num_images if num_images > 0 else 0
                
                stats[split] = {
                    'num_images': num_images,
                    'num_questions': num_questions,
                    'total_ocr_tokens': total_ocr_tokens,
                    'avg_ocr_tokens_per_image': avg_ocr_tokens
                }
                
                # Count answers for train and val sets
                if split in ['train', 'val']:
                    num_answers = sum(len(item.get('answers', [])) for item in dataset['data'])
                    avg_answers = num_answers / num_questions if num_questions > 0 else 0
                    stats[split]['num_answers'] = num_answers
                    stats[split]['avg_answers_per_question'] = avg_answers
            
            except Exception as e:
                stats[split] = {'error': str(e)}
        
        return stats
    
    def export_to_vqa_format(self, split: str = "train", output_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Export TextVQA data to a format compatible with VQA models.
        
        Args:
            split: Dataset split to export
            output_file: Optional path to save the exported data
            
        Returns:
            Dictionary in VQA format
        """
        dataset = self.load_split(split)
        
        # Prepare VQA formatted data
        vqa_data = []
        for item in dataset['data']:
            img_id = item['image_id']
            img_path = item['image_path_abs']
            
            for qa in item['questions']:
                question_id = qa['question_id']
                question_text = qa['question']
                
                # Find answers for this question
                answers = []
                for ans in item.get('answers', []):
                    if ans['question_id'] == question_id:
                        answers.append({
                            'answer': ans['answer'],
                            'answer_confidence': 'yes'
                        })
                
                vqa_item = {
                    'question_id': question_id,
                    'image_id': img_id,
                    'image_path': img_path,
                    'question': question_text,
                    'answers': answers
                }
                
                # Add OCR tokens as metadata
                if 'ocr_tokens' in item:
                    vqa_item['ocr_tokens'] = item['ocr_tokens']
                
                vqa_data.append(vqa_item)
        
        # Save to file if specified
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(vqa_data, f, indent=2)
        
        return vqa_data


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='TextVQA dataset parser')
    
    parser.add_argument('--data-dir', type=str, default='textvqa',
                        help='Path to TextVQA dataset directory')
    parser.add_argument('--split', type=str, default='train',
                        help='Dataset split to process (train, val, test)')
    parser.add_argument('--export', type=str, default=None,
                        help='Export data to VQA format and save to specified file')
    parser.add_argument('--stats', action='store_true',
                        help='Print dataset statistics')
    
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()
    
    # Initialize dataset
    dataset = TextVQADataset(args.data_dir)
    
    # Print statistics if requested
    if args.stats:
        stats = dataset.get_stats()
        print("\nTextVQA Dataset Statistics:")
        print("-" * 50)
        
        for split, split_stats in stats.items():
            print(f"\n{split.upper()} split:")
            for key, value in split_stats.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.2f}")
                else:
                    print(f"  {key}: {value}")
    
    # Export data if requested
    if args.export:
        vqa_data = dataset.export_to_vqa_format(args.split, args.export)
        print(f"\nExported {len(vqa_data)} question-answer pairs to {args.export}")
    
    # Print example data if no specific action is requested
    if not args.stats and not args.export:
        qa_pairs = dataset.get_question_answer_pairs(args.split)
        print(f"\nLoaded {len(qa_pairs)} question-answer pairs from {args.split} split")
        
        # Print a few examples
        for i, qa in enumerate(qa_pairs[:3]):
            print(f"\nExample {i+1}:")
            print(f"  Question: {qa['question']}")
            print(f"  Image ID: {qa['image_id']}")
            print(f"  Answers: {qa['answers']}")
            print(f"  OCR tokens: {qa['ocr_tokens'][:5]}..." if len(qa['ocr_tokens']) > 5 else f"  OCR tokens: {qa['ocr_tokens']}")


if __name__ == "__main__":
    main()