#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dataset-specific evaluation for visual reasoning tasks.

This script provides evaluation metrics for different dataset types:
- COCO: Segmentation metrics (IoU, precision, recall, F1)
- RefCOCO: Segmentation and localization metrics
- GQA: Visual reasoning question answering metrics
- VQA: Visual question answering metrics
"""

import os
import sys
import json
import logging
import argparse
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Any, Tuple, Optional, Union
from collections import defaultdict
import matplotlib.pyplot as plt
from tabulate import tabulate
from statistics import mean, median

# Import dataset-specific libraries
try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from pycocotools import mask as mask_utils
    COCO_AVAILABLE = True
except ImportError:
    COCO_AVAILABLE = False
    print("Warning: pycocotools not available. COCO evaluation will be limited.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Evaluator:
    """Base evaluator class with common evaluation utilities."""
    
    def __init__(self, dataset_name: str, predictions_path: str, 
                 ground_truth_path: str, output_dir: str = None):
        """
        Initialize the evaluator.
        
        Args:
            dataset_name: Name of the dataset (coco, refcoco, gqa, vqa)
            predictions_path: Path to the predictions file
            ground_truth_path: Path to the ground truth annotations
            output_dir: Directory to save evaluation results
        """
        self.dataset_name = dataset_name.lower()
        self.predictions_path = predictions_path
        self.ground_truth_path = ground_truth_path
        self.output_dir = output_dir or os.path.join(
            os.path.dirname(predictions_path), "evaluation_results"
        )
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load predictions and ground truth
        self.predictions = self._load_json(predictions_path)
        self.ground_truth = self._load_data(ground_truth_path)
        
        # Validation check
        if not self.predictions or not self.ground_truth:
            raise ValueError("Failed to load predictions or ground truth data")
        
        logger.info(f"Initialized {self.dataset_name} evaluator")
        logger.info(f"Loaded {len(self.predictions)} predictions")
    
    def _load_json(self, file_path: str) -> dict:
        """Load a JSON file."""
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading JSON file {file_path}: {e}")
            return {}
    
    def _load_data(self, file_path: str) -> Any:
        """Load dataset-specific data."""
        if self.dataset_name == "coco":
            return self._load_coco(file_path)
        elif self.dataset_name == "refcoco":
            return self._load_refcoco(file_path)
        elif self.dataset_name == "gqa":
            return self._load_gqa(file_path)
        elif self.dataset_name == "vqa":
            return self._load_vqa(file_path)
        elif self.dataset_name == "textvqa":
            return self._load_textvqa(file_path)
        else:
            return self._load_json(file_path)
    
    def _load_coco(self, file_path: str) -> Any:
        """Load COCO annotations."""
        if not COCO_AVAILABLE:
            logger.warning("pycocotools not available, loading as JSON")
            return self._load_json(file_path)
        try:
            return COCO(file_path)
        except Exception as e:
            logger.error(f"Error loading COCO annotations: {e}")
            return self._load_json(file_path)
    
    def _load_refcoco(self, file_path: str) -> Any:
        """Load RefCOCO annotations."""
        # RefCOCO typically uses COCO format with additional properties
        return self._load_coco(file_path)
    
    def _load_gqa(self, file_path: str) -> Any:
        """Load GQA annotations."""
        return self._load_json(file_path)
    
    def _load_vqa(self, file_path: str) -> Any:
        """Load VQA annotations."""
        return self._load_json(file_path)
        
    def _load_textvqa(self, file_path: str) -> Any:
        """Load TextVQA annotations."""
        return self._load_json(file_path)
    
    def evaluate(self) -> Dict[str, Any]:
        """
        Evaluate predictions against ground truth.
        
        Returns:
            Dict containing evaluation metrics
        """
        if self.dataset_name == "coco":
            return self.evaluate_coco()
        elif self.dataset_name == "refcoco":
            return self.evaluate_refcoco()
        elif self.dataset_name == "gqa":
            return self.evaluate_gqa()
        elif self.dataset_name == "vqa":
            return self.evaluate_vqa()
        elif self.dataset_name == "textvqa":
            return self.evaluate_textvqa()
        else:
            logger.error(f"Unsupported dataset: {self.dataset_name}")
            return {}
    
    def evaluate_coco(self) -> Dict[str, Any]:
        """
        Evaluate COCO instance segmentation predictions.
        
        Returns:
            Dict containing evaluation metrics (AP, AR, etc.)
        """
        if not COCO_AVAILABLE:
            logger.error("pycocotools required for COCO evaluation")
            return self._fallback_segmentation_evaluation()
        
        # Convert predictions to COCO format if needed
        if not isinstance(self.predictions, list):
            pred_list = self._convert_predictions_to_coco_format(self.predictions)
        else:
            pred_list = self.predictions
        
        # Create COCO results object
        coco_gt = self.ground_truth
        coco_dt = coco_gt.loadRes(pred_list)
        
        # Running COCO evaluation
        coco_eval = COCOeval(coco_gt, coco_dt, 'segm')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        # Extract and format metrics
        metrics = {
            'AP': coco_eval.stats[0],
            'AP50': coco_eval.stats[1],
            'AP75': coco_eval.stats[2],
            'APs': coco_eval.stats[3],
            'APm': coco_eval.stats[4],
            'APl': coco_eval.stats[5],
            'AR1': coco_eval.stats[6],
            'AR10': coco_eval.stats[7],
            'AR100': coco_eval.stats[8],
            'ARs': coco_eval.stats[9],
            'ARm': coco_eval.stats[10],
            'ARl': coco_eval.stats[11]
        }
        
        # Save results to file
        self._save_results(metrics, "coco_evaluation.json")
        self._plot_metrics(metrics, "COCO Evaluation Metrics")
        
        return metrics
    
    def evaluate_refcoco(self) -> Dict[str, Any]:
        """
        Evaluate RefCOCO localization and segmentation predictions.
        
        Returns:
            Dict containing evaluation metrics
        """
        results = {}
        
        # IoU thresholds for evaluation
        iou_thresholds = [0.5, 0.7, 0.9]
        
        # Evaluate localization (bounding box) accuracy
        loc_accuracy = defaultdict(list)
        loc_ious = []
        
        # Evaluate segmentation accuracy
        seg_ious = []
        
        # Process each prediction
        for pred in tqdm(self.predictions, desc="Evaluating RefCOCO"):
            image_id = pred.get('image_id')
            pred_box = pred.get('bbox')  # [x, y, width, height]
            pred_mask = pred.get('segmentation')
            
            # Get ground truth
            gt_annos = self._get_refcoco_gt_annotations(image_id, pred.get('ref_id'))
            if not gt_annos:
                continue
                
            gt_box = gt_annos.get('bbox')
            gt_mask = gt_annos.get('segmentation')
            
            # Calculate IoU for bounding box
            if pred_box and gt_box:
                box_iou = self._calculate_bbox_iou(pred_box, gt_box)
                loc_ious.append(box_iou)
                
                # Check if the prediction meets each threshold
                for threshold in iou_thresholds:
                    loc_accuracy[threshold].append(box_iou >= threshold)
            
            # Calculate IoU for segmentation mask
            if pred_mask and gt_mask:
                mask_iou = self._calculate_mask_iou(pred_mask, gt_mask)
                seg_ious.append(mask_iou)
        
        # Calculate metrics
        results['mean_box_iou'] = mean(loc_ious) if loc_ious else 0
        results['median_box_iou'] = median(loc_ious) if loc_ious else 0
        results['mean_mask_iou'] = mean(seg_ious) if seg_ious else 0
        results['median_mask_iou'] = median(seg_ious) if seg_ious else 0
        
        # Calculate accuracy at different thresholds
        for threshold in iou_thresholds:
            acc = mean(loc_accuracy[threshold]) if loc_accuracy[threshold] else 0
            results[f'accuracy_at_{int(threshold*100)}'] = acc
        
        # Save results to file
        self._save_results(results, "refcoco_evaluation.json")
        self._plot_refcoco_metrics(results)
        
        return results
    
    def evaluate_gqa(self) -> Dict[str, Any]:
        """
        Evaluate GQA visual reasoning question answering.
        
        Returns:
            Dict containing evaluation metrics
        """
        # Initialize counters
        total = 0
        correct = 0
        question_types = defaultdict(lambda: {'correct': 0, 'total': 0})
        
        # Process each prediction
        for pred in tqdm(self.predictions, desc="Evaluating GQA"):
            if 'question_id' not in pred:
                logger.warning("Missing question_id in prediction")
                continue
                
            # Get ground truth answer
            gt_answer = self._get_gqa_gt_answer(pred['question_id'])
            if not gt_answer:
                continue
                
            pred_answer = pred.get('answer', '')
            question_type = self._get_gqa_question_type(pred['question_id'])
            
            # Check if prediction is correct
            is_correct = self._normalize_answer(pred_answer) == self._normalize_answer(gt_answer)
            
            # Update counters
            total += 1
            correct += int(is_correct)
            
            # Update question type statistics
            question_types[question_type]['total'] += 1
            question_types[question_type]['correct'] += int(is_correct)
        
        # Calculate overall accuracy
        accuracy = correct / total if total > 0 else 0
        
        # Calculate per-type accuracy
        per_type_accuracy = {}
        for qtype, stats in question_types.items():
            per_type_accuracy[qtype] = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        
        # Calculate balanced accuracy (average of per-type accuracies)
        balanced_accuracy = mean(per_type_accuracy.values()) if per_type_accuracy else 0
        
        # Results
        results = {
            'accuracy': accuracy,
            'balanced_accuracy': balanced_accuracy,
            'question_types': per_type_accuracy,
            'total_questions': total
        }
        
        # Save results to file
        self._save_results(results, "gqa_evaluation.json")
        self._plot_gqa_metrics(results)
        
        return results
    
    def evaluate_vqa(self) -> Dict[str, Any]:
        """
        Evaluate VQA predictions.
        
        Returns:
            Dict containing evaluation metrics
        """
        # Initialize counters
        total = 0
        scores = []
        question_types = defaultdict(lambda: {'score': 0.0, 'total': 0})
        
        # Process each prediction
        for pred in tqdm(self.predictions, desc="Evaluating VQA"):
            if 'question_id' not in pred:
                logger.warning("Missing question_id in prediction")
                continue
                
            # Get ground truth answers (VQA has multiple answers per question)
            gt_answers = self._get_vqa_gt_answers(pred['question_id'])
            if not gt_answers:
                continue
                
            pred_answer = pred.get('answer', '')
            question_type = self._get_vqa_question_type(pred['question_id'])
            
            # Calculate VQA score
            vqa_score = self._calculate_vqa_score(pred_answer, gt_answers)
            
            # Update counters
            total += 1
            scores.append(vqa_score)
            
            # Update question type statistics
            question_types[question_type]['total'] += 1
            question_types[question_type]['score'] += vqa_score
        
        # Calculate overall accuracy
        mean_score = mean(scores) if scores else 0
        
        # Calculate per-type accuracy
        per_type_score = {}
        for qtype, stats in question_types.items():
            per_type_score[qtype] = stats['score'] / stats['total'] if stats['total'] > 0 else 0
        
        # Results
        results = {
            'mean_score': mean_score,
            'question_types': per_type_score,
            'total_questions': total
        }
        
        # Save results to file
        self._save_results(results, "vqa_evaluation.json")
        self._plot_vqa_metrics(results)
        
        return results
    
    def _fallback_segmentation_evaluation(self) -> Dict[str, Any]:
        """
        Fallback evaluation for segmentation when pycocotools is not available.
        """
        # Initialize counters
        ious = []
        precisions = []
        recalls = []
        f1_scores = []
        
        # Process each prediction
        for pred in tqdm(self.predictions, desc="Evaluating segmentation"):
            if 'image_id' not in pred or 'category_id' not in pred or 'segmentation' not in pred:
                continue
                
            # Get ground truth for this image and category
            gt_masks = self._get_gt_masks(pred['image_id'], pred['category_id'])
            if not gt_masks:
                continue
                
            # Calculate metrics for this prediction
            pred_mask = pred['segmentation']
            best_iou = 0
            best_precision = 0
            best_recall = 0
            best_f1 = 0
            
            for gt_mask in gt_masks:
                # Calculate intersection and union
                intersection = self._calculate_mask_intersection(pred_mask, gt_mask)
                union = self._calculate_mask_union(pred_mask, gt_mask)
                
                # Calculate IoU
                iou = intersection / union if union > 0 else 0
                
                # Calculate precision and recall
                precision = intersection / self._calculate_mask_area(pred_mask) if self._calculate_mask_area(pred_mask) > 0 else 0
                recall = intersection / self._calculate_mask_area(gt_mask) if self._calculate_mask_area(gt_mask) > 0 else 0
                
                # Calculate F1 score
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                # Keep the best metrics across all ground truth masks
                if iou > best_iou:
                    best_iou = iou
                    best_precision = precision
                    best_recall = recall
                    best_f1 = f1
            
            # Update counters
            ious.append(best_iou)
            precisions.append(best_precision)
            recalls.append(best_recall)
            f1_scores.append(best_f1)
        
        # Calculate overall metrics
        results = {
            'mean_iou': mean(ious) if ious else 0,
            'mean_precision': mean(precisions) if precisions else 0,
            'mean_recall': mean(recalls) if recalls else 0,
            'mean_f1': mean(f1_scores) if f1_scores else 0,
        }
        
        # Save results to file
        self._save_results(results, "segmentation_evaluation.json")
        self._plot_metrics(results, "Segmentation Metrics")
        
        return results
    
    def _calculate_bbox_iou(self, box1: List[float], box2: List[float]) -> float:
        """
        Calculate IoU between two bounding boxes.
        
        Args:
            box1: [x, y, width, height]
            box2: [x, y, width, height]
            
        Returns:
            IoU value
        """
        # Convert [x, y, width, height] to [x1, y1, x2, y2]
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[0] + box1[2], box1[1] + box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[0] + box2[2], box2[1] + box2[3]
        
        # Calculate intersection area
        x_left = max(b1_x1, b2_x1)
        y_top = max(b1_y1, b2_y1)
        x_right = min(b1_x2, b2_x2)
        y_bottom = min(b1_y2, b2_y2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
            
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union area
        b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
        union_area = b1_area + b2_area - intersection_area
        
        # Calculate IoU
        iou = intersection_area / union_area if union_area > 0 else 0
        
        return iou
    
    def _calculate_mask_iou(self, mask1: Any, mask2: Any) -> float:
        """
        Calculate IoU between two segmentation masks.
        
        Args:
            mask1: First segmentation mask
            mask2: Second segmentation mask
            
        Returns:
            IoU value
        """
        if COCO_AVAILABLE:
            # Convert masks to RLE if they're not already
            if isinstance(mask1, list):  # polygon format
                mask1 = mask_utils.frPyObjects(mask1, 1000, 1000)  # Assuming max dimensions
            if isinstance(mask2, list):  # polygon format
                mask2 = mask_utils.frPyObjects(mask2, 1000, 1000)  # Assuming max dimensions
                
            # Calculate intersection and union
            intersection = mask_utils.area(mask_utils.merge([mask1, mask2], intersect=True))
            union = mask_utils.area(mask_utils.merge([mask1, mask2], intersect=False))
            
            # Calculate IoU
            iou = intersection / union if union > 0 else 0
            
            return iou
        else:
            # Simplified calculation when pycocotools is not available
            intersection = self._calculate_mask_intersection(mask1, mask2)
            union = self._calculate_mask_union(mask1, mask2)
            return intersection / union if union > 0 else 0
    
    def _calculate_mask_intersection(self, mask1: Any, mask2: Any) -> float:
        """Simplified mask intersection calculation (approximation)."""
        # This is a placeholder - would need actual mask manipulation library
        return 0.0
    
    def _calculate_mask_union(self, mask1: Any, mask2: Any) -> float:
        """Simplified mask union calculation (approximation)."""
        # This is a placeholder - would need actual mask manipulation library
        return self._calculate_mask_area(mask1) + self._calculate_mask_area(mask2) - self._calculate_mask_intersection(mask1, mask2)
    
    def _calculate_mask_area(self, mask: Any) -> float:
        """Simplified mask area calculation (approximation)."""
        # This is a placeholder - would need actual mask manipulation library
        if COCO_AVAILABLE and isinstance(mask, dict) and 'counts' in mask and 'size' in mask:
            return mask_utils.area(mask)
        return 1.0  # Default value
    
    def _get_gt_masks(self, image_id: int, category_id: int) -> List[Any]:
        """Get ground truth masks for a specific image and category."""
        # This will need to be tailored to your specific dataset format
        return []
    
    def _get_refcoco_gt_annotations(self, image_id: int, ref_id: int) -> Dict[str, Any]:
        """Get ground truth annotations for a RefCOCO referring expression."""
        # This will need to be tailored to your specific dataset format
        return {}
    
    def _get_gqa_gt_answer(self, question_id: str) -> str:
        """Get ground truth answer for a GQA question."""
        # This will need to be tailored to your specific dataset format
        if isinstance(self.ground_truth, dict) and 'annotations' in self.ground_truth:
            annotations = self.ground_truth['annotations']
            for anno in annotations:
                if anno.get('question_id') == question_id:
                    return anno.get('answer', '')
        return ''
    
    def _get_gqa_question_type(self, question_id: str) -> str:
        """Get the question type for a GQA question."""
        # This will need to be tailored to your specific dataset format
        if isinstance(self.ground_truth, dict) and 'questions' in self.ground_truth:
            questions = self.ground_truth['questions']
            for q in questions:
                if q.get('question_id') == question_id:
                    return q.get('question_type', 'unknown')
        return 'unknown'
    
    def _get_vqa_gt_answers(self, question_id: str) -> List[str]:
        """Get ground truth answers for a VQA question."""
        # VQA typically has multiple answers per question
        answers = []
        if isinstance(self.ground_truth, dict) and 'annotations' in self.ground_truth:
            annotations = self.ground_truth['annotations']
            for anno in annotations:
                if anno.get('question_id') == question_id:
                    if 'answers' in anno:
                        answers = [a.get('answer', '') for a in anno['answers']]
                    break
        return answers
    
    def _get_vqa_question_type(self, question_id: str) -> str:
        """Get the question type for a VQA question."""
        if isinstance(self.ground_truth, dict) and 'questions' in self.ground_truth:
            questions = self.ground_truth['questions']
            for q in questions:
                if q.get('question_id') == question_id:
                    return q.get('question_type', 'unknown')
        return 'unknown'
        
    def _get_textvqa_gt_answers(self, question_id: str) -> List[str]:
        """Get ground truth answers for a TextVQA question."""
        # TextVQA format is similar to VQA but with different structure
        answers = []
        if isinstance(self.ground_truth, dict):
            if 'data' in self.ground_truth:
                # Handle primary annotation format
                for item in self.ground_truth['data']:
                    for qa in item.get('questions', []):
                        if qa.get('question_id') == question_id:
                            # Find answers for this question
                            for ans in item.get('answers', []):
                                if ans.get('question_id') == question_id:
                                    answers.append(ans.get('answer', ''))
                            break
            elif 'annotations' in self.ground_truth:
                # Handle alternative annotation format
                for anno in self.ground_truth['annotations']:
                    if anno.get('question_id') == question_id:
                        if 'answers' in anno:
                            answers = [a.get('answer', '') for a in anno['answers']]
                        break
        
        return answers
    
    def _get_textvqa_question_type(self, question_id: str) -> str:
        """
        Get the question type for a TextVQA question.
        
        TextVQA doesn't have explicit question types, so we infer them
        from the question text.
        """
        # Try to extract the question from different formats
        question_text = ""
        if isinstance(self.ground_truth, dict):
            if 'data' in self.ground_truth:
                for item in self.ground_truth['data']:
                    for qa in item.get('questions', []):
                        if qa.get('question_id') == question_id:
                            question_text = qa.get('question', '')
                            break
            elif 'questions' in self.ground_truth:
                for q in self.ground_truth['questions']:
                    if q.get('question_id') == question_id:
                        question_text = q.get('question', '')
                        break
                        
        # Determine question type based on first word
        if not question_text:
            return 'unknown'
            
        question_text = question_text.lower().strip()
        
        # Simple rule-based classification
        if question_text.startswith('what'):
            return 'what'
        elif question_text.startswith('where'):
            return 'where'
        elif question_text.startswith('who'):
            return 'who'
        elif question_text.startswith('how'):
            return 'how'
        elif question_text.startswith('when'):
            return 'when'
        elif question_text.startswith('why'):
            return 'why'
        elif question_text.startswith('which'):
            return 'which'
        elif question_text.startswith('is') or question_text.startswith('are') or question_text.startswith('does'):
            return 'yes/no'
        else:
            return 'other'
    
    def _get_textvqa_ocr_tokens(self, question_id: str) -> List[str]:
        """Get OCR tokens for a TextVQA question/image."""
        if isinstance(self.ground_truth, dict) and 'data' in self.ground_truth:
            for item in self.ground_truth['data']:
                for qa in item.get('questions', []):
                    if qa.get('question_id') == question_id:
                        # Found the question, get the image id
                        return item.get('ocr_tokens', [])
                        
        return []
    
    def _calculate_vqa_score(self, pred_answer: str, gt_answers: List[str]) -> float:
        """
        Calculate VQA score according to the VQA evaluation protocol.
        
        The score is min(# of humans that said answer / 3, 1)
        """
        pred_answer = self._normalize_answer(pred_answer)
        
        # Count matching answers
        matching_answers = 0
        for answer in gt_answers:
            if self._normalize_answer(answer) == pred_answer:
                matching_answers += 1
        
        # Calculate score
        score = min(matching_answers / 3.0, 1.0) if gt_answers else 0.0
        
        return score
    
    def _normalize_answer(self, answer: str) -> str:
        """Normalize answer string for comparison."""
        answer = answer.lower().strip()
        # Remove articles and punctuation (if needed)
        return answer
    
    def _convert_predictions_to_coco_format(self, predictions: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert predictions to COCO format if needed."""
        if isinstance(predictions, list):
            return predictions
            
        # Convert dict to list if needed
        pred_list = []
        for image_id, preds in predictions.items():
            for pred in preds:
                pred_list.append(pred)
        
        return pred_list
    
    def _save_results(self, results: Dict[str, Any], filename: str) -> None:
        """Save evaluation results to a JSON file."""
        output_path = os.path.join(self.output_dir, filename)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_path}")
        
        # Also save as readable text
        text_path = os.path.join(self.output_dir, filename.replace('.json', '.txt'))
        with open(text_path, 'w') as f:
            f.write(f"Evaluation results for {self.dataset_name.upper()}\n")
            f.write("=" * 50 + "\n\n")
            
            # Create a table of metrics
            table_data = []
            for key, value in results.items():
                if isinstance(value, dict):
                    f.write(f"{key}:\n")
                    for subkey, subvalue in value.items():
                        f.write(f"  {subkey}: {subvalue:.4f}\n")
                else:
                    table_data.append([key, f"{value:.4f}" if isinstance(value, float) else value])
            
            # Write the table
            if table_data:
                f.write(tabulate(table_data, headers=["Metric", "Value"], tablefmt="grid"))
                
        logger.info(f"Readable results saved to {text_path}")
    
    def _plot_metrics(self, metrics: Dict[str, Any], title: str) -> None:
        """Create a bar plot of the evaluation metrics."""
        # Filter metrics for plotting (only include numeric values that aren't dictionaries)
        plot_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float)) and not isinstance(v, dict)}
        
        if not plot_metrics:
            return
            
        plt.figure(figsize=(10, 6))
        bars = plt.bar(plot_metrics.keys(), plot_metrics.values())
        
        # Add data labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.3f}', ha='center', va='bottom')
        
        plt.title(title)
        plt.ylabel('Score')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save the plot
        plot_path = os.path.join(self.output_dir, f"{self.dataset_name}_metrics.png")
        plt.savefig(plot_path)
        plt.close()
        logger.info(f"Plot saved to {plot_path}")
    
    def _plot_refcoco_metrics(self, metrics: Dict[str, Any]) -> None:
        """Create plots for RefCOCO metrics."""
        # Create bar chart for accuracy at different IoU thresholds
        threshold_metrics = {k: v for k, v in metrics.items() if k.startswith('accuracy_at_')}
        
        if threshold_metrics:
            plt.figure(figsize=(8, 5))
            thresholds = [int(k.split('_')[-1]) / 100 for k in threshold_metrics.keys()]
            values = list(threshold_metrics.values())
            
            bars = plt.bar([str(t) for t in thresholds], values)
            
            # Add data labels
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                         f'{height:.3f}', ha='center', va='bottom')
            
            plt.title('RefCOCO Accuracy at Different IoU Thresholds')
            plt.xlabel('IoU Threshold')
            plt.ylabel('Accuracy')
            plt.ylim(0, 1.0)
            
            # Save the plot
            plot_path = os.path.join(self.output_dir, "refcoco_threshold_accuracy.png")
            plt.savefig(plot_path)
            plt.close()
        
        # Create general metrics plot
        general_metrics = {k: v for k, v in metrics.items() if k.startswith('mean_') or k.startswith('median_')}
        if general_metrics:
            self._plot_metrics(general_metrics, "RefCOCO Evaluation Metrics")
    
    def _plot_gqa_metrics(self, metrics: Dict[str, Any]) -> None:
        """Create plots for GQA metrics."""
        # Plot overall accuracy metrics
        overall_metrics = {
            'accuracy': metrics.get('accuracy', 0),
            'balanced_accuracy': metrics.get('balanced_accuracy', 0),
        }
        
        self._plot_metrics(overall_metrics, "GQA Overall Accuracy")
        
        # Plot accuracy by question type
        question_types = metrics.get('question_types', {})
        if question_types:
            # Sort by accuracy (descending)
            sorted_types = sorted(question_types.items(), key=lambda x: x[1], reverse=True)
            types = [t[0] for t in sorted_types]
            accuracies = [t[1] for t in sorted_types]
            
            # Select top N types if there are too many
            max_types = 15
            if len(types) > max_types:
                types = types[:max_types-1] + ['other']
                accuracies = accuracies[:max_types-1] + [mean(accuracies[max_types-1:])]
            
            plt.figure(figsize=(12, 8))
            bars = plt.bar(types, accuracies)
            
            # Add data labels
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                         f'{height:.3f}', ha='center', va='bottom')
            
            plt.title('GQA Accuracy by Question Type')
            plt.xlabel('Question Type')
            plt.ylabel('Accuracy')
            plt.ylim(0, 1.0)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            # Save the plot
            plot_path = os.path.join(self.output_dir, "gqa_accuracy_by_type.png")
            plt.savefig(plot_path)
            plt.close()
    
    def _plot_vqa_metrics(self, metrics: Dict[str, Any]) -> None:
        """Create plots for VQA metrics."""
        # Plot overall score
        overall_metrics = {
            'mean_score': metrics.get('mean_score', 0),
        }
        
        self._plot_metrics(overall_metrics, "VQA Overall Score")
        
        # Plot score by question type
        question_types = metrics.get('question_types', {})
        if question_types:
            # Sort by score (descending)
            sorted_types = sorted(question_types.items(), key=lambda x: x[1], reverse=True)
            types = [t[0] for t in sorted_types]
            scores = [t[1] for t in sorted_types]
            
            # Select top N types if there are too many
            max_types = 15
            if len(types) > max_types:
                types = types[:max_types-1] + ['other']
                scores = scores[:max_types-1] + [mean(scores[max_types-1:])]
            
            plt.figure(figsize=(12, 8))
            bars = plt.bar(types, scores)
            
            # Add data labels
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                         f'{height:.3f}', ha='center', va='bottom')
            
            plt.title('VQA Score by Question Type')
            plt.xlabel('Question Type')
            plt.ylabel('Score')
            plt.ylim(0, 1.0)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            # Save the plot
            plot_path = os.path.join(self.output_dir, "vqa_score_by_type.png")
            plt.savefig(plot_path)
            plt.close()
            
    def evaluate_textvqa(self) -> Dict[str, Any]:
        """
        Evaluate TextVQA predictions with OCR-aware metrics.
        
        Returns:
            Dict containing evaluation metrics
        """
        # Initialize counters
        total = 0
        scores = []
        exact_match_counts = 0
        ocr_relevance_counts = 0
        ocr_counts = []
        question_types = defaultdict(lambda: {'score': 0.0, 'total': 0})
        
        # Process each prediction
        for pred in tqdm(self.predictions, desc="Evaluating TextVQA"):
            if 'question_id' not in pred:
                logger.warning("Missing question_id in prediction")
                continue
                
            # Get ground truth answers (TextVQA has multiple answers per question)
            gt_answers = self._get_textvqa_gt_answers(pred['question_id'])
            ocr_tokens = self._get_textvqa_ocr_tokens(pred['question_id'])
            
            if not gt_answers:
                continue
                
            pred_answer = pred.get('answer', '')
            question_type = self._get_textvqa_question_type(pred['question_id'])
            
            # Calculate TextVQA score
            vqa_score = self._calculate_vqa_score(pred_answer, gt_answers)
            
            # Calculate exact match
            exact_match = any(self._normalize_answer(pred_answer) == self._normalize_answer(ans) for ans in gt_answers)
            
            # Calculate OCR relevance (if answer contains OCR tokens)
            has_ocr_relevance = False
            if ocr_tokens:
                normalized_pred = self._normalize_answer(pred_answer)
                for token in ocr_tokens:
                    normalized_token = self._normalize_answer(token)
                    if normalized_token and normalized_token in normalized_pred:
                        has_ocr_relevance = True
                        break
            
            # Update counters
            total += 1
            scores.append(vqa_score)
            exact_match_counts += int(exact_match)
            ocr_relevance_counts += int(has_ocr_relevance)
            
            # Count how many OCR tokens were used
            if ocr_tokens:
                ocr_counts.append(len(ocr_tokens))
            
            # Update question type statistics
            question_types[question_type]['total'] += 1
            question_types[question_type]['score'] += vqa_score
        
        # Calculate overall metrics
        mean_score = mean(scores) if scores else 0
        exact_match_accuracy = exact_match_counts / total if total > 0 else 0
        ocr_relevance = ocr_relevance_counts / total if total > 0 else 0
        avg_ocr_tokens = mean(ocr_counts) if ocr_counts else 0
        
        # Calculate per-type scores
        per_type_score = {}
        for qtype, stats in question_types.items():
            per_type_score[qtype] = stats['score'] / stats['total'] if stats['total'] > 0 else 0
        
        # Results
        results = {
            'mean_score': mean_score,
            'exact_match_accuracy': exact_match_accuracy,
            'ocr_relevance': ocr_relevance,
            'avg_ocr_tokens': avg_ocr_tokens,
            'question_types': per_type_score,
            'total_questions': total
        }
        
        # Save results to file
        self._save_results(results, "textvqa_evaluation.json")
        self._plot_textvqa_metrics(results)
        
        return results
        
    def _plot_textvqa_metrics(self, metrics: Dict[str, Any]) -> None:
        """Create plots for TextVQA metrics."""
        # Plot overall metrics
        overall_metrics = {
            'mean_score': metrics.get('mean_score', 0),
            'exact_match_accuracy': metrics.get('exact_match_accuracy', 0),
            'ocr_relevance': metrics.get('ocr_relevance', 0),
        }
        
        self._plot_metrics(overall_metrics, "TextVQA Overall Metrics")
        
        # Plot score by question type
        question_types = metrics.get('question_types', {})
        if question_types:
            # Sort by score (descending)
            sorted_types = sorted(question_types.items(), key=lambda x: x[1], reverse=True)
            types = [t[0] for t in sorted_types]
            scores = [t[1] for t in sorted_types]
            
            # Select top N types if there are too many
            max_types = 15
            if len(types) > max_types:
                types = types[:max_types-1] + ['other']
                scores = scores[:max_types-1] + [mean(scores[max_types-1:])]
            
            plt.figure(figsize=(12, 8))
            bars = plt.bar(types, scores)
            
            # Add data labels
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                         f'{height:.3f}', ha='center', va='bottom')
            
            plt.title('TextVQA Score by Question Type')
            plt.xlabel('Question Type')
            plt.ylabel('Score')
            plt.ylim(0, 1.0)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            # Save the plot
            plot_path = os.path.join(self.output_dir, "textvqa_score_by_type.png")
            plt.savefig(plot_path)
            plt.close()


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate predictions on various datasets')
    
    parser.add_argument('--dataset', required=True, type=str, choices=['coco', 'refcoco', 'gqa', 'vqa', 'textvqa'],
                        help='Dataset type to evaluate')
    parser.add_argument('--predictions', required=True, type=str,
                        help='Path to predictions file (JSON format)')
    parser.add_argument('--ground-truth', required=True, type=str,
                        help='Path to ground truth annotations')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save evaluation results')
    
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()
    
    try:
        # Initialize evaluator based on dataset type
        evaluator = Evaluator(
            dataset_name=args.dataset,
            predictions_path=args.predictions,
            ground_truth_path=args.ground_truth,
            output_dir=args.output_dir
        )
        
        # Run evaluation
        metrics = evaluator.evaluate()
        
        # Print summary of results
        print("\nEvaluation Results Summary:")
        print("-" * 50)
        
        for key, value in metrics.items():
            if isinstance(value, dict):
                print(f"{key}:")
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, float):
                        print(f"  {subkey}: {subvalue:.4f}")
                    else:
                        print(f"  {subkey}: {subvalue}")
            elif isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()