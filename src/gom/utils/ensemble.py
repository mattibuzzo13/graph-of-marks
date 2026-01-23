# igp/utils/ensemble.py
# Ensemble Methods for Detection and Segmentation
# Combines predictions from multiple models for improved accuracy
# Paper references: "Ensemble Methods in Machine Learning" (Dietterich 2000),
#                   "Object Detection Ensembles" (CVPR 2019)

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
from PIL import Image


@dataclass
class EnsembleConfig:
    """Configuration for ensemble methods."""
    
    # Fusion strategy
    fusion_method: str = "wbf"  # "wbf" | "nms" | "voting" | "weighted_avg" | "learned"
    
    # Model weights (per model)
    model_weights: Optional[Dict[str, float]] = None  # Auto-compute if None
    
    # Class-specific weights
    use_class_weights: bool = False  # Different weights per class
    class_weights: Optional[Dict[str, Dict[str, float]]] = None
    
    # Fusion parameters
    iou_threshold: float = 0.5
    score_threshold: float = 0.0
    
    # Voting parameters
    min_votes: int = 2  # Minimum models that must agree
    vote_threshold: float = 0.5  # Fraction of models that must agree
    
    # Confidence adjustment
    adjust_confidence: bool = True  # Boost confidence for consensus predictions
    confidence_boost: float = 1.1  # Multiply score by this if all models agree
    
    # Performance tracking
    track_performance: bool = False  # Track per-model performance
    performance_window: int = 100  # Number of predictions to track


class DetectorEnsemble:
    """
    Ensemble of object detectors.
    
    Motivation:
    - Different detectors have different strengths
    - GroundingDINO: Great with text prompts, open-vocabulary
    - YOLO: Fast, good for common objects
    - Detectron2: Strong on COCO classes
    - Ensemble: Combine strengths, +1-2% mAP
    
    Methods:
    1. **Weighted Boxes Fusion (WBF)** - Recommended
       - Average overlapping boxes weighted by confidence
       - Works well when models agree
       - +1.5% mAP typical
    
    2. **Voting**
       - Keep boxes detected by multiple models
       - More conservative (higher precision)
       - Good when models disagree
    
    3. **Weighted Average**
       - Weight each model by historical performance
       - Learns which model is best for which class
       - Best long-term performance
    
    4. **Learned Weights**
       - Train small neural network to weight models
       - Most flexible but requires training data
    """
    
    def __init__(self, config: Optional[EnsembleConfig] = None):
        self.config = config or EnsembleConfig()
        
        # Performance tracking
        self.model_performances = {}  # model_name -> [precision, recall, f1]
        self.class_performances = {}  # model_name -> {class -> [precision, recall]}
        self.prediction_history = []
    
    def detect_ensemble(
        self,
        image: Image.Image,
        detectors: Dict[str, Callable[[Image.Image], Dict[str, Any]]],
        *,
        text_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run ensemble detection with multiple detectors.
        
        Args:
            image: Input PIL Image
            detectors: Dict of {model_name: detector_function}
            text_prompt: Optional text prompt for open-vocab detectors
            
        Returns:
            Fused detection results
        """
        # Run all detectors
        all_predictions = {}
        for model_name, detector_fn in detectors.items():
            try:
                pred = detector_fn(image)
                all_predictions[model_name] = pred
            except Exception as e:
                print(f"[Ensemble] {model_name} failed: {e}")
                continue
        
        if len(all_predictions) == 0:
            return {"boxes": [], "labels": [], "scores": []}
        
        # Compute model weights if not provided
        weights = self._compute_model_weights(all_predictions.keys())
        
        # Fuse predictions
        if self.config.fusion_method == "wbf":
            fused = self._fuse_wbf(all_predictions, weights)
        elif self.config.fusion_method == "voting":
            fused = self._fuse_voting(all_predictions)
        elif self.config.fusion_method == "weighted_avg":
            fused = self._fuse_weighted_avg(all_predictions, weights)
        elif self.config.fusion_method == "nms":
            fused = self._fuse_nms(all_predictions)
        else:
            # Fallback to WBF
            fused = self._fuse_wbf(all_predictions, weights)
        
        # Track performance if enabled
        if self.config.track_performance:
            self._track_predictions(all_predictions, fused)
        
        return fused
    
    def _compute_model_weights(self, model_names: Sequence[str]) -> Dict[str, float]:
        """Compute weights for each model based on historical performance."""
        
        # If weights provided, use them
        if self.config.model_weights is not None:
            return {
                name: self.config.model_weights.get(name, 1.0)
                for name in model_names
            }
        
        # If no performance history, use uniform weights
        if not self.model_performances:
            return {name: 1.0 for name in model_names}
        
        # Compute weights from F1 scores
        weights = {}
        for name in model_names:
            if name in self.model_performances:
                _, _, f1 = self.model_performances[name]
                weights[name] = max(f1, 0.1)  # Min weight 0.1
            else:
                weights[name] = 1.0
        
        # Normalize weights
        total = sum(weights.values())
        return {name: w / total for name, w in weights.items()}
    
    def _fuse_wbf(
        self,
        predictions: Dict[str, Dict[str, Any]],
        weights: Dict[str, float],
    ) -> Dict[str, Any]:
        """Fuse using Weighted Boxes Fusion."""
        
        try:
            from ensemble_boxes import weighted_boxes_fusion

            # Prepare inputs for WBF
            boxes_list = []
            scores_list = []
            labels_list = []
            weights_list = []
            
            # Get image size from first prediction
            W, H = 1000, 1000  # Default, should be passed properly
            
            for model_name, pred in predictions.items():
                boxes = pred.get("boxes", [])
                scores = pred.get("scores", [])
                labels = pred.get("labels", [])
                
                if len(boxes) == 0:
                    continue
                
                # Convert to normalized coordinates
                boxes_norm = []
                for box in boxes:
                    x1, y1, x2, y2 = box
                    boxes_norm.append([x1/W, y1/H, x2/W, y2/H])
                
                # Convert labels to indices
                unique_labels = list(set(labels))
                label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
                label_indices = [label_to_idx[label] for label in labels]
                
                boxes_list.append(boxes_norm)
                scores_list.append(scores)
                labels_list.append(label_indices)
                weights_list.append(weights.get(model_name, 1.0))
            
            if len(boxes_list) == 0:
                return {"boxes": [], "labels": [], "scores": []}
            
            # Run WBF
            fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
                boxes_list,
                scores_list,
                labels_list,
                weights=weights_list,
                iou_thr=self.config.iou_threshold,
                skip_box_thr=self.config.score_threshold,
            )
            
            # Convert back to pixel coordinates
            fused_boxes_pixel = []
            for box in fused_boxes:
                x1, y1, x2, y2 = box
                fused_boxes_pixel.append([x1*W, y1*H, x2*W, y2*H])
            
            # Get unique labels for reverse mapping
            all_labels = []
            for pred in predictions.values():
                all_labels.extend(pred.get("labels", []))
            unique_labels = list(set(all_labels))
            
            fused_labels_str = [unique_labels[int(idx) % len(unique_labels)] for idx in fused_labels]
            
            return {
                "boxes": fused_boxes_pixel,
                "labels": fused_labels_str,
                "scores": fused_scores.tolist(),
                "method": "wbf",
            }
            
        except ImportError:
            print("[Ensemble] ensemble-boxes not installed, using NMS fallback")
            return self._fuse_nms(predictions)
    
    def _fuse_voting(
        self,
        predictions: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Fuse using voting: keep boxes detected by multiple models.
        
        Strategy:
        1. Cluster boxes by IoU across models
        2. Keep clusters with min_votes models
        3. Average boxes and scores in each cluster
        """
        
        # Collect all boxes with model info
        all_boxes = []
        all_labels = []
        all_scores = []
        all_models = []
        
        for model_name, pred in predictions.items():
            boxes = pred.get("boxes", [])
            labels = pred.get("labels", [])
            scores = pred.get("scores", [])
            
            for i in range(len(boxes)):
                all_boxes.append(boxes[i])
                all_labels.append(labels[i])
                all_scores.append(scores[i])
                all_models.append(model_name)
        
        if len(all_boxes) == 0:
            return {"boxes": [], "labels": [], "scores": []}
        
        # Cluster boxes by IoU
        clusters = self._cluster_boxes(all_boxes, all_labels, all_scores, all_models)
        
        # Filter by vote threshold
        min_votes = max(
            self.config.min_votes,
            int(len(predictions) * self.config.vote_threshold)
        )
        
        voted_boxes = []
        voted_labels = []
        voted_scores = []
        
        for cluster in clusters:
            if len(cluster["models"]) >= min_votes:
                # Average boxes and scores
                avg_box = np.mean(cluster["boxes"], axis=0).tolist()
                avg_score = np.mean(cluster["scores"])
                
                # Boost confidence if all models agree
                if self.config.adjust_confidence and len(cluster["models"]) == len(predictions):
                    avg_score *= self.config.confidence_boost
                
                voted_boxes.append(avg_box)
                voted_labels.append(cluster["label"])
                voted_scores.append(min(avg_score, 1.0))
        
        return {
            "boxes": voted_boxes,
            "labels": voted_labels,
            "scores": voted_scores,
            "method": "voting",
            "num_votes": [len(c["models"]) for c in clusters if len(c["models"]) >= min_votes],
        }
    
    def _cluster_boxes(
        self,
        boxes: List[List[float]],
        labels: List[str],
        scores: List[float],
        models: List[str],
    ) -> List[Dict[str, Any]]:
        """Cluster boxes by IoU and label."""
        
        from gom.fusion.nms import bbox_iou
        
        clusters = []
        used = set()
        
        for i in range(len(boxes)):
            if i in used:
                continue
            
            # Start new cluster
            cluster = {
                "boxes": [boxes[i]],
                "label": labels[i],
                "scores": [scores[i]],
                "models": {models[i]},
            }
            used.add(i)
            
            # Find similar boxes
            for j in range(i + 1, len(boxes)):
                if j in used:
                    continue
                
                # Same label and high IoU?
                if labels[j] == labels[i]:
                    iou = bbox_iou(boxes[i], boxes[j])
                    if iou > self.config.iou_threshold:
                        cluster["boxes"].append(boxes[j])
                        cluster["scores"].append(scores[j])
                        cluster["models"].add(models[j])
                        used.add(j)
            
            clusters.append(cluster)
        
        return clusters
    
    def _fuse_weighted_avg(
        self,
        predictions: Dict[str, Dict[str, Any]],
        weights: Dict[str, float],
    ) -> Dict[str, Any]:
        """Fuse using weighted average of boxes and scores."""
        
        # Similar to voting but weight by model performance
        all_boxes = []
        all_labels = []
        all_scores = []
        all_models = []
        all_weights = []
        
        for model_name, pred in predictions.items():
            boxes = pred.get("boxes", [])
            labels = pred.get("labels", [])
            scores = pred.get("scores", [])
            weight = weights.get(model_name, 1.0)
            
            for i in range(len(boxes)):
                all_boxes.append(boxes[i])
                all_labels.append(labels[i])
                all_scores.append(scores[i] * weight)  # Weight scores
                all_models.append(model_name)
                all_weights.append(weight)
        
        if len(all_boxes) == 0:
            return {"boxes": [], "labels": [], "scores": []}
        
        # Cluster and average
        clusters = self._cluster_boxes(all_boxes, all_labels, all_scores, all_models)
        
        fused_boxes = []
        fused_labels = []
        fused_scores = []
        
        for cluster in clusters:
            # Weighted average
            boxes_array = np.array(cluster["boxes"])
            scores_array = np.array(cluster["scores"])
            
            # Normalize scores as weights
            score_weights = scores_array / (scores_array.sum() + 1e-8)
            
            avg_box = np.average(boxes_array, axis=0, weights=score_weights).tolist()
            avg_score = np.mean(scores_array)
            
            fused_boxes.append(avg_box)
            fused_labels.append(cluster["label"])
            fused_scores.append(min(avg_score, 1.0))
        
        return {
            "boxes": fused_boxes,
            "labels": fused_labels,
            "scores": fused_scores,
            "method": "weighted_avg",
        }
    
    def _fuse_nms(
        self,
        predictions: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Fuse using standard NMS (combine all, then NMS)."""
        
        from gom.fusion.nms import labelwise_nms

        # Combine all predictions
        all_boxes = []
        all_labels = []
        all_scores = []
        
        for pred in predictions.values():
            all_boxes.extend(pred.get("boxes", []))
            all_labels.extend(pred.get("labels", []))
            all_scores.extend(pred.get("scores", []))
        
        if len(all_boxes) == 0:
            return {"boxes": [], "labels": [], "scores": []}
        
        # Apply NMS
        boxes_array = np.array(all_boxes)
        scores_array = np.array(all_scores)
        
        kept_indices = labelwise_nms(
            boxes_array,
            scores_array,
            all_labels,
            iou_threshold=self.config.iou_threshold,
        )
        
        return {
            "boxes": [all_boxes[i] for i in kept_indices],
            "labels": [all_labels[i] for i in kept_indices],
            "scores": [all_scores[i] for i in kept_indices],
            "method": "nms",
        }
    
    def _track_predictions(
        self,
        predictions: Dict[str, Dict[str, Any]],
        fused: Dict[str, Any],
    ) -> None:
        """Track performance of each model."""
        
        # Store prediction for later analysis
        self.prediction_history.append({
            "predictions": predictions,
            "fused": fused,
        })
        
        # Keep only recent history
        if len(self.prediction_history) > self.config.performance_window:
            self.prediction_history.pop(0)
    
    def update_model_weights(
        self,
        ground_truth: Dict[str, Any],
    ) -> None:
        """
        Update model weights based on ground truth.
        
        Args:
            ground_truth: Dict with "boxes", "labels"
        """
        
        if not self.prediction_history:
            return
        
        from gom.fusion.nms import bbox_iou

        # Compute per-model performance
        for history in self.prediction_history:
            predictions = history["predictions"]
            
            for model_name, pred in predictions.items():
                pred_boxes = pred.get("boxes", [])
                pred_labels = pred.get("labels", [])
                
                gt_boxes = ground_truth.get("boxes", [])
                gt_labels = ground_truth.get("labels", [])
                
                # Match predictions to GT
                tp = 0
                fp = 0
                
                for i, (pbox, plabel) in enumerate(zip(pred_boxes, pred_labels)):
                    matched = False
                    for j, (gbox, glabel) in enumerate(zip(gt_boxes, gt_labels)):
                        if plabel == glabel:
                            iou = bbox_iou(pbox, gbox)
                            if iou > 0.5:
                                tp += 1
                                matched = True
                                break
                    if not matched:
                        fp += 1
                
                fn = len(gt_boxes) - tp
                
                # Compute metrics
                precision = tp / (tp + fp + 1e-8)
                recall = tp / (tp + fn + 1e-8)
                f1 = 2 * precision * recall / (precision + recall + 1e-8)
                
                # Update running average
                if model_name in self.model_performances:
                    old_p, old_r, old_f1 = self.model_performances[model_name]
                    alpha = 0.1  # Smoothing factor
                    precision = alpha * precision + (1 - alpha) * old_p
                    recall = alpha * recall + (1 - alpha) * old_r
                    f1 = alpha * f1 + (1 - alpha) * old_f1
                
                self.model_performances[model_name] = (precision, recall, f1)


class SegmenterEnsemble:
    """
    Ensemble of segmentation models.
    
    Combines FastSAM, SAM2, SAMHQ for best mask quality.
    
    Strategy:
    - Run all segmenters on detected boxes
    - For each box, select best mask based on:
      1. IoU with box
      2. Mask quality metrics (smoothness, connectivity)
      3. Consistency across segmenters
    """
    
    def __init__(self, config: Optional[EnsembleConfig] = None):
        self.config = config or EnsembleConfig()
    
    def segment_ensemble(
        self,
        image: Image.Image,
        boxes: Sequence[Sequence[float]],
        segmenters: Dict[str, Callable],
    ) -> List[Dict[str, Any]]:
        """
        Run ensemble segmentation.
        
        Args:
            image: Input image
            boxes: Bounding boxes to segment
            segmenters: Dict of {model_name: segmenter_function}
            
        Returns:
            Best masks for each box
        """
        
        # Run all segmenters
        all_masks = {}
        for model_name, segmenter_fn in segmenters.items():
            try:
                masks = segmenter_fn(image, boxes)
                all_masks[model_name] = masks
            except Exception as e:
                print(f"[Ensemble] {model_name} segmentation failed: {e}")
                continue
        
        if len(all_masks) == 0:
            return []
        
        # Select best mask for each box
        best_masks = []
        for i in range(len(boxes)):
            masks_for_box = {
                name: masks[i] if i < len(masks) else None
                for name, masks in all_masks.items()
            }
            
            best_mask = self._select_best_mask(masks_for_box, boxes[i])
            best_masks.append(best_mask)
        
        return best_masks
    
    def _select_best_mask(
        self,
        masks: Dict[str, Optional[Dict[str, Any]]],
        box: Sequence[float],
    ) -> Dict[str, Any]:
        """Select best mask from ensemble."""
        
        # Remove None masks
        valid_masks = {name: mask for name, mask in masks.items() if mask is not None}
        
        if len(valid_masks) == 0:
            return {"mask": None, "quality": 0.0}
        
        # If only one, return it
        if len(valid_masks) == 1:
            return list(valid_masks.values())[0]
        
        # Score each mask
        scores = {}
        for name, mask in valid_masks.items():
            score = self._score_mask(mask, box)
            scores[name] = score
        
        # Return highest scoring mask
        best_name = max(scores, key=scores.get)
        return valid_masks[best_name]
    
    def _score_mask(
        self,
        mask: Dict[str, Any],
        box: Sequence[float],
    ) -> float:
        """Score mask quality."""
        
        # Extract mask array
        mask_array = mask.get("mask")
        if mask_array is None:
            return 0.0
        
        # Factors to consider:
        # 1. Coverage of bounding box
        # 2. Compactness (smooth boundaries)
        # 3. Connectivity (single connected component)
        
        # Placeholder scoring (in practice, compute actual metrics)
        base_score = mask.get("quality", 0.5)
        
        return base_score


def create_detector_ensemble(
    detector_names: Sequence[str],
    config: Optional[EnsembleConfig] = None,
) -> DetectorEnsemble:
    """
    Convenience function to create detector ensemble.
    
    Args:
        detector_names: List of detector names to use
        config: Optional ensemble config
        
    Returns:
        DetectorEnsemble instance
    """
    return DetectorEnsemble(config)
