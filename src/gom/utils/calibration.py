# igp/utils/calibration.py
# Confidence Calibration for Detection and Segmentation
# Ensures predicted confidence scores reflect true accuracy
# Paper references: "On Calibration of Modern Neural Networks" (ICML 2017),
#                   "Calibrating Deep Neural Networks" (NeurIPS 2019)

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np


@dataclass
class CalibrationConfig:
    """Configuration for confidence calibration."""
    
    # Calibration method
    method: str = "temperature"  # "temperature" | "platt" | "isotonic" | "beta"
    
    # Temperature scaling parameters
    temperature: float = 1.5  # Initial temperature (>1 = softer, <1 = sharper)
    learn_temperature: bool = True  # Learn temperature from validation set
    
    # Platt scaling parameters (logistic regression)
    platt_max_iter: int = 100
    
    # Isotonic regression parameters
    isotonic_increasing: bool = True  # Monotonicity constraint
    
    # Beta calibration parameters
    beta_parameters: Tuple[float, float] = (1.0, 1.0)  # (a, b) for Beta distribution
    
    # Calibration data
    calibration_cache_path: Optional[str] = None  # Path to save/load calibration parameters


class ConfidenceCalibrator:
    """
    Confidence Calibration for object detection.
    
    Problem:
    - Neural networks often produce overconfident or underconfident predictions
    - A prediction with 90% confidence might only be correct 70% of the time
    - This hurts decision-making and threshold selection
    
    Solution:
    - Learn a calibration function from validation data
    - Transform raw confidence scores to calibrated probabilities
    - Calibrated scores better reflect true accuracy
    
    Methods:
    1. Temperature Scaling: Scale logits by temperature T
       - Simple, effective, single parameter
       - p_cal = softmax(logits / T)
    
    2. Platt Scaling: Logistic regression on scores
       - Learn A, B such that p_cal = sigmoid(A * logit + B)
       - More flexible than temperature
    
    3. Isotonic Regression: Non-parametric monotonic mapping
       - Learn arbitrary monotonic function
       - Most flexible, but can overfit
    
    4. Beta Calibration: Beta distribution mapping
       - Generalizes Platt scaling
       - Better for skewed distributions
    """
    
    def __init__(self, config: Optional[CalibrationConfig] = None):
        self.config = config or CalibrationConfig()
        self.is_fitted = False
        
        # Learned parameters
        self.temperature = self.config.temperature
        self.platt_A = 1.0
        self.platt_B = 0.0
        self.isotonic_f = None
        self.beta_a = self.config.beta_parameters[0]
        self.beta_b = self.config.beta_parameters[1]
    
    def fit(
        self,
        predictions: Sequence[Dict[str, Any]],
        ground_truth: Sequence[Dict[str, Any]],
    ) -> None:
        """
        Fit calibration parameters from validation data.
        
        Args:
            predictions: List of prediction dicts with "scores" and "labels"
            ground_truth: List of ground truth dicts with "labels" and "boxes"
        """
        # Collect (score, correctness) pairs
        scores = []
        correct = []
        
        for pred, gt in zip(predictions, ground_truth):
            pred_scores = pred.get("scores", [])
            pred_labels = pred.get("labels", [])
            pred_boxes = pred.get("boxes", [])
            
            gt_labels = gt.get("labels", [])
            gt_boxes = gt.get("boxes", [])
            
            # Match predictions to ground truth
            matches = self._match_predictions_to_gt(
                pred_boxes, pred_labels, gt_boxes, gt_labels
            )
            
            for i, (score, label) in enumerate(zip(pred_scores, pred_labels)):
                scores.append(score)
                correct.append(matches[i])  # 1 if matched, 0 otherwise
        
        if len(scores) == 0:
            print("[Calibration] Warning: No predictions to calibrate")
            return
        
        scores = np.array(scores)
        correct = np.array(correct)
        
        # Fit calibration model
        if self.config.method == "temperature":
            self._fit_temperature(scores, correct)
        elif self.config.method == "platt":
            self._fit_platt(scores, correct)
        elif self.config.method == "isotonic":
            self._fit_isotonic(scores, correct)
        elif self.config.method == "beta":
            self._fit_beta(scores, correct)
        else:
            print(f"[Calibration] Unknown method: {self.config.method}")
            return
        
        self.is_fitted = True
        print(f"[Calibration] Fitted {self.config.method} calibration on {len(scores)} predictions")
        
        # Save to cache if specified
        if self.config.calibration_cache_path:
            self.save(self.config.calibration_cache_path)
    
    def calibrate_scores(self, scores: Sequence[float]) -> np.ndarray:
        """
        Calibrate confidence scores.
        
        Args:
            scores: Raw confidence scores from detector
            
        Returns:
            Calibrated confidence scores
        """
        if not self.is_fitted:
            print("[Calibration] Warning: Calibrator not fitted, returning original scores")
            return np.array(scores)
        
        scores = np.array(scores)
        
        if self.config.method == "temperature":
            return self._calibrate_temperature(scores)
        elif self.config.method == "platt":
            return self._calibrate_platt(scores)
        elif self.config.method == "isotonic":
            return self._calibrate_isotonic(scores)
        elif self.config.method == "beta":
            return self._calibrate_beta(scores)
        else:
            return scores
    
    def calibrate_predictions(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calibrate confidence scores in prediction dict.
        
        Args:
            predictions: Dict with "scores", "boxes", "labels"
            
        Returns:
            Predictions with calibrated scores
        """
        calibrated = predictions.copy()
        scores = predictions.get("scores", [])
        
        if len(scores) > 0:
            calibrated_scores = self.calibrate_scores(scores)
            calibrated["scores"] = calibrated_scores.tolist()
        
        return calibrated
    
    def _match_predictions_to_gt(
        self,
        pred_boxes: Sequence[Sequence[float]],
        pred_labels: Sequence[str],
        gt_boxes: Sequence[Sequence[float]],
        gt_labels: Sequence[str],
        iou_threshold: float = 0.5,
    ) -> np.ndarray:
        """
        Match predictions to ground truth using IoU.
        
        Returns:
            Binary array indicating which predictions are correct (matched to GT)
        """
        from gom.fusion.nms import bbox_iou
        
        n_pred = len(pred_boxes)
        matches = np.zeros(n_pred, dtype=int)
        
        if len(gt_boxes) == 0:
            return matches  # No GT, all predictions are false positives
        
        pred_boxes = np.array(pred_boxes)
        gt_boxes = np.array(gt_boxes)
        
        # Compute IoU matrix
        iou_matrix = np.zeros((n_pred, len(gt_boxes)))
        for i in range(n_pred):
            for j in range(len(gt_boxes)):
                iou_matrix[i, j] = bbox_iou(pred_boxes[i], gt_boxes[j])
        
        # Greedy matching (highest IoU first)
        matched_gt = set()
        for i in range(n_pred):
            best_j = -1
            best_iou = iou_threshold
            
            for j in range(len(gt_boxes)):
                if j in matched_gt:
                    continue
                if iou_matrix[i, j] > best_iou and pred_labels[i] == gt_labels[j]:
                    best_iou = iou_matrix[i, j]
                    best_j = j
            
            if best_j >= 0:
                matches[i] = 1
                matched_gt.add(best_j)
        
        return matches
    
    # ================== Temperature Scaling ==================
    
    def _fit_temperature(self, scores: np.ndarray, correct: np.ndarray) -> None:
        """Fit temperature parameter using cross-entropy minimization."""
        if not self.config.learn_temperature:
            return
        
        # Convert scores to logits (inverse sigmoid)
        logits = np.log(scores / (1 - scores + 1e-8) + 1e-8)
        
        # Find temperature that minimizes negative log-likelihood
        from scipy.optimize import minimize_scalar
        
        def nll(T):
            calibrated_probs = 1 / (1 + np.exp(-logits / T))
            # Binary cross-entropy
            loss = -np.mean(
                correct * np.log(calibrated_probs + 1e-8) +
                (1 - correct) * np.log(1 - calibrated_probs + 1e-8)
            )
            return loss
        
        result = minimize_scalar(nll, bounds=(0.1, 10.0), method='bounded')
        self.temperature = result.x
        
        print(f"[Calibration] Learned temperature: {self.temperature:.3f}")
    
    def _calibrate_temperature(self, scores: np.ndarray) -> np.ndarray:
        """Apply temperature scaling."""
        logits = np.log(scores / (1 - scores + 1e-8) + 1e-8)
        calibrated_logits = logits / self.temperature
        calibrated_scores = 1 / (1 + np.exp(-calibrated_logits))
        return np.clip(calibrated_scores, 0, 1)
    
    # ================== Platt Scaling ==================
    
    def _fit_platt(self, scores: np.ndarray, correct: np.ndarray) -> None:
        """Fit Platt scaling using logistic regression."""
        from scipy.optimize import minimize
        
        logits = np.log(scores / (1 - scores + 1e-8) + 1e-8)
        
        def nll(params):
            A, B = params
            calibrated_probs = 1 / (1 + np.exp(-(A * logits + B)))
            loss = -np.mean(
                correct * np.log(calibrated_probs + 1e-8) +
                (1 - correct) * np.log(1 - calibrated_probs + 1e-8)
            )
            return loss
        
        result = minimize(
            nll,
            x0=[1.0, 0.0],
            method='BFGS',
            options={'maxiter': self.config.platt_max_iter}
        )
        
        self.platt_A, self.platt_B = result.x
        print(f"[Calibration] Learned Platt parameters: A={self.platt_A:.3f}, B={self.platt_B:.3f}")
    
    def _calibrate_platt(self, scores: np.ndarray) -> np.ndarray:
        """Apply Platt scaling."""
        logits = np.log(scores / (1 - scores + 1e-8) + 1e-8)
        calibrated_logits = self.platt_A * logits + self.platt_B
        calibrated_scores = 1 / (1 + np.exp(-calibrated_logits))
        return np.clip(calibrated_scores, 0, 1)
    
    # ================== Isotonic Regression ==================
    
    def _fit_isotonic(self, scores: np.ndarray, correct: np.ndarray) -> None:
        """Fit isotonic regression (non-parametric)."""
        from sklearn.isotonic import IsotonicRegression
        
        self.isotonic_f = IsotonicRegression(
            increasing=self.config.isotonic_increasing,
            out_of_bounds='clip'
        )
        self.isotonic_f.fit(scores, correct)
        
        print("[Calibration] Fitted isotonic regression")
    
    def _calibrate_isotonic(self, scores: np.ndarray) -> np.ndarray:
        """Apply isotonic regression."""
        if self.isotonic_f is None:
            return scores
        return self.isotonic_f.predict(scores)
    
    # ================== Beta Calibration ==================
    
    def _fit_beta(self, scores: np.ndarray, correct: np.ndarray) -> None:
        """Fit beta calibration (generalized Platt scaling)."""
        from scipy.optimize import minimize
        
        def nll(params):
            a, b, c = params
            # Beta calibration: p_cal = p^a / (p^a + (1-p)^b)
            calibrated_probs = scores**a / (scores**a + (1 - scores)**b + 1e-8)
            loss = -np.mean(
                correct * np.log(calibrated_probs + 1e-8) +
                (1 - correct) * np.log(1 - calibrated_probs + 1e-8)
            )
            return loss
        
        result = minimize(
            nll,
            x0=[1.0, 1.0, 0.0],
            method='L-BFGS-B',
            bounds=[(0.1, 10), (0.1, 10), (-5, 5)]
        )
        
        self.beta_a, self.beta_b, self.beta_c = result.x
        print(f"[Calibration] Learned Beta parameters: a={self.beta_a:.3f}, b={self.beta_b:.3f}")
    
    def _calibrate_beta(self, scores: np.ndarray) -> np.ndarray:
        """Apply beta calibration."""
        calibrated = scores**self.beta_a / (
            scores**self.beta_a + (1 - scores)**self.beta_b + 1e-8
        )
        return np.clip(calibrated, 0, 1)
    
    # ================== Save/Load ==================
    
    def save(self, path: str) -> None:
        """Save calibration parameters to file."""
        params = {
            "method": self.config.method,
            "is_fitted": self.is_fitted,
            "temperature": self.temperature,
            "platt_A": self.platt_A,
            "platt_B": self.platt_B,
            "isotonic_f": self.isotonic_f,
            "beta_a": self.beta_a,
            "beta_b": self.beta_b,
        }
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(params, f)
        
        print(f"[Calibration] Saved parameters to {path}")
    
    def load(self, path: str) -> None:
        """Load calibration parameters from file."""
        with open(path, 'rb') as f:
            params = pickle.load(f)
        
        self.is_fitted = params["is_fitted"]
        self.temperature = params["temperature"]
        self.platt_A = params["platt_A"]
        self.platt_B = params["platt_B"]
        self.isotonic_f = params["isotonic_f"]
        self.beta_a = params["beta_a"]
        self.beta_b = params["beta_b"]
        
        print(f"[Calibration] Loaded {params['method']} calibration from {path}")


def compute_calibration_metrics(
    predictions: Sequence[Dict[str, Any]],
    ground_truth: Sequence[Dict[str, Any]],
    num_bins: int = 10,
) -> Dict[str, float]:
    """
    Compute calibration metrics (ECE, MCE, etc.).
    
    Args:
        predictions: List of predictions with "scores", "labels", "boxes"
        ground_truth: List of ground truth with "labels", "boxes"
        num_bins: Number of bins for ECE calculation
        
    Returns:
        Dict with "ece" (Expected Calibration Error), "mce" (Maximum Calibration Error)
    """
    calibrator = ConfidenceCalibrator()
    
    # Collect (score, correctness) pairs
    scores = []
    correct = []
    
    for pred, gt in zip(predictions, ground_truth):
        pred_scores = pred.get("scores", [])
        pred_labels = pred.get("labels", [])
        pred_boxes = pred.get("boxes", [])
        
        gt_labels = gt.get("labels", [])
        gt_boxes = gt.get("boxes", [])
        
        matches = calibrator._match_predictions_to_gt(
            pred_boxes, pred_labels, gt_boxes, gt_labels
        )
        
        scores.extend(pred_scores)
        correct.extend(matches)
    
    scores = np.array(scores)
    correct = np.array(correct)
    
    # Compute ECE and MCE
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    mce = 0.0
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (scores > bin_lower) & (scores <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(correct[in_bin])
            avg_confidence_in_bin = np.mean(scores[in_bin])
            
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            mce = max(mce, np.abs(avg_confidence_in_bin - accuracy_in_bin))
    
    return {
        "ece": float(ece),
        "mce": float(mce),
        "num_predictions": len(scores),
    }
