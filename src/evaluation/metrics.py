import logging
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, precision_recall_curve
)

logger = logging.getLogger(__name__)

class PerformanceMetrics:
    """
    Performance metrics calculator for bot detection.
    
    This class computes various performance metrics to evaluate
    model accuracy, precision, recall, and other key measures.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the performance metrics calculator.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config
        self.metrics = config.get("metrics", ["accuracy", "precision", "recall", "f1", "auc"])
        
        logger.info(f"Initialized PerformanceMetrics with metrics: {self.metrics}")
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                        y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calculate performance metrics.
        
        Args:
            y_true: Array of true labels (0 for human, 1 for bot)
            y_pred: Array of predicted labels (0 for human, 1 for bot)
            y_prob: Optional array of predicted probabilities (for ROC-AUC)
            
        Returns:
            Dictionary mapping metric names to values
        """
        metrics_dict = {}
        
        try:
            # Basic classification metrics
            if "accuracy" in self.metrics:
                metrics_dict["accuracy"] = float(accuracy_score(y_true, y_pred))
            
            if "precision" in self.metrics:
                metrics_dict["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
            
            if "recall" in self.metrics:
                metrics_dict["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
            
            if "f1" in self.metrics:
                metrics_dict["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
            
            # Confusion matrix
            if "confusion_matrix" in self.metrics:
                cm = confusion_matrix(y_true, y_pred)
                metrics_dict["true_negatives"] = float(cm[0, 0])
                metrics_dict["false_positives"] = float(cm[0, 1])
                metrics_dict["false_negatives"] = float(cm[1, 0])
                metrics_dict["true_positives"] = float(cm[1, 1])
            
            # ROC-AUC (requires probabilities)
            if "auc" in self.metrics and y_prob is not None:
                if len(np.unique(y_true)) > 1:  # AUC requires at least one positive and one negative sample
                    metrics_dict["auc"] = float(roc_auc_score(y_true, y_prob))
                else:
                    metrics_dict["auc"] = 0.5  # Default value when only one class is present
            
            # Calculate specificity (true negative rate)
            if "specificity" in self.metrics:
                cm = confusion_matrix(y_true, y_pred)
                tn, fp = float(cm[0, 0]), float(cm[0, 1])
                if tn + fp > 0:
                    metrics_dict["specificity"] = tn / (tn + fp)
                else:
                    metrics_dict["specificity"] = 0.0
            
            logger.info(f"Calculated performance metrics: {metrics_dict}")
            return metrics_dict
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}", exc_info=True)
            return {}
    
    def calculate_expert_metrics(self, y_true: np.ndarray, 
                               expert_predictions: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
        """
        Calculate performance metrics for each expert.
        
        Args:
            y_true: Array of true labels (0 for human, 1 for bot)
            expert_predictions: Dictionary mapping expert names to predicted probabilities
            
        Returns:
            Dictionary mapping expert names to metric dictionaries
        """
        expert_metrics = {}
        
        for expert_name, y_prob in expert_predictions.items():
            try:
                # Convert probabilities to binary predictions
                y_pred = (y_prob > 0.5).astype(int)
                
                # Calculate metrics
                metrics = self.calculate_metrics(y_true, y_pred, y_prob)
                expert_metrics[expert_name] = metrics
                
                logger.info(f"Calculated metrics for expert {expert_name}: {metrics}")
            except Exception as e:
                logger.error(f"Error calculating metrics for expert {expert_name}: {str(e)}", exc_info=True)
                expert_metrics[expert_name] = {}
        
        return expert_metrics
    
    def calculate_threshold_metrics(self, y_true: np.ndarray, y_prob: np.ndarray, 
                                  thresholds: Optional[List[float]] = None) -> Dict[str, List[Tuple[float, Dict[str, float]]]]:
        """
        Calculate metrics at different decision thresholds.
        
        Args:
            y_true: Array of true labels (0 for human, 1 for bot)
            y_prob: Array of predicted probabilities
            thresholds: Optional list of thresholds to evaluate (default: [0.1, 0.2, ..., 0.9])
            
        Returns:
            Dictionary mapping metric sets to lists of (threshold, metrics) tuples
        """
        if thresholds is None:
            thresholds = np.arange(0.1, 1.0, 0.1)
        
        threshold_metrics = {
            "all": []
        }
        
        try:
            for threshold in thresholds:
                # Apply threshold
                y_pred = (y_prob >= threshold).astype(int)
                
                # Calculate metrics
                metrics = self.calculate_metrics(y_true, y_pred, y_prob)
                threshold_metrics["all"].append((float(threshold), metrics))
                
                logger.debug(f"Calculated metrics at threshold {threshold}: {metrics}")
            
            # Calculate precision-recall curve
            precision, recall, pr_thresholds = precision_recall_curve(y_true, y_prob)
            threshold_metrics["pr_curve"] = [
                (float(t), {"precision": float(p), "recall": float(r)})
                for p, r, t in zip(precision, recall, pr_thresholds)
            ]
            
            logger.info(f"Calculated metrics for {len(thresholds)} thresholds")
            return threshold_metrics
            
        except Exception as e:
            logger.error(f"Error calculating threshold metrics: {str(e)}", exc_info=True)
            return {"all": []}
    
    def calculate_sliced_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                               y_prob: np.ndarray, slices: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
        """
        Calculate metrics on different data slices.
        
        Args:
            y_true: Array of true labels (0 for human, 1 for bot)
            y_pred: Array of predicted labels
            y_prob: Array of predicted probabilities
            slices: Dictionary mapping slice names to boolean masks
            
        Returns:
            Dictionary mapping slice names to metric dictionaries
        """
        sliced_metrics = {}
        
        try:
            # Calculate overall metrics
            overall_metrics = self.calculate_metrics(y_true, y_pred, y_prob)
            sliced_metrics["overall"] = overall_metrics
            
            # Calculate metrics for each slice
            for slice_name, mask in slices.items():
                if not any(mask):  # Skip empty slices
                    logger.warning(f"Slice {slice_name} is empty, skipping")
                    continue
                
                slice_metrics = self.calculate_metrics(
                    y_true[mask], y_pred[mask], y_prob[mask] if y_prob is not None else None
                )
                sliced_metrics[slice_name] = slice_metrics
                
                logger.info(f"Calculated metrics for slice {slice_name}: {slice_metrics}")
            
            return sliced_metrics
            
        except Exception as e:
            logger.error(f"Error calculating sliced metrics: {str(e)}", exc_info=True)
            return {"overall": {}}
    
    def calculate_confidence_calibration(self, y_true: np.ndarray, y_prob: np.ndarray, 
                                      n_bins: int = 10) -> List[Dict[str, float]]:
        """
        Calculate confidence calibration metrics.
        
        Args:
            y_true: Array of true labels (0 for human, 1 for bot)
            y_prob: Array of predicted probabilities
            n_bins: Number of bins for calibration analysis
            
        Returns:
            List of dictionaries containing bin_start, bin_end, accuracy, confidence, and samples
        """
        try:
            # Create bins
            bin_edges = np.linspace(0, 1, n_bins + 1)
            bin_indices = np.digitize(y_prob, bin_edges) - 1
            bin_indices = np.clip(bin_indices, 0, n_bins - 1)
            
            calibration_data = []
            
            for i in range(n_bins):
                mask = (bin_indices == i)
                if not any(mask):
                    continue
                
                bin_start = bin_edges[i]
                bin_end = bin_edges[i + 1]
                
                # Calculate metrics for this bin
                bin_y_true = y_true[mask]
                bin_y_prob = y_prob[mask]
                bin_y_pred = (bin_y_prob >= 0.5).astype(int)
                
                # Accuracy in this bin
                accuracy = accuracy_score(bin_y_true, bin_y_pred)
                
                # Average confidence in this bin
                confidence = np.mean(bin_y_prob)
                
                # Number of samples in this bin
                n_samples = np.sum(mask)
                
                calibration_data.append({
                    "bin_start": float(bin_start),
                    "bin_end": float(bin_end),
                    "accuracy": float(accuracy),
                    "confidence": float(confidence),
                    "samples": int(n_samples)
                })
            
            logger.info(f"Calculated calibration data for {len(calibration_data)} bins")
            return calibration_data
            
        except Exception as e:
            logger.error(f"Error calculating confidence calibration: {str(e)}", exc_info=True)
            return []