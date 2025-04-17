import logging
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)

class DynamicWeighting:
    """
    Dynamic weighting system for adjusting expert influence based on performance.
    
    This class implements various strategies for dynamically adjusting the 
    weights of experts in the ensemble based on their recent performance.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the dynamic weighting system.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config
        self.strategy = config.get("strategy", "performance_based")
        self.learning_rate = config.get("learning_rate", 0.1)
        self.window_size = config.get("window_size", 100)  # Number of samples to consider for recent performance
        
        # Initialize expert weights
        self.weights: Dict[str, float] = {}
        self.performance_history: Dict[str, List[float]] = {}
        self.confidence_history: Dict[str, List[float]] = {}
        
        logger.info(f"Initialized DynamicWeighting with strategy: {self.strategy}")
    
    def initialize_weights(self, expert_names: List[str]):
        """
        Initialize weights for all experts.
        
        Args:
            expert_names: List of expert names
        """
        # Start with equal weights
        weight = 1.0 / len(expert_names) if expert_names else 0.0
        self.weights = {name: weight for name in expert_names}
        
        # Initialize performance history
        self.performance_history = {name: [] for name in expert_names}
        self.confidence_history = {name: [] for name in expert_names}
        
        logger.info(f"Initialized weights for {len(expert_names)} experts")
    
    def update_weights(self, expert_performances: Dict[str, float], 
                      expert_confidences: Dict[str, float]):
        """
        Update weights based on recent expert performance.
        
        Args:
            expert_performances: Dictionary mapping expert names to performance scores
            expert_confidences: Dictionary mapping expert names to confidence scores
            
        Returns:
            Updated weights dictionary
        """
        # Update performance history
        for name, performance in expert_performances.items():
            if name in self.performance_history:
                self.performance_history[name].append(performance)
                # Keep only the most recent window_size samples
                if len(self.performance_history[name]) > self.window_size:
                    self.performance_history[name] = self.performance_history[name][-self.window_size:]
        
        # Update confidence history
        for name, confidence in expert_confidences.items():
            if name in self.confidence_history:
                self.confidence_history[name].append(confidence)
                # Keep only the most recent window_size samples
                if len(self.confidence_history[name]) > self.window_size:
                    self.confidence_history[name] = self.confidence_history[name][-self.window_size:]
        
        # Apply weighting strategy
        if self.strategy == "performance_based":
            self._update_performance_based()
        elif self.strategy == "confidence_weighted":
            self._update_confidence_weighted(expert_confidences)
        elif self.strategy == "adaptive":
            self._update_adaptive(expert_performances, expert_confidences)
        else:
            logger.warning(f"Unknown weighting strategy: {self.strategy}, using performance_based")
            self._update_performance_based()
        
        return self.weights
    
    def _update_performance_based(self):
        """Update weights based solely on historical performance"""
        # Calculate average performance for each expert
        avg_performances = {}
        for name, history in self.performance_history.items():
            if history:
                avg_performances[name] = np.mean(history)
            else:
                avg_performances[name] = 0.5  # Default if no history
        
        # Convert to weights (higher performance = higher weight)
        total_performance = sum(avg_performances.values())
        if total_performance > 0:
            new_weights = {name: perf / total_performance 
                          for name, perf in avg_performances.items()}
            
            # Apply learning rate for smooth transitions
            for name in self.weights:
                old_weight = self.weights.get(name, 0.0)
                new_weight = new_weights.get(name, 0.0)
                self.weights[name] = old_weight + self.learning_rate * (new_weight - old_weight)
            
            # Normalize weights to sum to 1
            total_weight = sum(self.weights.values())
            if total_weight > 0:
                self.weights = {name: weight / total_weight for name, weight in self.weights.items()}
        
        logger.debug(f"Updated performance-based weights: {self.weights}")
    
    def _update_confidence_weighted(self, expert_confidences: Dict[str, float]):
        """Update weights based on confidence scores"""
        # Use confidence scores directly as weights
        total_confidence = sum(expert_confidences.values())
        if total_confidence > 0:
            new_weights = {name: conf / total_confidence 
                          for name, conf in expert_confidences.items()}
            
            # Apply learning rate for smooth transitions
            for name in self.weights:
                if name in new_weights:
                    old_weight = self.weights.get(name, 0.0)
                    new_weight = new_weights.get(name, 0.0)
                    self.weights[name] = old_weight + self.learning_rate * (new_weight - old_weight)
            
            # Normalize weights to sum to 1
            total_weight = sum(self.weights.values())
            if total_weight > 0:
                self.weights = {name: weight / total_weight for name, weight in self.weights.items()}
        
        logger.debug(f"Updated confidence-weighted weights: {self.weights}")
    
    def _update_adaptive(self, expert_performances: Dict[str, float], 
                        expert_confidences: Dict[str, float]):
        """Update weights using both performance and confidence"""
        # Calculate combined score (performance * confidence)
        combined_scores = {}
        for name in self.weights:
            performance = expert_performances.get(name, 0.5)
            confidence = expert_confidences.get(name, 0.0)
            combined_scores[name] = performance * confidence
        
        # Convert to weights
        total_score = sum(combined_scores.values())
        if total_score > 0:
            new_weights = {name: score / total_score 
                          for name, score in combined_scores.items()}
            
            # Apply learning rate for smooth transitions
            for name in self.weights:
                old_weight = self.weights.get(name, 0.0)
                new_weight = new_weights.get(name, 0.0)
                self.weights[name] = old_weight + self.learning_rate * (new_weight - old_weight)
            
            # Normalize weights to sum to 1
            total_weight = sum(self.weights.values())
            if total_weight > 0:
                self.weights = {name: weight / total_weight for name, weight in self.weights.items()}
        
        logger.debug(f"Updated adaptive weights: {self.weights}")
    
    def get_weights(self) -> Dict[str, float]:
        """
        Get the current expert weights.
        
        Returns:
            Dictionary mapping expert names to weights
        """
        return self.weights
    
    def compute_expert_performance(self, predictions: Dict[str, np.ndarray], 
                                  true_labels: np.ndarray) -> Dict[str, float]:
        """
        Compute performance scores for each expert.
        
        Args:
            predictions: Dictionary mapping expert names to predicted probabilities
            true_labels: Array of true labels (0 for human, 1 for bot)
            
        Returns:
            Dictionary mapping expert names to performance scores
        """
        performances = {}
        
        for name, preds in predictions.items():
            try:
                # Use AUC-ROC as performance metric
                score = roc_auc_score(true_labels, preds)
                performances[name] = score
            except Exception as e:
                logger.error(f"Error computing performance for expert {name}: {str(e)}", exc_info=True)
                performances[name] = 0.5  # Default to random guessing
        
        return performances