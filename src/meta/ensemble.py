import logging
from typing import Dict, Any, List, Tuple, Optional
import os
import json

import numpy as np
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
import joblib

from ..experts import BaseExpert, EXPERT_CLASSES

logger = logging.getLogger(__name__)

class ExpertEnsemble:
    """
    Meta-learning ensemble that combines predictions from multiple expert models.
    
    This class manages the collection of expert models and combines their outputs
    using a stacking meta-learner to produce a final bot detection decision.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the expert ensemble with configuration parameters.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config
        self.meta_config = config.get("meta_learning", {})
        self.experts_config = config.get("experts", {})
        
        self.ensemble_method = self.meta_config.get("ensemble_method", "stacking")
        self.weighting_strategy = self.meta_config.get("weighting_strategy", "dynamic")
        self.confidence_threshold = self.meta_config.get("confidence_threshold", 0.75)
        
        # Initialize experts
        self.experts: Dict[str, BaseExpert] = {}
        self._initialize_experts()
        
        # Initialize meta-learner
        self.meta_learner = None
        self.is_trained = False
        
        logger.info(f"Initialized ExpertEnsemble with {len(self.experts)} experts")
    
    def _initialize_experts(self):
        """Initialize all expert models from configuration"""
        for expert_name, expert_config in self.experts_config.items():
            if not expert_config.get("enabled", True):
                logger.info(f"Skipping disabled expert: {expert_name}")
                continue
                
            expert_class = EXPERT_CLASSES.get(expert_name)
            if not expert_class:
                logger.warning(f"Unknown expert type: {expert_name}")
                continue
                
            try:
                logger.info(f"Initializing expert: {expert_name}")
                expert = expert_class(expert_config)
                self.experts[expert_name] = expert
            except Exception as e:
                logger.error(f"Error initializing expert {expert_name}: {str(e)}", exc_info=True)
    
    def build_meta_learner(self):
        """
        Build the meta-learner model based on the ensemble method.
        
        Returns:
            Meta-learner model
        """
        if self.ensemble_method == "stacking":
            # Create a stacking ensemble using all experts as base estimators
            estimators = []
            
            for name, expert in self.experts.items():
                # Create a wrapper for the expert to make it compatible with sklearn
                estimator = ExpertEstimatorWrapper(expert)
                estimators.append((name, estimator))
            
            # Use logistic regression as the final estimator
            meta_learner = StackingClassifier(
                estimators=estimators,
                final_estimator=LogisticRegression(C=1.0, class_weight='balanced'),
                cv=5,
                stack_method='predict_proba',
                n_jobs=-1
            )
            
            logger.info(f"Built stacking meta-learner with {len(estimators)} experts")
            return meta_learner
        
        elif self.ensemble_method == "weighted_average":
            # Use a simple weighted average ensemble
            # Initialize with equal weights
            weights = {name: 1.0 / len(self.experts) for name in self.experts.keys()}
            meta_learner = WeightedAverageEnsemble(weights=weights)
            
            logger.info(f"Built weighted average meta-learner with {len(weights)} experts")
            return meta_learner
        
        else:
            logger.warning(f"Unknown ensemble method: {self.ensemble_method}, falling back to weighted average")
            weights = {name: 1.0 / len(self.experts) for name in self.experts.keys()}
            return WeightedAverageEnsemble(weights=weights)
    
    def train(self, X_dict: Dict[str, np.ndarray], y: np.ndarray) -> Dict[str, Any]:
        """
        Train the ensemble model on the provided data.
        
        Args:
            X_dict: Dictionary mapping expert names to their respective feature arrays
            y: Training labels (0 for human, 1 for bot)
            
        Returns:
            Dictionary containing training metrics
        """
        logger.info(f"Training expert ensemble on {len(y)} samples")
        
        # Train individual experts first
        expert_metrics = {}
        expert_predictions = {}
        
        for name, expert in self.experts.items():
            if name not in X_dict:
                logger.warning(f"No training data for expert {name}, skipping")
                continue
                
            try:
                X_expert = X_dict[name]
                metrics = expert.train(X_expert, y)
                expert_metrics[name] = metrics
                
                # Get predictions for meta-learner training
                expert_predictions[name] = expert.predict_probability(X_expert).reshape(-1, 1)
            except Exception as e:
                logger.error(f"Error training expert {name}: {str(e)}", exc_info=True)
        
        # Create meta-learner if needed
        if self.meta_learner is None:
            self.meta_learner = self.build_meta_learner()
        
        # Train meta-learner
        if self.ensemble_method == "stacking":
            # Stacking is trained directly on features
            X_stacked = np.hstack([X_dict[name] for name in self.experts.keys() if name in X_dict])
            self.meta_learner.fit(X_stacked, y)
        else:
            # Weighted average is trained on expert predictions
            X_meta = np.hstack([pred for pred in expert_predictions.values()])
            self.meta_learner.fit(X_meta, y)
        
        self.is_trained = True
        
        # Calculate overall metrics
        if self.ensemble_method == "stacking":
            X_stacked = np.hstack([X_dict[name] for name in self.experts.keys() if name in X_dict])
            y_pred = self.meta_learner.predict(X_stacked)
        else:
            X_meta = np.hstack([pred for pred in expert_predictions.values()])
            y_pred = self.meta_learner.predict(X_meta)
            
        accuracy = np.mean(y_pred == y)
        
        return {
            "accuracy": float(accuracy),
            "expert_metrics": expert_metrics
        }
    
    def predict(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a session and predict if it's a bot or human.
        
        Args:
            session_data: Dictionary containing session data
            
        Returns:
            Dictionary containing:
                - probability: float, probability of being a bot
                - is_bot: bool, True if the session is classified as a bot
                - confidence: float, confidence in the prediction
                - expert_results: dict of individual expert results
                - explanation: list of reasons for the prediction
        """
        # Get predictions from each expert
        expert_results = {}
        expert_probabilities = {}
        expert_confidences = {}
        expert_explanations = {}
        
        for name, expert in self.experts.items():
            try:
                result = expert.analyze_session(session_data)
                expert_results[name] = result
                expert_probabilities[name] = result.get("probability", 0.5)
                expert_confidences[name] = result.get("confidence", 0.0)
                expert_explanations[name] = result.get("explanation", [])
            except Exception as e:
                logger.error(f"Error getting prediction from expert {name}: {str(e)}", exc_info=True)
                expert_results[name] = {
                    "probability": 0.5,
                    "confidence": 0.0,
                    "explanation": [f"Error: {str(e)}"]
                }
                expert_probabilities[name] = 0.5
                expert_confidences[name] = 0.0
                expert_explanations[name] = [f"Error: {str(e)}"]
        
        # Combine expert predictions
        if self.is_trained and self.meta_learner:
            if self.ensemble_method == "stacking":
                # For stacking, we would need to extract features again
                # In a real system, we'd cache these values
                # For simplicity, using the weighted average approach here
                weights = self._calculate_weights(expert_confidences)
                probability = sum(prob * weights[name] for name, prob in expert_probabilities.items())
            else:
                # For weighted average, use the meta-learner directly
                X_meta = np.array([prob for prob in expert_probabilities.values()]).reshape(1, -1)
                probability = self.meta_learner.predict_proba(X_meta)[0, 1]
        else:
            # If not trained, use simple average
            probability = sum(expert_probabilities.values()) / len(expert_probabilities)
        
        # Calculate overall confidence
        if self.weighting_strategy == "dynamic":
            weights = self._calculate_weights(expert_confidences)
            confidence = sum(conf * weights[name] for name, conf in expert_confidences.items())
        else:
            confidence = sum(expert_confidences.values()) / len(expert_confidences)
        
        # Determine final classification
        is_bot = probability > 0.5
        
        # Generate overall explanation
        explanation = self._generate_explanation(expert_explanations, expert_confidences, is_bot)
        
        return {
            "probability": float(probability),
            "is_bot": bool(is_bot),
            "confidence": float(confidence),
            "expert_results": expert_results,
            "explanation": explanation
        }
    
    def _calculate_weights(self, confidences: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate weights for each expert based on confidence scores.
        
        Args:
            confidences: Dictionary mapping expert names to confidence scores
            
        Returns:
            Dictionary mapping expert names to weights
        """
        # Filter out zero confidence scores
        valid_confidences = {name: conf for name, conf in confidences.items() if conf > 0}
        
        if not valid_confidences:
            # If all confidences are zero, use equal weights
            return {name: 1.0 / len(confidences) for name in confidences.keys()}
        
        # Normalize confidence scores to get weights
        total_confidence = sum(valid_confidences.values())
        weights = {name: conf / total_confidence for name, conf in valid_confidences.items()}
        
        # Add zero weights for experts with zero confidence
        for name in confidences:
            if name not in weights:
                weights[name] = 0.0
        
        return weights
    
    def _generate_explanation(self, expert_explanations: Dict[str, List[str]], 
                             expert_confidences: Dict[str, float],
                             is_bot: bool) -> List[str]:
        """
        Generate an overall explanation based on expert explanations and confidences.
        
        Args:
            expert_explanations: Dictionary mapping expert names to lists of explanations
            expert_confidences: Dictionary mapping expert names to confidence scores
            is_bot: Boolean indicating if the session is classified as a bot
            
        Returns:
            List of explanation strings
        """
        explanations = []
        
        # Add summary statement
        if is_bot:
            explanations.append("Session identified as likely bot traffic.")
        else:
            explanations.append("Session identified as likely human traffic.")
        
        # Get top expert explanations based on confidence
        expert_items = list(expert_confidences.items())
        expert_items.sort(key=lambda x: x[1], reverse=True)
        
        # Add explanations from top 3 most confident experts
        for name, _ in expert_items[:3]:
            expert_name = name.replace("_expert", "").replace("_", " ").title()
            expert_explanation = expert_explanations.get(name, [])
            
            if expert_explanation:
                top_reason = expert_explanation[0]
                explanations.append(f"{expert_name}: {top_reason}")
        
        return explanations
    
    def save(self, path: str) -> None:
        """
        Save the ensemble model and all experts to disk.
        
        Args:
            path: Directory path to save the models
        """
        try:
            os.makedirs(path, exist_ok=True)
            
            # Save each expert
            for name, expert in self.experts.items():
                expert_path = os.path.join(path, f"{name}.model")
                expert.save(expert_path)
            
            # Save meta-learner
            if self.meta_learner:
                meta_path = os.path.join(path, "meta_learner.model")
                if self.ensemble_method == "stacking":
                    joblib.dump(self.meta_learner, meta_path)
                else:
                    # For weighted average, just save the weights
                    with open(meta_path, 'w') as f:
                        json.dump(self.meta_learner.weights, f)
            
            logger.info(f"Saved ensemble model to {path}")
        except Exception as e:
            logger.error(f"Error saving ensemble model: {str(e)}", exc_info=True)
    
    def load(self, path: str) -> None:
        """
        Load the ensemble model and all experts from disk.
        
        Args:
            path: Directory path to load the models from
        """
        try:
            # Load each expert
            for name, expert in self.experts.items():
                expert_path = os.path.join(path, f"{name}.model")
                if os.path.exists(expert_path):
                    expert.load(expert_path)
                else:
                    logger.warning(f"Expert model not found at {expert_path}")
            
            # Load meta-learner
            meta_path = os.path.join(path, "meta_learner.model")
            if os.path.exists(meta_path):
                if self.ensemble_method == "stacking":
                    self.meta_learner = joblib.load(meta_path)
                else:
                    # For weighted average, load the weights
                    with open(meta_path, 'r') as f:
                        weights = json.load(f)
                    self.meta_learner = WeightedAverageEnsemble(weights=weights)
                
                self.is_trained = True
                logger.info(f"Loaded ensemble model from {path}")
            else:
                logger.warning(f"Meta-learner model not found at {meta_path}")
        except Exception as e:
            logger.error(f"Error loading ensemble model: {str(e)}", exc_info=True)


class ExpertEstimatorWrapper:
    """
    Wrapper class to make expert models compatible with scikit-learn's API.
    """
    
    def __init__(self, expert: BaseExpert):
        """
        Initialize the wrapper with an expert model.
        
        Args:
            expert: The expert model to wrap
        """
        self.expert = expert
    
    def fit(self, X, y):
        """
        Train the expert model.
        
        Args:
            X: Training features
            y: Training labels
            
        Returns:
            self
        """
        self.expert.train(X, y)
        return self
    
    def predict(self, X):
        """
        Make predictions using the expert model.
        
        Args:
            X: Features to predict on
            
        Returns:
            Array of predictions
        """
        return self.expert.predict(X)
    
    def predict_proba(self, X):
        """
        Predict probabilities using the expert model.
        
        Args:
            X: Features to predict on
            
        Returns:
            Array of probabilities with shape (n_samples, 2)
        """
        proba = self.expert.predict_probability(X)
        # Convert to (n_samples, 2) shape for compatibility with sklearn
        return np.column_stack((1 - proba, proba))


class WeightedAverageEnsemble:
    """
    Simple weighted average ensemble for combining expert predictions.
    """
    
    def __init__(self, weights: Dict[str, float]):
        """
        Initialize the ensemble with weights for each expert.
        
        Args:
            weights: Dictionary mapping expert names to weights
        """
        self.weights = weights
    
    def fit(self, X, y):
        """
        Train the ensemble (adjust weights based on expert performance).
        
        Args:
            X: Expert predictions with shape (n_samples, n_experts)
            y: True labels
            
        Returns:
            self
        """
        # Placeholder for more sophisticated weight learning
        # For now, just normalizing the weights
        total_weight = sum(self.weights.values())
        if total_weight > 0:
            self.weights = {name: weight / total_weight for name, weight in self.weights.items()}
        return self
    
    def predict(self, X):
        """
        Make predictions using weighted average of expert predictions.
        
        Args:
            X: Expert predictions with shape (n_samples, n_experts)
            
        Returns:
            Array of binary predictions
        """
        proba = self.predict_proba(X)
        return (proba[:, 1] > 0.5).astype(int)
    
    def predict_proba(self, X):
        """
        Predict probabilities using weighted average of expert probabilities.
        
        Args:
            X: Expert probabilities with shape (n_samples, n_experts)
            
        Returns:
            Array of probabilities with shape (n_samples, 2)
        """
        # Compute weighted average
        weight_values = np.array(list(self.weights.values()))
        weighted_proba = np.dot(X, weight_values) / np.sum(weight_values)
        
        # Convert to (n_samples, 2) shape for compatibility with sklearn
        return np.column_stack((1 - weighted_proba, weighted_proba))