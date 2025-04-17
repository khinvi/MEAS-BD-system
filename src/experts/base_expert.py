from abc import ABC, abstractmethod
import logging
from typing import Dict, Any, List, Tuple, Optional

import numpy as np

logger = logging.getLogger(__name__)

class BaseExpert(ABC):
    """
    Abstract base class for all expert models in the MEAS-BD system.
    
    Each expert specializes in analyzing a specific aspect of user behavior
    to determine whether the session is likely a bot or a human.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the expert with configuration parameters.
        
        Args:
            config: Dictionary containing configuration parameters for the expert
        """
        self.config = config
        self.name = self.__class__.__name__
        self.enabled = config.get("enabled", True)
        self.model = None
        self.is_trained = False
        self.feature_importance = {}
        logger.info(f"Initializing {self.name} with config: {config}")
    
    @abstractmethod
    def extract_features(self, session_data: Dict[str, Any]) -> np.ndarray:
        """
        Extract relevant features from the session data.
        
        Args:
            session_data: Dictionary containing session data
            
        Returns:
            Numpy array of extracted features
        """
        pass
    
    @abstractmethod
    def build_model(self) -> Any:
        """
        Build and return the model for this expert.
        
        Returns:
            The built model
        """
        pass
    
    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Train the expert model on the provided data.
        
        Args:
            X: Training features
            y: Training labels (0 for human, 1 for bot)
            
        Returns:
            Dictionary containing training metrics
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X: Features to predict on
            
        Returns:
            Array of predictions (0 for human, 1 for bot)
        """
        pass
    
    def predict_probability(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the probability of the session being a bot.
        
        Args:
            X: Features to predict on
            
        Returns:
            Array of probabilities (0-1)
        """
        if not self.is_trained:
            logger.warning(f"{self.name} is not trained yet, returning default probability")
            return np.full(X.shape[0], 0.5)
        
        return self._predict_probability_implementation(X)
    
    @abstractmethod
    def _predict_probability_implementation(self, X: np.ndarray) -> np.ndarray:
        """
        Implementation of probability prediction specific to each expert.
        
        Args:
            X: Features to predict on
            
        Returns:
            Array of probabilities (0-1)
        """
        pass
    
    def analyze_session(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a session and return bot probability and explanation.
        
        Args:
            session_data: Dictionary containing session data
            
        Returns:
            Dictionary containing:
                - probability: float, probability of being a bot
                - confidence: float, confidence in the prediction
                - explanation: list of reasons for the prediction
                - features: dict of extracted features and their values
        """
        if not self.enabled:
            logger.info(f"{self.name} is disabled, skipping analysis")
            return {
                "probability": 0.5,
                "confidence": 0.0,
                "explanation": [f"{self.name} is disabled"],
                "features": {}
            }
        
        try:
            features = self.extract_features(session_data)
            
            # Reshape for single sample prediction
            if len(features.shape) == 1:
                features = features.reshape(1, -1)
                
            probability = self.predict_probability(features)[0]
            explanation = self.generate_explanation(features, probability, session_data)
            confidence = self.calculate_confidence(features, probability)
            
            feature_dict = self.features_to_dict(features)
            
            return {
                "probability": float(probability),
                "confidence": float(confidence),
                "explanation": explanation,
                "features": feature_dict
            }
        except Exception as e:
            logger.error(f"Error in {self.name} analysis: {str(e)}", exc_info=True)
            return {
                "probability": 0.5,
                "confidence": 0.0,
                "explanation": [f"Error in {self.name}: {str(e)}"],
                "features": {}
            }
    
    @abstractmethod
    def generate_explanation(self, features: np.ndarray, probability: float, 
                           session_data: Dict[str, Any]) -> List[str]:
        """
        Generate human-readable explanations for the prediction.
        
        Args:
            features: Extracted features
            probability: Predicted bot probability
            session_data: Original session data
            
        Returns:
            List of explanation strings
        """
        pass
    
    def calculate_confidence(self, features: np.ndarray, probability: float) -> float:
        """
        Calculate confidence in the prediction based on feature values and model certainty.
        
        Args:
            features: Extracted features
            probability: Predicted bot probability
            
        Returns:
            Confidence score (0-1)
        """
        # Default implementation - higher confidence when probability is further from 0.5
        return 2 * abs(probability - 0.5)
    
    def features_to_dict(self, features: np.ndarray) -> Dict[str, float]:
        """
        Convert feature array to dictionary for easier interpretation.
        
        Args:
            features: Extracted features
            
        Returns:
            Dictionary mapping feature names to values
        """
        # Default implementation - should be overridden by subclasses
        return {f"feature_{i}": float(val) for i, val in enumerate(features.flatten())}
    
    def save(self, path: str) -> None:
        """
        Save the expert model to disk.
        
        Args:
            path: Path to save the model
        """
        logger.info(f"Saving {self.name} to {path}")
        # Implementation depends on model type
        pass
    
    def load(self, path: str) -> None:
        """
        Load the expert model from disk.
        
        Args:
            path: Path to load the model from
        """
        logger.info(f"Loading {self.name} from {path}")
        # Implementation depends on model type
        pass