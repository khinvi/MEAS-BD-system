import logging
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    Preprocessor for bot detection data.
    
    This class handles data normalization, feature selection, and missing value
    imputation for different expert models.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the data preprocessor.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config
        self.normalization = config.get("normalization", "standard")
        self.handle_missing = config.get("handle_missing", "impute")
        self.feature_selection = config.get("feature_selection", "auto")
        
        # Initialize preprocessing components
        self.scalers: Dict[str, Any] = {}
        self.imputers: Dict[str, Any] = {}
        self.feature_selectors: Dict[str, Any] = {}
        
        logger.info(f"Initialized DataPreprocessor with normalization: {self.normalization}")
    
    def fit_transform(self, features: Dict[str, np.ndarray], 
                     labels: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Fit preprocessors to the data and transform it.
        
        Args:
            features: Dictionary mapping expert names to feature arrays
            labels: Array of labels (0 for human, 1 for bot)
            
        Returns:
            Dictionary of preprocessed features
        """
        preprocessed = {}
        
        for expert_name, X in features.items():
            try:
                # Handle missing values
                X_imputed = self._handle_missing_values(X, expert_name)
                
                # Normalize features
                X_normalized = self._normalize_features(X_imputed, expert_name)
                
                # Select features
                X_selected = self._select_features(X_normalized, labels, expert_name)
                
                preprocessed[expert_name] = X_selected
                
                logger.info(f"Preprocessed features for {expert_name}: {X.shape} -> {X_selected.shape}")
            except Exception as e:
                logger.error(f"Error preprocessing features for {expert_name}: {str(e)}", exc_info=True)
                preprocessed[expert_name] = X  # Use original features as fallback
        
        return preprocessed
    
    def transform(self, features: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Transform new data using fitted preprocessors.
        
        Args:
            features: Dictionary mapping expert names to feature arrays
            
        Returns:
            Dictionary of preprocessed features
        """
        preprocessed = {}
        
        for expert_name, X in features.items():
            try:
                # Check if preprocessors are fitted
                if expert_name not in self.imputers or expert_name not in self.scalers:
                    logger.warning(f"Preprocessors not fitted for {expert_name}, using original features")
                    preprocessed[expert_name] = X
                    continue
                
                # Handle missing values
                X_imputed = self.imputers[expert_name].transform(X)
                
                # Normalize features
                X_normalized = self.scalers[expert_name].transform(X_imputed)
                
                # Select features
                if expert_name in self.feature_selectors and self.feature_selectors[expert_name] is not None:
                    X_selected = self.feature_selectors[expert_name].transform(X_normalized)
                else:
                    X_selected = X_normalized
                
                preprocessed[expert_name] = X_selected
                
                logger.info(f"Transformed features for {expert_name}: {X.shape} -> {X_selected.shape}")
            except Exception as e:
                logger.error(f"Error transforming features for {expert_name}: {str(e)}", exc_info=True)
                preprocessed[expert_name] = X  # Use original features as fallback
        
        return preprocessed
    
    def _handle_missing_values(self, X: np.ndarray, expert_name: str) -> np.ndarray:
        """
        Handle missing values in the data.
        
        Args:
            X: Feature array
            expert_name: Name of the expert
            
        Returns:
            Array with missing values handled
        """
        if self.handle_missing == "drop":
            # Not applicable for numpy arrays, would need pandas
            logger.warning("Drop strategy not applicable for numpy arrays, using imputation")
            self.handle_missing = "impute"
        
        if self.handle_missing == "impute":
            # Create and fit imputer if not already done
            if expert_name not in self.imputers:
                self.imputers[expert_name] = SimpleImputer(strategy='mean')
                return self.imputers[expert_name].fit_transform(X)
            else:
                return self.imputers[expert_name].transform(X)
        
        # Default: do nothing
        return X
    
    def _normalize_features(self, X: np.ndarray, expert_name: str) -> np.ndarray:
        """
        Normalize features.
        
        Args:
            X: Feature array
            expert_name: Name of the expert
            
        Returns:
            Normalized feature array
        """
        if self.normalization == "standard":
            # Create and fit scaler if not already done
            if expert_name not in self.scalers:
                self.scalers[expert_name] = StandardScaler()
                return self.scalers[expert_name].fit_transform(X)
            else:
                return self.scalers[expert_name].transform(X)
        
        elif self.normalization == "minmax":
            # Create and fit scaler if not already done
            if expert_name not in self.scalers:
                self.scalers[expert_name] = MinMaxScaler()
                return self.scalers[expert_name].fit_transform(X)
            else:
                return self.scalers[expert_name].transform(X)
        
        # Default: do nothing
        return X
    
    def _select_features(self, X: np.ndarray, y: Optional[np.ndarray], 
                        expert_name: str) -> np.ndarray:
        """
        Select most informative features.
        
        Args:
            X: Feature array
            y: Labels (optional, required for fitting)
            expert_name: Name of the expert
            
        Returns:
            Selected feature array
        """
        if self.feature_selection == "auto" and y is not None:
            # Determine number of features to select
            k = max(3, X.shape[1] // 2)  # At least 3 features, at most half
            
            # Create and fit selector if not already done
            if expert_name not in self.feature_selectors:
                self.feature_selectors[expert_name] = SelectKBest(f_classif, k=k)
                return self.feature_selectors[expert_name].fit_transform(X, y)
            else:
                return self.feature_selectors[expert_name].transform(X)
        
        # Default: do nothing
        return X
    
    def get_feature_importances(self) -> Dict[str, Dict[int, float]]:
        """
        Get feature importances from feature selection.
        
        Returns:
            Dictionary mapping expert names to dictionaries of feature indices and scores
        """
        importances = {}
        
        for expert_name, selector in self.feature_selectors.items():
            if selector is not None and hasattr(selector, 'scores_'):
                # Get indices of selected features
                mask = selector.get_support()
                indices = np.where(mask)[0]
                
                # Get scores for selected features
                scores = selector.scores_[mask]
                
                # Create dictionary mapping feature indices to scores
                feature_scores = {int(idx): float(score) for idx, score in zip(indices, scores)}
                importances[expert_name] = feature_scores
        
        return importances