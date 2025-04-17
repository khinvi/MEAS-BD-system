import logging
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import pairwise_distances

logger = logging.getLogger(__name__)

class ActiveLearning:
    """
    Active learning system for selecting the most informative samples for labeling.
    
    This class implements various strategies for selecting unlabeled samples that,
    when labeled, would provide the most information gain for model training.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the active learning system.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config
        self.strategy = config.get("strategy", "uncertainty_sampling")
        self.batch_size = config.get("batch_size", 10)
        self.uncertainty_threshold = config.get("uncertainty_threshold", 0.3)
        
        logger.info(f"Initialized ActiveLearning with strategy: {self.strategy}")
    
    def select_samples(self, X: np.ndarray, predictions: np.ndarray, 
                      labeled_indices: List[int] = None) -> List[int]:
        """
        Select samples for labeling based on the active learning strategy.
        
        Args:
            X: Feature matrix for all samples
            predictions: Predicted probabilities from current model
            labeled_indices: Indices of already labeled samples
            
        Returns:
            List of indices for samples to be labeled
        """
        if labeled_indices is None:
            labeled_indices = []
            
        # Find unlabeled indices
        all_indices = set(range(len(X)))
        unlabeled_indices = list(all_indices - set(labeled_indices))
        
        if not unlabeled_indices:
            logger.warning("No unlabeled samples available for selection")
            return []
        
        # Apply selection strategy
        if self.strategy == "uncertainty_sampling":
            selected = self._uncertainty_sampling(predictions, unlabeled_indices)
        elif self.strategy == "diversity_sampling":
            selected = self._diversity_sampling(X, unlabeled_indices)
        elif self.strategy == "hybrid":
            selected = self._hybrid_sampling(X, predictions, unlabeled_indices)
        else:
            logger.warning(f"Unknown sampling strategy: {self.strategy}, using uncertainty_sampling")
            selected = self._uncertainty_sampling(predictions, unlabeled_indices)
        
        logger.info(f"Selected {len(selected)} samples for labeling using {self.strategy} strategy")
        return selected
    
    def _uncertainty_sampling(self, predictions: np.ndarray, 
                             unlabeled_indices: List[int]) -> List[int]:
        """
        Select samples with highest prediction uncertainty.
        
        Args:
            predictions: Predicted probabilities from current model
            unlabeled_indices: Indices of unlabeled samples
            
        Returns:
            List of selected indices
        """
        # Calculate uncertainty (distance from decision boundary at 0.5)
        uncertainties = np.abs(predictions - 0.5)
        unlabeled_uncertainties = uncertainties[unlabeled_indices]
        
        # Sort by uncertainty (lower = more uncertain)
        sorted_indices = np.argsort(unlabeled_uncertainties)
        selected_local_indices = sorted_indices[:self.batch_size]
        
        # Map back to original indices
        selected_indices = [unlabeled_indices[i] for i in selected_local_indices]
        
        return selected_indices
    
    def _diversity_sampling(self, X: np.ndarray, unlabeled_indices: List[int]) -> List[int]:
        """
        Select diverse samples using clustering.
        
        Args:
            X: Feature matrix for all samples
            unlabeled_indices: Indices of unlabeled samples
            
        Returns:
            List of selected indices
        """
        # Extract features for unlabeled samples
        X_unlabeled = X[unlabeled_indices]
        
        # Determine number of clusters
        n_clusters = min(self.batch_size, len(X_unlabeled))
        
        if n_clusters < 2:
            return unlabeled_indices[:1]
        
        # Cluster the unlabeled samples
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(X_unlabeled)
        
        # Select the samples closest to each cluster center
        selected_local_indices = []
        
        for i in range(n_clusters):
            # Get indices of samples in this cluster
            cluster_samples = np.where(cluster_labels == i)[0]
            
            if len(cluster_samples) == 0:
                continue
                
            # Get cluster center
            center = kmeans.cluster_centers_[i]
            
            # Calculate distances to center
            distances = np.linalg.norm(X_unlabeled[cluster_samples] - center, axis=1)
            
            # Select sample closest to center
            closest_index = cluster_samples[np.argmin(distances)]
            selected_local_indices.append(closest_index)
        
        # Map back to original indices
        selected_indices = [unlabeled_indices[i] for i in selected_local_indices]
        
        return selected_indices
    
    def _hybrid_sampling(self, X: np.ndarray, predictions: np.ndarray, 
                        unlabeled_indices: List[int]) -> List[int]:
        """
        Hybrid approach combining uncertainty and diversity.
        
        Args:
            X: Feature matrix for all samples
            predictions: Predicted probabilities from current model
            unlabeled_indices: Indices of unlabeled samples
            
        Returns:
            List of selected indices
        """
        # First, select candidates with high uncertainty
        uncertainties = np.abs(predictions - 0.5)
        unlabeled_uncertainties = uncertainties[unlabeled_indices]
        
        # Get indices of samples with uncertainty above threshold
        uncertain_local_indices = np.where(unlabeled_uncertainties < self.uncertainty_threshold)[0]
        
        if len(uncertain_local_indices) == 0:
            # If no samples meet uncertainty threshold, fall back to pure uncertainty sampling
            logger.info("No samples meet uncertainty threshold, falling back to uncertainty sampling")
            return self._uncertainty_sampling(predictions, unlabeled_indices)
        
        # Get original indices of uncertain samples
        uncertain_indices = [unlabeled_indices[i] for i in uncertain_local_indices]
        
        # Then, select diverse samples from the uncertain ones
        X_uncertain = X[uncertain_indices]
        
        # Determine number of clusters
        n_clusters = min(self.batch_size, len(X_uncertain))
        
        if n_clusters < 2:
            return uncertain_indices[:self.batch_size]
        
        # Cluster the uncertain samples
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(X_uncertain)
        
        # Select the samples closest to each cluster center
        selected_local_indices = []
        
        for i in range(n_clusters):
            # Get indices of samples in this cluster
            cluster_samples = np.where(cluster_labels == i)[0]
            
            if len(cluster_samples) == 0:
                continue
                
            # Get cluster center
            center = kmeans.cluster_centers_[i]
            
            # Calculate distances to center
            distances = np.linalg.norm(X_uncertain[cluster_samples] - center, axis=1)
            
            # Select sample closest to center
            closest_index = cluster_samples[np.argmin(distances)]
            selected_local_indices.append(closest_index)
        
        # Map back to original indices
        selected_indices = [uncertain_indices[i] for i in selected_local_indices]
        
        return selected_indices
    
    def estimate_value_of_sample(self, X: np.ndarray, predictions: np.ndarray, 
                               model: Any, sample_index: int) -> float:
        """
        Estimate the value of labeling a particular sample.
        
        Args:
            X: Feature matrix for all samples
            predictions: Predicted probabilities from current model
            model: Current model
            sample_index: Index of the sample to evaluate
            
        Returns:
            Estimated value score (higher = more valuable)
        """
        # Extract sample features
        sample = X[sample_index].reshape(1, -1)
        
        # Calculate uncertainty
        uncertainty = np.abs(predictions[sample_index] - 0.5)
        uncertainty_value = 1.0 - 2 * uncertainty  # Highest at decision boundary
        
        # Calculate representativeness (average distance to other samples)
        distances = pairwise_distances(sample, X)
        representativeness = np.mean(distances)
        representativeness_value = 1.0 - representativeness  # Higher for more representative samples
        
        # Calculate expected model improvement
        # This is a simplified estimate using prediction variance as a proxy
        # In a real implementation, would use more sophisticated methods such as expected error reduction
        expected_improvement = uncertainty_value
        
        # Combine factors with weights
        # These weights could be tuned based on specific problem characteristics
        value = (0.6 * uncertainty_value + 
                0.2 * representativeness_value + 
                0.2 * expected_improvement)
        
        return value
    
    def query_by_committee(self, X: np.ndarray, models: List[Any], 
                         unlabeled_indices: List[int]) -> List[int]:
        """
        Select samples with highest disagreement among ensemble models.
        
        Args:
            X: Feature matrix for all samples
            models: List of trained models
            unlabeled_indices: Indices of unlabeled samples
            
        Returns:
            List of selected indices
        """
        if not models:
            logger.warning("No models provided for query by committee")
            return []
        
        # Get predictions from each model
        all_predictions = []
        for model in models:
            try:
                if hasattr(model, 'predict_proba'):
                    probs = model.predict_proba(X[unlabeled_indices])
                    # Extract probability of positive class
                    preds = probs[:, 1] if probs.shape[1] > 1 else probs
                else:
                    preds = model.predict(X[unlabeled_indices])
                all_predictions.append(preds)
            except Exception as e:
                logger.error(f"Error getting predictions from model: {str(e)}", exc_info=True)
        
        if not all_predictions:
            logger.warning("No valid predictions obtained from models")
            return []
        
        # Convert to array
        all_predictions = np.array(all_predictions)
        
        # Calculate disagreement (standard deviation of predictions)
        disagreement = np.std(all_predictions, axis=0)
        
        # Select samples with highest disagreement
        sorted_indices = np.argsort(disagreement)[::-1]  # Sort in descending order
        selected_local_indices = sorted_indices[:self.batch_size]
        
        # Map back to original indices
        selected_indices = [unlabeled_indices[i] for i in selected_local_indices]
        
        return selected_indices