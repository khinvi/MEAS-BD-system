import logging
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, MultiHeadAttention, LayerNormalization, Dropout
from tensorflow.keras.optimizers import Adam

from .base_expert import BaseExpert

logger = logging.getLogger(__name__)

class NavigationSequenceExpert(BaseExpert):
    """
    Expert model that analyzes navigation sequences and browsing patterns.
    
    This expert focuses on the sequence and pattern of page visits and
    interactions, looking for patterns that distinguish human browsing
    from automated bot behavior.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Get configuration parameters
        self.embedding_dim = config.get("embedding_dim", 128)
        self.num_heads = config.get("num_heads", 4)
        self.num_layers = config.get("num_layers", 2)
        self.dropout = config.get("dropout", 0.1)
        self.learning_rate = config.get("learning_rate", 0.0005)
        
        # Set input dimensions
        self.max_sequence_length = 50  # Maximum number of navigation events to analyze
        
        # Navigation action types
        self.action_types = [
            "pageview", "click", "form_submit", "search", "add_to_cart", 
            "checkout", "login", "register", "back", "forward"
        ]
        
        # Page categories
        self.page_categories = [
            "home", "product_list", "product_detail", "cart", "checkout",
            "account", "login", "register", "search_results", "other"
        ]
        
        if self.enabled:
            self.model = self.build_model()
    
    def extract_features(self, session_data: Dict[str, Any]) -> np.ndarray:
        """
        Extract navigation sequence features from the session data.
        
        Features include:
        - Sequence of page visits
        - Time spent on each page
        - Navigation patterns (back/forward/direct)
        - Form interactions
        - Page category transitions
        
        Args:
            session_data: Dictionary containing session data
            
        Returns:
            Numpy array of extracted navigation features
        """
        try:
            # Extract navigation events
            events = session_data.get("events", [])
            navigation_events = self._extract_navigation_events(events)
            
            # Sort events by timestamp
            navigation_events.sort(key=lambda e: e.get("timestamp", 0))
            
            # Take the most recent max_sequence_length events
            navigation_events = navigation_events[-self.max_sequence_length:]
            
            # Initialize feature array
            # Each event has:
            # - One-hot encoded action type (len(self.action_types))
            # - One-hot encoded page category (len(self.page_categories))
            # - URL embedding (simplified as URL hash feature)
            # - Time spent on page
            # - Is direct navigation (vs back/forward)
            feature_dim = len(self.action_types) + len(self.page_categories) + 3
            features = np.zeros((self.max_sequence_length, feature_dim))
            
            # Populate features
            for i, event in enumerate(navigation_events):
                if i >= self.max_sequence_length:
                    break
                
                # Action type (one-hot)
                action_type = event.get("type", "other")
                if action_type in self.action_types:
                    action_idx = self.action_types.index(action_type)
                    features[i, action_idx] = 1.0
                
                # Page category (one-hot)
                page_category = event.get("pageCategory", "other")
                if page_category in self.page_categories:
                    category_idx = self.page_categories.index(page_category)
                    features[i, len(self.action_types) + category_idx] = 1.0
                
                # URL hash (normalized)
                url = event.get("url", "")
                url_feature = hash(url) % 10000 / 10000  # Normalize to 0-1
                features[i, len(self.action_types) + len(self.page_categories)] = url_feature
                
                # Time on page (normalized)
                time_on_page = 0
                if i < len(navigation_events) - 1:
                    time_on_page = (navigation_events[i+1].get("timestamp", 0) - 
                                    event.get("timestamp", 0))
                features[i, len(self.action_types) + len(self.page_categories) + 1] = min(time_on_page / 300, 1.0)  # Normalize to 0-5 minutes
                
                # Navigation type
                nav_type = event.get("navigationType", "direct")
                is_direct = 1.0 if nav_type == "direct" else 0.0
                features[i, len(self.action_types) + len(self.page_categories) + 2] = is_direct
            
            # Add batch dimension for model input
            return features.reshape(1, self.max_sequence_length, feature_dim)
            
        except Exception as e:
            logger.error(f"Error extracting navigation features: {str(e)}", exc_info=True)
            feature_dim = len(self.action_types) + len(self.page_categories) + 3
            return np.zeros((1, self.max_sequence_length, feature_dim))
    
    def _extract_navigation_events(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract navigation-related events from all events"""
        navigation_types = [
            "pageview", "click", "form_submit", "search", "add_to_cart", 
            "checkout", "login", "register", "back", "forward"
        ]
        
        return [event for event in events if event.get("type") in navigation_types]
    
    def build_model(self) -> tf.keras.Model:
        """
        Build and compile the Transformer model for navigation sequence analysis.
        
        Returns:
            Compiled Keras Transformer model
        """
        # Input shape
        feature_dim = len(self.action_types) + len(self.page_categories) + 3
        inputs = Input(shape=(self.max_sequence_length, feature_dim))
        
        # First dense layer to project to embedding dimension
        x = Dense(self.embedding_dim, activation='relu')(inputs)
        
        # Transformer layers
        for _ in range(self.num_layers):
            # Multi-head attention
            attention_output = MultiHeadAttention(
                num_heads=self.num_heads, key_dim=self.embedding_dim // self.num_heads
            )(x, x)
            
            # Add & Norm
            x = LayerNormalization()(attention_output + x)
            
            # Feed-forward network
            ffn_output = Dense(self.embedding_dim * 2, activation='relu')(x)
            ffn_output = Dense(self.embedding_dim)(ffn_output)
            
            # Add & Norm
            x = LayerNormalization()(ffn_output + x)
            
            # Dropout
            x = Dropout(self.dropout)(x)
        
        # Global average pooling
        x = tf.reduce_mean(x, axis=1)
        
        # Output layers
        x = Dense(64, activation='relu')(x)
        x = Dropout(self.dropout)(x)
        outputs = Dense(1, activation='sigmoid')(x)
        
        # Create and compile model
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info(f"Built Transformer model with {self.num_layers} layers and {self.num_heads} heads")
        return model
    
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Train the Transformer model on the provided data.
        
        Args:
            X: Training features of shape (n_samples, max_sequence_length, feature_dim)
            y: Training labels (0 for human, 1 for bot)
            
        Returns:
            Dictionary containing training metrics
        """
        logger.info(f"Training {self.name} on {X.shape[0]} samples")
        
        # Ensure correct input shape
        if len(X.shape) != 3:
            raise ValueError(f"Expected 3D input (samples, sequence_length, features), got {X.shape}")
        
        # Train model
        history = self.model.fit(
            X, y,
            epochs=self.config.get("epochs", 30),
            batch_size=self.config.get("batch_size", 64),
            validation_split=0.2,
            verbose=2
        )
        
        self.is_trained = True
        
        # Return training metrics
        return {
            "accuracy": float(history.history['accuracy'][-1]),
            "loss": float(history.history['loss'][-1]),
            "val_accuracy": float(history.history['val_accuracy'][-1]),
            "val_loss": float(history.history['val_loss'][-1])
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make binary predictions using the trained model.
        
        Args:
            X: Features to predict on
            
        Returns:
            Array of binary predictions (0 for human, 1 for bot)
        """
        if not self.is_trained:
            logger.warning(f"{self.name} is not trained yet, returning default prediction")
            return np.zeros(X.shape[0])
        
        probabilities = self.model.predict(X)
        return (probabilities > 0.5).astype(int).flatten()
    
    def _predict_probability_implementation(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the probability of the session being a bot.
        
        Args:
            X: Features to predict on
            
        Returns:
            Array of probabilities (0-1)
        """
        return self.model.predict(X).flatten()
    
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
        explanations = []
        events = session_data.get("events", [])
        
        # Analyze navigation patterns
        
        # Check for direct targeting of purchase pages
        page_views = [e for e in events if e.get("type") == "pageview"]
        if len(page_views) >= 2:
            # Check if user went directly from homepage to checkout
            first_page = page_views[0].get("pageCategory", "")
            checkout_pages = [e for e in page_views if e.get("pageCategory") in ["cart", "checkout"]]
            
            if first_page == "home" and len(checkout_pages) > 0:
                first_checkout_idx = page_views.index(checkout_pages[0])
                if first_checkout_idx <= 2:  # Very quick navigation to checkout
                    explanations.append("Suspiciously direct navigation to checkout pages")
        
        # Check for unusual speed in browsing
        if len(page_views) >= 3:
            timestamps = [e.get("timestamp", 0) for e in page_views]
            time_diffs = np.diff(timestamps)
            if np.mean(time_diffs) < 2 and len(page_views) > 5:  # Less than 2 seconds per page on average
                explanations.append("Pages browsed too quickly for human reading")
        
        # Check for lack of natural browsing patterns
        if len(page_views) >= 5:
            # Humans typically view multiple product pages before purchase
            product_pages = [e for e in page_views if e.get("pageCategory") == "product_detail"]
            if len(product_pages) <= 1 and any(e.get("pageCategory") == "checkout" for e in page_views):
                explanations.append("Unusual purchase behavior - minimal product browsing")
        
        # Check for repetitive navigation patterns
        if len(page_views) >= 4:
            page_categories = [e.get("pageCategory", "other") for e in page_views]
            
            # Check for repeated identical sequences
            for seq_len in range(2, min(5, len(page_categories) // 2 + 1)):
                for i in range(len(page_categories) - 2*seq_len + 1):
                    seq1 = page_categories[i:i+seq_len]
                    seq2 = page_categories[i+seq_len:i+2*seq_len]
                    if seq1 == seq2:
                        explanations.append(f"Detected repetitive navigation pattern")
                        break
        
        # If no specific patterns detected but probability is high
        if not explanations and probability > 0.7:
            explanations.append("Navigation pattern resembles automated behavior")
        elif not explanations:
            explanations.append("Navigation pattern consistent with human browsing")
        
        return explanations
    
    def features_to_dict(self, features: np.ndarray) -> Dict[str, float]:
        """
        Convert feature array to dictionary for easier interpretation.
        
        Args:
            features: Extracted features
            
        Returns:
            Dictionary mapping feature names to values
        """
        # Extract summary statistics rather than all sequence features
        features = features.reshape(self.max_sequence_length, -1)
        
        feature_dict = {}
        
        # Count action types
        action_counts = {}
        for i, action in enumerate(self.action_types):
            count = np.sum(features[:, i])
            if count > 0:
                action_counts[action] = float(count)
        
        # Count page categories
        category_counts = {}
        for i, category in enumerate(self.page_categories):
            count = np.sum(features[:, len(self.action_types) + i])
            if count > 0:
                category_counts[category] = float(count)
        
        # Time on page statistics
        time_on_page = features[:, len(self.action_types) + len(self.page_categories) + 1]
        time_on_page = time_on_page[time_on_page > 0]  # Only consider non-zero values
        
        if len(time_on_page) > 0:
            feature_dict["avg_time_on_page"] = float(np.mean(time_on_page))
            feature_dict["min_time_on_page"] = float(np.min(time_on_page))
            feature_dict["max_time_on_page"] = float(np.max(time_on_page))
        
        # Navigation type statistics
        direct_nav = features[:, len(self.action_types) + len(self.page_categories) + 2]
        feature_dict["direct_navigation_ratio"] = float(np.mean(direct_nav))
        
        return {
            "action_counts": action_counts,
            "category_counts": category_counts,
            **feature_dict
        }
    
    def save(self, path: str) -> None:
        """
        Save the expert model to disk.
        
        Args:
            path: Path to save the model
        """
        if self.model is not None:
            self.model.save(path)
            logger.info(f"Saved {self.name} model to {path}")
    
    def load(self, path: str) -> None:
        """
        Load the expert model from disk.
        
        Args:
            path: Path to load the model from
        """
        try:
            self.model = tf.keras.models.load_model(path)
            self.is_trained = True
            logger.info(f"Loaded {self.name} model from {path}")
        except Exception as e:
            logger.error(f"Error loading model from {path}: {str(e)}", exc_info=True)