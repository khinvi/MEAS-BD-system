import logging
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

from .base_expert import BaseExpert

logger = logging.getLogger(__name__)

class TemporalPatternExpert(BaseExpert):
    """
    Expert model that analyzes temporal patterns in user sessions.
    
    This expert focuses on the timing and sequence of user actions, looking for 
    patterns that distinguish human behavior from automated bots.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.hidden_size = config.get("hidden_size", 64)
        self.num_layers = config.get("num_layers", 2)
        self.dropout = config.get("dropout", 0.2)
        self.learning_rate = config.get("learning_rate", 0.001)
        self.sequence_length = config.get("sequence_length", 20)
        self.feature_names = [
            "time_between_pageviews",
            "time_on_page",
            "action_frequency",
            "session_duration",
            "time_of_day",
            "day_of_week",
            "consistency_score"
        ]
        
        if self.enabled:
            self.model = self.build_model()
    
    def extract_features(self, session_data: Dict[str, Any]) -> np.ndarray:
        """
        Extract temporal features from the session data.
        
        Features include:
        - Time between pageviews
        - Time spent on each page
        - Frequency of actions
        - Overall session duration
        - Time of day patterns
        - Day of week
        - Consistency in timing patterns
        
        Args:
            session_data: Dictionary containing session data
            
        Returns:
            Numpy array of extracted temporal features
        """
        try:
            # Extract timestamps from events
            events = session_data.get("events", [])
            timestamps = np.array([event.get("timestamp", 0) for event in events], dtype=np.float32)
            
            if len(timestamps) < 2:
                # Not enough events to analyze
                logger.warning("Not enough events to analyze temporal patterns")
                return np.zeros((1, self.sequence_length, len(self.feature_names)))
            
            # Sort timestamps
            timestamps.sort()
            
            # Calculate time differences between events
            time_diffs = np.diff(timestamps)
            
            # Pad or truncate to sequence_length
            if len(time_diffs) >= self.sequence_length:
                time_diffs = time_diffs[:self.sequence_length]
            else:
                time_diffs = np.pad(time_diffs, (0, self.sequence_length - len(time_diffs)))
            
            # Calculate time on page for each page view
            page_views = [event for event in events if event.get("type") == "pageview"]
            page_times = []
            
            for i in range(len(page_views) - 1):
                page_times.append(page_views[i+1].get("timestamp", 0) - page_views[i].get("timestamp", 0))
            
            # Pad or truncate page times
            if len(page_times) >= self.sequence_length:
                page_times = page_times[:self.sequence_length]
            else:
                page_times = np.pad(page_times, (0, self.sequence_length - len(page_times)))
            
            # Calculate action frequency (events per second)
            session_duration = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 1
            action_frequency = len(events) / max(session_duration, 1)
            
            # Extract time of day and day of week
            time_of_day = np.array([self._extract_time_of_day(event.get("timestamp", 0)) 
                                   for event in events[:self.sequence_length]])
            day_of_week = np.array([self._extract_day_of_week(event.get("timestamp", 0)) 
                                   for event in events[:self.sequence_length]])
            
            # Pad if needed
            if len(time_of_day) < self.sequence_length:
                time_of_day = np.pad(time_of_day, (0, self.sequence_length - len(time_of_day)))
                day_of_week = np.pad(day_of_week, (0, self.sequence_length - len(day_of_week)))
            
            # Calculate consistency score (standard deviation of time differences)
            consistency_score = np.std(time_diffs) if len(time_diffs) > 1 else 0
            
            # Normalize features
            time_diffs_norm = self._normalize_array(time_diffs, 0, 10)  # Normalize to 0-10 second range
            page_times_norm = self._normalize_array(page_times, 0, 60)  # Normalize to 0-60 second range
            action_frequency_norm = min(action_frequency / 5, 1)  # Normalize to 0-5 actions per second
            
            # Create feature matrix (sequence_length x num_features)
            features = np.zeros((self.sequence_length, len(self.feature_names)))
            
            features[:, 0] = time_diffs_norm
            features[:, 1] = page_times_norm
            features[:, 2] = action_frequency_norm
            features[:, 3] = session_duration / 3600  # Normalize to hours
            features[:, 4] = time_of_day
            features[:, 5] = day_of_week
            features[:, 6] = consistency_score
            
            # Add batch dimension for LSTM input
            return features.reshape(1, self.sequence_length, len(self.feature_names))
            
        except Exception as e:
            logger.error(f"Error extracting temporal features: {str(e)}", exc_info=True)
            return np.zeros((1, self.sequence_length, len(self.feature_names)))
    
    def _normalize_array(self, arr, min_val, max_val):
        """Normalize array to 0-1 range with clipping"""
        return np.clip(arr / max_val, 0, 1)
    
    def _extract_time_of_day(self, timestamp):
        """Extract time of day (0-1) from timestamp"""
        # This is a simplified implementation - in production, would use proper datetime conversion
        seconds_in_day = 24 * 60 * 60
        return (timestamp % seconds_in_day) / seconds_in_day
    
    def _extract_day_of_week(self, timestamp):
        """Extract day of week (0-6) from timestamp"""
        # This is a simplified implementation - in production, would use proper datetime conversion
        seconds_in_week = 7 * 24 * 60 * 60
        return ((timestamp // (24 * 60 * 60)) % 7) / 6  # Normalize to 0-1
    
    def build_model(self) -> tf.keras.Model:
        """
        Build and compile the LSTM model for temporal pattern analysis.
        
        Returns:
            Compiled Keras LSTM model
        """
        model = Sequential()
        
        # Input layer
        model.add(LSTM(self.hidden_size, 
                      return_sequences=True if self.num_layers > 1 else False,
                      input_shape=(self.sequence_length, len(self.feature_names))))
        model.add(Dropout(self.dropout))
        
        # Additional LSTM layers
        for i in range(1, self.num_layers):
            return_sequences = i < self.num_layers - 1
            model.add(LSTM(self.hidden_size, return_sequences=return_sequences))
            model.add(Dropout(self.dropout))
        
        # Output layer
        model.add(Dense(1, activation='sigmoid'))
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info(f"Built LSTM model with {self.num_layers} layers and {self.hidden_size} hidden units")
        return model
    
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Train the LSTM model on the provided data.
        
        Args:
            X: Training features of shape (n_samples, sequence_length, n_features)
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
        
        # Reshape features to access values
        features = features.reshape(self.sequence_length, len(self.feature_names))
        
        # Analyze time between actions
        time_diffs = features[:, 0]
        if np.std(time_diffs) < 0.1 and np.mean(time_diffs) > 0:
            explanations.append("Suspiciously consistent timing between actions")
        
        if np.mean(time_diffs) < 0.05 and len(session_data.get("events", [])) > 10:
            explanations.append("Actions performed too quickly for human interaction")
        
        # Analyze page view times
        page_times = features[:, 1]
        if np.mean(page_times) < 0.1 and len(session_data.get("events", [])) > 5:
            explanations.append("Pages viewed too quickly to read content")
        
        # Analyze action frequency
        action_frequency = features[0, 2]
        if action_frequency > 0.8:
            explanations.append("Very high frequency of actions")
        
        # Analyze consistency score
        consistency_score = features[0, 6]
        if consistency_score < 0.1 and len(session_data.get("events", [])) > 10:
            explanations.append("Unnaturally consistent timing patterns")
        
        # If no specific patterns detected but probability is high
        if not explanations and probability > 0.7:
            explanations.append("Temporal pattern resembles automated behavior")
        elif not explanations:
            explanations.append("Temporal pattern consistent with human behavior")
        
        return explanations
    
    def features_to_dict(self, features: np.ndarray) -> Dict[str, float]:
        """
        Convert feature array to dictionary for easier interpretation.
        
        Args:
            features: Extracted features
            
        Returns:
            Dictionary mapping feature names to values
        """
        # Reshape features to access values
        features = features.reshape(self.sequence_length, len(self.feature_names))
        
        return {
            "avg_time_between_actions": float(np.mean(features[:, 0])),
            "std_time_between_actions": float(np.std(features[:, 0])),
            "avg_time_on_page": float(np.mean(features[:, 1])),
            "action_frequency": float(features[0, 2]),
            "session_duration": float(features[0, 3]),
            "avg_time_of_day": float(np.mean(features[:, 4])),
            "consistency_score": float(features[0, 6])
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