import logging
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam

from .base_expert import BaseExpert

logger = logging.getLogger(__name__)

class InputBehaviorExpert(BaseExpert):
    """
    Expert model that analyzes user input behavior.
    
    This expert focuses on mouse movements, click patterns, scrolling behavior,
    and keystroke dynamics to distinguish between human and bot behavior.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Get configuration parameters
        self.features = config.get("features", [
            "mouse_movement", "click_patterns", "scroll_behavior", "keystroke_dynamics"
        ])
        self.conv_layers = config.get("conv_layers", 3)
        self.filters = config.get("filters", [32, 64, 128])
        self.learning_rate = config.get("learning_rate", 0.0008)
        
        # Set input dimensions
        self.sequence_length = 200  # Number of events to consider
        self.num_features = 8      # Features per event
        
        if self.enabled:
            self.model = self.build_model()
    
    def extract_features(self, session_data: Dict[str, Any]) -> np.ndarray:
        """
        Extract input behavior features from the session data.
        
        Features include:
        - Mouse movement patterns (speed, acceleration, curvature)
        - Click patterns (double clicks, click locations)
        - Scroll behavior (speed, frequency, direction changes)
        - Keystroke dynamics (typing speed, rhythm, errors)
        
        Args:
            session_data: Dictionary containing session data
            
        Returns:
            Numpy array of extracted input behavior features
        """
        try:
            # Initialize feature array
            features = np.zeros((self.sequence_length, self.num_features))
            
            # Extract mouse events
            mouse_events = self._extract_mouse_events(session_data)
            if "mouse_movement" in self.features and mouse_events:
                features = self._process_mouse_events(mouse_events, features)
            
            # Extract click events
            click_events = self._extract_click_events(session_data)
            if "click_patterns" in self.features and click_events:
                features = self._process_click_events(click_events, features)
            
            # Extract scroll events
            scroll_events = self._extract_scroll_events(session_data)
            if "scroll_behavior" in self.features and scroll_events:
                features = self._process_scroll_events(scroll_events, features)
            
            # Extract keyboard events
            keyboard_events = self._extract_keyboard_events(session_data)
            if "keystroke_dynamics" in self.features and keyboard_events:
                features = self._process_keyboard_events(keyboard_events, features)
            
            # Add batch dimension for CNN input
            return features.reshape(1, self.sequence_length, self.num_features)
            
        except Exception as e:
            logger.error(f"Error extracting input behavior features: {str(e)}", exc_info=True)
            return np.zeros((1, self.sequence_length, self.num_features))
    
    def _extract_mouse_events(self, session_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract mouse movement events from session data"""
        events = session_data.get("events", [])
        return [event for event in events if event.get("type") == "mousemove"]
    
    def _extract_click_events(self, session_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract mouse click events from session data"""
        events = session_data.get("events", [])
        return [event for event in events if event.get("type") in ["click", "dblclick"]]
    
    def _extract_scroll_events(self, session_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract scroll events from session data"""
        events = session_data.get("events", [])
        return [event for event in events if event.get("type") == "scroll"]
    
    def _extract_keyboard_events(self, session_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract keyboard events from session data"""
        events = session_data.get("events", [])
        return [event for event in events if event.get("type") in ["keydown", "keyup"]]
    
    def _process_mouse_events(self, mouse_events: List[Dict[str, Any]], features: np.ndarray) -> np.ndarray:
        """Process mouse movement events to extract features"""
        if len(mouse_events) < 2:
            return features
        
        # Sort events by timestamp
        mouse_events.sort(key=lambda e: e.get("timestamp", 0))
        
        # Take the most recent sequence_length events or all if less
        mouse_events = mouse_events[-self.sequence_length:] if len(mouse_events) > self.sequence_length else mouse_events
        
        for i, event in enumerate(mouse_events):
            if i >= self.sequence_length:
                break
                
            x = event.get("x", 0)
            y = event.get("y", 0)
            
            # Normalize coordinates to 0-1 range
            screen_width = event.get("screenWidth", 1920)
            screen_height = event.get("screenHeight", 1080)
            
            x_norm = x / screen_width
            y_norm = y / screen_height
            
            # Calculate velocity if possible
            vel_x, vel_y = 0, 0
            if i > 0:
                prev_event = mouse_events[i-1]
                time_diff = event.get("timestamp", 0) - prev_event.get("timestamp", 0)
                if time_diff > 0:
                    prev_x = prev_event.get("x", 0)
                    prev_y = prev_event.get("y", 0)
                    vel_x = (x - prev_x) / time_diff
                    vel_y = (y - prev_y) / time_diff
            
            # Normalize velocity
            vel_x_norm = np.clip(vel_x / 1000, -1, 1)
            vel_y_norm = np.clip(vel_y / 1000, -1, 1)
            
            # Store features
            features[i, 0] = x_norm
            features[i, 1] = y_norm
            features[i, 2] = vel_x_norm
            features[i, 3] = vel_y_norm
        
        return features
    
    def _process_click_events(self, click_events: List[Dict[str, Any]], features: np.ndarray) -> np.ndarray:
        """Process click events to extract features"""
        if not click_events:
            return features
            
        # Sort events by timestamp
        click_events.sort(key=lambda e: e.get("timestamp", 0))
        
        # Calculate click intervals
        click_intervals = []
        for i in range(1, len(click_events)):
            interval = click_events[i].get("timestamp", 0) - click_events[i-1].get("timestamp", 0)
            click_intervals.append(interval)
        
        # Calculate statistics
        avg_interval = np.mean(click_intervals) if click_intervals else 0
        std_interval = np.std(click_intervals) if len(click_intervals) > 1 else 0
        
        # Detect double clicks
        double_clicks = sum(1 for interval in click_intervals if interval < 300) / max(len(click_intervals), 1)
        
        # Store click features across all events (global features)
        for i in range(min(len(click_events), self.sequence_length)):
            event = click_events[i]
            
            # Get event type
            is_double_click = event.get("type") == "dblclick"
            
            # Store features
            features[i, 4] = avg_interval / 1000  # Normalize to seconds
            features[i, 5] = std_interval / 1000  # Normalize to seconds
            features[i, 6] = double_clicks
            features[i, 7] = 1 if is_double_click else 0
        
        return features
    
    def _process_scroll_events(self, scroll_events: List[Dict[str, Any]], features: np.ndarray) -> np.ndarray:
        """Process scroll events to extract features"""
        # In a real implementation, would extract scroll behavior features
        # For now, using placeholder implementation
        return features
    
    def _process_keyboard_events(self, keyboard_events: List[Dict[str, Any]], features: np.ndarray) -> np.ndarray:
        """Process keyboard events to extract features"""
        # In a real implementation, would extract keystroke dynamics features
        # For now, using placeholder implementation
        return features
    
    def build_model(self) -> tf.keras.Model:
        """
        Build and compile the CNN model for input behavior analysis.
        
        Returns:
            Compiled Keras CNN model
        """
        model = Sequential()
        
        # Input layer and first Conv layer
        model.add(Conv1D(self.filters[0], kernel_size=3, activation='relu', 
                        input_shape=(self.sequence_length, self.num_features),
                        padding='same'))
        model.add(MaxPooling1D(pool_size=2))
        
        # Additional Conv layers
        for i in range(1, min(self.conv_layers, len(self.filters))):
            model.add(Conv1D(self.filters[i], kernel_size=3, activation='relu', padding='same'))
            model.add(MaxPooling1D(pool_size=2))
        
        # Flatten and Dense layers
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.2))
        
        # Output layer
        model.add(Dense(1, activation='sigmoid'))
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info(f"Built CNN model with {self.conv_layers} conv layers")
        return model
    
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Train the CNN model on the provided data.
        
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
        
        # Reshape features for analysis
        features = features.reshape(self.sequence_length, self.num_features)
        
        # Analyze mouse movement
        mouse_x = features[:, 0]
        mouse_y = features[:, 1]
        vel_x = features[:, 2]
        vel_y = features[:, 3]
        
        # Check for unnatural straight line movements
        if np.corrcoef(mouse_x[mouse_x != 0], mouse_y[mouse_y != 0])[0, 1] > 0.95:
            explanations.append("Mouse movement follows suspiciously straight lines")
        
        # Check for consistent velocity
        if np.std(vel_x[vel_x != 0]) < 0.1 and np.std(vel_y[vel_y != 0]) < 0.1:
            explanations.append("Mouse movement velocity is unnaturally consistent")
        
        # Analyze click patterns
        avg_click_interval = np.mean(features[:, 4][features[:, 4] != 0])
        std_click_interval = np.mean(features[:, 5][features[:, 5] != 0])
        
        if avg_click_interval > 0 and std_click_interval / avg_click_interval < 0.1:
            explanations.append("Click timing is suspiciously regular")
        
        # If no specific patterns detected but probability is high
        if not explanations and probability > 0.7:
            explanations.append("Input behavior patterns resemble automated interaction")
        elif not explanations:
            explanations.append("Input behavior consistent with human interaction")
        
        return explanations
    
    def features_to_dict(self, features: np.ndarray) -> Dict[str, float]:
        """
        Convert feature array to dictionary for easier interpretation.
        
        Args:
            features: Extracted features
            
        Returns:
            Dictionary mapping feature names to values
        """
        # Reshape features for analysis
        features = features.reshape(self.sequence_length, self.num_features)
        
        # Calculate summary statistics
        mouse_vel_x = features[:, 2]
        mouse_vel_y = features[:, 3]
        mouse_vel_magnitude = np.sqrt(mouse_vel_x**2 + mouse_vel_y**2)
        
        click_intervals = features[:, 4][features[:, 4] != 0]
        
        return {
            "avg_mouse_velocity": float(np.mean(mouse_vel_magnitude)),
            "std_mouse_velocity": float(np.std(mouse_vel_magnitude)),
            "avg_click_interval": float(np.mean(click_intervals) if len(click_intervals) > 0 else 0),
            "std_click_interval": float(np.std(click_intervals) if len(click_intervals) > 1 else 0),
            "double_click_ratio": float(np.mean(features[:, 6][features[:, 6] != 0])) if np.any(features[:, 6] != 0) else 0
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