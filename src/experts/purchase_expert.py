import logging
from typing import Dict, Any, List, Optional

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
import joblib

from .base_expert import BaseExpert

logger = logging.getLogger(__name__)

class PurchasePatternExpert(BaseExpert):
    """
    Expert model that analyzes purchase and checkout patterns.
    
    This expert focuses on behaviors during product selection, cart interactions,
    and checkout processes to distinguish between human and bot purchasing.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Get configuration parameters
        self.feature_list = config.get("features", [
            "checkout_speed", "payment_pattern", "shipping_info", "product_selection"
        ])
        self.n_estimators = config.get("n_estimators", 150)
        self.learning_rate = config.get("learning_rate", 0.05)
        self.max_depth = config.get("max_depth", 8)
        
        # Initialize feature names
        self.feature_names = [
            # Checkout speed features
            "time_to_checkout", "form_fill_speed", "payment_entry_speed",
            
            # Payment pattern features
            "payment_method_type", "saved_payment_used", "express_checkout_used",
            
            # Shipping info features
            "shipping_address_consistency", "billing_shipping_match", "address_change_count",
            
            # Product selection features
            "product_view_count", "add_to_cart_speed", "size_change_count",
            "target_product_ratio", "cart_abandon_attempts"
        ]
        
        if self.enabled:
            self.model = self.build_model()
    
    def extract_features(self, session_data: Dict[str, Any]) -> np.ndarray:
        """
        Extract purchase pattern features from the session data.
        
        Features include:
        - Checkout process timing
        - Payment method patterns
        - Shipping information characteristics
        - Product selection behavior
        
        Args:
            session_data: Dictionary containing session data
            
        Returns:
            Numpy array of extracted purchase pattern features
        """
        try:
            # Initialize features array
            features = np.zeros(len(self.feature_names))
            
            # Extract events
            events = session_data.get("events", [])
            
            # Sort events by timestamp
            events.sort(key=lambda e: e.get("timestamp", 0))
            
            # Extract checkout speed features if available
            if "checkout_speed" in self.feature_list:
                features = self._extract_checkout_speed_features(events, features)
            
            # Extract payment pattern features if available
            if "payment_pattern" in self.feature_list:
                features = self._extract_payment_pattern_features(session_data, features)
            
            # Extract shipping info features if available
            if "shipping_info" in self.feature_list:
                features = self._extract_shipping_info_features(session_data, features)
            
            # Extract product selection features if available
            if "product_selection" in self.feature_list:
                features = self._extract_product_selection_features(events, features)
            
            return features.reshape(1, -1)
            
        except Exception as e:
            logger.error(f"Error extracting purchase pattern features: {str(e)}", exc_info=True)
            return np.zeros((1, len(self.feature_names)))
    
    def _extract_checkout_speed_features(self, events: List[Dict[str, Any]], features: np.ndarray) -> np.ndarray:
        """Extract features related to checkout speed"""
        # Get relevant events
        add_to_cart_events = [e for e in events if e.get("type") == "add_to_cart"]
        checkout_events = [e for e in events if e.get("type") == "checkout"]
        form_events = [e for e in events if e.get("type") == "form_field"]
        payment_events = [e for e in events if e.get("type") == "payment_entry"]
        
        # Calculate time to checkout (from first add to cart to checkout)
        if add_to_cart_events and checkout_events:
            first_add_to_cart = min(e.get("timestamp", 0) for e in add_to_cart_events)
            first_checkout = min(e.get("timestamp", 0) for e in checkout_events)
            
            time_to_checkout = first_checkout - first_add_to_cart
            # Normalize to 0-1 (assuming reasonable range is 0-300 seconds)
            features[0] = min(time_to_checkout / 300, 1.0)
        
        # Calculate form fill speed
        if form_events and len(form_events) >= 2:
            form_timestamps = [e.get("timestamp", 0) for e in form_events]
            form_time_diffs = np.diff(form_timestamps)
            avg_form_time = np.mean(form_time_diffs)
            
            # Normalize to 0-1 (assuming reasonable range is 1-30 seconds per field)
            features[1] = min(avg_form_time / 30, 1.0)
        
        # Calculate payment entry speed
        if payment_events and len(payment_events) >= 2:
            payment_timestamps = [e.get("timestamp", 0) for e in payment_events]
            payment_time_diffs = np.diff(payment_timestamps)
            avg_payment_time = np.mean(payment_time_diffs)
            
            # Normalize to 0-1 (assuming reasonable range is 0.5-20 seconds per field)
            features[2] = min(avg_payment_time / 20, 1.0)
        
        return features
    
    def _extract_payment_pattern_features(self, session_data: Dict[str, Any], features: np.ndarray) -> np.ndarray:
        """Extract features related to payment patterns"""
        checkout_data = session_data.get("checkout", {})
        
        # Payment method type (categorical, encoded as numerical)
        payment_method = checkout_data.get("paymentMethod", "")
        payment_methods = {"credit_card": 0.2, "paypal": 0.4, "apple_pay": 0.6, "other": 0.8}
        features[3] = payment_methods.get(payment_method.lower(), 1.0)
        
        # Saved payment used
        features[4] = 1.0 if checkout_data.get("savedPaymentUsed", False) else 0.0
        
        # Express checkout used
        features[5] = 1.0 if checkout_data.get("expressCheckout", False) else 0.0
        
        return features
    
    def _extract_shipping_info_features(self, session_data: Dict[str, Any], features: np.ndarray) -> np.ndarray:
        """Extract features related to shipping information"""
        checkout_data = session_data.get("checkout", {})
        user_data = session_data.get("user", {})
        
        # Shipping address consistency with user profile
        user_address = user_data.get("address", "")
        shipping_address = checkout_data.get("shippingAddress", "")
        
        if user_address and shipping_address:
            # Simple string similarity (would be more sophisticated in production)
            address_similarity = self._calculate_string_similarity(user_address, shipping_address)
            features[6] = address_similarity
        
        # Billing and shipping address match
        billing_address = checkout_data.get("billingAddress", "")
        if billing_address and shipping_address:
            address_match = self._calculate_string_similarity(billing_address, shipping_address)
            features[7] = address_match
        
        # Address change count
        address_changes = checkout_data.get("addressChanges", 0)
        features[8] = min(address_changes / 5, 1.0)  # Normalize to 0-5 changes
        
        return features
    
    def _extract_product_selection_features(self, events: List[Dict[str, Any]], features: np.ndarray) -> np.ndarray:
        """Extract features related to product selection behavior"""
        # Count product views
        product_views = [e for e in events if e.get("type") == "product_view"]
        features[9] = min(len(product_views) / 20, 1.0)  # Normalize to 0-20 views
        
        # Calculate add to cart speed
        product_view_events = [e for e in events if e.get("type") == "product_view"]
        add_to_cart_events = [e for e in events if e.get("type") == "add_to_cart"]
        
        if product_view_events and add_to_cart_events:
            add_to_cart_speeds = []
            
            for add_event in add_to_cart_events:
                product_id = add_event.get("productId")
                if not product_id:
                    continue
                    
                # Find the latest product view for this product
                matching_views = [e for e in product_view_events 
                                 if e.get("productId") == product_id and 
                                 e.get("timestamp", 0) < add_event.get("timestamp", 0)]
                
                if matching_views:
                    latest_view = max(matching_views, key=lambda e: e.get("timestamp", 0))
                    time_diff = add_event.get("timestamp", 0) - latest_view.get("timestamp", 0)
                    add_to_cart_speeds.append(time_diff)
            
            if add_to_cart_speeds:
                avg_add_speed = np.mean(add_to_cart_speeds)
                # Normalize to 0-1 (assuming reasonable range is 5-120 seconds)
                features[10] = min(avg_add_speed / 120, 1.0)
        
        # Count size change events
        size_changes = [e for e in events if e.get("type") == "size_change"]
        features[11] = min(len(size_changes) / 10, 1.0)  # Normalize to 0-10 changes
        
        # Calculate target product ratio (limited edition products vs. regular)
        product_ids = [e.get("productId") for e in add_to_cart_events if e.get("productId")]
        target_products = [pid for pid in product_ids if self._is_target_product(pid)]
        
        if product_ids:
            target_ratio = len(target_products) / len(product_ids)
            features[12] = target_ratio
        
        # Count cart abandon attempts
        cart_abandons = [e for e in events if e.get("type") == "cart_abandon"]
        features[13] = min(len(cart_abandons) / 3, 1.0)  # Normalize to 0-3 abandons
        
        return features
    
    def _calculate_string_similarity(self, str1: str, str2: str) -> float:
        """
        Calculate simple string similarity.
        This is a simplified version - production would use more sophisticated algorithms.
        """
        # Convert to lowercase and remove common punctuation
        str1 = str1.lower().replace(",", "").replace(".", "")
        str2 = str2.lower().replace(",", "").replace(".", "")
        
        # Split into words
        words1 = set(str1.split())
        words2 = set(str2.split())
        
        # Calculate Jaccard similarity
        if not words1 or not words2:
            return 0.0
            
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _is_target_product(self, product_id: str) -> bool:
        """
        Determine if a product is a target product (limited edition).
        This is a placeholder implementation.
        """
        # In a real implementation, would check against a database of limited edition product IDs
        # For now, using a simplified approach where product IDs containing "limited" are targets
        return "limited" in product_id.lower() if isinstance(product_id, str) else False
    
    def build_model(self) -> GradientBoostingClassifier:
        """
        Build the Gradient Boosting model for purchase pattern analysis.
        
        Returns:
            Gradient Boosting classifier
        """
        model = GradientBoostingClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            random_state=42
        )
        
        logger.info(f"Built Gradient Boosting model with {self.n_estimators} estimators")
        return model
    
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Train the Gradient Boosting model on the provided data.
        
        Args:
            X: Training features
            y: Training labels (0 for human, 1 for bot)
            
        Returns:
            Dictionary containing training metrics
        """
        logger.info(f"Training {self.name} on {X.shape[0]} samples")
        
        # Train model
        self.model.fit(X, y)
        
        # Calculate feature importance
        importances = self.model.feature_importances_
        self.feature_importance = {
            name: float(importance) 
            for name, importance in zip(self.feature_names, importances)
        }
        
        # Compute training metrics
        y_pred = self.model.predict(X)
        accuracy = np.mean(y_pred == y)
        
        self.is_trained = True
        
        # Return training metrics
        return {
            "accuracy": float(accuracy),
            "feature_importance": self.feature_importance
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
        
        return self.model.predict(X)
    
    def _predict_probability_implementation(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the probability of the session being a bot.
        
        Args:
            X: Features to predict on
            
        Returns:
            Array of probabilities (0-1)
        """
        return self.model.predict_proba(X)[:, 1]
    
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
        features = features.flatten()
        
        # Analyze checkout speed
        time_to_checkout = features[0] * 300  # Denormalize
        if time_to_checkout < 30 and time_to_checkout > 0:
            explanations.append("Checkout completed suspiciously quickly")
        
        form_fill_speed = features[1] * 30  # Denormalize
        if form_fill_speed < 3 and form_fill_speed > 0:
            explanations.append("Form fields filled too quickly for manual typing")
        
        # Analyze product selection
        product_views = features[9] * 20  # Denormalize
        add_to_cart_speed = features[10] * 120  # Denormalize
        
        if product_views < 2 and features[12] > 0.8:
            explanations.append("Targeted limited edition products without browsing")
        
        if add_to_cart_speed < 5 and add_to_cart_speed > 0:
            explanations.append("Products added to cart suspiciously quickly after viewing")
        
        # Analyze shipping information
        address_similarity = features[6]
        address_match = features[7]
        
        if address_similarity < 0.3 and features[8] == 0:
            explanations.append("Shipping address inconsistent with user profile")
        
        # If no specific patterns detected but probability is high
        if not explanations and probability > 0.7:
            explanations.append("Purchase pattern matches known bot behavior")
        elif not explanations:
            explanations.append("Purchase pattern consistent with legitimate customer")
        
        return explanations
    
    def features_to_dict(self, features: np.ndarray) -> Dict[str, float]:
        """
        Convert feature array to dictionary for easier interpretation.
        
        Args:
            features: Extracted features
            
        Returns:
            Dictionary mapping feature names to values
        """
        return {
            name: float(value) 
            for name, value in zip(self.feature_names, features.flatten())
        }
    
    def save(self, path: str) -> None:
        """
        Save the expert model to disk.
        
        Args:
            path: Path to save the model
        """
        if self.model is not None:
            joblib.dump(self.model, path)
            logger.info(f"Saved {self.name} model to {path}")
    
    def load(self, path: str) -> None:
        """
        Load the expert model from disk.
        
        Args:
            path: Path to load the model from
        """
        try:
            self.model = joblib.load(path)
            self.is_trained = True
            logger.info(f"Loaded {self.name} model from {path}")
        except Exception as e:
            logger.error(f"Error loading model from {path}: {str(e)}", exc_info=True)