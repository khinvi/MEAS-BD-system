import logging
from typing import Dict, Any, List, Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

from .base_expert import BaseExpert

logger = logging.getLogger(__name__)

class TechnicalFingerprintExpert(BaseExpert):
    """
    Expert model that analyzes technical fingerprints of devices and browsers.
    
    This expert focuses on browser characteristics, HTTP headers, canvas fingerprints,
    and other technical signals that can differentiate between human users and bots.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Get configuration parameters
        self.feature_list = config.get("features", [
            "browser_fingerprint", "headers", "canvas_fingerprint", "webrtc_info"
        ])
        self.n_estimators = config.get("n_estimators", 200)
        self.max_depth = config.get("max_depth", 15)
        self.min_samples_split = config.get("min_samples_split", 10)
        
        # Features
        self.feature_names = []
        self._initialize_feature_names()
        
        if self.enabled:
            self.model = self.build_model()
    
    def _initialize_feature_names(self):
        """Initialize the list of feature names based on enabled features"""
        if "browser_fingerprint" in self.feature_list:
            self.feature_names.extend([
                "user_agent_hash", "browser_type", "browser_version",
                "os_type", "os_version", "device_type", "device_memory",
                "cpu_cores", "touchpoints", "has_webgl", "has_webgl2"
            ])
            
        if "headers" in self.feature_list:
            self.feature_names.extend([
                "accept_language", "accept_encoding", "accept_header",
                "connection_header", "cookies_enabled", "do_not_track",
                "headers_order_hash"
            ])
            
        if "canvas_fingerprint" in self.feature_list:
            self.feature_names.extend([
                "canvas_hash", "webgl_vendor", "webgl_renderer",
                "canvas_text_metrics_hash"
            ])
            
        if "webrtc_info" in self.feature_list:
            self.feature_names.extend([
                "ip_address_hash", "has_local_network", "ip_addresses_count",
                "vpn_detected"
            ])
            
        # Add feature to detect inconsistencies
        self.feature_names.append("inconsistency_score")
    
    def extract_features(self, session_data: Dict[str, Any]) -> np.ndarray:
        """
        Extract technical fingerprint features from the session data.
        
        Args:
            session_data: Dictionary containing session data
            
        Returns:
            Numpy array of extracted features
        """
        try:
            # Initialize features array
            features = np.zeros(len(self.feature_names))
            
            # Extract browser fingerprint information
            if "browser_fingerprint" in self.feature_list:
                self._extract_browser_fingerprint(session_data, features)
                
            # Extract headers information
            if "headers" in self.feature_list:
                self._extract_headers_info(session_data, features)
                
            # Extract canvas fingerprint
            if "canvas_fingerprint" in self.feature_list:
                self._extract_canvas_fingerprint(session_data, features)
                
            # Extract WebRTC information
            if "webrtc_info" in self.feature_list:
                self._extract_webrtc_info(session_data, features)
                
            # Calculate inconsistency score
            features[-1] = self._calculate_inconsistency_score(session_data)
            
            return features.reshape(1, -1)
            
        except Exception as e:
            logger.error(f"Error extracting technical fingerprint features: {str(e)}", exc_info=True)
            return np.zeros((1, len(self.feature_names)))
    
    def _extract_browser_fingerprint(self, session_data: Dict[str, Any], features: np.ndarray):
        """Extract browser fingerprint information"""
        fingerprint = session_data.get("browserFingerprint", {})
        
        # Fill in feature array
        start_idx = self.feature_names.index("user_agent_hash") if "user_agent_hash" in self.feature_names else 0
        
        # User agent hash (simplified simulation - would be actual hash in production)
        user_agent = fingerprint.get("userAgent", "")
        features[start_idx] = hash(user_agent) % 1000 / 1000  # Normalize to 0-1
        
        # Browser type (categorical, encoded as numerical)
        browser_types = {"chrome": 0.1, "firefox": 0.2, "safari": 0.3, "edge": 0.4, "other": 0.5}
        features[start_idx + 1] = browser_types.get(fingerprint.get("browserType", "other").lower(), 0.5)
        
        # Browser version (normalized)
        browser_version = fingerprint.get("browserVersion", 0)
        features[start_idx + 2] = min(browser_version / 100, 1.0)  # Normalize to 0-1
        
        # OS type (categorical, encoded as numerical)
        os_types = {"windows": 0.1, "macos": 0.2, "linux": 0.3, "android": 0.4, "ios": 0.5, "other": 0.6}
        features[start_idx + 3] = os_types.get(fingerprint.get("osType", "other").lower(), 0.6)
        
        # OS version (normalized)
        os_version = fingerprint.get("osVersion", 0)
        features[start_idx + 4] = min(os_version / 20, 1.0)  # Normalize to 0-1
        
        # Device type (categorical, encoded as numerical)
        device_types = {"desktop": 0.1, "mobile": 0.2, "tablet": 0.3, "other": 0.4}
        features[start_idx + 5] = device_types.get(fingerprint.get("deviceType", "other").lower(), 0.4)
        
        # Device memory (GB, normalized)
        device_memory = fingerprint.get("deviceMemory", 0)
        features[start_idx + 6] = min(device_memory / 16, 1.0)  # Normalize to 0-1
        
        # CPU cores
        cpu_cores = fingerprint.get("cpuCores", 0)
        features[start_idx + 7] = min(cpu_cores / 16, 1.0)  # Normalize to 0-1
        
        # Touch points
        touchpoints = fingerprint.get("touchPoints", 0)
        features[start_idx + 8] = min(touchpoints / 10, 1.0)  # Normalize to 0-1
        
        # WebGL support
        features[start_idx + 9] = 1.0 if fingerprint.get("hasWebGL", False) else 0.0
        features[start_idx + 10] = 1.0 if fingerprint.get("hasWebGL2", False) else 0.0
    
    def _extract_headers_info(self, session_data: Dict[str, Any], features: np.ndarray):
        """Extract HTTP headers information"""
        headers = session_data.get("headers", {})
        
        start_idx = self.feature_names.index("accept_language") if "accept_language" in self.feature_names else 0
        
        # Accept-Language header (hash, normalized)
        accept_language = headers.get("Accept-Language", "")
        features[start_idx] = hash(accept_language) % 1000 / 1000  # Normalize to 0-1
        
        # Accept-Encoding header (hash, normalized)
        accept_encoding = headers.get("Accept-Encoding", "")
        features[start_idx + 1] = hash(accept_encoding) % 1000 / 1000
        
        # Accept header (hash, normalized)
        accept_header = headers.get("Accept", "")
        features[start_idx + 2] = hash(accept_header) % 1000 / 1000
        
        # Connection header (binary)
        connection_header = headers.get("Connection", "").lower()
        features[start_idx + 3] = 1.0 if connection_header == "keep-alive" else 0.0
        
        # Cookies enabled
        features[start_idx + 4] = 1.0 if session_data.get("cookiesEnabled", False) else 0.0
        
        # Do Not Track
        features[start_idx + 5] = 1.0 if headers.get("DNT") == "1" else 0.0
        
        # Headers order hash (would be actual implementation in production)
        header_names = list(headers.keys())
        features[start_idx + 6] = hash("".join(header_names)) % 1000 / 1000
    
    def _extract_canvas_fingerprint(self, session_data: Dict[str, Any], features: np.ndarray):
        """Extract canvas fingerprint information"""
        canvas_data = session_data.get("canvasFingerprint", {})
        
        start_idx = self.feature_names.index("canvas_hash") if "canvas_hash" in self.feature_names else 0
        
        # Canvas hash (simplified simulation)
        canvas_hash = canvas_data.get("hash", "")
        features[start_idx] = hash(canvas_hash) % 1000 / 1000
        
        # WebGL vendor and renderer information
        webgl_info = canvas_data.get("webgl", {})
        vendor = webgl_info.get("vendor", "")
        renderer = webgl_info.get("renderer", "")
        
        features[start_idx + 1] = hash(vendor) % 1000 / 1000
        features[start_idx + 2] = hash(renderer) % 1000 / 1000
        
        # Text metrics hash
        text_metrics = canvas_data.get("textMetricsHash", "")
        features[start_idx + 3] = hash(text_metrics) % 1000 / 1000
    
    def _extract_webrtc_info(self, session_data: Dict[str, Any], features: np.ndarray):
        """Extract WebRTC information"""
        webrtc_data = session_data.get("webrtcInfo", {})
        
        start_idx = self.feature_names.index("ip_address_hash") if "ip_address_hash" in self.feature_names else 0
        
        # IP address hash
        ip_address = webrtc_data.get("ipAddress", "")
        features[start_idx] = hash(ip_address) % 1000 / 1000
        
        # Local network detection
        features[start_idx + 1] = 1.0 if webrtc_data.get("hasLocalNetwork", False) else 0.0
        
        # Number of IP addresses
        ip_count = len(webrtc_data.get("ipAddresses", []))
        features[start_idx + 2] = min(ip_count / 10, 1.0)
        
        # VPN detection
        features[start_idx + 3] = 1.0 if webrtc_data.get("vpnDetected", False) else 0.0
    
    def _calculate_inconsistency_score(self, session_data: Dict[str, Any]) -> float:
        """
        Calculate a score that represents inconsistencies in the technical fingerprint.
        Higher score indicates more inconsistencies, which is suspicious.
        """
        inconsistencies = 0
        
        # Example inconsistency checks (would be more extensive in production)
        fingerprint = session_data.get("browserFingerprint", {})
        headers = session_data.get("headers", {})
        
        # Check user agent consistency with reported browser
        user_agent = headers.get("User-Agent", "").lower()
        browser_type = fingerprint.get("browserType", "").lower()
        
        if browser_type == "chrome" and "chrome" not in user_agent:
            inconsistencies += 1
        elif browser_type == "firefox" and "firefox" not in user_agent:
            inconsistencies += 1
        elif browser_type == "safari" and "safari" not in user_agent:
            inconsistencies += 1
            
        # Check OS consistency
        os_type = fingerprint.get("osType", "").lower()
        
        if os_type == "windows" and "windows" not in user_agent:
            inconsistencies += 1
        elif os_type == "macos" and "mac" not in user_agent:
            inconsistencies += 1
        elif os_type == "linux" and "linux" not in user_agent:
            inconsistencies += 1
            
        # Check for mismatch between timezone and Accept-Language
        timezone = session_data.get("timezone", "")
        accept_language = headers.get("Accept-Language", "").lower()
        
        if "us" in timezone and not any(lang in accept_language for lang in ["en-us", "en_us"]):
            inconsistencies += 1
            
        # Normalize to 0-1 scale (assuming max 5 inconsistencies)
        return min(inconsistencies / 5, 1.0)
    
    def build_model(self) -> RandomForestClassifier:
        """
        Build the Random Forest model for technical fingerprint analysis.
        
        Returns:
            Random Forest classifier
        """
        model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            random_state=42
        )
        
        logger.info(f"Built Random Forest model with {self.n_estimators} trees")
        return model
    
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Train the Random Forest model on the provided data.
        
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
        
        # Check for suspicious technical characteristics
        
        # Inconsistency score
        inconsistency_score = features.flatten()[-1]
        if inconsistency_score > 0.4:
            explanations.append("Detected inconsistencies in browser fingerprint data")
        
        # Header anomalies
        headers = session_data.get("headers", {})
        if "headers" in self.feature_list:
            if not headers.get("Accept-Language"):
                explanations.append("Missing Accept-Language header")
            
            user_agent = headers.get("User-Agent", "")
            if not user_agent or len(user_agent) < 20:
                explanations.append("Suspicious User-Agent header")
        
        # Canvas fingerprint anomalies
        if "canvas_fingerprint" in self.feature_list:
            canvas_data = session_data.get("canvasFingerprint", {})
            if not canvas_data.get("hash"):
                explanations.append("Canvas fingerprint not available")
        
        # WebRTC anomalies
        if "webrtc_info" in self.feature_list:
            webrtc_data = session_data.get("webrtcInfo", {})
            if webrtc_data.get("vpnDetected", False):
                explanations.append("VPN or proxy detected")
        
        # If no specific patterns detected but probability is high
        if not explanations and probability > 0.7:
            explanations.append("Technical fingerprint matches known bot patterns")
        elif not explanations:
            explanations.append("Technical fingerprint consistent with legitimate browsers")
        
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