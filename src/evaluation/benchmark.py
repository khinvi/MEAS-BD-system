import logging
import time
from typing import Dict, Any, List, Tuple, Optional
import numpy as np

from ..meta import ExpertEnsemble
from ..data import SyntheticDataGenerator, DataPreprocessor
from .metrics import PerformanceMetrics

logger = logging.getLogger(__name__)

class BenchmarkSystem:
    """
    Benchmark system for evaluating bot detection performance.
    
    This class runs automated benchmarks to evaluate the system's performance
    across different bot types, configurations, and scenarios.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the benchmark system.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config
        self.evaluation_config = config.get("evaluation", {})
        self.cross_validation_folds = self.evaluation_config.get("cross_validation_folds", 5)
        self.test_size = self.evaluation_config.get("test_size", 0.2)
        
        # Initialize components
        self.metrics = PerformanceMetrics(self.evaluation_config)
        
        logger.info(f"Initialized BenchmarkSystem with {self.cross_validation_folds} CV folds")
    
    def run_benchmark(self, ensemble: ExpertEnsemble) -> Dict[str, Any]:
        """
        Run a comprehensive benchmark of the detection system.
        
        Args:
            ensemble: The expert ensemble to benchmark
            
        Returns:
            Dictionary containing benchmark results
        """
        logger.info("Starting benchmark run")
        start_time = time.time()
        
        # Generate synthetic data
        data_config = self.config.get("data", {}).get("synthetic_data", {})
        data_generator = SyntheticDataGenerator(data_config)
        
        sessions, labels = data_generator.generate_dataset()
        logger.info(f"Generated {len(sessions)} synthetic sessions ({sum(labels)} bots, {len(labels) - sum(labels)} humans)")
        
        # Extract features for each expert
        features = data_generator.generate_feature_vectors(sessions)
        
        # Preprocess data
        preproc_config = self.config.get("data", {}).get("preprocessing", {})
        preprocessor = DataPreprocessor(preproc_config)
        
        # Convert labels to numpy array
        y = np.array(labels)
        
        # Split data into train and test sets
        n_samples = len(labels)
        n_test = int(n_samples * self.test_size)
        n_train = n_samples - n_test
        
        # Use random indices for split
        indices = np.random.permutation(n_samples)
        train_indices = indices[:n_train]
        test_indices = indices[n_train:]
        
        # Create train and test feature sets
        X_train = {expert: features[expert][train_indices] for expert in features}
        X_test = {expert: features[expert][test_indices] for expert in features}
        y_train = y[train_indices]
        y_test = y[test_indices]
        
        # Preprocess features
        X_train_proc = preprocessor.fit_transform(X_train, y_train)
        X_test_proc = preprocessor.transform(X_test)
        
        # Train the ensemble
        train_metrics = ensemble.train(X_train_proc, y_train)
        logger.info(f"Trained ensemble with metrics: {train_metrics}")
        
        # Make predictions on test set
        expert_predictions = {}
        for expert_name, expert in ensemble.experts.items():
            if expert_name not in X_test_proc:
                logger.warning(f"No test data for expert {expert_name}, skipping")
                continue
                
            try:
                X_expert = X_test_proc[expert_name]
                predictions = expert.predict_probability(X_expert)
                expert_predictions[expert_name] = predictions
            except Exception as e:
                logger.error(f"Error getting predictions from expert {expert_name}: {str(e)}", exc_info=True)
        
        # Create sessions from test features for ensemble prediction
        test_sessions = [sessions[i] for i in test_indices]
        
        # Get ensemble predictions
        ensemble_predictions = np.zeros(len(test_sessions))
        ensemble_results = []
        
        for i, session in enumerate(test_sessions):
            try:
                result = ensemble.predict(session)
                ensemble_predictions[i] = result.get("probability", 0.5)
                ensemble_results.append(result)
            except Exception as e:
                logger.error(f"Error getting ensemble prediction: {str(e)}", exc_info=True)
                ensemble_predictions[i] = 0.5
                ensemble_results.append({"probability": 0.5, "is_bot": False, "confidence": 0.0})
        
        # Calculate metrics
        ensemble_pred_binary = (ensemble_predictions >= 0.5).astype(int)
        overall_metrics = self.metrics.calculate_metrics(y_test, ensemble_pred_binary, ensemble_predictions)
        expert_metrics = self.metrics.calculate_expert_metrics(y_test, expert_predictions)
        
        # Calculate threshold metrics
        threshold_metrics = self.metrics.calculate_threshold_metrics(y_test, ensemble_predictions)
        
        # Calculate metrics by bot type
        bot_types = ["simple_script", "advanced_script", "browser_automation", 
                    "headless_browser", "proxy_rotating", "human_mimicking"]
        
        bot_type_metrics = {}
        for bot_type in bot_types:
            # Create mask for this bot type
            bot_type_indices = []
            for i, session in enumerate(test_sessions):
                if y_test[i] == 1:  # If it's a bot
                    # Check bot type (simplified)
                    if session.get("botType", "") == bot_type:
                        bot_type_indices.append(i)
            
            if not bot_type_indices:
                logger.warning(f"No test samples for bot type {bot_type}, skipping")
                continue
                
            # Convert to numpy array
            bot_type_mask = np.zeros(len(test_sessions), dtype=bool)
            bot_type_mask[bot_type_indices] = True
            
            # Calculate metrics for this bot type
            y_test_slice = y_test[bot_type_mask]
            y_pred_slice = ensemble_pred_binary[bot_type_mask]
            y_prob_slice = ensemble_predictions[bot_type_mask]
            
            bot_metrics = self.metrics.calculate_metrics(y_test_slice, y_pred_slice, y_prob_slice)
            bot_type_metrics[bot_type] = bot_metrics
        
        # Calculate confidence calibration
        calibration = self.metrics.calculate_confidence_calibration(y_test, ensemble_predictions)
        
        # Calculate timing metrics
        total_time = time.time() - start_time
        
        # Compile results
        benchmark_results = {
            "overall_metrics": overall_metrics,
            "expert_metrics": expert_metrics,
            "threshold_metrics": threshold_metrics,
            "bot_type_metrics": bot_type_metrics,
            "calibration": calibration,
            "timing": {
                "total_benchmark_time": total_time
            },
            "dataset_info": {
                "total_samples": n_samples,
                "train_samples": n_train,
                "test_samples": n_test,
                "bot_samples": int(sum(y)),
                "human_samples": int(n_samples - sum(y))
            }
        }
        
        logger.info(f"Benchmark completed in {total_time:.2f} seconds")
        return benchmark_results
    
    def run_cross_validation(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run cross-validation benchmark.
        
        Args:
            config: System configuration
            
        Returns:
            Dictionary containing cross-validation results
        """
        logger.info(f"Starting {self.cross_validation_folds}-fold cross-validation")
        start_time = time.time()
        
        # Generate synthetic data
        data_config = config.get("data", {}).get("synthetic_data", {})
        data_generator = SyntheticDataGenerator(data_config)
        
        sessions, labels = data_generator.generate_dataset()
        logger.info(f"Generated {len(sessions)} synthetic sessions for CV")
        
        # Extract features for each expert
        features = data_generator.generate_feature_vectors(sessions)
        
        # Convert labels to numpy array
        y = np.array(labels)
        
        # Initialize metrics collectors
        fold_metrics = []
        
        # Run cross-validation
        n_samples = len(labels)
        indices = np.random.permutation(n_samples)
        fold_size = n_samples // self.cross_validation_folds
        
        for fold in range(self.cross_validation_folds):
            logger.info(f"Running CV fold {fold+1}/{self.cross_validation_folds}")
            
            # Create train/test split for this fold
            test_indices = indices[fold * fold_size:(fold + 1) * fold_size]
            train_indices = np.setdiff1d(indices, test_indices)
            
            # Create train and test feature sets
            X_train = {expert: features[expert][train_indices] for expert in features}
            X_test = {expert: features[expert][test_indices] for expert in features}
            y_train = y[train_indices]
            y_test = y[test_indices]
            
            # Preprocess features
            preproc_config = config.get("data", {}).get("preprocessing", {})
            preprocessor = DataPreprocessor(preproc_config)
            X_train_proc = preprocessor.fit_transform(X_train, y_train)
            X_test_proc = preprocessor.transform(X_test)
            
            # Initialize and train ensemble
            ensemble = ExpertEnsemble(config)
            train_metrics = ensemble.train(X_train_proc, y_train)
            
            # Make predictions on test set
            expert_predictions = {}
            for expert_name, expert in ensemble.experts.items():
                if expert_name not in X_test_proc:
                    logger.warning(f"No test data for expert {expert_name} in fold {fold+1}, skipping")
                    continue
                    
                try:
                    X_expert = X_test_proc[expert_name]
                    predictions = expert.predict_probability(X_expert)
                    expert_predictions[expert_name] = predictions
                except Exception as e:
                    logger.error(f"Error getting predictions from expert {expert_name} in fold {fold+1}: {str(e)}", exc_info=True)
            
            # Create sessions from test features for ensemble prediction
            test_sessions = [sessions[i] for i in test_indices]
            
            # Get ensemble predictions
            ensemble_predictions = np.zeros(len(test_sessions))
            
            for i, session in enumerate(test_sessions):
                try:
                    result = ensemble.predict(session)
                    ensemble_predictions[i] = result.get("probability", 0.5)
                except Exception as e:
                    logger.error(f"Error getting ensemble prediction in fold {fold+1}: {str(e)}", exc_info=True)
                    ensemble_predictions[i] = 0.5
            
            # Calculate metrics
            ensemble_pred_binary = (ensemble_predictions >= 0.5).astype(int)
            fold_overall_metrics = self.metrics.calculate_metrics(y_test, ensemble_pred_binary, ensemble_predictions)
            fold_expert_metrics = self.metrics.calculate_expert_metrics(y_test, expert_predictions)
            
            # Save fold results
            fold_metrics.append({
                "fold": fold + 1,
                "train_size": len(train_indices),
                "test_size": len(test_indices),
                "overall_metrics": fold_overall_metrics,
                "expert_metrics": fold_expert_metrics
            })
        
        # Calculate average metrics across folds
        avg_metrics = {}
        for metric in fold_metrics[0]["overall_metrics"]:
            values = [fold["overall_metrics"].get(metric, 0) for fold in fold_metrics]
            avg_metrics[metric] = float(np.mean(values))
            
        # Calculate standard deviations
        std_metrics = {}
        for metric in fold_metrics[0]["overall_metrics"]:
            values = [fold["overall_metrics"].get(metric, 0) for fold in fold_metrics]
            std_metrics[metric] = float(np.std(values))
        
        # Calculate timing metrics
        total_time = time.time() - start_time
        
        # Compile results
        cv_results = {
            "avg_metrics": avg_metrics,
            "std_metrics": std_metrics,
            "fold_metrics": fold_metrics,
            "timing": {
                "total_cv_time": total_time,
                "avg_fold_time": total_time / self.cross_validation_folds
            },
            "dataset_info": {
                "total_samples": n_samples,
                "fold_size": fold_size,
                "bot_samples": int(sum(y)),
                "human_samples": int(n_samples - sum(y))
            }
        }
        
        logger.info(f"Cross-validation completed in {total_time:.2f} seconds")
        return cv_results