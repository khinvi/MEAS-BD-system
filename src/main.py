import os
import json
import logging
import argparse
from datetime import datetime
from typing import Dict, Any, Optional

from .meta import ExpertEnsemble
from .data import SyntheticDataGenerator, DataPreprocessor, DataStorage
from .evaluation import BenchmarkSystem, ResultVisualizer
from .experts import EXPERT_CLASSES

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('sneaker_bot_detection.log')
    ]
)

logger = logging.getLogger(__name__)

class BotDetectionSystem:
    """
    Main system for sneaker bot detection.
    
    This class orchestrates the entire bot detection pipeline, including
    data generation, model training, evaluation, and result visualization.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the bot detection system.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = self._load_config(config_path)
        
        # Initialize components
        self.ensemble = None
        self.data_storage = None
        self.benchmark = None
        self.visualizer = None
        
        logger.info(f"Initialized BotDetectionSystem with config from {config_path}")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from file"""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded configuration from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}", exc_info=True)
            # Fallback to default configuration
            logger.warning("Using default configuration")
            return {
                "system": {
                    "name": "MEAS-BD",
                    "version": "0.1.0",
                    "log_level": "INFO"
                },
                "experts": {
                    "temporal_expert": {"enabled": True},
                    "navigation_expert": {"enabled": True},
                    "input_expert": {"enabled": True},
                    "technical_expert": {"enabled": True},
                    "purchase_expert": {"enabled": True}
                },
                "meta_learning": {
                    "ensemble_method": "stacking",
                    "weighting_strategy": "dynamic"
                },
                "data": {
                    "synthetic_data": {
                        "enabled": True,
                        "human_samples": 100,
                        "bot_samples": 100
                    },
                    "storage": {
                        "type": "sqlite",
                        "path": "data/sessions.db"
                    }
                },
                "evaluation": {
                    "metrics": ["accuracy", "precision", "recall", "f1", "auc"],
                    "cross_validation_folds": 5,
                    "test_size": 0.2
                }
            }
    
    def initialize(self):
        """Initialize all system components"""
        try:
            # Initialize ensemble
            self.ensemble = ExpertEnsemble(self.config)
            
            # Initialize data storage
            storage_config = self.config.get("data", {}).get("storage", {})
            self.data_storage = DataStorage(storage_config)
            
            # Initialize benchmark system
            self.benchmark = BenchmarkSystem(self.config)
            
            # Initialize visualizer
            self.visualizer = ResultVisualizer(self.config.get("evaluation", {}))
            
            logger.info("All components initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Error initializing components: {str(e)}", exc_info=True)
            return False
    
    def train(self) -> bool:
        """
        Train the bot detection system using synthetic data.
        
        Returns:
            True if training was successful, False otherwise
        """
        try:
            logger.info("Starting system training")
            
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
            import numpy as np
            y = np.array(labels)
            
            # Preprocess features
            X_proc = preprocessor.fit_transform(features, y)
            
            # Train the ensemble
            train_metrics = self.ensemble.train(X_proc, y)
            
            logger.info(f"Training completed with metrics: {train_metrics}")
            
            # Save training metrics
            self.data_storage.save_model_metrics(
                self.config.get("system", {}).get("version", "0.1.0"),
                {f"train_{k}": v for k, v in train_metrics.get("accuracy", 0.0)}
            )
            
            return True
        except Exception as e:
            logger.error(f"Error during training: {str(e)}", exc_info=True)
            return False
    
    def evaluate(self) -> Optional[Dict[str, Any]]:
        """
        Evaluate the bot detection system.
        
        Returns:
            Dictionary containing evaluation results, or None if evaluation failed
        """
        try:
            logger.info("Starting system evaluation")
            
            # Run benchmark
            benchmark_results = self.benchmark.run_benchmark(self.ensemble)
            
            # Save metrics
            self.data_storage.save_model_metrics(
                self.config.get("system", {}).get("version", "0.1.0"),
                benchmark_results.get("overall_metrics", {})
            )
            
            # Generate visualizations
            output_dir = os.path.join("results", 
                                    f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            os.makedirs(output_dir, exist_ok=True)
            
            self.visualizer.generate_summary_dashboard(benchmark_results, save_dir=output_dir)
            
            # Save full results
            results_path = os.path.join(output_dir, "benchmark_results.json")
            with open(results_path, 'w') as f:
                json.dump(benchmark_results, f, indent=2)
            
            logger.info(f"Evaluation completed, results saved to {output_dir}")
            
            return benchmark_results
        except Exception as e:
            logger.error(f"Error during evaluation: {str(e)}", exc_info=True)
            return None
    
    def analyze_session(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a session and determine if it's a bot.
        
        Args:
            session_data: Dictionary containing session data
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            logger.info(f"Analyzing session {session_data.get('sessionId', 'unknown')}")
            
            # Check if ensemble is initialized
            if self.ensemble is None:
                logger.error("Ensemble not initialized, cannot analyze session")
                return {"error": "System not initialized"}
            
            # Run analysis
            result = self.ensemble.predict(session_data)
            
            # Save session and result
            if self.data_storage is not None:
                self.data_storage.save_session(session_data, result)
            
            logger.info(f"Session analyzed: bot probability {result.get('probability', 0.5):.3f}, confidence {result.get('confidence', 0.0):.3f}")
            
            return result
        except Exception as e:
            logger.error(f"Error analyzing session: {str(e)}", exc_info=True)
            return {"error": str(e)}
    
    def cross_validate(self) -> Optional[Dict[str, Any]]:
        """
        Run cross-validation to evaluate model stability.
        
        Returns:
            Dictionary containing cross-validation results, or None if failed
        """
        try:
            logger.info("Starting cross-validation")
            
            # Run cross-validation
            cv_results = self.benchmark.run_cross_validation(self.config)
            
            # Generate visualizations
            output_dir = os.path.join("results", 
                                    f"cv_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            os.makedirs(output_dir, exist_ok=True)
            
            # Plot CV results
            cv_plot = self.visualizer.plot_cross_validation_results(
                cv_results,
                metrics_to_plot=["accuracy", "precision", "recall", "f1"],
                save_path=os.path.join(output_dir, "cv_results.png")
            )
            
            # Save full results
            results_path = os.path.join(output_dir, "cv_results.json")
            with open(results_path, 'w') as f:
                json.dump(cv_results, f, indent=2)
            
            logger.info(f"Cross-validation completed, results saved to {output_dir}")
            
            return cv_results
        except Exception as e:
            logger.error(f"Error during cross-validation: {str(e)}", exc_info=True)
            return None
    
    def save_model(self, path: str) -> bool:
        """
        Save the trained model to disk.
        
        Args:
            path: Directory path to save the model
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Saving model to {path}")
            
            # Create directory if it doesn't exist
            os.makedirs(path, exist_ok=True)
            
            # Save ensemble
            self.ensemble.save(path)
            
            # Save configuration
            config_path = os.path.join(path, "config.json")
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            
            logger.info(f"Model saved successfully to {path}")
            return True
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}", exc_info=True)
            return False
    
    def load_model(self, path: str) -> bool:
        """
        Load a trained model from disk.
        
        Args:
            path: Directory path to load the model from
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Loading model from {path}")
            
            # Load configuration
            config_path = os.path.join(path, "config.json")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    self.config = json.load(f)
            
            # Initialize ensemble with loaded config
            self.ensemble = ExpertEnsemble(self.config)
            
            # Load ensemble
            self.ensemble.load(path)
            
            logger.info(f"Model loaded successfully from {path}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}", exc_info=True)
            return False
    
    def run_demo(self):
        """Run a demonstration of the system with synthetic sessions"""
        try:
            logger.info("Running system demonstration")
            
            # Generate a small set of synthetic data
            demo_config = {
                "human_samples": 5,
                "bot_samples": 5,
                "noise_level": 0.1
            }
            data_generator = SyntheticDataGenerator(demo_config)
            
            sessions, labels = data_generator.generate_dataset()
            
            # Analyze each session
            results = []
            for i, session in enumerate(sessions):
                true_label = "Bot" if labels[i] == 1 else "Human"
                
                # Add session ID if not present
                if "sessionId" not in session:
                    session["sessionId"] = f"demo-{i+1}"
                
                result = self.analyze_session(session)
                
                # Determine if classification was correct
                predicted_label = "Bot" if result.get("is_bot", False) else "Human"
                is_correct = predicted_label == true_label
                
                results.append({
                    "session_id": session.get("sessionId"),
                    "true_label": true_label,
                    "predicted_label": predicted_label,
                    "probability": result.get("probability", 0.5),
                    "confidence": result.get("confidence", 0.0),
                    "is_correct": is_correct
                })
                
                # Print result
                print(f"Session {session.get('sessionId')}: "
                     f"True: {true_label}, Predicted: {predicted_label}, "
                     f"Probability: {result.get('probability', 0.5):.3f}, "
                     f"Confidence: {result.get('confidence', 0.0):.3f}, "
                     f"{'✓' if is_correct else '✗'}")
                
                # Print expert breakdown
                expert_results = result.get("expert_results", {})
                for expert_name, expert_result in expert_results.items():
                    expert_prob = expert_result.get("probability", 0.5)
                    expert_conf = expert_result.get("confidence", 0.0)
                    print(f"  - {expert_name}: Prob={expert_prob:.3f}, Conf={expert_conf:.3f}")
                
                # Print top explanations
                explanations = result.get("explanation", [])
                if explanations:
                    print("  Explanations:")
                    for expl in explanations[:3]:  # Show top 3
                        print(f"  - {expl}")
                
                print()
            
            # Calculate overall accuracy
            accuracy = sum(1 for r in results if r["is_correct"]) / len(results)
            print(f"\nOverall accuracy: {accuracy:.2f}")
            
            logger.info("Demonstration completed")
            return results
        except Exception as e:
            logger.error(f"Error running demonstration: {str(e)}", exc_info=True)
            return None


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Sneaker Bot Detection System')
    parser.add_argument('--config', type=str, default='config/default_config.json',
                      help='Path to configuration file')
    parser.add_argument('--mode', type=str, default='demo',
                      choices=['train', 'evaluate', 'demo', 'cross-validate'],
                      help='Mode to run the system in')
    parser.add_argument('--save-model', type=str, default=None,
                      help='Path to save the trained model')
    parser.add_argument('--load-model', type=str, default=None,
                      help='Path to load a trained model from')
    
    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_args()
    
    # Initialize system
    system = BotDetectionSystem(args.config)
    if not system.initialize():
        logger.error("Failed to initialize system")
        return 1
    
    # Load model if specified
    if args.load_model:
        if not system.load_model(args.load_model):
            logger.error("Failed to load model")
            return 1
    
    # Run in specified mode
    if args.mode == 'train':
        if not system.train():
            logger.error("Training failed")
            return 1
    elif args.mode == 'evaluate':
        results = system.evaluate()
        if results is None:
            logger.error("Evaluation failed")
            return 1
    elif args.mode == 'cross-validate':
        results = system.cross_validate()
        if results is None:
            logger.error("Cross-validation failed")
            return 1
    elif args.mode == 'demo':
        results = system.run_demo()
        if results is None:
            logger.error("Demo failed")
            return 1
    
    # Save model if specified
    if args.save_model:
        if not system.save_model(args.save_model):
            logger.error("Failed to save model")
            return 1
    
    logger.info("System executed successfully")
    return 0


if __name__ == "__main__":
    exit(main())