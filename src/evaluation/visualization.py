import logging
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix

logger = logging.getLogger(__name__)

class ResultVisualizer:
    """
    Visualization tools for bot detection results.
    
    This class generates various plots and visualizations to analyze
    performance metrics, compare experts, and evaluate model behavior.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the result visualizer.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config
        self.output_dir = config.get("output_dir", "results/plots")
        self.dpi = config.get("dpi", 100)
        self.figsize = config.get("figsize", (10, 6))
        self.color_palette = config.get("color_palette", "viridis")
        
        # Set up matplotlib style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette(self.color_palette)
        
        logger.info(f"Initialized ResultVisualizer with output directory: {self.output_dir}")
    
    def plot_roc_curve(self, y_true: np.ndarray, y_score: Dict[str, np.ndarray], 
                     title: str = "ROC Curve", save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot ROC curve for multiple models/experts.
        
        Args:
            y_true: Array of true labels (0 for human, 1 for bot)
            y_score: Dictionary mapping model names to predicted probabilities
            title: Plot title
            save_path: Path to save the plot (if None, will not save)
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Plot ROC curve for each model
        for name, scores in y_score.items():
            fpr, tpr, _ = roc_curve(y_true, scores)
            ax.plot(fpr, tpr, label=f"{name}")
            
        # Plot diagonal line
        ax.plot([0, 1], [0, 1], 'k--', label='Random')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(title)
        ax.legend(loc="lower right")
        
        if save_path:
            fig.savefig(save_path, bbox_inches='tight')
            logger.info(f"Saved ROC curve to {save_path}")
        
        return fig
    
    def plot_precision_recall_curve(self, y_true: np.ndarray, y_score: Dict[str, np.ndarray],
                                  title: str = "Precision-Recall Curve", 
                                  save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot precision-recall curve for multiple models/experts.
        
        Args:
            y_true: Array of true labels (0 for human, 1 for bot)
            y_score: Dictionary mapping model names to predicted probabilities
            title: Plot title
            save_path: Path to save the plot (if None, will not save)
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Plot precision-recall curve for each model
        for name, scores in y_score.items():
            precision, recall, _ = precision_recall_curve(y_true, scores)
            ax.plot(recall, precision, label=f"{name}")
            
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(title)
        ax.legend(loc="lower left")
        
        if save_path:
            fig.savefig(save_path, bbox_inches='tight')
            logger.info(f"Saved precision-recall curve to {save_path}")
        
        return fig
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                            title: str = "Confusion Matrix", 
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot confusion matrix.
        
        Args:
            y_true: Array of true labels (0 for human, 1 for bot)
            y_pred: Array of predicted labels
            title: Plot title
            save_path: Path to save the plot (if None, will not save)
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(8, 6), dpi=self.dpi)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Plot heatmap
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(title)
        ax.set_xticklabels(['Human', 'Bot'])
        ax.set_yticklabels(['Human', 'Bot'])
        
        if save_path:
            fig.savefig(save_path, bbox_inches='tight')
            logger.info(f"Saved confusion matrix to {save_path}")
        
        return fig
    
    def plot_expert_comparison(self, metrics: Dict[str, Dict[str, float]],
                             metric_name: str = "accuracy",
                             title: Optional[str] = None,
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot comparison of expert performance.
        
        Args:
            metrics: Dictionary mapping expert names to metric dictionaries
            metric_name: Name of the metric to compare
            title: Plot title (if None, will use metric name)
            save_path: Path to save the plot (if None, will not save)
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Extract values for the specified metric
        experts = []
        values = []
        
        for expert_name, expert_metrics in metrics.items():
            if metric_name in expert_metrics:
                experts.append(expert_name)
                values.append(expert_metrics[metric_name])
        
        # Sort by value
        sorted_idx = np.argsort(values)
        experts = [experts[i] for i in sorted_idx]
        values = [values[i] for i in sorted_idx]
        
        # Plot horizontal bar chart
        ax.barh(experts, values, color=sns.color_palette(self.color_palette, len(experts)))
        
        ax.set_xlim([0, 1.0])
        ax.set_xlabel(metric_name.capitalize())
        ax.set_title(title or f"Expert Comparison - {metric_name.capitalize()}")
        
        # Add value labels
        for i, v in enumerate(values):
            ax.text(v + 0.01, i, f"{v:.3f}", va='center')
        
        if save_path:
            fig.savefig(save_path, bbox_inches='tight')
            logger.info(f"Saved expert comparison to {save_path}")
        
        return fig
    
    def plot_metric_by_threshold(self, threshold_metrics: List[Tuple[float, Dict[str, float]]],
                              metrics_to_plot: List[str] = ["accuracy", "precision", "recall", "f1"],
                              title: str = "Metrics by Threshold",
                              save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot metrics as a function of threshold.
        
        Args:
            threshold_metrics: List of (threshold, metrics) tuples
            metrics_to_plot: List of metric names to include
            title: Plot title
            save_path: Path to save the plot (if None, will not save)
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Extract thresholds and metrics
        thresholds = [t for t, _ in threshold_metrics]
        
        # Plot each metric
        for metric_name in metrics_to_plot:
            metric_values = [m.get(metric_name, 0) for _, m in threshold_metrics]
            ax.plot(thresholds, metric_values, marker='o', label=metric_name.capitalize())
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Threshold')
        ax.set_ylabel('Metric Value')
        ax.set_title(title)
        ax.legend()
        ax.grid(True)
        
        if save_path:
            fig.savefig(save_path, bbox_inches='tight')
            logger.info(f"Saved threshold metrics to {save_path}")
        
        return fig
    
    def plot_calibration_curve(self, calibration_data: List[Dict[str, float]],
                             title: str = "Confidence Calibration",
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot calibration curve comparing predicted confidence to actual accuracy.
        
        Args:
            calibration_data: List of dictionaries with bin info
            title: Plot title
            save_path: Path to save the plot (if None, will not save)
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Extract bin data
        bin_centers = [(d["bin_start"] + d["bin_end"]) / 2 for d in calibration_data]
        accuracies = [d["accuracy"] for d in calibration_data]
        confidences = [d["confidence"] for d in calibration_data]
        sample_counts = [d["samples"] for d in calibration_data]
        
        # Calculate size for scatter points based on sample count
        sizes = [50 * (count / max(sample_counts)) + 10 for count in sample_counts]
        
        # Plot points
        scatter = ax.scatter(confidences, accuracies, s=sizes, alpha=0.7, 
                          c=confidences, cmap='viridis', label='Bin')
        
        # Plot perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Predicted Confidence')
        ax.set_ylabel('Observed Accuracy')
        ax.set_title(title)
        ax.legend()
        ax.grid(True)
        
        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Confidence')
        
        if save_path:
            fig.savefig(save_path, bbox_inches='tight')
            logger.info(f"Saved calibration curve to {save_path}")
        
        return fig
    
    def plot_bot_type_comparison(self, bot_type_metrics: Dict[str, Dict[str, float]],
                               metric_name: str = "f1",
                               title: Optional[str] = None,
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot comparison of performance across different bot types.
        
        Args:
            bot_type_metrics: Dictionary mapping bot types to metric dictionaries
            metric_name: Name of the metric to compare
            title: Plot title (if None, will use metric name)
            save_path: Path to save the plot (if None, will not save)
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Extract values for the specified metric
        bot_types = []
        values = []
        
        for bot_type, metrics in bot_type_metrics.items():
            if metric_name in metrics:
                # Format bot type name
                display_name = bot_type.replace('_', ' ').title()
                bot_types.append(display_name)
                values.append(metrics[metric_name])
        
        # Sort by value
        sorted_idx = np.argsort(values)
        bot_types = [bot_types[i] for i in sorted_idx]
        values = [values[i] for i in sorted_idx]
        
        # Plot horizontal bar chart
        bars = ax.barh(bot_types, values, color=sns.color_palette(self.color_palette, len(bot_types)))
        
        ax.set_xlim([0, 1.0])
        ax.set_xlabel(metric_name.capitalize())
        ax.set_title(title or f"Bot Type Comparison - {metric_name.capitalize()}")
        
        # Add value labels
        for i, v in enumerate(values):
            ax.text(v + 0.01, i, f"{v:.3f}", va='center')
        
        if save_path:
            fig.savefig(save_path, bbox_inches='tight')
            logger.info(f"Saved bot type comparison to {save_path}")
        
        return fig
    
    def plot_cross_validation_results(self, cv_results: Dict[str, Any],
                                   metrics_to_plot: List[str] = ["accuracy", "precision", "recall", "f1"],
                                   title: str = "Cross-Validation Results",
                                   save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot cross-validation results across folds.
        
        Args:
            cv_results: Cross-validation results dictionary
            metrics_to_plot: List of metric names to include
            title: Plot title
            save_path: Path to save the plot (if None, will not save)
            
        Returns:
            Matplotlib figure
        """
        # Extract fold metrics
        fold_metrics = cv_results.get("fold_metrics", [])
        if not fold_metrics:
            logger.warning("No fold metrics found in CV results")
            return None
        
        # Create subplot for each metric
        n_metrics = len(metrics_to_plot)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 6), dpi=self.dpi)
        
        # If only one metric, axes is not a list
        if n_metrics == 1:
            axes = [axes]
        
        # Plot each metric
        for i, metric_name in enumerate(metrics_to_plot):
            ax = axes[i]
            
            # Extract values for this metric across folds
            fold_nums = [m["fold"] for m in fold_metrics]
            metric_values = [m["overall_metrics"].get(metric_name, 0) for m in fold_metrics]
            
            # Plot bars
            ax.bar(fold_nums, metric_values)
            
            # Add average line
            avg_value = cv_results.get("avg_metrics", {}).get(metric_name, 0)
            ax.axhline(y=avg_value, color='r', linestyle='--', 
                    label=f'Avg: {avg_value:.3f}')
            
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('Fold')
            ax.set_ylabel(metric_name.capitalize())
            ax.set_title(f"{metric_name.capitalize()} by Fold")
            ax.set_xticks(fold_nums)
            ax.legend()
            
            # Add value labels
            for j, v in enumerate(metric_values):
                ax.text(j + 1, v + 0.02, f"{v:.3f}", ha='center')
        
        plt.tight_layout()
        fig.suptitle(title, fontsize=16, y=1.05)
        
        if save_path:
            fig.savefig(save_path, bbox_inches='tight')
            logger.info(f"Saved cross-validation results to {save_path}")
        
        return fig
    
    def generate_summary_dashboard(self, benchmark_results: Dict[str, Any],
                                save_dir: Optional[str] = None) -> Dict[str, plt.Figure]:
        """
        Generate a comprehensive dashboard of visualizations from benchmark results.
        
        Args:
            benchmark_results: Benchmark results dictionary
            save_dir: Directory to save plots (if None, will not save)
            
        Returns:
            Dictionary mapping plot names to figures
        """
        if save_dir is None:
            save_dir = self.output_dir
        
        dashboard = {}
        
        try:
            # Extract key data
            overall_metrics = benchmark_results.get("overall_metrics", {})
            expert_metrics = benchmark_results.get("expert_metrics", {})
            threshold_metrics = benchmark_results.get("threshold_metrics", {}).get("all", [])
            bot_type_metrics = benchmark_results.get("bot_type_metrics", {})
            calibration = benchmark_results.get("calibration", [])
            
            # Generate visualizations
            if expert_metrics:
                # Expert comparison for different metrics
                for metric in ["accuracy", "precision", "recall", "f1", "auc"]:
                    if all(metric in metrics for metrics in expert_metrics.values()):
                        fig = self.plot_expert_comparison(
                            expert_metrics, metric, 
                            title=f"Expert Comparison - {metric.capitalize()}",
                            save_path=f"{save_dir}/expert_comparison_{metric}.png" if save_dir else None
                        )
                        dashboard[f"expert_comparison_{metric}"] = fig
            
            if threshold_metrics:
                # Metrics by threshold
                fig = self.plot_metric_by_threshold(
                    threshold_metrics,
                    metrics_to_plot=["accuracy", "precision", "recall", "f1"],
                    title="Metrics by Threshold",
                    save_path=f"{save_dir}/metrics_by_threshold.png" if save_dir else None
                )
                dashboard["metrics_by_threshold"] = fig
            
            if bot_type_metrics:
                # Bot type comparison for different metrics
                for metric in ["accuracy", "precision", "recall", "f1"]:
                    if all(metric in metrics for metrics in bot_type_metrics.values()):
                        fig = self.plot_bot_type_comparison(
                            bot_type_metrics, metric,
                            title=f"Bot Type Detection Performance - {metric.capitalize()}",
                            save_path=f"{save_dir}/bot_type_comparison_{metric}.png" if save_dir else None
                        )
                        dashboard[f"bot_type_comparison_{metric}"] = fig
            
            if calibration:
                # Calibration curve
                fig = self.plot_calibration_curve(
                    calibration,
                    title="Confidence Calibration",
                    save_path=f"{save_dir}/calibration_curve.png" if save_dir else None
                )
                dashboard["calibration_curve"] = fig
            
            logger.info(f"Generated {len(dashboard)} summary dashboard visualizations")
            return dashboard
        
        except Exception as e:
            logger.error(f"Error generating summary dashboard: {str(e)}", exc_info=True)
            return dashboard