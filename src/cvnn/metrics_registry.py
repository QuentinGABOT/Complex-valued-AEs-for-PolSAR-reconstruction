"""
Metrics registry for evaluation functions based on task and pipeline type.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple
import torch
import numpy as np

from cvnn.utils import setup_logging

logger = setup_logging(__name__)


# Metrics registry structure:
# {task: {pipeline_type: {metric_name: metric_function}}}
_METRICS_REGISTRY: Dict[str, Dict[str, Dict[str, Callable]]] = {}


def register_metric(
    task: str,
    pipeline_type: str, 
    metric_name: str,
    metric_function: Callable
) -> None:
    """
    Register a metric function for a specific task and pipeline type.
    
    Args:
        task: Task name (e.g., "reconstruction", "segmentation")
        pipeline_type: Pipeline type (e.g., "complex_dual_real", "real_real")
        metric_name: Name of the metric (e.g., "mse", "psnr", "ssim")
        metric_function: Function that computes the metric
    """
    if task not in _METRICS_REGISTRY:
        _METRICS_REGISTRY[task] = {}
    
    if pipeline_type not in _METRICS_REGISTRY[task]:
        _METRICS_REGISTRY[task][pipeline_type] = {}
    
    _METRICS_REGISTRY[task][pipeline_type][metric_name] = metric_function
    logger.debug(f"Registered metric: {task}.{pipeline_type}.{metric_name}")


def get_metric_function(
    task: str,
    pipeline_type: str,
    metric_name: str
) -> Optional[Callable]:
    """
    Get a metric function for specific task and pipeline type.
    
    Args:
        task: Task name
        pipeline_type: Pipeline type  
        metric_name: Metric name
        
    Returns:
        Metric function or None if not found
    """
    try:
        return _METRICS_REGISTRY[task][pipeline_type][metric_name]
    except KeyError:
        logger.warning(
            f"Metric not found: {task}.{pipeline_type}.{metric_name}"
        )
        return None


def get_available_metrics(
    task: str,
    pipeline_type: str
) -> List[str]:
    """
    Get list of available metrics for a task and pipeline type.
    
    Args:
        task: Task name
        pipeline_type: Pipeline type
        
    Returns:
        List of available metric names
    """
    try:
        return list(_METRICS_REGISTRY[task][pipeline_type].keys())
    except KeyError:
        return []


def compute_metrics(
    task: str,
    pipeline_type: str,
    predictions: torch.Tensor,
    targets: torch.Tensor,
    metric_names: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Compute multiple metrics for given predictions and targets.
    
    Args:
        task: Task name
        pipeline_type: Pipeline type
        predictions: Model predictions
        targets: Ground truth targets
        metric_names: List of metrics to compute (all if None)
        
    Returns:
        Dictionary mapping metric names to computed values
    """
    if metric_names is None:
        metric_names = get_available_metrics(task, pipeline_type)
    
    results = {}
    for metric_name in metric_names:
        metric_fn = get_metric_function(task, pipeline_type, metric_name)
        if metric_fn is not None:
            try:
                value = metric_fn(predictions, targets)
                results[metric_name] = float(value)
            except Exception as e:
                logger.error(f"Error computing {metric_name}: {e}")
                results[metric_name] = float('nan')
        else:
            logger.warning(f"Metric {metric_name} not available")
    
    return results


# Standard metric implementations
def mse_metric(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Mean Squared Error for real-valued data."""
    return torch.nn.functional.mse_loss(predictions, targets).item()


def complex_mse_metric(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Mean Squared Error for complex-valued data."""
    diff = predictions - targets
    return torch.mean(torch.abs(diff) ** 2).item()


def psnr_metric(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Peak Signal-to-Noise Ratio for real-valued data."""
    mse = torch.nn.functional.mse_loss(predictions, targets)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse)).item()


def complex_psnr_metric(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Peak Signal-to-Noise Ratio for complex-valued data."""
    mse = complex_mse_metric(predictions, targets)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(1.0 / np.sqrt(mse))


def accuracy_metric(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Classification accuracy."""
    pred_classes = torch.argmax(predictions, dim=1)
    correct = (pred_classes == targets).float()
    return torch.mean(correct).item()


def iou_metric(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Intersection over Union for segmentation."""
    pred_classes = torch.argmax(predictions, dim=1)
    intersection = torch.logical_and(pred_classes, targets)
    union = torch.logical_or(pred_classes, targets)
    iou = torch.sum(intersection) / torch.sum(union)
    return iou.item()


# Additional metric implementations for segmentation
def dice_metric(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Dice coefficient for segmentation."""
    pred_classes = torch.argmax(predictions, dim=1)
    intersection = torch.logical_and(pred_classes, targets).float().sum()
    union = pred_classes.float().sum() + targets.float().sum()
    if union == 0:
        return 1.0
    dice = (2 * intersection) / union
    return dice.item()


def precision_metric(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Precision metric for classification/segmentation."""
    pred_classes = torch.argmax(predictions, dim=1)
    # Calculate macro precision
    unique_classes = torch.unique(targets)
    precision_sum = 0.0
    valid_classes = 0
    
    for cls in unique_classes:
        pred_mask = (pred_classes == cls)
        true_mask = (targets == cls)
        
        true_positive = torch.logical_and(pred_mask, true_mask).float().sum()
        predicted_positive = pred_mask.float().sum()
        
        if predicted_positive > 0:
            precision_sum += (true_positive / predicted_positive).item()
            valid_classes += 1
    
    return precision_sum / valid_classes if valid_classes > 0 else 0.0


def recall_metric(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Recall metric for classification/segmentation."""
    pred_classes = torch.argmax(predictions, dim=1)
    # Calculate macro recall
    unique_classes = torch.unique(targets)
    recall_sum = 0.0
    valid_classes = 0
    
    for cls in unique_classes:
        pred_mask = (pred_classes == cls)
        true_mask = (targets == cls)
        
        true_positive = torch.logical_and(pred_mask, true_mask).float().sum()
        actual_positive = true_mask.float().sum()
        
        if actual_positive > 0:
            recall_sum += (true_positive / actual_positive).item()
            valid_classes += 1
    
    return recall_sum / valid_classes if valid_classes > 0 else 0.0


def f1_metric(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """F1 score metric for classification/segmentation."""
    precision = precision_metric(predictions, targets)
    recall = recall_metric(predictions, targets)
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * (precision * recall) / (precision + recall)


# SSIM implementation for real-valued data
def ssim_metric(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """SSIM metric for real-valued reconstruction."""
    from cvnn.evaluate import ssim  # Import the existing SSIM function
    return ssim(predictions, targets).item()


def complex_ssim_metric(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Complex SSIM metric for complex-valued reconstruction."""
    from cvnn.evaluate import complex_ssim  # Import the existing complex SSIM function
    return complex_ssim(predictions, targets).item()


# Register standard metrics for common tasks
def _register_standard_metrics():
    """Register commonly used metrics for different tasks and pipeline types."""
    
    # Reconstruction metrics for real-valued pipelines
    reconstruction_real_metrics = {
        "mse": mse_metric,
        "psnr": psnr_metric,
        "ssim": ssim_metric,
        "mae": mse_metric,  # For now, use MSE implementation
    }
    
    # Reconstruction metrics for complex-valued pipelines
    reconstruction_complex_metrics = {
        "mse": complex_mse_metric,
        "psnr": complex_psnr_metric,
        "ssim": complex_ssim_metric,
        "mae": complex_mse_metric,  # For now, use complex MSE implementation
    }
    
    # Register for real-valued reconstruction pipelines
    for pipeline in ["real_real", "complex_amplitude_real"]:
        for name, func in reconstruction_real_metrics.items():
            register_metric("reconstruction", pipeline, name, func)
    
    # Register for complex-valued reconstruction pipelines
    for pipeline in ["complex_dual_real", "complex", "split"]:
        for name, func in reconstruction_complex_metrics.items():
            register_metric("reconstruction", pipeline, name, func)
    
    # Segmentation metrics (comprehensive set)
    segmentation_metrics = {
        "accuracy": accuracy_metric,
        "iou": iou_metric,
        "dice": dice_metric,
        "precision": precision_metric,
        "recall": recall_metric,
        "f1": f1_metric,
    }
    
    # Register for all segmentation pipelines
    for pipeline in ["real_real", "complex_amplitude_real", "complex_dual_real", "complex", "split"]:
        for name, func in segmentation_metrics.items():
            register_metric("segmentation", pipeline, name, func)
    
    # Classification metrics (reuse segmentation metrics)
    classification_metrics = {
        "accuracy": accuracy_metric,
        "precision": precision_metric,
        "recall": recall_metric,
        "f1": f1_metric,
    }
    
    # Register for all classification pipelines
    for pipeline in ["real_real", "complex_amplitude_real", "complex_dual_real", "complex", "split"]:
        for name, func in classification_metrics.items():
            register_metric("classification", pipeline, name, func)
    
    # Generation metrics (comprehensive for both real and complex)
    generation_real_metrics = {
        "fid": mse_metric,  # Placeholder - proper FID would need implementation
        "inception_score": mse_metric,  # Placeholder - proper IS would need implementation
        "mse": mse_metric,
    }
    
    generation_complex_metrics = {
        "fid": complex_mse_metric,  # Placeholder for complex FID
        "inception_score": complex_mse_metric,  # Placeholder for complex IS
        "mse": complex_mse_metric,
    }
    
    # Register for real generation pipelines
    for pipeline in ["real_real", "complex_amplitude_real"]:
        for name, func in generation_real_metrics.items():
            register_metric("generation", pipeline, name, func)
    
    # Register for complex generation pipelines
    for pipeline in ["complex_dual_real", "complex", "split"]:
        for name, func in generation_complex_metrics.items():
            register_metric("generation", pipeline, name, func)


# Initialize standard metrics
_register_standard_metrics()


def list_tasks() -> List[str]:
    """List all registered tasks."""
    return list(_METRICS_REGISTRY.keys())


def list_pipeline_types_for_task(task: str) -> List[str]:
    """List all pipeline types registered for a task."""
    return list(_METRICS_REGISTRY.get(task, {}).keys())


def get_metrics_info() -> Dict[str, Any]:
    """
    Get comprehensive information about all registered metrics.
    
    Returns:
        Nested dictionary with task/pipeline/metric organization
    """
    info = {}
    for task in _METRICS_REGISTRY:
        info[task] = {}
        for pipeline in _METRICS_REGISTRY[task]:
            info[task][pipeline] = list(_METRICS_REGISTRY[task][pipeline].keys())
    return info


class MetricsRegistry:
    """
    Unified metrics registry for CVNN evaluation.
    
    Provides a clean, object-oriented interface to the metrics registry
    functions with task-specific configuration and error handling.
    """
    
    def __init__(self, task: str, cfg: Any):
        """
        Initialize MetricsRegistry for a specific task.
        
        Args:
            task: Task name (e.g., "reconstruction", "segmentation")
            cfg: Configuration object with evaluation settings
        """
        self.task = task
        self.cfg = cfg
        
        # Validate task
        if task not in _METRICS_REGISTRY:
            available_tasks = list(_METRICS_REGISTRY.keys())
            raise ValueError(f"Unknown task '{task}'. Available tasks: {available_tasks}")
        
        # Determine pipeline type from config
        self.pipeline_type = self._infer_pipeline_type(cfg)
        
        # Validate pipeline type for task
        available_pipelines = list(_METRICS_REGISTRY[task].keys())
        if self.pipeline_type not in available_pipelines:
            logger.warning(f"Pipeline '{self.pipeline_type}' not found for task '{task}'. "
                         f"Available: {available_pipelines}. Using fallback.")
            self.pipeline_type = available_pipelines[0] if available_pipelines else "real_real"
        
        logger.info(f"MetricsRegistry initialized: task={task}, pipeline={self.pipeline_type}")
    
    def _infer_pipeline_type(self, cfg: Any) -> str:
        """
        Infer pipeline type from configuration.
        
        Args:
            cfg: Configuration object
            
        Returns:
            Pipeline type string
        """
        try:
            # Try to get pipeline type from config
            if hasattr(cfg, 'pipeline_type'):
                return cfg.pipeline_type
            
            # Infer from model configuration
            if hasattr(cfg, 'model'):
                model_cfg = cfg.model
                
                # Check if using complex-valued model
                if hasattr(model_cfg, 'use_complex') and model_cfg.use_complex:
                    return "complex"
                elif hasattr(model_cfg, 'complex_mode') and model_cfg.complex_mode:
                    return "complex"
                
                # Check for specific architecture hints
                if hasattr(model_cfg, 'name') and 'complex' in str(model_cfg.name).lower():
                    return "complex"
            
            # Default to real-valued
            return "real_real"
            
        except Exception as e:
            logger.warning(f"Could not infer pipeline type: {e}. Using default 'real_real'")
            return "real_real"
    
    def get_available_metrics(self) -> List[str]:
        """
        Get list of available metrics for the current task and pipeline.
        
        Returns:
            List of metric names
        """
        return list(_METRICS_REGISTRY[self.task][self.pipeline_type].keys())
    
    def compute_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """
        Compute all configured metrics for predictions vs targets.
        
        Args:
            predictions: Model predictions tensor
            targets: Ground truth targets tensor
            
        Returns:
            Dictionary mapping metric names to computed values
        """
        results = {}
        
        # Get metrics to compute from config
        metrics_to_compute = []
        if hasattr(self.cfg, 'evaluation') and hasattr(self.cfg.evaluation, 'metrics'):
            metrics_to_compute = self.cfg.evaluation.metrics
        else:
            # Use all available metrics if not specified
            metrics_to_compute = self.get_available_metrics()
        
        # Compute each metric
        for metric_name in metrics_to_compute:
            try:
                metric_func = get_metric_function(self.task, self.pipeline_type, metric_name)
                if metric_func is not None:
                    value = metric_func(predictions, targets)
                    results[metric_name] = float(value)
                    logger.debug(f"Computed {metric_name}: {value}")
                else:
                    logger.warning(f"Metric '{metric_name}' not found for {self.task}.{self.pipeline_type}")
            except Exception as e:
                logger.error(f"Error computing metric '{metric_name}': {e}")
                # Continue with other metrics
        
        return results
    
    def compute_single_metric(self, 
                            metric_name: str,
                            predictions: torch.Tensor, 
                            targets: torch.Tensor) -> float:
        """
        Compute a single metric.
        
        Args:
            metric_name: Name of the metric to compute
            predictions: Model predictions tensor
            targets: Ground truth targets tensor
            
        Returns:
            Computed metric value
        """
        metric_func = get_metric_function(self.task, self.pipeline_type, metric_name)
        if metric_func is None:
            raise ValueError(f"Metric '{metric_name}' not available for {self.task}.{self.pipeline_type}")
        
        return float(metric_func(predictions, targets))
    
    def register_custom_metric(self, metric_name: str, metric_function: Callable) -> None:
        """
        Register a custom metric for the current task and pipeline.
        
        Args:
            metric_name: Name of the custom metric
            metric_function: Function that computes the metric
        """
        register_metric(self.task, self.pipeline_type, metric_name, metric_function)
        logger.info(f"Registered custom metric: {self.task}.{self.pipeline_type}.{metric_name}")
    
    def get_task_info(self) -> Dict[str, Any]:
        """
        Get information about the current task configuration.
        
        Returns:
            Dictionary with task information
        """
        return {
            "task": self.task,
            "pipeline_type": self.pipeline_type,
            "available_metrics": self.get_available_metrics(),
            "configured_metrics": getattr(self.cfg.evaluation, 'metrics', []) 
                                if hasattr(self.cfg, 'evaluation') else []
        }
    
    def __repr__(self) -> str:
        """String representation of the MetricsRegistry."""
        return f"MetricsRegistry(task='{self.task}', pipeline='{self.pipeline_type}')"
