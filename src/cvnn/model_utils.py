"""
Model building utilities for consistent model creation across experiments.
"""

# Standard library imports
from typing import Any, Callable, Dict

# Third-party imports
import torch.nn as nn
import torchinfo

# Local imports
from cvnn.config_utils import get_model_params
from cvnn.models import AutoEncoder, LatentAutoEncoder
from cvnn.models.models import UNet
from cvnn.utils import setup_logging

logger = setup_logging(__name__)

# Model Registry for task-specific model builders
MODEL_REGISTRY: Dict[str, Callable] = {}


def register_model_builder(task_name: str):
    """Decorator to register model building functions."""

    def decorator(func: Callable):
        MODEL_REGISTRY[task_name] = func
        return func

    return decorator


def build_autoencoder(
    cfg: Dict[str, Any], model_class: str = "AutoEncoder"
) -> nn.Module:
    """
    Build an autoencoder model based on configuration.

    Args:
        cfg: Configuration dictionary
        model_class: Class name ("AutoEncoder" or "LatentAutoEncoder")

    Returns:
        Configured autoencoder model
    """
    model_params = get_model_params(cfg)

    # Rename input_channels to num_channels for AutoEncoder compatibility
    model_params["num_channels"] = model_params.pop("input_channels")
    # Rename patch_size to input_size for AutoEncoder compatibility
    model_params["input_size"] = model_params.pop("patch_size")

    if model_class == "AutoEncoder":
        # Remove latent_dim if present (not needed for basic AutoEncoder)
        model_params.pop("latent_dim", None)
        model_params.pop("num_classes", None)
        return AutoEncoder(**model_params)

    elif model_class == "LatentAutoEncoder":
        # latent_dim is required for LatentAutoEncoder
        if "latent_dim" not in model_params:
            raise ValueError("LatentAutoEncoder requires latent_dim in config")
        model_params.pop("num_classes", None)
        return LatentAutoEncoder(**model_params)

    else:
        raise ValueError(f"Unknown autoencoder class: {model_class}")


def build_unet(cfg: Dict[str, Any]) -> nn.Module:
    """
    Build a UNet model for segmentation based on configuration.

    Args:
        cfg: Configuration dictionary

    Returns:
        Configured UNet model
    """
    model_params = get_model_params(cfg)

    # Rename input_channels to num_channels for UNet compatibility
    model_params["num_channels"] = model_params.pop("input_channels")

    # UNet requires num_classes - check multiple possible sources
    if "num_classes" not in model_params:
        # Try to get from model config directly
        model_cfg = cfg.get("model", {})
        num_classes = (
            model_cfg.get("num_classes") or 
            model_cfg.get("inferred_num_classes")
        )
        if num_classes:
            model_params["num_classes"] = num_classes
        else:
            raise ValueError("UNet requires num_classes in config (expected in model.num_classes or model.inferred_num_classes)")

    # UNet-specific parameter mapping
    model_params["input_size"] = model_params.pop("patch_size")

    # Remove autoencoder-specific parameters
    model_params.pop("latent_dim", None)

    return UNet(**model_params)


def build_classifier(cfg: Dict[str, Any]) -> nn.Module:
    """
    Build a simple MLP classifier for classification tasks.

    Args:
        cfg: Configuration dictionary

    Returns:
        Configured SimpleClassifier model
    """
    model_params = get_model_params(cfg)
    num_features = model_params.get("num_features", 256)
    num_classes = model_params.get("num_classes", 10)

    class SimpleClassifier(nn.Module):
        """Simple MLP classifier for classification tasks."""

        def __init__(self, num_features: int, num_classes: int):
            """Initialize the classifier.

            Args:
                num_features: Number of input features.
                num_classes: Number of output classes.
            """
            super().__init__()
            self.classifier = nn.Sequential(
                nn.Linear(num_features, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(32, num_classes),
            )

        def forward(self, x):
            """Forward pass through the classifier.

            Args:
                x: Input tensor.

            Returns:
                torch.Tensor: Classification logits.
            """
            # Flatten input if needed
            if x.dim() > 2:
                x = x.view(x.size(0), -1)
            return self.classifier(x)

    return SimpleClassifier(num_features, num_classes)


def build_model_from_config(cfg: Dict[str, Any], task: str) -> nn.Module:
    """
    Build appropriate model based on task and configuration.

    Args:
        cfg: Configuration dictionary
        task: Task name ("reconstruction", "segmentation", etc.)

    Returns:
        Configured model for the task
    """
    logger.info(f"Building model for task: {task}")

    if task == "reconstruction":
        model_class = cfg.get("model", {}).get("class", "AutoEncoder")
        return build_autoencoder(cfg, model_class)

    elif task == "segmentation":
        return build_unet(cfg)

    elif task == "generation":
        # Generation typically uses autoencoders
        return build_autoencoder(cfg, "AutoEncoder")

    elif task == "classification":
        return build_classifier(cfg)

    else:
        raise ValueError(f"Unknown task for model building: {task}")


def get_model_summary(model: nn.Module, input_size: tuple, device: str = "cpu") -> str:
    """
    Get a string summary of the model architecture.

    Args:
        model: PyTorch model
        input_size: Input tensor size (channels, height, width)
        device: Device for summary computation

    Returns:
        String representation of model summary
    """
    try:
        summary = torchinfo.summary(
            model, input_size=input_size, device=device, verbose=0
        )
        return str(summary)
    except ImportError:
        logger.warning("torchinfo not available, using simple summary")
        return f"Model: {model.__class__.__name__}\nParameters: {sum(p.numel() for p in model.parameters()):,}"
    except Exception as e:
        logger.warning(f"Could not generate model summary: {e}")
        return f"Model: {model.__class__.__name__}"
