"""
Configuration utility functions for consistent config access patterns.
"""

# Standard library imports
from pathlib import Path
from typing import Any, Dict, Optional

# Local imports
from cvnn.utils import setup_logging

logger = setup_logging(__name__)


def get_model_params(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract common model parameters from config.

    Returns:
        Dictionary with standardized model parameters
    """
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})

    # Handle input channels with fallback for both naming conventions
    input_channels = (
        data_cfg.get("inferred_input_channels") or 
        model_cfg.get("input_channels") or 
        data_cfg.get("input_channels") or
        2  # Default fallback for testing
    )

    # Handle patch size with fallback
    patch_size = data_cfg.get("patch_size") or model_cfg.get("patch_size")
    if patch_size is None:
        raise ValueError("No patch_size found in config (expected in data.patch_size or model.patch_size)")

    params = {
        "input_channels": input_channels,
        "patch_size": patch_size,
        "num_layers": model_cfg.get("num_layers"),
        "channels_width": model_cfg.get("channels_width"),
        "activation": model_cfg.get("activation"),
        "layer_mode": model_cfg.get("layer_mode"),
        "normalization_layer": model_cfg.get("normalization_layer"),
        "downsampling_layer": model_cfg.get("downsampling_layer"),
        "upsampling_layer": model_cfg.get("upsampling_layer"),
        "residual": model_cfg.get("residual"),
        "dropout": model_cfg.get("dropout"),
        "projection_layer": model_cfg.get("projection_layer", None),
        "projection_config": model_cfg.get("projection", {}),
        "gumbel_softmax": model_cfg.get("gumbel_softmax", None),
    }

    # Add task-specific parameters
    if "inferred_num_classes" in model_cfg:
        params["num_classes"] = model_cfg["inferred_num_classes"]

    if "latent_dim" in model_cfg:
        params["latent_dim"] = model_cfg["latent_dim"]
    
    return params


def get_wandb_config(cfg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Extract WandB configuration if present."""
    logging_cfg = cfg.get("logging", {})
    if "wandb" not in logging_cfg:
        return None
    return logging_cfg["wandb"]


def get_logdir(cfg: Dict[str, Any]) -> Optional[Path]:
    """Extract log directory from config."""
    logging_cfg = cfg.get("logging", {})
    if "logdir" in logging_cfg:
        return Path(logging_cfg["logdir"])
    return None


def validate_required_config_sections(cfg: Dict[str, Any]) -> None:
    """Validate that required config sections are present."""
    required_sections = ["data", "model"]

    for section in required_sections:
        if section not in cfg:
            raise ValueError(f"Missing required config section: {section}")

    # Validate required data fields
    data_required = ["dataset"]
    for field in data_required:
        if field not in cfg["data"]:
            raise ValueError(f"Missing required data config field: {field}")

    # Validate dataset has name
    if "name" not in cfg["data"]["dataset"]:
        raise ValueError("Missing required dataset name in data.dataset.name")


def update_config_with_inferred_values(cfg: Dict[str, Any], **kwargs) -> None:
    """
    Update config with inferred values from data pipeline.

    Args:
        cfg: Configuration dictionary to update
        **kwargs: Key-value pairs to set in appropriate config sections
    """
    # Map of parameter names to their config locations
    location_map = {
        "inferred_input_channels": ("data", "inferred_input_channels"),
        "inferred_num_classes": ("model", "inferred_num_classes"),
    }

    for key, value in kwargs.items():
        if key in location_map:
            section, param = location_map[key]
            if section not in cfg:
                cfg[section] = {}
            cfg[section][param] = value
            logger.debug(f"Updated config: {section}.{param} = {value}")
        else:
            logger.warning(f"Unknown config parameter for update: {key}")
