"""
Transform registry for automatic transform selection based on configuration.
"""

# Standard library imports
from typing import Any, Dict, List, Optional, Type

# Third-party imports
import torch.nn as nn
import torchcvnn.transforms as c_transforms
from torchvision import transforms

# Local imports
from cvnn.utils import setup_logging

logger = setup_logging(__name__)


# Available transforms by layer mode
_TRANSFORM_REGISTRY: Dict[str, Dict[str, Type]] = {
    "real": {
        "resize": transforms.Resize,
        "normalize": transforms.Normalize,
        "randomhorizontalflip": transforms.RandomHorizontalFlip,
        "randomverticalflip": transforms.RandomVerticalFlip,
        "randomrotation": transforms.RandomRotation,
        "colorjitter": transforms.ColorJitter,
        "centercrop": transforms.CenterCrop,
        "pad": transforms.Pad,
    },
    "complex": {
        "resize": c_transforms.FFTResize,
        "fftresize": c_transforms.FFTResize,
        "spatialresize": c_transforms.SpatialResize, 
        "centercrop": c_transforms.CenterCrop,
        "padifneeded": c_transforms.PadIfNeeded,
        "randomphase": c_transforms.RandomPhase,
        "randomhorizontalflip": transforms.RandomHorizontalFlip,
        "randomverticalflip": transforms.RandomVerticalFlip,
        # Note: Complex normalize doesn't exist in torchcvnn based on the list
    },
}


def register_transform(
    layer_mode: str, transform_name: str, transform_class: Type
) -> None:
    """
    Register a transform for a specific layer mode.

    Args:
        layer_mode: Layer mode ("real" or "complex")
        transform_name: Name of the transform (e.g., "resize", "normalize")
        transform_class: Transform class to register
    """
    if layer_mode not in _TRANSFORM_REGISTRY:
        _TRANSFORM_REGISTRY[layer_mode] = {}

    _TRANSFORM_REGISTRY[layer_mode][transform_name] = transform_class
    logger.debug(f"Registered transform: {layer_mode}.{transform_name}")


def get_transform_class(layer_mode: str, transform_name: str) -> Type:
    """
    Get transform class for a specific layer mode.

    Args:
        layer_mode: Layer mode ("real" or "complex")
        transform_name: Name of transform (e.g., "resize")

    Returns:
        Transform class

    Raises:
        ValueError: If layer mode or transform name is unknown
    """
    # Determine which registry to use based on layer mode
    if layer_mode in ["real", "split"]:  # Real and split use real transforms
        registry_key = "real"
    elif layer_mode == "complex":
        registry_key = "complex"
    else:
        raise ValueError(f"Unknown layer_mode: {layer_mode}")

    if registry_key not in _TRANSFORM_REGISTRY:
        raise ValueError(f"No transforms registered for layer_mode: {layer_mode}")

    transforms_for_mode = _TRANSFORM_REGISTRY[registry_key]
    if transform_name not in transforms_for_mode:
        available = list(transforms_for_mode.keys())
        raise ValueError(
            f"Transform '{transform_name}' not available for layer_mode '{layer_mode}'. "
            f"Available: {available}"
        )

    return transforms_for_mode[transform_name]


def _get_dataset_base_transforms(dataset_type: str, num_channels: int, min_value: float, max_value: float) -> List:
    """Get dataset-specific base transforms based on dataset type."""
    if dataset_type in ["sar", "polsar"]:
        # SAR/PolSAR datasets need PolSAR and LogAmplitude
        return [
            c_transforms.PolSAR(out_channel=num_channels),
            c_transforms.LogAmplitude(max_value=max_value, min_value=min_value),
        ]
    else:
        # Other dataset types - no special preprocessing
        return []


def _get_mode_conversion_transforms(layer_mode: str, real_pipeline_type: Optional[str] = None) -> List:
    """Get mode-specific conversion transforms."""
    if layer_mode in ["complex", "split"]:
        # Complex modes use complex64 tensor
        return [c_transforms.ToTensor(dtype="complex64")]
    
    elif layer_mode == "real":
        if real_pipeline_type == "real_real":
            # Real data, real model
            return [c_transforms.ToTensor(dtype="float32")]
        elif real_pipeline_type == "complex_amplitude_real":
            # Complex -> amplitude -> real
            return [
                c_transforms.ToTensor(dtype="complex64"),
                c_transforms.Amplitude(dtype="float32")
            ]
        elif real_pipeline_type == "complex_dual_real":
            # Complex -> real/imag concatenation -> real
            return [
                c_transforms.ToTensor(dtype="complex64"),
                c_transforms.RealImaginary(dtype="float32")
            ]
        else:
            raise ValueError(f"Unknown real_pipeline_type: {real_pipeline_type}")
    
    else:
        raise ValueError(f"Unknown layer_mode: {layer_mode}")


def build_transform_pipeline(cfg: Dict[str, Any]) -> transforms.Compose:
    """
    Build complete transform pipeline based on configuration.

    Args:
        cfg: Configuration dictionary

    Returns:
        Composed transform pipeline
    """
    # Extract configuration
    dataset_name = cfg["data"]["dataset"]["name"]
    layer_mode = cfg.get("model", {}).get("layer_mode", "complex")
    real_pipeline_type = cfg.get("data", {}).get("real_pipeline_type", None)

    if dataset_name in ["ALOSDataset", "PolSFDataset", "Sethi"]: # need to rework this by using the inferred num_channels variable
        num_channels = 4  # PolSAR datasets
    else:
        num_channels = 3
    
    if dataset_name == "Sethi": # need to create a function that add this information in the config file
        min_value = 4.7*10**-10
        max_value = 3.3*10**-8
    elif dataset_name == "Bretigny":
        min_value = 0.015
        max_value = 2.895
    else:
        min_value = 0.009
        max_value = 0.992
    # Get dataset type - import here to avoid circular import
    try:
        from cvnn.data import get_dataset_info
        dataset_info = get_dataset_info(dataset_name)
        dataset_type = dataset_info.get("type", "unknown")
    except ImportError:
        # Fallback if circular import still exists
        dataset_type = "unknown"
        logger.warning("Could not get dataset info due to import issues")
    
    # Start with dataset-specific base transforms
    transform_list = _get_dataset_base_transforms(dataset_type, num_channels, min_value, max_value)

    # Add mode-specific conversion transforms
    mode_transforms = _get_mode_conversion_transforms(layer_mode, real_pipeline_type)
    transform_list.extend(mode_transforms)
    
    # Add configurable transforms from config
    additional_transforms = cfg.get("data", {}).get("transforms", [])
    for transform_config in additional_transforms:
        transform_name = transform_config["name"].lower()
        transform_params = transform_config.get("params", {})
        
        # Convert size parameter to tuple if it's a list (for transforms that expect tuples)
        if "size" in transform_params and isinstance(transform_params["size"], list):
            transform_params = transform_params.copy()  # Don't modify original
            transform_params["size"] = tuple(transform_params["size"])
        
        try:
            transform_class = get_transform_class(layer_mode, transform_name)
            transform_instance = transform_class(**transform_params)
            transform_list.append(transform_instance)
            logger.debug(f"Added transform: {transform_name} with params: {transform_params}")
        except Exception as e:
            logger.warning(f"Failed to add transform {transform_name}: {e}")
    
    return transforms.Compose(transform_list)


def list_available_transforms(layer_mode: str) -> List[str]:
    """
    List all available transforms for a layer mode.

    Args:
        layer_mode: Layer mode to query ("real" or "complex")

    Returns:
        List of available transform names
    """
    registry_key = "real" if layer_mode in ["real", "split"] else "complex"
    
    if registry_key not in _TRANSFORM_REGISTRY:
        return []

    return list(_TRANSFORM_REGISTRY[registry_key].keys())


def list_supported_layer_modes() -> List[str]:
    """List all supported layer modes."""
    return ["real", "complex", "split"]


def validate_transform_config(cfg: Dict[str, Any]) -> bool:
    """
    Validate transform configuration.

    Args:
        cfg: Configuration dictionary

    Returns:
        True if configuration is valid

    Raises:
        ValueError: If configuration is invalid
    """
    # Check required fields
    if "data" not in cfg:
        raise ValueError("Missing 'data' section in config")
    
    if "dataset" not in cfg["data"] or "name" not in cfg["data"]["dataset"]:
        raise ValueError("Missing dataset name in config")
    
    layer_mode = cfg.get("model", {}).get("layer_mode", "complex")
    if layer_mode not in list_supported_layer_modes():
        raise ValueError(f"Unsupported layer_mode: {layer_mode}")
    
    # Validate real pipeline type if in real mode
    if layer_mode == "real":
        real_pipeline_type = cfg.get("data", {}).get("real_pipeline_type")
        valid_types = ["real_real", "complex_amplitude_real", "complex_dual_real"]
        if real_pipeline_type not in valid_types:
            raise ValueError(f"Invalid real_pipeline_type: {real_pipeline_type}. Must be one of: {valid_types}")
    
    # Validate additional transforms
    additional_transforms = cfg.get("data", {}).get("transforms", [])
    available_transforms = list_available_transforms(layer_mode)
    
    for transform_config in additional_transforms:
        if "name" not in transform_config:
            raise ValueError("Transform config missing 'name' field")
        
        transform_name = transform_config["name"].lower()
        if transform_name not in available_transforms:
            raise ValueError(
                f"Transform '{transform_name}' not available for layer_mode '{layer_mode}'. "
                f"Available: {available_transforms}"
            )
    
    return True


def get_transform_info(layer_mode: str, transform_name: str) -> Dict[str, Any]:
    """
    Get information about a specific transform.

    Args:
        layer_mode: Layer mode ("real" or "complex")
        transform_name: Transform name

    Returns:
        Dictionary with transform information
    """
    try:
        transform_class = get_transform_class(layer_mode, transform_name)
        return {
            "class_name": transform_class.__name__,
            "module": transform_class.__module__,
            "doc": transform_class.__doc__,
            "layer_mode": layer_mode,
        }
    except ValueError:
        return {}