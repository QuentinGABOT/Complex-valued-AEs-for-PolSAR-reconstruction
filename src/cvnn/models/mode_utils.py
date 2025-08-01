"""
Mode-aware utilities for handling real vs complex-valued model components.

This module provides centralized functions for selecting appropriate
components (activations, losses, normalizations, initializations) based
on the layer_mode parameter. It ensures seamless switching between
PyTorch and torchcvnn components while maintaining API consistency.

Key Functions:
    is_real_mode: Check if layer mode uses real arithmetic
    get_activation: Get mode-appropriate activation function
    get_loss_function: Get mode-appropriate loss function
    get_normalization: Get mode-appropriate normalization layer
    init_weights_mode_aware: Initialize weights appropriately for mode
    validate_layer_mode: Validate layer mode parameter

Usage Example:
    >>> from cvnn.models.mode_utils import get_activation, is_real_mode
    >>> activation = get_activation("ReLU", layer_mode="complex")
    >>> is_real = is_real_mode("real")
"""

# Standard library imports
import warnings
from typing import Optional, Union

# Third-party imports
import torch
import torch.nn as nn
import torchcvnn.nn.modules as c_nn


def is_real_mode(layer_mode: str) -> bool:
    """Check if the layer mode corresponds to real-valued operations.

    Real modes use standard PyTorch components, while complex modes
    use torchcvnn components that handle complex arithmetic.

    Args:
        layer_mode (str): The layer mode to check. Must be one of:
            - "real": Standard real-valued neural networks
            - "complex": Full complex-valued networks
            - "split": Split complex representations

    Returns:
        bool: True if mode uses real-valued operations (real),
              False if complex-valued operations (complex/split)

    Raises:
        ValueError: If layer_mode is not one of the supported modes

    Example:
        >>> is_real_mode("real")
        True
        >>> is_real_mode("complex")
        False
    """
    if layer_mode == "real":
        return True
    elif layer_mode in ["complex", "split"]:
        return False
    else:
        raise ValueError(
            f"Invalid layer_mode '{layer_mode}'. Must be one of: real, complex, split"
        )


def get_activation(activation_name: str, layer_mode: str) -> nn.Module:
    """Get appropriate activation function based on layer mode.

    Automatically selects between PyTorch and torchcvnn activations
    based on the layer mode. Uses generic names that work across modes.

    Args:
        activation_name (str): Generic activation name. Supported:
            - "ReLU": Maps to ReLU (real) or modReLU (complex)
            - "Tanh": Maps to Tanh (real) or Tanh (complex)
            - "Sigmoid": Maps to Sigmoid (real) or Sigmoid (complex)
            - "LeakyReLU": Maps to LeakyReLU (real) or LeakyReLU (complex)
            - "CReLU": Only available for complex modes
        layer_mode (str): The layer mode (real/complex/split)

    Returns:
        nn.Module: Instantiated activation module appropriate for the mode

    Raises:
        ValueError: If activation_name is unknown or incompatible with layer_mode

    Example:
        >>> # Real mode gets torch.nn.ReLU
        >>> activation = get_activation("ReLU", layer_mode="real")
        >>> # Complex mode gets torchcvnn.nn.modules.modReLU
        >>> activation = get_activation("ReLU", layer_mode="complex")
    """
    # Mapping from generic/complex activation names to real equivalents
    COMPLEX_TO_REAL_ACTIVATIONS = {
        "modReLU": "ReLU",
        "CReLU": "ReLU",
        "zReLU": "ReLU",
        "modTanh": "Tanh",
        "modSigmoid": "Sigmoid",
        "modELU": "ELU",
        "modLeakyReLU": "LeakyReLU",
        # Direct mappings for standard activations
        "ReLU": "ReLU",
        "Tanh": "Tanh",
        "Sigmoid": "Sigmoid",
        "ELU": "ELU",
        "LeakyReLU": "LeakyReLU",
        "GELU": "GELU",
        "Swish": "SiLU",  # Swish is SiLU in PyTorch
        "Mish": "Mish",
    }

    if is_real_mode(layer_mode):
        # For real modes, use PyTorch activations
        if activation_name in COMPLEX_TO_REAL_ACTIVATIONS:
            real_activation_name = COMPLEX_TO_REAL_ACTIVATIONS[activation_name]
            try:
                activation_cls = getattr(nn, real_activation_name)
                return activation_cls()
            except AttributeError:
                raise ValueError(
                    f"Real activation '{real_activation_name}' not found in torch.nn. "
                    f"Available activations: {list(COMPLEX_TO_REAL_ACTIVATIONS.values())}"
                )
        else:
            # Unknown activation - check if it's available in real form
            try:
                activation_cls = getattr(nn, activation_name)
                warnings.warn(
                    f"Using real activation '{activation_name}' directly for real mode '{layer_mode}'. "
                    f"Consider using a mapped activation from: {list(COMPLEX_TO_REAL_ACTIVATIONS.keys())}"
                )
                return activation_cls()
            except AttributeError:
                raise ValueError(
                    f"Activation '{activation_name}' not found for real mode '{layer_mode}'. "
                    f"Available mappings: {list(COMPLEX_TO_REAL_ACTIVATIONS.keys())} "
                    f"or standard PyTorch activations."
                )
    else:
        # For complex modes, use torchcvnn activations
        try:
            activation_cls = getattr(c_nn, activation_name)
            return activation_cls()
        except AttributeError:
            # Check if we can map to a complex activation
            if activation_name in COMPLEX_TO_REAL_ACTIVATIONS:
                # This is a standard activation, but user wants complex mode
                warnings.warn(
                    f"Activation '{activation_name}' may not be available in torchcvnn. "
                    f"For complex mode '{layer_mode}', consider using complex activations like 'modReLU'."
                )
            raise ValueError(
                f"Complex activation '{activation_name}' not found in torchcvnn. "
                f"Available complex activations depend on your torchcvnn installation."
            )


def get_loss_function(
    loss_name: str, layer_mode: str, ignore_index: Optional[int] = None
) -> nn.Module:
    """Get appropriate loss function based on layer mode.

    Automatically selects between PyTorch and torchcvnn loss functions
    based on the layer mode. Uses generic names for consistency.

    Args:
        loss_name (str): Generic loss function name. Supported:
            - "MSE": Mean Squared Error loss
            - "L1": L1/Mean Absolute Error loss
            - "CrossEntropy": Cross-entropy loss (real modes only)
            - "BCE": Binary Cross-Entropy loss
            - "BCEWithLogits": BCE with logits loss
            - "Huber": Huber/Smooth L1 loss
        layer_mode (str): The layer mode (real/complex/split)
        ignore_index (Optional[int]): Index to ignore for losses that support it (e.g., CrossEntropy)

    Returns:
        nn.Module: Instantiated loss function appropriate for the mode

    Raises:
        ValueError: If loss_name is unknown or incompatible with layer_mode

    Example:
        >>> # Real mode gets torch.nn.MSELoss
        >>> loss = get_loss_function("MSE", layer_mode="real")
        >>> # Complex mode gets torchcvnn.nn.modules.ComplexMSELoss
        >>> loss = get_loss_function("MSE", layer_mode="complex")
        >>> # CrossEntropy with ignore_index
        >>> loss = get_loss_function("CrossEntropy", layer_mode="real", ignore_index=0)

    Note:
        Some losses (like CrossEntropy) may only be available for certain modes.
        The function will raise an error for incompatible combinations.
    """
    # Mapping from generic loss names to specific implementations
    LOSS_MAPPINGS = {
        "MSE": {"complex": "ComplexMSELoss", "real": "MSELoss"},
        "L1": {"complex": "ComplexL1Loss", "real": "L1Loss"},
        "CrossEntropy": {
            "complex": "ComplexCrossEntropyLoss",
            "real": "CrossEntropyLoss",
        },
        "BCE": {"complex": "ComplexBCELoss", "real": "BCELoss"},
        "BCEWithLogits": {
            "complex": "ComplexBCEWithLogitsLoss",
            "real": "BCEWithLogitsLoss",
        },
        "Huber": {"complex": "ComplexHuberLoss", "real": "HuberLoss"},
    }

    if loss_name not in LOSS_MAPPINGS:
        raise ValueError(
            f"Unknown loss '{loss_name}'. "
            f"Available losses: {list(LOSS_MAPPINGS.keys())}"
        )

    loss_mapping = LOSS_MAPPINGS[loss_name]

    if is_real_mode(layer_mode):
        # Use PyTorch loss functions for real modes
        real_loss_name = loss_mapping["real"]
        try:
            loss_cls = getattr(nn, real_loss_name)
            # Pass ignore_index to losses that support it
            if ignore_index is not None and loss_name == "CrossEntropy":
                return loss_cls(ignore_index=ignore_index)
            else:
                return loss_cls()
        except AttributeError:
            raise ValueError(f"Real loss '{real_loss_name}' not found in torch.nn")
    else:
        # Use torchcvnn loss functions for complex modes
        complex_loss_name = loss_mapping["complex"]
        try:
            loss_cls = getattr(c_nn, complex_loss_name)
            # Pass ignore_index to complex losses that support it
            if ignore_index is not None and loss_name == "CrossEntropy":
                return loss_cls(ignore_index=ignore_index)
            else:
                return loss_cls()
        except AttributeError:
            # Try to suggest alternative or provide helpful error
            available_losses = [name for name in dir(c_nn) if "Loss" in name]
            raise ValueError(
                f"Complex loss '{complex_loss_name}' not found in torchcvnn. "
                f"Available complex losses in torchcvnn: {available_losses}"
            )


def get_normalization(
    norm_type: str,
    layer_mode: str,
    num_features: int,
    normalized_shape: Optional[tuple] = None,
) -> nn.Module:
    """Get appropriate normalization layer based on layer mode.

    Automatically selects between PyTorch and torchcvnn normalization
    layers based on the layer mode and input tensor requirements.

    Args:
        norm_type (str): Type of normalization. Supported:
            - "batch": Batch normalization
            - "layer": Layer normalization
            - "instance": Instance normalization
            - "group": Group normalization
            - "none" or None: No normalization (returns Identity)
        layer_mode (str): The layer mode (real/complex/split)
        num_features (int): Number of features/channels for normalization
        normalized_shape (Optional[tuple]): Shape for LayerNorm. If None,
            defaults to (num_features,)

    Returns:
        nn.Module: Instantiated normalization layer appropriate for the mode

    Raises:
        ValueError: If norm_type is unknown

    Example:
        >>> # Real mode gets torch.nn.BatchNorm2d
        >>> norm = get_normalization("batch", "real", num_features=64)
        >>> # Complex mode gets torchcvnn.nn.modules.BatchNorm2d
        >>> norm = get_normalization("batch", "complex", num_features=64)

    Note:
        The function automatically handles the different parameter names
        and requirements between PyTorch and torchcvnn normalization layers.
    """

    if num_features is None:
        raise ValueError(
            f"num_features is required for normalization type '{norm_type}'"
        )

    if norm_type is None or norm_type.lower() == "none":
        return nn.Identity()

    # Mapping from normalization types to implementations
    NORM_MAPPINGS = {
        "batch": {
            "complex": lambda: c_nn.BatchNorm2d(num_features=num_features),
            "real": lambda: nn.BatchNorm2d(num_features=num_features),
        },
        "layer": {
            "complex": lambda: c_nn.LayerNorm(
                normalized_shape=normalized_shape or (num_features,)
            ),
            "real": lambda: nn.LayerNorm(
                normalized_shape=normalized_shape or (num_features,)
            ),
        },
        "instance": {
            "complex": lambda: c_nn.InstanceNorm2d(num_features=num_features),
            "real": lambda: nn.InstanceNorm2d(num_features=num_features),
        },
        "group": {
            "complex": lambda: c_nn.GroupNorm(
                num_groups=min(32, num_features), num_channels=num_features
            ),
            "real": lambda: nn.GroupNorm(
                num_groups=min(32, num_features), num_channels=num_features
            ),
        },
    }

    norm_type_lower = norm_type.lower()
    if norm_type_lower not in NORM_MAPPINGS:
        raise ValueError(
            f"Unknown normalization type '{norm_type}'. "
            f"Available types: {list(NORM_MAPPINGS.keys())}"
        )

    norm_mapping = NORM_MAPPINGS[norm_type_lower]

    if is_real_mode(layer_mode):
        return norm_mapping["real"]()
    else:
        return norm_mapping["complex"]()


def get_downsampling(
    downsampling: str, layer_mode: str = "complex", factor: int = 2
) -> nn.Module:
    """Retrieve downsampling layer by name with mode awareness.

    Args:
        downsampling: Downsampling method (e.g., "maxpool", "avgpool", "none")
        layer_mode: Layer mode to determine real vs complex implementation
        factor: Downsampling factor

    Returns:
        Instantiated downsampling module
    """
    from .mode_utils import is_real_mode

    if downsampling is None or downsampling == "none":
        return nn.Identity()

    is_real = is_real_mode(layer_mode)

    if downsampling == "maxpool":
        return (
            nn.MaxPool2d(kernel_size=2, stride=factor)
            if is_real
            else c_nn.MaxPool2d(kernel_size=2, stride=factor)
        )
    elif downsampling == "avgpool":
        return (
            nn.AvgPool2d(kernel_size=2, stride=factor)
            if is_real
            else c_nn.AvgPool2d(kernel_size=2, stride=factor)
        )
    else:
        raise ValueError(f"Unsupported downsampling method: {downsampling}")


def get_upsampling(
    upsampling: str,
    layer_mode: str = "complex",
    factor: int = 2,
    in_channels: int = None,
    out_channels: int = None,
) -> nn.Module:
    """Retrieve upsampling layer by name with mode awareness.

    Args:
        upsampling: Upsampling method (e.g., "nearest", "bilinear", "transpose")
        layer_mode: Layer mode to determine real vs complex implementation
        factor: Upsampling factor
        in_channels: Input channels (required for transpose)
        out_channels: Output channels (required for transpose)

    Returns:
        Instantiated upsampling module
    """
    from .mode_utils import is_real_mode

    if upsampling is None or upsampling == "none":
        return nn.Identity()

    is_real = is_real_mode(layer_mode)

    if upsampling == "nearest":
        return (
            nn.Upsample(scale_factor=factor, mode="nearest")
            if is_real
            else c_nn.Upsample(scale_factor=factor, mode="nearest")
        )
    elif upsampling == "bilinear":
        return (
            nn.Upsample(scale_factor=factor, mode="bilinear", align_corners=True)
            if is_real
            else c_nn.Upsample(scale_factor=factor, mode="bilinear", align_corners=True)
        )
    elif upsampling == "transpose":
        if in_channels is None or out_channels is None:
            raise ValueError(
                "in_channels and out_channels are required for transpose upsampling"
            )
        return (
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=2,
                stride=factor,
            )
            if is_real
            else c_nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=2,
                stride=factor,
            )
        )
    else:
        raise ValueError(f"Unsupported upsampling method: {upsampling}")


def init_weights_mode_aware(module: nn.Module, layer_mode: str) -> None:
    """Initialize weights appropriately based on layer mode.

    Applies mode-appropriate weight initialization strategies:
    - Real modes: Standard PyTorch Kaiming normal initialization
    - Complex modes: Complex-aware Kaiming normal initialization

    Args:
        module (nn.Module): PyTorch module to initialize. Should be a layer
            with weights (Linear, Conv2d, ConvTranspose2d, etc.)
        layer_mode (str): The layer mode (real/complex/split)

    Note:
        Only initializes weights for supported layer types (Linear, Conv2d,
        ConvTranspose2d). Other layer types are ignored.

    Example:
        >>> layer = nn.Linear(10, 5)
        >>> init_weights_mode_aware(layer, "complex")
        >>> # Weights are now initialized for complex-valued operations
    """
    if (
        isinstance(module, nn.Linear)
        or isinstance(module, nn.Conv2d)
        or isinstance(module, nn.ConvTranspose2d)
    ):

        if is_real_mode(layer_mode):
            # Use standard PyTorch initialization for real modes
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.01)
        else:
            # Use complex initialization for complex modes
            c_nn.init.complex_kaiming_normal_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                module.bias.data.fill_(0.01)


def validate_layer_mode(layer_mode: str) -> None:
    """Validate that layer_mode is a supported value.

    Checks that the provided layer_mode is one of the supported modes
    and raises a clear error message if not.

    Args:
        layer_mode (str): The layer mode to validate

    Raises:
        ValueError: If layer_mode is not one of the supported modes

    Example:
        >>> validate_layer_mode("complex")  # No error
        >>> validate_layer_mode("invalid")  # Raises ValueError

    Note:
        This function is automatically called by other mode utilities,
        but can also be used for early validation in user code.
    """
    valid_modes = ["complex", "real", "split"]
    if layer_mode not in valid_modes:
        raise ValueError(
            f"Invalid layer_mode '{layer_mode}'. Must be one of: {', '.join(valid_modes)}"
        )


def validate_mode_consistency(model: nn.Module) -> None:
    """Validate that all components in a model use consistent layer modes.

    This is a placeholder function for future implementation of model-wide
    consistency validation. Could be extended to check that all layers
    in a model use compatible modes and tensor types.

    Args:
        model (nn.Module): PyTorch model to validate

    Note:
        Currently this function does not perform any validation but provides
        a placeholder for future consistency checking features.
    """
    # This is a placeholder for now - we could implement more sophisticated
    # consistency checking if needed
    pass


def get_dropout(
    dropout_prob: float, layer_mode: str, inplace: bool = False
) -> Optional[nn.Module]:
    """Get mode-appropriate dropout layer.

    Returns a dropout layer appropriate for the given layer mode and dropout
    probability. For real modes, uses standard PyTorch dropout. For complex
    modes, uses torchcvnn complex dropout if available, otherwise falls back
    to real dropout with a warning.

    Args:
        dropout_prob (float): Dropout probability. If 0.0, returns None (no dropout).
        layer_mode (str): The layer mode (real/complex/split)
        inplace (bool, optional): Whether to do the operation in-place. Defaults to False.

    Returns:
        Optional[nn.Module]: Dropout layer instance or None if dropout_prob is 0.0

    Raises:
        ValueError: If layer_mode is invalid

    Example:
        >>> # Real mode dropout
        >>> dropout = get_dropout(0.5, layer_mode="real")
        >>> # Complex mode dropout
        >>> dropout = get_dropout(0.3, layer_mode="complex")
        >>> # No dropout
        >>> dropout = get_dropout(0.0, layer_mode="real")  # Returns None

    Note:
        Complex dropout may not be available in all torchcvnn versions.
        In such cases, falls back to real dropout with a warning.
    """
    validate_layer_mode(layer_mode)

    if dropout_prob == 0.0:
        return None

    if dropout_prob < 0.0 or dropout_prob > 1.0:
        raise ValueError(
            f"Dropout probability must be between 0.0 and 1.0, got {dropout_prob}"
        )

    if is_real_mode(layer_mode):
        # Use standard PyTorch dropout for real modes
        return nn.Dropout2d(p=dropout_prob, inplace=inplace)
    else:
        return c_nn.Dropout2d(p=dropout_prob)
