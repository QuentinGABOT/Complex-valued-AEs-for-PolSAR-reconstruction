# Standard library imports
from typing import List, Optional

# Third-party imports
import torch
import torch.nn as nn
import torchcvnn.nn.modules as c_nn
from torch import Tensor
from torch.utils.checkpoint import checkpoint_sequential

# Local imports
from .base import BaseAutoEncoder, BaseComplexModel
from .blocks import Down, LatentBottleneck, Up
from .conv import DoubleConv, SingleConv
from .linear import BaseLinear
from .utils import (
    get_activation,
    init_weights_mode_aware,
)

# COMPLEX_DTYPE: torch.dtype = torch.complex64
# REAL_DTYPE: torch.dtype = torch.float32
DOWNSAMPLING_FACTOR = 2
UPSAMPLING_FACTOR = 2

__all__ = [
    "AutoEncoder",
    "LatentAutoEncoder",
    "UNet",
]


class AutoEncoder(BaseAutoEncoder, BaseComplexModel):
    """Autoencoder with downsampling and upsampling layers using complex convolutions."""

    def __init__(
        self,
        num_channels: int,
        num_layers: int,
        channels_width: int,
        input_size: int,
        activation: str,
        upsampling_layer: str,
        layer_mode: str = "complex",
        normalization_layer: Optional[str] = None,
        downsampling_layer: Optional[str] = None,
        residual: bool = False,
        dropout: float = 0.0,
        projection_layer: Optional[str] = None,
        projection_config: Optional[dict] = None,
        gumbel_softmax: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Initialize the autoencoder."""
        super().__init__(
            num_channels=num_channels,
            num_layers=num_layers,
            channels_width=channels_width,
            input_size=input_size,
            activation=activation,
            layer_mode=layer_mode,
            normalization_layer=normalization_layer,
            downsampling_layer=downsampling_layer,
            upsampling_layer=upsampling_layer,
            residual=residual,
            dropout=dropout,
            projection_layer=projection_layer,
            projection_config=projection_config,
            gumbel_softmax=gumbel_softmax,
            **kwargs,
        )
        self.convnet = ConvNet(
            num_channels=num_channels,
            num_layers=num_layers,
            channels_width=channels_width,
            input_size=input_size,
            activation=activation,
            layer_mode=layer_mode,
            normalization_layer=normalization_layer,
            downsampling_layer=downsampling_layer,
            upsampling_layer=upsampling_layer,
            residual=residual,
            skip_connections=False,
            upsampling=True,
            dropout=dropout,
            projection_layer=projection_layer,
            projection_config=projection_config,
            gumbel_softmax=gumbel_softmax,
        )
        self.convnet.apply(lambda m: init_weights_mode_aware(m, layer_mode))

    def encode(self, x: Tensor) -> Tensor:
        """Encode input to latent representation."""
        list_probs = []
        for enc in self.convnet.encoder:
            if isinstance(enc, Down):
                x, prob = enc(x)
            else:
                # For SingleConv or other final layers, just pass through
                x = enc(x)
                prob = None  # No probability for non-downsampling layers
            list_probs.append(prob)
        return x

    def decode(self, z: Tensor) -> Tensor:
        """Decode latent representation to output."""
        for dec in self.convnet.decoder:
            if isinstance(dec, Up):
                z = dec(z, None)
            else:
                # For SingleConv or other final layers, just pass through
                z = dec(z)
        return z

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through encoder and decoder."""
        list_probs = []

        for enc in self.convnet.encoder:
            if isinstance(enc, Down):
                x, prob = enc(x)
            else:
                # For SingleConv or other final layers, just pass through
                x = enc(x)
                prob = None  # No probability for non-downsampling layers

            list_probs.append(prob)

        for dec, prob in zip(self.convnet.decoder, list_probs[::-1]):
            if isinstance(dec, Up):
                x = dec(x, prob=prob)
            else:
                # For SingleConv or other final layers, just pass through
                x = dec(x)
        return x


class LatentAutoEncoder(BaseAutoEncoder, BaseComplexModel):
    """Autoencoder with a latent vector bottleneck between encoder and decoder."""

    def __init__(
        self,
        num_channels: int,
        num_layers: int,
        channels_width: int,
        input_size: int,
        activation: str,
        latent_dim: int,
        upsampling_layer: str,
        layer_mode: str = "complex",
        normalization_layer: str = None,
        downsampling_layer: str = None,
        residual: bool = False,
        dropout: float = 0.0,
        projection_layer: str = None,
        projection_config: Optional[dict] = None,
        gumbel_softmax: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Initialize the latent autoencoder."""
        super().__init__(
            num_channels=num_channels,
            num_layers=num_layers,
            channels_width=channels_width,
            input_size=input_size,
            activation=activation,
            latent_dim=latent_dim,
            layer_mode=layer_mode,
            normalization_layer=normalization_layer,
            downsampling_layer=downsampling_layer,
            upsampling_layer=upsampling_layer,
            residual=residual,
            dropout=dropout,
            projection_layer=projection_layer,
            projection_config=projection_config,
            gumbel_softmax=gumbel_softmax,
            **kwargs,
        )

        self.convnet = ConvNet(
            num_channels=num_channels,
            num_layers=num_layers,
            channels_width=channels_width,
            input_size=input_size,
            activation=activation,
            layer_mode=layer_mode,
            normalization_layer=normalization_layer,
            downsampling_layer=downsampling_layer,
            upsampling_layer=upsampling_layer,
            residual=residual,
            skip_connections=False,
            upsampling=True,
            bottleneck=True,
            latent_dim=latent_dim,
            dropout=dropout,
            projection_layer=projection_layer,
            projection_config=projection_config,
            gumbel_softmax=gumbel_softmax,
        )

        self.convnet.apply(lambda m: init_weights_mode_aware(m, layer_mode))

    def encode(self, x: Tensor) -> Tensor:
        """Encode input to latent representation."""
        
        list_probs = []
        for enc in self.convnet.encoder:
            if isinstance(enc, Down):
                x, prob = enc(x)
            else:
                # For SingleConv or other final layers, just pass through
                x = enc(x)
                prob = None  # No probability for non-downsampling layers
            list_probs.append(prob)
        return x

    def bottleneck(self, x: Tensor) -> Tensor:
        """Get latent representation from bottleneck."""
        return self.convnet.bottleneck(x)

    def decode(self, z: Tensor) -> Tensor:
        """Decode latent representation to output."""
        for dec in self.convnet.decoder:
            if isinstance(dec, Up):
                z = dec(z, None)
            else:
                # For SingleConv or other final layers, just pass through
                z = dec(z)
        return z

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through encoder, bottleneck, and decoder."""
        list_probs = []
        for enc in self.convnet.encoder:
            if isinstance(enc, Down):
                x, prob = enc(x)
            else:
                # For SingleConv or other final layers, just pass through
                x = enc(x)
                prob = None  # No probability for non-downsampling layers
            list_probs.append(prob)

        x = self.convnet.bottleneck(x)

        for dec, prob in zip(self.convnet.decoder, list_probs[::-1]):
            if isinstance(dec, Up):
                x = dec(x, prob=prob)
            else:
                # For SingleConv or other final layers, just pass through
                x = dec(x)
        return x


class UNet(BaseAutoEncoder, BaseComplexModel):
    """UNet model with downsampling and upsampling layers using complex convolutions."""

    def __init__(
        self,
        num_channels: int,
        num_layers: int,
        channels_width: int,
        input_size: int,
        activation: str,
        num_classes: Optional[int] = None,
        layer_mode: str = "complex",
        normalization_layer: str = None,
        downsampling_layer: str = "maxpool",
        upsampling_layer: str = "nearest",
        residual: bool = False,
        projection_layer: str = "amplitude",
        projection_config: Optional[dict] = None,
        dropout: float = 0.0,
        gumbel_softmax: Optional[str] = None,
    ) -> None:
        """Initialize the UNet model."""
        super().__init__(
            num_channels=num_channels,
            num_layers=num_layers,
            channels_width=channels_width,
            input_size=input_size,
            activation=activation,
            layer_mode=layer_mode,
            normalization_layer=normalization_layer,
            downsampling_layer=downsampling_layer,
            upsampling_layer=upsampling_layer,
            residual=residual,
            num_classes=num_classes,
            projection_layer=projection_layer,
            projection_config=projection_config,
            dropout=dropout,
            gumbel_softmax=gumbel_softmax,
        )

        self.convnet = ConvNet(
            num_channels=num_channels,
            num_layers=num_layers,
            channels_width=channels_width,
            input_size=input_size,
            activation=activation,
            num_classes=num_classes,
            layer_mode=layer_mode,
            normalization_layer=normalization_layer,
            downsampling_layer=downsampling_layer,
            upsampling_layer=upsampling_layer,
            residual=residual,
            skip_connections=True,
            upsampling=True,
            projection_layer=projection_layer,
            projection_config=projection_config,
            dropout=dropout,
            gumbel_softmax=gumbel_softmax,
        )

    def encode(self, x: Tensor) -> Tensor:
        """Encode input to latent representation."""
        list_probs = []
        for enc in self.convnet.encoder:
            if isinstance(enc, Down):
                x, prob = enc(x)
            else:
                # For SingleConv or other final layers, just pass through
                x = enc(x)
                prob = None  # No probability for non-downsampling layers
            list_probs.append(prob)
        return self.convnet.bridge(x)

    def decode(self, z: Tensor) -> Tensor:
        """Decode latent representation to output."""
        for dec, in zip(self.convnet.decoder):
            if isinstance(dec, Up):
                z = dec(z, None, None)
            else:
                # For SingleConv or other final layers, just pass through
                z = dec(z)
        return z

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through encoder and decoder."""
        list_skip_connections = []
        list_probs = []

        list_skip_connections.append(None)  # Initialize with None to match list_probs and decoder size
        for enc in self.convnet.encoder:
            if isinstance(enc, Down):
                x, prob = enc(x)
            else:
                # For SingleConv or other final layers, just pass through
                x = enc(x)
                prob = None  # No probability for non-downsampling layers
            list_probs.append(prob)
            list_skip_connections.append(x)

        x, prob = self.convnet.bridge(x)
        list_probs.append(prob)

        for dec, prob, skip in zip(self.convnet.decoder, list_probs[::-1], list_skip_connections[::-1]):
            if isinstance(dec, Up):
                x = dec(x1=x, x2=skip, prob=prob)
            else:
                # For SingleConv or other final layers, just pass through
                x, x_projected = dec(x)
        return x, x_projected


class ConvNet(nn.Module):
    def __init__(
        self,
        num_channels: int,
        num_layers: int,
        channels_width: int,
        input_size: int,
        activation: str,
        num_classes: Optional[int] = None,
        projection_layer: Optional[str] = None,
        projection_config: Optional[dict] = None,
        gumbel_softmax: Optional[str] = None,
        normalization_layer: Optional[str] = None,
        downsampling_layer: Optional[str] = None,
        upsampling_layer: Optional[str] = None,
        dropout: float = 0.0,
        latent_dim: int = 0,
        residual: bool = False,
        skip_connections=True,
        upsampling=True,
        bottleneck=False,
        layer_mode: str = "complex",
    ) -> None:
        """ConvNet with downsampling and upsampling layers using complex convolutions."""
        super().__init__()
        assert bottleneck or upsampling, "Either dense or upsampling must be provided"

        # Encoder with doubzling channels
        current_channels = channels_width
        encoder_layers = []
        bridge_layers = []
        bottleneck_layers = []
        decoder_layers = []

        activation_fn = get_activation(activation, layer_mode)
        encoder_layers.append(
            SingleConv(
                in_ch=num_channels, out_ch=current_channels, conv_mode=layer_mode, projection=None
            )
        )

        for i in range(1, num_layers + 1):
            out_channels = channels_width * 2**i
            if i < num_layers:
                encoder_layers.append(
                    Down(
                        in_channels=current_channels,
                        out_channels=out_channels,
                        activation=activation_fn,
                        layer_mode=layer_mode,
                        normalization=normalization_layer,
                        downsampling=downsampling_layer,
                        downsampling_factor=DOWNSAMPLING_FACTOR,
                        residual=residual,
                        dropout=dropout,
                        projection=projection_layer,
                        projection_config=projection_config,
                        gumbel_softmax=gumbel_softmax,
                    )
                )
            else:
                if skip_connections:
                    bridge_layers.append(
                        Down(
                            in_channels=current_channels,
                            out_channels=out_channels,
                            activation=activation_fn,
                            layer_mode=layer_mode,
                            normalization=normalization_layer,
                            downsampling=downsampling_layer,
                            downsampling_factor=DOWNSAMPLING_FACTOR,
                            residual=residual,
                            dropout=dropout,
                            projection=projection_layer,
                            projection_config=projection_config,
                            gumbel_softmax=gumbel_softmax,
                        )
                    )
                else:
                    encoder_layers.append(
                        Down(
                            in_channels=current_channels,
                            out_channels=out_channels,
                            activation=activation_fn,
                            layer_mode=layer_mode,
                            normalization=normalization_layer,
                            downsampling=downsampling_layer,
                            downsampling_factor=DOWNSAMPLING_FACTOR,
                            residual=residual,
                            dropout=dropout,
                            projection=projection_layer,
                            projection_config=projection_config,
                            gumbel_softmax=gumbel_softmax,
                        )
                    )
            current_channels = out_channels

        self.encoder = nn.Sequential(*encoder_layers)

        self.bridge = nn.Sequential(*bridge_layers)

        if bottleneck:
            # Calculate size at this point after downsampling
            # For LatentAutoEncoder (skip_connections=False), all num_layers Down layers are in encoder
            # For UNet (skip_connections=True), num_layers-1 Down layers are in encoder, 1 in bridge
            num_downsampling_layers = (
                num_layers if not skip_connections else num_layers - 1
            )
            bottleneck_input_size = input_size // (
                DOWNSAMPLING_FACTOR**num_downsampling_layers
            )
            bottleneck_layers.append(
                LatentBottleneck(
                    in_channels=current_channels,
                    input_size=bottleneck_input_size,
                    activation=activation_fn,
                    latent_dim=latent_dim,
                    layer_mode=layer_mode,
                    normalization=normalization_layer,
                    downsampling=downsampling_layer,
                    projection=projection_layer,
                    projection_config=projection_config,
                    gumbel_softmax=gumbel_softmax,
                    upsampling=upsampling_layer,
                    residual=residual,
                    dropout=dropout,
                    upsampling_factor=UPSAMPLING_FACTOR,
                    downsampling_factor=DOWNSAMPLING_FACTOR,
                )
            )

        self.bottleneck = nn.Sequential(*bottleneck_layers)

        if upsampling:
            # Decoder with halving channels
            for i in range(num_layers - 1, -1, -1):
                out_channels = channels_width * 2**i
                decoder_layers.append(
                    Up(
                        in_channels=current_channels,
                        out_channels=out_channels,
                        activation=activation_fn,
                        layer_mode=layer_mode,
                        normalization=normalization_layer,
                        upsampling=upsampling_layer,
                        upsampling_factor=UPSAMPLING_FACTOR,
                        skip_connections=skip_connections,
                        residual=residual,
                        gumbel_softmax=gumbel_softmax,
                    )
                )
                current_channels = out_channels

            if num_classes is not None:
                # Final output layer for classification/segmentation
                decoder_layers.append(
                    SingleConv(
                        in_ch=current_channels,
                        out_ch=num_classes,
                        kernel_size=1,  # 1x1 conv for output
                        padding=0,
                        conv_mode=layer_mode,
                        projection="amplitude" if projection_layer is None else projection_layer, # we need to ensure the output is real-valued
                        projection_config=projection_config,
                    )
                )
            else:
                # Final output layer for reconstruction
                decoder_layers.append(
                    SingleConv(
                        in_ch=current_channels,
                        out_ch=num_channels,
                        kernel_size=1,  # 1x1 conv for output
                        padding=0,
                        conv_mode=layer_mode,
                        projection=None,
                        projection_config=projection_config,
                    )
                )

        self.decoder = nn.Sequential(*decoder_layers)

    def use_checkpointing(self) -> None:
        """Wrap encoder and decoder with checkpointing to save memory."""
        # wrap with CheckpointSequential to preserve Module type
        encoder_modules = list(self.encoder.children())
        decoder_modules = list(self.decoder.children())
        self.encoder = CheckpointSequential(encoder_modules)
        self.bridge = (
            CheckpointSequential(self.bridge)
            if hasattr(self, "bridge") and len(self.bridge) > 0
            else nn.Identity()
        )
        self.bottleneck = (
            CheckpointSequential(self.bottleneck)
            if hasattr(self, "bottleneck") and len(self.bottleneck) > 0
            else nn.Identity()
        )
        self.decoder = CheckpointSequential(decoder_modules)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the ConvNet."""
        # Pass through encoder
        encoded = self.encoder(x)

        # Pass through bridge if it exists
        if hasattr(self, "bridge") and len(self.bridge) > 0:
            encoded = self.bridge(encoded)

        # Pass through bottleneck if it exists
        if hasattr(self, "bottleneck") and len(self.bottleneck) > 0:
            encoded = self.bottleneck(encoded)

        # Pass through decoder if it exists
        if hasattr(self, "decoder") and len(self.decoder) > 0:
            decoded = self.decoder(encoded)
            return decoded
        else:
            return encoded


class CheckpointSequential(nn.Module):
    """Wrap a sequence of modules with checkpointing."""

    def __init__(self, modules: List[nn.Module]) -> None:
        super().__init__()
        self.seq = nn.Sequential(*modules)
        self.length = len(modules)

    def forward(self, x: Tensor) -> Tensor:
        return checkpoint_sequential(self.seq, self.length, x)
