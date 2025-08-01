# Standard library imports
from typing import Optional

# Third-party imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchcvnn.nn.modules as c_nn
from torch import Tensor

# Local imports
from .conv import DoubleConv, SingleConv
from .linear import DoubleLinear, SingleLinear
from .utils import get_downsampling, get_dropout, get_upsampling, is_real_mode
from .learn_poly_sampling.layers import PolyphaseInvariantUp2D, PolyphaseInvariantDown2D


class Down(nn.Module):
    """
    Downscaling block for U-Net architecture.
    
    Applies downsampling (pooling or strided convolution) followed by
    double convolution with optional dropout.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        activation: Activation function module
        layer_mode: Layer mode ('real' or 'complex')
        normalization: Type of normalization ('batch', 'instance', etc.)
        downsampling: Downsampling method ('maxpool', 'avgpool', etc.)
        downsampling_factor: Factor for spatial dimension reduction
        residual: Whether to use residual connections
        dropout: Dropout probability (0.0 to disable)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: nn.Module,
        layer_mode: str,
        projection: Optional[str] = None,
        projection_config: Optional[dict] = None,
        normalization: Optional[str] = None,
        downsampling: Optional[str] = None,
        downsampling_factor: int = 2,
        residual: bool = False,
        dropout: float = 0.0,
        gumbel_softmax: Optional[str] = None,
    ) -> None:
        """Initialize downscaling block."""
        super().__init__()

        if downsampling in ["LPD", "LPD_F"]:
            lpd_conv = DoubleConv(in_ch=in_channels,
                                  out_ch=out_channels,
                                  conv_mode=layer_mode,
                                  activation=activation,
                                  normalization=None,
                                  residual=residual)
            padding_mode = "circular"
        else:
            lpd_conv = None
            padding_mode = "reflect"
        self.down = get_downsampling(downsampling=downsampling, 
                                     projection=projection, 
                                     projection_config=projection_config,
                                     factor=downsampling_factor, 
                                     layer_mode=layer_mode, 
                                     conv=lpd_conv, 
                                     in_channels=in_channels, 
                                     out_channels=out_channels,
                                     gumbel_softmax_type=gumbel_softmax)
        if downsampling is None:
            stride = downsampling_factor
            padding = 1
        else:
            stride = 1
            padding = "same"

        self.conv = DoubleConv(
            in_ch=in_channels,
            out_ch=out_channels,
            conv_mode=layer_mode,
            activation=activation,
            normalization=normalization,
            residual=residual,
            stride=stride,
            padding=padding,
            padding_mode=padding_mode,
        )

        # Add dropout layer if requested
        self.dropout = get_dropout(dropout, layer_mode)

    def forward(self, x: Tensor) -> Tensor:
        if isinstance(self.down, PolyphaseInvariantDown2D):
            x, prob = self.down(x, ret_prob=True)
        else:
            prob = None
            x = self.down(x)
        x = self.conv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x, prob


class Up(nn.Module):
    """
    Upscaling block for U-Net architecture.
    
    Applies upsampling (transpose convolution or interpolation) followed by
    double convolution. Supports skip connections for U-Net style networks.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        activation: Activation function module
        layer_mode: Layer mode ('real' or 'complex')
        upsampling: Upsampling method ('transpose', 'interpolate', etc.)
        skip_connections: Whether to concatenate skip connections
        normalization: Type of normalization ('batch', 'instance', etc.)
        upsampling_factor: Factor for spatial dimension increase
        residual: Whether to use residual connections
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: nn.Module,
        layer_mode: str,
        upsampling: str,
        skip_connections: bool = False,
        normalization: Optional[str] = None,
        upsampling_factor: int = 2,
        residual: bool = False,
        gumbel_softmax: Optional[str] = False,
    ) -> None:
        """Initialize upscaling block."""
        super().__init__()

        self.up = get_upsampling(
            upsampling, 
            layer_mode=layer_mode, 
            factor=upsampling_factor, 
            in_channels=in_channels, 
            out_channels=out_channels,
            gumbel_softmax_type=gumbel_softmax
        )

        # Handle channel count for conv layer input
        if upsampling == "transpose":
            # ConvTranspose already reduces channels to out_channels
            in_channels = out_channels

        if skip_connections:
            in_channels += out_channels

        self.conv = DoubleConv(
            in_ch=in_channels,
            out_ch=out_channels,
            conv_mode=layer_mode,
            activation=activation,
            normalization=normalization,
            residual=residual,
            padding="same",
            padding_mode="circular"
        )

    def forward(self, x1: Tensor, x2: Tensor = None, prob = None) -> Tensor:
        """Apply upsampling and convolution."""
        if isinstance(self.up, PolyphaseInvariantUp2D):
            x1 = self.up(x1, prob=prob)
        else:
            x1 = self.up(x1)
        x = concat(x1, x2)
        return self.conv(x)


class LatentBottleneck(nn.Module):
    """
    Latent bottleneck layer for autoencoder architectures.
    
    Compresses spatial feature maps to a lower-dimensional latent space
    representation and reconstructs back to original dimensions. Supports
    both real and complex-valued data.
    
    Args:
        in_channels: Number of input channels
        latent_dim: Dimensionality of latent space
        input_size: Spatial size of input (assumes square)
        activation: Activation function module
        layer_mode: Layer mode ('real' or 'complex')
        normalization: Type of normalization ('batch', 'instance', etc.)
    """

    def __init__(
        self,
        in_channels: int,
        latent_dim: int,
        input_size: int,
        activation: nn.Module,
        layer_mode: str = "complex",
        normalization: Optional[str] = None,
        dropout: Optional[float] = 0.0,
        downsampling: Optional[str] = None,
        downsampling_factor: int = 2,
        projection: Optional[str] = None,
        projection_config: Optional[dict] = None,
        gumbel_softmax: Optional[str] = None,
        upsampling: Optional[str] = None,
        upsampling_factor: int = 2,
        residual: bool = False,
    ) -> None:
        """Initialize latent bottleneck layer."""
        super().__init__()
        self.input_size = input_size
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.normalization = normalization

        # Global average pooling to reduce spatial dimensions
        # self.avg_pool = c_nn.AvgPool2d(
        #     kernel_size=2,
        #     stride=2,
        #     padding=0,
        #     ceil_mode=True,
        # )
        # lpd_conv = DoubleConv(in_ch=in_channels,
        #                 out_ch=in_channels*2,
        #                 conv_mode=layer_mode,
        #                 activation=activation,
        #                 normalization=None,
        #                 residual=residual)

        # self.down = get_downsampling(downsampling=downsampling, 
        #                              projection=projection, 
        #                              projection_config=projection_config,
        #                              factor=downsampling_factor, 
        #                              layer_mode=layer_mode, 
        #                              conv=lpd_conv, 
        #                              in_channels=in_channels, 
        #                              out_channels=in_channels*2,
        #                              gumbel_softmax_type=gumbel_softmax)

        # self.encoder = DoubleLinear(
        #     in_ch=in_channels,
        #     out_ch=latent_dim//2,
        #     mid_ch=latent_dim,
        #     linear_mode=layer_mode,
        #     activation=activation,
        #     normalization=None,
        # )
        # self.decoder = DoubleLinear(
        #     in_ch=latent_dim//2,
        #     out_ch=in_channels,
        #     mid_ch=latent_dim,
        #     linear_mode=layer_mode,
        #     activation=None,
        #     normalization=None,
        # )
        self.encoder = SingleLinear(
            in_ch=in_channels*input_size*input_size,
            out_ch=in_channels*input_size*input_size,
            linear_mode=layer_mode,
            activation=activation,
            normalization=normalization,
        )
        self.decoder = SingleLinear(
            in_ch=in_channels*input_size*input_size,
            out_ch=in_channels*input_size*input_size,
            linear_mode=layer_mode,
            activation=None,
            normalization=normalization,
        )

        # Unflatten to restore spatial dimensions
        self.unflatten = nn.Unflatten(
            dim=1, unflattened_size=(in_channels, input_size, input_size)
        )
        # self.unflatten = nn.Unflatten(
        #     dim=1, unflattened_size=(in_channels, input_size, input_size)
        # )

        # self.upsample = get_upsampling(
        #     upsampling="bilinear",
        #     layer_mode=layer_mode,
        #     factor=input_size,
        #     in_channels=in_channels,
        #     out_channels=in_channels,
        # )
        # self.up = get_upsampling(
        #     upsampling, 
        #     layer_mode=layer_mode, 
        #     factor=upsampling_factor, 
        #     in_channels=in_channels, 
        #     out_channels=in_channels*2,
        #     gumbel_softmax_type=gumbel_softmax
        # )

        #self.dropout = get_dropout(0.25, layer_mode, use_2d=False)
        self.dropout = None

    def encode(self, x: Tensor) -> Tensor:
        """
        Encode input to latent space.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Latent representation of shape (B, latent_dim)
        """
        # Flatten input
        # x, prob = self.down(x, ret_prob=True)
        encoded = torch.flatten(x, start_dim=1)
        # Apply encoder
        # latent = self.encoder(encoded)
        if self.dropout is not None:
            # Apply dropout to latent space
            encoded = self.dropout(encoded)

        return encoded

    def decode(self, latent: Tensor) -> Tensor:
        """
        Decode latent space back to input shape.
        
        Args:
            latent: Latent tensor of shape (B, latent_dim)
            prob: Probability tensor from downsampling
            
        Returns:
            Reconstructed tensor of shape (B, C, H, W)
        """
        # Apply decoder
        # decoded = self.decoder(latent)
        # Reshape back to spatial dimensions
        output = self.unflatten(latent)

        # output = self.up(output, prob=prob)
        return output


    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the bottleneck.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Reconstructed tensor of shape (B, C, H, W)
        """
        latent = self.encode(x)
        output = self.decode(latent)
        return output
        #return x


def concat(x1, x2):
    """
    Concatenate two tensors with automatic padding for size matching.
    
    Pads x1 to match x2's spatial dimensions, then concatenates along
    the channel dimension. Used in U-Net style architectures.
    
    Args:
        x1: First tensor (CHW format)
        x2: Second tensor (CHW format) or None
        
    Returns:
        Concatenated tensor along channel dimension, or x1 if x2 is None
    """
    if x2 is None:
        return x1
    else:
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return x
