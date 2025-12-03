from dataclasses import dataclass


@dataclass
class DownstreamModelConfig:
    """Configuration for the downstream model."""
    name: str  # Name of downstream model, used for model creation
    dropout: float  # Dropout rate (all models)
    num_bins: int  # Number of output channels (imaging models)
    balance_image_classes: bool  # Whether to balance image classes (imaging models)
    num_channels: int = 256  # Different meaning in different archs (all models)
    input_dropout: float = 0  # Dropout rate for the input representation (all models)
    loss: str = "cross_entropy"  # Loss function to use e.g cross_entropy, focal, dice (all models)
    num_mlp_layers: int = 1  # Number of layers in MLP (quantity classifiers and regressors)
    image_size: int = 128  # Size of the output image, assumed square (imaging models and size predictors)
    flow_steps: int = 1000  # Number of flow steps for the model (Flow matching)
    use_residual_in_unet_flowmatching: bool = False  # Use residual connection in the unet (Flow matching)
