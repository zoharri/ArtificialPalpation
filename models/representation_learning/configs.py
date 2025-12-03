"""
Configuration classes for representation learning models.

This module defines the configuration dataclasses used by representation learning models.
The main configuration class is RepresentationModelConfig which contains all the
parameters needed to configure different types of representation learning models.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class RepresentationModelConfig:
    """
    Configuration class for representation learning models.
    """
    name: str  # Name of the representation learning model
    force_size: int  # Size of the force vectors
    locations_size: int  # Size of the location vectors
    representation_size: int  # Size of the hidden state/representation
    gru_tbptt_step_size: Optional[int]  # Step size for truncated backpropagation through time (ReconstructionModel)
    input_num_random_samples: int  # Number of random samples for input (ReconstructionModel)
    reconstruction_num_random_samples: int  # Number of random samples for reconstruction (ReconstructionModel)
    pe_max_freq_log2: int  # Maximum frequency for positional encoding (ReconstructionModel)
    force_predictor_hidden: int  # Size of the hidden layer in the force predictor (ReconstructionModel)
    decoder_input_embed_dim: int  # Embedding dimension for the decoder input (ReconstructionModel)
    gru_input_embed_dim: int  # Embedding dimension for the GRU input (ReconstructionModel)
    consistency_reg_weight: float  # Weight for consistency regularization (ReconstructionModel)
    mask_percentage: float  # Percentage of the trajectory to mask, 0 to disable (ReconstructionModel)
    trajectory_level_mask: bool  # Use trajectory-level masking instead of step-level (ReconstructionModel)
    dropout: float = 0  # Dropout rate
    trajectory_length: int = -1  # Length of each trajectory, populated automatically (ReconstructionModel)
    arch: str = "gru"  # Architecture type (e.g., "gru", "transformer") (ReconstructionModel)
    transformer_num_layers: int = 1  # Number of layers in the transformer (ReconstructionModel)
    transformer_nhead: int = 1  # Number of attention heads in the transformer (ReconstructionModel)
    transformer_ff_dim: int = 128  # Dimension of the feedforward network in the transformer (ReconstructionModel)
    transformer_use_causal_mask: bool = False  # Use causal mask in the transformer (ReconstructionModel)
    forcemap_use_last_force: Optional[int] = None  # Number of last forces to use for heatmap generation (ForceMapModel)
    bandwidth_scale: float = 0.2  # Bandwidth scaling factor for KDE (ForceMapModel)
    image_size: int = 128  # Size of the output image (assumed square) (ForceMapModel)
    forcemap_use_norm: bool = False  # Whether to normalize forces (ForceMapModel)
