"""
Map Representation Predictors

This module contains predictors that work with 2D map representations (e.g., force maps)
and predict images using various architectures.

Available models:
- MapRepresentationPred: Abstract base class
- UNetMapRepPred: U-Net based predictor with skip connections
- CNNMapRepPred: CNN based predictor with convolutional layers
- AutoCNNMapRepPred: Autoencoder CNN based predictor with latent compression
"""

from .autocnn.autocnn_map_rep_pred import AutoCNNMapRepPred
from .cnn.cnn_map_rep_pred import CNNMapRepPred
from .map_representation_pred import MapRepresentationPred
from .unet.unet_map_rep_pred import UNetMapRepPred

__all__ = [
    'MapRepresentationPred',
    'UNetMapRepPred',
    'CNNMapRepPred',
    'AutoCNNMapRepPred'
]
