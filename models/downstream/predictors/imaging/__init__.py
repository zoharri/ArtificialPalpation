"""
Imaging Predictors

This module contains models that predict images from representations using various architectures.

Available modules:
- image_pred: Abstract base class for image prediction models
- vec_representation: Vector representation based predictors (ADM, Flow Matching, Transposed Conv)
- map_representation: 2D map representation based predictors (UNet, CNN, AutoCNN)
"""

from .image_pred import ImagePred
from .map_representation import MapRepresentationPred, UNetMapRepPred, CNNMapRepPred, AutoCNNMapRepPred
from .vec_representation import FlowMatchingImagePred, TransposedConvImagePred

__all__ = [
    'ImagePred',
    'FlowMatchingImagePred',
    'TransposedConvImagePred',
    'MapRepresentationPred',
    'UNetMapRepPred',
    'CNNMapRepPred',
    'AutoCNNMapRepPred'
]
