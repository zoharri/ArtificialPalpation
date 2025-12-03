"""
Downstream Predictors

This module contains all downstream models that take representations as input
and predict various outputs (images, regressors, etc.).

Available modules:
- imaging: Image prediction models (ADM, Flow Matching, Transposed Conv, UNet, CNN, AutoCNN)
- quantity_regression: Regression models (Lump Area, Lump Center)
- quantity_classification: Classification models (Phantom Index)
"""

from .downstream_model import DownstreamModel
from .imaging import *
from .quantity_classification import *
from .quantity_regression import *

__all__ = [
    'DownstreamModel',
    # Imaging predictors
    'ImagePred',
    'FlowMatchingImagePred',
    'TransposedConvImagePred',
    'MapRepresentationPred',
    'UNetMapRepPred',
    'CNNMapRepPred',
    'AutoCNNMapRepPred',
    # Quantity regression predictors
    'MLPLumpAreaReg',
    'MLPLumpCenterReg',
    # Quantity classification predictors
    'MLPPhantomIndexClassifier'
]
