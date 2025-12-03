"""
Representation Learning Module

This module provides models for learning representations from tactile data.
It includes both learnable models (like GRU-based reconstruction) and non-learnable models (like force maps).

Structure:
- models/: Contains different model implementations
  - force_reconstruction/: Models that reconstruct forces from locations
    - vec_representation/: Vector-based representation models
  - non_learnable/: Models that don't require training
- positional_encoding/: Utilities for positional encoding
- configs.py: Configuration classes for representation learning models
- representation_learning_model.py: Abstract base class for all representation learning models
"""

from .configs import RepresentationModelConfig
# Import specific model implementations
from .models.force_reconstruction.vec_representation.reconstruction_model import ReconstructionModel
from .models.non_learnable.force_map import ForceMapModel
from .representation_learning_model import RepresentationLearningModel

__all__ = [
    'RepresentationLearningModel',
    'RepresentationModelConfig',
    'ReconstructionModel',
    'ForceMapModel',
]
