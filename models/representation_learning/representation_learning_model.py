from abc import ABC, abstractmethod
from typing import Tuple, Dict

import torch

from models.abstract_model import AbstractModel
from models.representation_learning.configs import RepresentationModelConfig


class RepresentationLearningModel(AbstractModel, ABC):
    """
    Abstract base class for representation learning models.
    
    This class defines the interface that all representation learning models must implement.
    Representation learning models take tactile data (forces and locations) as input and
    learn to produce meaningful representations that can be used by downstream tasks.
    
    Subclasses should implement:
    - inference(): Perform inference to generate representations
    - compute_loss(): Compute the loss for training
    - representation_size: Property returning the size of the learned representation
    """

    @abstractmethod
    def inference(self, forces: torch.Tensor, locations: torch.Tensor, **kwargs) -> Tuple[
        torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Perform inference using the model.
        This method should be overridden in subclasses to implement specific inference logic.
        """
        raise NotImplementedError()

    @abstractmethod
    def compute_loss(self, inference_results: Dict[str, torch.Tensor], forces: torch.Tensor, locations: torch.Tensor,
                     **kwargs) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute the loss for the model.
        This method should be overridden in subclasses to implement specific loss computation logic.
        """
        raise NotImplementedError()

    def forward(self, forces: torch.Tensor, locations: torch.Tensor, **kwargs) -> Tuple[
        torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        representation, inference_results = self.inference(forces, locations, **kwargs)
        loss, loss_info = self.compute_loss(inference_results, forces, locations, **kwargs)
        return representation, loss, loss_info, inference_results

    @property
    @abstractmethod
    def representation_size(self) -> int:
        """
        Returns the size of the representation produced by the model.
        This should be overridden in subclasses to return the specific representation size.
        """
        raise NotImplementedError()

    @property
    def config_type(self):
        """
        Returns the type of configuration for this model.
        """
        return RepresentationModelConfig
