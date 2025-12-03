"""
Non-learnable force map model for representation learning.

This module contains the ForceMapModel, which generates force maps using
Kernel Density Estimation (KDE). This is a non-learnable model that doesn't
require training and uses statistical methods to create representations
from tactile data.
"""

from typing import Tuple, Dict

import numpy as np
import torch
from scipy.stats import gaussian_kde

from models.representation_learning.representation_learning_model import RepresentationLearningModel


def kde_force_heatmap(last_pos, forces, grid_res=100, bandwidth_scale=0.3):
    """
    Generate a force heatmap using Kernel Density Estimation (KDE).
    
    This function creates a 2D heatmap representing the distribution of forces
    at different spatial locations using KDE. The heatmap can be used as a
    representation of the tactile interaction.
    
    Args:
        last_pos: Last positions of the tactile interaction
        forces: Force values corresponding to the positions
        grid_res: Resolution of the output grid
        bandwidth_scale: Scaling factor for the KDE bandwidth
        
    Returns:
        A 2D numpy array representing the force heatmap
    """
    positions = last_pos.T  # (2, N)

    # Apply KDE with bandwidth scaling
    kde = gaussian_kde(positions, weights=forces, bw_method=bandwidth_scale)

    x_min, x_max = last_pos[:, 0].min(), last_pos[:, 0].max()
    y_min, y_max = last_pos[:, 1].min(), last_pos[:, 1].max()

    x_grid, y_grid = np.meshgrid(
        np.linspace(x_min, x_max, grid_res),
        np.linspace(y_min, y_max, grid_res)
    )

    grid_coords = np.vstack([x_grid.ravel(), y_grid.ravel()])
    z = kde(grid_coords).reshape(grid_res, grid_res)

    return z


class ForceMapModel(RepresentationLearningModel):
    """
    A non-learnable model that generates force maps using Kernel Density Estimation.
    
    This model doesn't require training and uses statistical methods to create
    representations from tactile data. It generates 2D force heatmaps that
    represent the distribution of forces at different spatial locations.
    
    The model is useful for creating baseline representations or for cases where
    training a neural network is not feasible or desired.
    """

    def __init__(self, config_path: str):
        super(ForceMapModel, self).__init__(config_path)

    def inference(self, forces: torch.Tensor, locations: torch.Tensor, **kwargs) -> Tuple[
        torch.Tensor, Dict[str, torch.Tensor]]:
        last_pos = locations.reshape(locations.size(0), -1, self.config.trajectory_length, locations.size(2),
                                     locations.size(3)).mean(-2)[:, :, -1, :2]
        if self.config.use_norm_forces:
            forces = torch.norm(forces, dim=-1, keepdim=True)
        else:
            forces = forces[:, :, :, -1:]

        if self.config.heatmap_last_forces is not None:
            forces = forces.reshape(forces.size(0), -1, self.config.trajectory_length, forces.size(2))
            forces = forces[:, :, -self.config.heatmap_last_forces:, :]
        else:
            forces = forces
        forces = torch.max(forces.abs(), dim=-1)[0]
        forces = torch.mean(forces, dim=-1)
        force_maps = []
        for i in range(len(last_pos)):
            force_map = kde_force_heatmap(last_pos[i].cpu().numpy(), forces[i].cpu().numpy(),
                                          grid_res=self.config.image_size,
                                          bandwidth_scale=self.config.bandwidth_scale)
            force_map = torch.flip(torch.tensor(force_map, device=self.device),
                                   dims=(-1,)).float()  # Add batch dimension
            force_maps.append(force_map.unsqueeze(0))
        force_map = torch.cat(force_maps, dim=0)
        return force_map.reshape(force_map.shape[0], -1), {"force_map": force_map}

    def compute_loss(self, inference_results: Dict[str, torch.Tensor], forces: torch.Tensor,
                     locations: torch.Tensor,
                     **kwargs) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # In this case, we don't compute a loss as this is not learnable
        return torch.tensor(0.0, device=self.device), {}

    @property
    def representation_size(self) -> int:
        """
        Returns the size of the representation produced by the model.
        In this case, it is the size of the force map.
        """
        return self.config.image_size * self.config.image_size  # Assuming square grid
