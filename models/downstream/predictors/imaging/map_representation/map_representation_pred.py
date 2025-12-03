from abc import abstractmethod, ABC
from typing import Dict, Tuple, Optional

import matplotlib
import matplotlib.pyplot as plt
from torch import nn

from models.downstream.predictors.imaging.image_pred import ImagePred
from models.downstream.utils import *

matplotlib.use('Agg')


class MapRepresentationPred(ImagePred, ABC):
    """
    Abstract base class for models that predict images from 2D representations using various architectures.
    
    This class handles the common logic for:
    - Reshaping flattened representations back to 2D format
    - Computing image prediction losses
    - Visualizing predictions with the original representation
    """

    def __init__(self, config_path: str, representation_size: int):
        super().__init__(config_path, representation_size)
        # Calculate the width and height of the 2D representation
        self.width = self.height = int(np.sqrt(self.representation_size))

        # Initialize the image predictor architecture
        self.image_predictor = self._build_image_predictor()

    @abstractmethod
    def _build_image_predictor(self) -> nn.Module:
        """
        Build the specific image predictor architecture (UNet, CNN, AutoCNN, etc.).
        This method should be overridden in subclasses.
        
        Returns:
            nn.Module: The image predictor network
        """
        raise NotImplementedError()

    def inference(self, representation: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Perform inference by reshaping the representation to 2D and passing through the image predictor.
        
        Args:
            representation: Flattened representation tensor [B, representation_size]
            
        Returns:
            Tuple of predicted images and inference results dictionary
        """
        # Reshape representation to 2D format [B, 1, H, W]
        representation_2d = representation.view(-1, 1, self.height, self.width)

        # Predict images using the specific architecture
        predicted_images = self.image_predictor(representation_2d)

        return predicted_images, {
            "predicted_images": predicted_images,
            "representation": representation_2d
        }

    def visualize_predictions(self, inference_results: Dict[str, torch.Tensor], target: torch.Tensor,
                              **kwargs) -> Optional[plt.Figure]:
        """
        Visualize predictions alongside ground truth and the input representation.
        
        Shows three columns: Model Image, Predicted Image, and Input Representation
        """
        predictions = inference_results["predicted_images"]
        representation = inference_results["representation"]
        num_images_to_plot = min(8, len(predictions))

        fig, axes = plt.subplots(num_images_to_plot, 3, figsize=(18, 4 * num_images_to_plot))

        # Handle single image case
        if num_images_to_plot == 1:
            axes = axes.reshape(1, -1)

        for i, (model_image, predicted_image) in enumerate(
                zip(target[:num_images_to_plot], predictions[:num_images_to_plot])):
            # Plot ground truth model image
            model_image = model_image.float() / (self.config.num_bins - 1)
            axes[i, 0].imshow(model_image.cpu().detach().numpy(), cmap='gray', vmin=0, vmax=1)
            axes[i, 0].set_title('Model Image')
            axes[i, 0].axis('off')

            # Plot predicted image
            predicted_image = undiscretize_image(predicted_image, bins=self.config.num_bins)
            axes[i, 1].imshow(predicted_image.cpu().detach().numpy(), cmap='gray', vmin=0, vmax=1)
            axes[i, 1].set_title('Predicted Image')
            axes[i, 1].axis('off')

            # Plot input representation
            representation_image = representation[i].squeeze(0).cpu().detach().numpy()
            axes[i, 2].imshow(representation_image, cmap='gray')
            axes[i, 2].set_title('Input Representation')
            axes[i, 2].axis('off')

        plt.tight_layout()
        return fig
