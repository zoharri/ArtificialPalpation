from torch import nn

from models.downstream.predictors.imaging.map_representation.map_representation_pred import MapRepresentationPred


class CNNImagePredictor(nn.Module):
    """
    A CNN architecture for image-to-image prediction using only convolutional layers.
    This model progressively processes the input representation through multiple
    convolutional layers without downsampling, maintaining spatial dimensions.
    """

    def __init__(self, input_channels: int, output_channels: int, hidden_size: int, dropout: float = 0.0):
        super(CNNImagePredictor, self).__init__()

        # Progressive CNN layers with residual-like connections
        self.conv_layers = nn.Sequential(
            # First layer: expand channels
            nn.Conv2d(input_channels, hidden_size // 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),

            # Second layer: increase complexity
            nn.Conv2d(hidden_size // 4, hidden_size // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),

            # Third layer: full hidden size
            nn.Conv2d(hidden_size // 2, hidden_size, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),

            # Fourth layer: maintain complexity
            nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),

            # Fifth layer: reduce complexity
            nn.Conv2d(hidden_size, hidden_size // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),

            # Output layer: predict final image
            nn.Conv2d(hidden_size // 2, output_channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        """
        Forward pass through the CNN layers.
        
        Args:
            x: Input tensor [B, input_channels, H, W]
            
        Returns:
            Output tensor [B, output_channels, H, W]
        """
        return self.conv_layers(x)


class CNNMapRepPred(MapRepresentationPred):
    """
    A representation map predictor using CNN architecture.
    
    This model takes a 2D representation (e.g., force map) and predicts an image
    using a series of convolutional layers that maintain spatial dimensions
    throughout the network.
    """

    def _build_image_predictor(self) -> nn.Module:
        """
        Build a CNN architecture for image prediction.
        
        Returns:
            CNNImagePredictor: CNN model for image prediction
        """
        return CNNImagePredictor(
            input_channels=1,  # Single channel input (grayscale representation)
            output_channels=self.config.num_bins,  # Multi-class output
            hidden_size=self.config.hidden_size,
            dropout=self.config.dropout
        )
