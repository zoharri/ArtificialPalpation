import torch


def discretize_image(image, bins):
    # Assuming image pixel values are scaled between 0 and 1
    discretized = (image * (bins - 1)).long()
    return discretized


def calc_loc_error(predicted_images, real_model_image):
    real_lump = real_model_image == 2
    B, H, W = real_lump.shape
    mask = real_lump.to(torch.float32)

    # Create coordinate grids
    y_coords = torch.arange(H, dtype=torch.float32, device=mask.device)
    x_coords = torch.arange(W, dtype=torch.float32, device=mask.device)
    y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')  # shape [H, W]
    y_grid = y_grid.unsqueeze(0).expand(B, -1, -1)  # [B, H, W]
    x_grid = x_grid.unsqueeze(0).expand(B, -1, -1)  # [B, H, W]
    mass = mask.sum(dim=(1, 2)) + 1e-6  # prevent division by zero
    y_center = (mask * y_grid).sum(dim=(1, 2)) / mass
    x_center = (mask * x_grid).sum(dim=(1, 2)) / mass
    real_locations = torch.stack([y_center, x_center], dim=1)  # shape [B, 2]

    pred_lump = predicted_images == 2
    mask = pred_lump.to(torch.float32)
    mass = mask.sum(dim=(1, 2)) + 1e-6
    y_center = (mask * y_grid).sum(dim=(1, 2)) / mass
    x_center = (mask * x_grid).sum(dim=(1, 2)) / mass
    pred_locations = torch.stack([y_center, x_center], dim=1)  # shape [B, 2]

    return torch.norm(real_locations - pred_locations, dim=1)


def dice_loss(predicted_images, real_model_images, num_bins: int) -> torch.Tensor:
    """
    Compute the Dice loss for image segmentation.
    """
    smooth = 1e-6
    predicted_images = torch.softmax(predicted_images, dim=1)
    predicted_images = predicted_images.view(-1, num_bins, 128, 128)
    real_model_images = torch.nn.functional.one_hot(real_model_images, num_classes=num_bins).permute(0, 3, 1,
                                                                                                     2).float()
    real_model_images = real_model_images.view(-1, num_bins, 128, 128)
    intersection = (predicted_images * real_model_images).sum(dim=(2, 3))
    union = predicted_images.sum(dim=(2, 3)) + real_model_images.sum(dim=(2, 3))
    dice_score = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice_score.mean()  # Return the Dice loss as 1 - Dice score


def is_power_of_2(x):
    return (x != 0) and (x & (x - 1) == 0)
