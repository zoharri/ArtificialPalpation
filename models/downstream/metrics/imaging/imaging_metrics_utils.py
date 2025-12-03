import torch


def compute_confusion_matrix(predicted: torch.Tensor, gt: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Compute a confusion matrix for predicted and ground truth tensors.

    Parameters:
        predicted (torch.Tensor): Predicted class indices
        gt (torch.Tensor): Ground truth class indices
        num_classes (int): Number of classes

    Returns:
        torch.Tensor: Confusion matrix of shape [num_classes, num_classes]
                      Rows = true classes, Columns = predicted classes
    """
    # Flatten the tensors
    predicted_flat = predicted.view(-1)  # makes the tensor a row of numbers
    gt_flat = gt.view(-1)

    # Initialize confusion matrix
    confusion_matrix = torch.zeros((1, num_classes, num_classes), dtype=torch.float32)
    # Fill confusion matrix
    for true_class in range(num_classes):
        for pred_class in range(num_classes):
            mask = (gt_flat == true_class) & (predicted_flat == pred_class)
            confusion_matrix[0, true_class, pred_class] = torch.sum(mask).float() / gt.shape[0]

    return confusion_matrix


def compute_loc_error(predicted: torch.Tensor, gt: torch.Tensor, predicted_lumps: torch.Tensor,
                      true_lumps: torch.Tensor, lump_value: int = 2) -> torch.Tensor:
    """
    Compute the location error between predicted and ground truth tensors.

    Parameters
    ----------
    predicted (torch.Tensor): Predicted class indices
    gt (torch.Tensor): Ground truth class indices
    predicted_lumps (torch.Tensor): the amount of lumps present in the predicted tensor, shape [B].
    true_lumps (torch.Tensor): the amount of lumps present in the ground truth tensor, shape [B].
    lump_idx (int): the index of the lump

    Returns
    -------
    loc_error (torch.Tensor): A tensor which holds the loc_error between predicted and ground truth tensors.

    """

    B = gt.shape[0]
    loc_error = 0
    for i in range(B):
        if true_lumps[i] == 0 and predicted_lumps[i] == 0:
            continue  # perfect guess

        if true_lumps[i] == 0 or predicted_lumps[i] == 0:
            loc_error += 10  # penalty
            continue

        pred_center = torch.argwhere(predicted[i] == lump_value).float().mean(dim=0)
        real_center = torch.argwhere(gt[i] == lump_value).float().mean(dim=0)

        H, W = gt.shape[-2:]

        pred_center = mri_pixel_to_mm(pred_center, W, H)
        real_center = mri_pixel_to_mm(real_center, W, H)

        loc_error += torch.linalg.norm(pred_center - real_center).item()

    loc_error = torch.tensor(loc_error, dtype=torch.float32)

    return loc_error


def mri_pixel_to_mm(com: torch.Tensor, new_size_x: int, new_size_y: int) -> torch.Tensor:
    """
    Correct the coordinates for voxel spacing differences
    in MRI scans so that distances are expressed in millimeters.

    Parameters
    ----------
    com : (torch.Tensor): holds the point in the (y, x) format.
    new_size_x : (int)
        The size of the volume along the x-axis (width) in voxels.
    new_size_y : (int)
        The size of the volume along the y-axis (height) in voxels.

    Returns
    -------
    new_com (torch.Tensor):
        Corrected coordinates in millimeters (y_mm, x_mm).
    """

    original_size = 80

    # the numbers are the distance between pixel in the mri scans
    new_spacing_x = 0.763889 * (original_size / new_size_x)
    new_spacing_y = 0.763889 * (original_size / new_size_y)

    y, x = com
    new_com = (y * new_spacing_y, x * new_spacing_x)

    new_com = torch.tensor(new_com, dtype=torch.float32)

    return new_com
