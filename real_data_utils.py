import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import h5py
import numpy as np
import torch
import torchvision.transforms.functional as TF
from sklearn.neighbors import KNeighborsClassifier


@dataclass
class TrialData:
    traj_press_forces: np.ndarray
    traj_press_angles: np.ndarray
    traj_press_dx: np.ndarray
    traj_press_dy: np.ndarray
    traj_names: List
    xela: np.ndarray
    xela_z: np.ndarray
    xela_norm: np.ndarray
    ee_positions: np.ndarray
    ee_rotations: np.ndarray
    traj_name_indices: dict
    num_sensors: int
    num_trajectories: int
    trajectory_length: int
    lump_center: np.ndarray


@dataclass
class DataFilter:
    press_force: Optional[List[float]] = None
    press_angle: Optional[List[float]] = None
    press_dx: Optional[List[float]] = None
    press_dy: Optional[List[float]] = None


def name_to_lump_center(name: str):
    return np.array([0, 0, 0])


def filter_data(trial_data: TrialData, data_filter: DataFilter) -> TrialData:
    """
    Filter the data based on the data_filter.
    :param trial_data: trial data
    :param data_filter: data filter
    :return: filtered trial data
    """
    traj_indices = np.arange(trial_data.num_trajectories)
    if data_filter.press_force is not None:
        traj_indices = np.where(~np.isin(trial_data.traj_press_forces, data_filter.press_force))[0]
    if data_filter.press_angle is not None:
        traj_indices = np.where(~np.isin(trial_data.traj_press_angles, data_filter.press_angle))[0]
    if data_filter.press_dx is not None:
        traj_indices = np.where(~np.isin(trial_data.traj_press_dx, data_filter.press_dx))[0]
    if data_filter.press_dy is not None:
        traj_indices = np.where(~np.isin(trial_data.traj_press_dy, data_filter.press_dy))[0]

    return TrialData(trial_data.traj_press_forces[traj_indices], trial_data.traj_press_angles[traj_indices],
                     trial_data.traj_press_dx[traj_indices], trial_data.traj_press_dy[traj_indices],
                     [trial_data.traj_names[i] for i in traj_indices], trial_data.xela[traj_indices],
                     trial_data.xela_z[traj_indices], trial_data.xela_norm[traj_indices],
                     trial_data.ee_positions[traj_indices], trial_data.ee_rotations[traj_indices],
                     {name: i for i, name in enumerate(trial_data.traj_names)}, trial_data.num_sensors,
                     len(traj_indices), trial_data.trajectory_length, trial_data.lump_center)


def write_traj_video_from_h5file(h5_file_path: Path, traj_name: str, out_path: Path):
    """
    Extracts camera frames of a specific trajectory from an HDF5 file and writes them to a video file.

    Parameters
    ----------
    h5_file_path : Path
        Path to the input HDF5 file containing trajectory data.
    traj_name : str
        Name of the trajectory group within the HDF5 file, which contains the camera frames and timestamps.
    out_path : Path
        Path to the output video file to be written (should have `.mp4` extension).
    """
    with h5py.File(str(h5_file_path), "r") as f:
        camera_frames = f[traj_name]["camera"]["frames"][:]
        camera_timestamps = f[traj_name]["camera"]["timestamps"][:]
        # calc fps from timestamps (nano seconds)
        fps = 1 / np.mean(np.diff(camera_timestamps))

    # write video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(out_path), fourcc, fps, (camera_frames.shape[2], camera_frames.shape[1]))
    for frame in camera_frames:
        out.write(frame)
    out.release()
    print(f"Video written to {out_path}")


def load_h5_data(h5_file_path: Path, max_traj_length: int = 500, shift_origin: bool = True,
                 rel_forces: bool = True,
                 shift_orientation: bool = False,
                 data_filter: Optional[DataFilter] = None, num_trajectories_to_keep: int = -1,
                 close_to_lump_trajs: bool = False):
    """
    Load data from an HDF5 file containing multiple trajectories.
    h5_file_path : Path
        Path to the input HDF5 file containing trajectory data.
    max_traj_length : int
        Maximum length of the trajectories to pad to.
    shift_origin : bool
        Whether to shift the origin of the end effector positions
    rel_forces : bool
        Whether to use relative forces (forces relative to the first position in each trajectory)
    shift_orientation : bool
        Whether to shift the orientation of the end effector positions
    data_filter : DataFilter
        Data filter to filter the data
    num_trajectories_to_keep : int
        Number of trajectories to subsample
    close_to_lump_trajs : bool
        Whether to select the closest to the lump center
    Returns
    -------
    TrialData
        A dataclass containing the loaded data.
    """

    traj_press_forces = []
    traj_press_angles = []
    traj_press_dx = []
    traj_press_dy = []
    traj_names = []
    all_xela_data = []
    all_ee_positions = []
    all_ee_rotations = []
    file_name_indices = {}
    with h5py.File(str(h5_file_path), "r") as f:
        for i, traj_name in enumerate(f.keys()):
            traj_names.append(traj_name)
            file_name_indices[traj_name] = i
            traj_press_forces.append(f[traj_name].attrs["force"])
            traj_press_angles.append(f[traj_name].attrs["angle"])
            traj_press_dx.append(f[traj_name].attrs["dx"])
            traj_press_dy.append(f[traj_name].attrs["dy"])
            xela_data = f[traj_name]["xela"]["data"][:]
            ee_position = f[traj_name]["robot"]["ee_position"][:]
            ee_rotation = f[traj_name]["robot"]["ee_rotation"][:]  # rotation is a 3x3 matrix

            if len(ee_rotation.shape) == 3:
                x_angle = np.arctan2(ee_rotation[:, 2, 1], ee_rotation[:, 2, 2])
                y_angle = np.arctan2(-ee_rotation[:, 2, 0],
                                     np.sqrt(ee_rotation[:, 2, 1] ** 2 + ee_rotation[:, 2, 2] ** 2))
                z_angle = np.arctan2(ee_rotation[:, 1, 0], ee_rotation[:, 0, 0])
                ee_rotation = np.stack([x_angle, y_angle, z_angle], axis=1)

            assert xela_data.shape[0] <= max_traj_length and ee_position.shape[
                0] <= max_traj_length, f"Trajectory {traj_name} is longer than max_traj_length"

            # xela_data is of shape KxN pad it with last value to get to 500XN
            if xela_data.shape[0] == 0:
                print(f"Trajectory {traj_name} in {h5_file_path} has no data, skipping")
            xela_data = np.pad(xela_data, ((0, max_traj_length - xela_data.shape[0]), (0, 0)), mode="edge")
            ee_position = np.pad(ee_position, ((0, max_traj_length - ee_position.shape[0]), (0, 0)), mode="edge")
            ee_rotation = np.pad(ee_rotation, ((0, max_traj_length - ee_rotation.shape[0]), (0, 0)), mode="edge")

            all_xela_data.append(xela_data)
            all_ee_positions.append(ee_position)
            all_ee_rotations.append(ee_rotation)

    traj_press_forces = np.array(traj_press_forces)
    traj_press_angles = np.array(traj_press_angles)
    traj_press_dx = np.array(traj_press_dx)
    traj_press_dy = np.array(traj_press_dy)
    all_xela_data = np.array(all_xela_data)
    all_xela_data = all_xela_data.reshape(
        (all_xela_data.shape[0], all_xela_data.shape[1], all_xela_data.shape[2] // 3, 3))
    if rel_forces:
        # make forces relative to the first position in each trajectory
        all_xela_data = all_xela_data - all_xela_data[:, :1, :, :]
    all_ee_positions = np.array(all_ee_positions)
    if shift_origin:
        all_ee_positions = all_ee_positions - np.mean(np.mean(all_ee_positions, axis=0, keepdims=True), axis=1,
                                                      keepdims=True)

    all_ee_rotations = np.array(all_ee_rotations)
    if shift_orientation:
        all_ee_rotations = all_ee_rotations - np.mean(np.mean(all_ee_rotations, axis=0, keepdims=True), axis=1,
                                                      keepdims=True)
    all_xela_z = all_xela_data[:, :, :, 2]
    all_xela_norm = np.sqrt(np.sum(all_xela_data ** 2, axis=3))
    num_sensors = all_xela_data.shape[2]
    num_trajectories = len(traj_names)
    trajectory_length = all_xela_data.shape[1]

    trial_data = TrialData(traj_press_forces, traj_press_angles, traj_press_dx, traj_press_dy, traj_names,
                           all_xela_data,
                           all_xela_z, all_xela_norm, all_ee_positions, all_ee_rotations, file_name_indices,
                           num_sensors,
                           num_trajectories,
                           trajectory_length, name_to_lump_center(h5_file_path.stem))

    if data_filter is not None:
        trial_data = filter_data(trial_data, data_filter)
    if num_trajectories_to_keep > 0:
        trial_data = subsample_data(trial_data, num_trajectories_to_keep, close_to_lump_trajs)

    return trial_data


def subsample_data(trial_data: TrialData, num_trajectories: int, close_to_lump_trajs: bool = False):
    """
    Subsample the trial data to num_trajectories, either randomly or by selecting the closest to the lump center.
    :param trial_data: trial data
    :param num_trajectories: number of trajectories to subsample
    :param close_to_lump_trajs: whether to select the closest to the lump center
    :return: subsampled trial data
    """
    if num_trajectories > trial_data.num_trajectories:
        raise ValueError(
            f"num_trajs ({num_trajectories}) is greater than the number of trajectories in the group ({trial_data.num_trajectories}).")
    if close_to_lump_trajs:
        # get lump center
        lump_center = trial_data.lump_center
        all_locations = trial_data.ee_positions[:, -1, :]
        all_distances = np.linalg.norm(all_locations - np.expand_dims(lump_center, axis=0), axis=-1)
        traj_indices = np.argsort(all_distances)[:num_trajectories]
        # shuffle the indices
        traj_indices = np.random.permutation(traj_indices).tolist()
    else:
        traj_indices = np.random.choice(range(0, trial_data.num_trajectories),
                                        size=num_trajectories,
                                        replace=False).tolist()

    return TrialData(trial_data.traj_press_forces[traj_indices], trial_data.traj_press_angles[traj_indices],
                     trial_data.traj_press_dx[traj_indices], trial_data.traj_press_dy[traj_indices],
                     [trial_data.traj_names[i] for i in traj_indices], trial_data.xela[traj_indices],
                     trial_data.xela_z[traj_indices], trial_data.xela_norm[traj_indices],
                     trial_data.ee_positions[traj_indices], trial_data.ee_rotations[traj_indices],
                     {name: i for i, name in enumerate(trial_data.traj_names)}, trial_data.num_sensors,
                     num_trajectories, trial_data.trajectory_length, trial_data.lump_center)


def get_nn_indices(trial_data_1: TrialData, trial_data_2: TrialData, num_neighbors: int = 1):
    """
    Get the nearest neighbors indices of the end effector positions and orientations of trial_data_2 in trial_data_1.
    :param trial_data_1: trial data 1
    :param trial_data_2: trial data 2
    :param num_neighbors: number of neighbors to find
    :return: indices of the nearest neighbors
    """
    ee_pos_1 = trial_data_1.ee_positions.reshape(-1, 3)
    ee_pos_2 = trial_data_2.ee_positions.reshape(-1, 3)

    # rotation matrices
    ee_ori_1 = trial_data_1.ee_rotations.reshape(-1, 3 * 3)
    ee_ori_2 = trial_data_2.ee_rotations.reshape(-1, 3 * 3)

    ee_vec_1 = np.concatenate([ee_pos_1, ee_ori_1], axis=1)
    ee_vec_2 = np.concatenate([ee_pos_2, ee_ori_2], axis=1)

    nn = KNeighborsClassifier(n_neighbors=num_neighbors, algorithm='kd_tree')
    return nn.fit(ee_vec_1, np.arange(ee_vec_1.shape[0])).kneighbors(ee_vec_2, return_distance=False)


def create_mri_image(model_name: str, synthetic: bool = False, mri_images_dir: Path = None,
                     randomly_choose_mri: bool = False, merge01_image_class: bool = False):
    """
    Create an MRI image based on the model name and whether it is synthetic or not.
    :param model_name: Model name in the format 'phantom_insert_orientation'
    :param synthetic: Whether to create a synthetic MRI image or load a real one
    :param mri_images_dir: Directory containing the real MRI images
    :param randomly_choose_mri: Whether to randomly choose an MRI out of two images or choose the first one
    :param Merge01_image_class: merge class 0 1 in the image
    :return: MRI image as a torch tensor
    """
    phantom_name, insert_name, orientation = model_name.split("_")
    orientation = int(orientation)

    if synthetic:
        mri_scan = create_synthetic_mri_image(insert_name)
    else:
        insert_name_lowercase = insert_name.lower()
        if not randomly_choose_mri:
            mri_number = 1
        else:
            inserts_with_missing_idx = {"st14d15", "st14c13"}
            if insert_name_lowercase in inserts_with_missing_idx:
                mri_number = int(np.random.choice([0, 1]))
            else:
                mri_number = int(np.random.choice([0, 1, 2]))

        mri_file_path = mri_images_dir / f"{insert_name_lowercase}_{mri_number}"
        if not mri_file_path.with_suffix('.png').exists():
            print(f"Warning: MRI file {mri_file_path.with_suffix('.png')} does not exist")
            mri_image_orig = np.zeros((128, 128))
        else:
            mri_image_orig = cv2.imread(str(mri_file_path.with_suffix('.png')))
            mri_image_orig = cv2.cvtColor(mri_image_orig, cv2.COLOR_BGR2GRAY)

        mri_image_orig = torch.from_numpy(mri_image_orig)
        mri_scan = torch.zeros(*mri_image_orig.shape)

        if merge01_image_class:
            # The logic before
            mri_scan[(mri_image_orig == 255)] = 0
            mri_scan[(mri_image_orig == 255)] = 2
        else:
            mri_scan[(mri_image_orig == 127) | (mri_image_orig == 255)] = 0
            mri_scan[(mri_image_orig == 127)] = 1
            mri_scan[(mri_image_orig == 255)] = 2

    angle = -orientation * 45
    mri_scan_rotated = rotate_tensor(mri_scan, angle, fill=0)

    return mri_scan_rotated


def create_synthetic_mri_image(insert_name):
    """
    Create a synthetic MRI image based on the insert name.
    :param insert_name: Insert name in the format 'LumpRadiusAngleDistance'
    :return: Synthetic MRI image as a torch tensor
    """
    match = re.match(r"([a-zA-Z]+)(\d+)([a-zA-Z]+)(\d+)", insert_name)
    if match:
        lump_radius = int(match.group(2)) / 2
        if match.group(3) == "D":
            angle = 22.5
        elif match.group(3) == "C":
            angle = 45
        else:
            raise ValueError("String does not match the expected format")
        dist = int(match.group(4))
    else:
        raise ValueError("String does not match the expected format")
    angle = angle - 45  # the 0 orientation is shifted anti-clockwise
    y = -dist * np.cos(np.deg2rad(angle))
    x = dist * np.sin(np.deg2rad(angle))
    # create a synthetic image with a '1' circle of radius 30 center (0,0), '2' circle of radius lump_radius center (x,y) and the rest '0'
    xs = torch.linspace(-40, 40, 128)
    ys = torch.linspace(-40, 40, 128)
    Y, X = torch.meshgrid(ys, xs, indexing='ij')  # corrected indexing for image-like coordinate system

    image = torch.zeros_like(X)

    # Circle 1: radius 30 at (0, 0) -> value 1
    circle1 = (X ** 2 + Y ** 2) <= 30 ** 2
    image[circle1] = 1

    # Circle 2: radius lump_radius at (x, y) -> value 2
    circle2 = ((X - x) ** 2 + (Y - y) ** 2) <= lump_radius ** 2
    image[circle2] = 2

    # Now turn the image to RGB by setting '2' to green '1' to blue and '0' to white
    image_rgb = torch.ones((4, 128, 128))  # white background
    image_rgb[0][(image == 1) | (image == 2)] = 0
    image_rgb[1][image == 1] = 0
    image_rgb[2][image == 2] = 0
    return image_rgb


def rotate_locations(trial_data: TrialData, angle: float) -> TrialData:
    """
    Rotate the end effector positions and orientations by a given angle around the z-axis at the center-of-mass of the trajectory.
    :param trial_data:
        The trial data to rotate.
    :param angle:
        The angle in degrees to rotate the data.
    :return:
        The rotated trial data.
    """

    # calculate the center of mass of the trajectory based of axis 0 and 1
    # set the com to the mean in x but 34.7% in y
    com_x = np.mean(trial_data.ee_positions, axis=(0, 1))[0]
    max_y = np.max(trial_data.ee_positions, axis=(0, 1))[1]
    min_y = np.min(trial_data.ee_positions, axis=(0, 1))[1]
    com_y = min_y + (max_y - min_y) * (1 - 0.347)
    com = np.array([com_x, com_y, 0])
    # create the rotation matrix
    angle = np.deg2rad(angle)
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle), 0],
                                [np.sin(angle), np.cos(angle), 0],
                                [0, 0, 1]])
    # rotate the end effector positions
    rotated_positions = np.dot(trial_data.ee_positions - com, rotation_matrix.T) + com
    # rotate the end effector orientations
    rotated_orientations = np.dot(trial_data.ee_rotations, rotation_matrix.T)
    # create a new trial data object with the rotated data
    rotated_trial_data = TrialData(
        traj_press_forces=trial_data.traj_press_forces,
        traj_press_angles=trial_data.traj_press_angles,
        traj_press_dx=trial_data.traj_press_dx,
        traj_press_dy=trial_data.traj_press_dy,
        traj_names=trial_data.traj_names,
        xela=trial_data.xela,
        xela_z=trial_data.xela_z,
        xela_norm=trial_data.xela_norm,
        ee_positions=rotated_positions,
        ee_rotations=rotated_orientations,
        traj_name_indices=trial_data.traj_name_indices,
        num_sensors=trial_data.num_sensors,
        num_trajectories=trial_data.num_trajectories,
        trajectory_length=trial_data.trajectory_length,
        lump_center=trial_data.lump_center
    )
    return rotated_trial_data


def is_hard_trial(trial_data: TrialData, thresh_num_hard_trajectories: int, thresh_hard_traj_force: int) -> bool:
    forces_diff = np.abs(trial_data.xela_z - trial_data.xela_z[:, :1]).max(axis=(1, 2))
    forces_diff_thresh = forces_diff > thresh_hard_traj_force
    return forces_diff_thresh.sum() > thresh_num_hard_trajectories


def flip_x_locations(trial_data: TrialData) -> TrialData:
    """
    Flip the end effector positions and orientations around the x-axis, including flipping the xela data.
    :param trial_data: The trial data to flip
    :return: The flipped trial data
    """

    # Calc the center of mass and the y axis orientation
    from scipy.spatial.transform import Rotation as R
    positions = trial_data.ee_positions  # shape (N, K, 3)
    orientations = trial_data.ee_rotations  # shape (N, K, 3), Euler angles (XYZ)

    x_axis = np.array([1, 0, 0])  # x-axis in world frame
    y_axis = np.array([0, 1, 0])  # x-axis in world frame
    x_axis = x_axis / np.linalg.norm(x_axis)  # shape (3,)
    y_axis = y_axis / np.linalg.norm(y_axis)
    z_axis = np.cross(x_axis, y_axis)
    z_axis = z_axis / np.linalg.norm(z_axis)

    R_world_to_local = np.stack([x_axis, y_axis, z_axis], axis=1)  # (3, 3)
    R_local_to_world = R_world_to_local.T  # (3, 3)

    # Flatten to (N*K, 3)
    N, K, _ = positions.shape
    positions_flat = positions.reshape(-1, 3)
    orientations_flat = orientations.reshape(-1, 3)

    # --- Flip positions ---
    positions_local = (R_world_to_local.T @ positions_flat.T).T  # (N*K, 3)
    positions_local[:, 0] *= -1
    flipped_positions_flat = (R_local_to_world @ positions_local.T).T  # (N*K, 3)
    flipped_positions = flipped_positions_flat.reshape(N, K, 3)

    # --- Flip orientations ---
    rot = R.from_euler('xyz', orientations_flat)  # (N*K,)
    rot_matrices = rot.as_matrix()  # (N*K, 3, 3)

    # Transform rotations into local frame
    rot_local_matrices = R_world_to_local.T @ rot_matrices @ R_world_to_local  # (N*K, 3, 3)
    rot_local = R.from_matrix(rot_local_matrices)
    eul_local = rot_local.as_euler('xyz')  # (N*K, 3)
    eul_local[:, 0] *= -1
    eul_local[:, 2] *= -1

    # Transform back to world frame
    rot_flipped_local = R.from_euler('xyz', eul_local)
    rot_flipped_local_matrices = rot_flipped_local.as_matrix()
    rot_flipped_world_matrices = R_local_to_world @ rot_flipped_local_matrices @ R_world_to_local
    flipped_orientations_flat = R.from_matrix(rot_flipped_world_matrices).as_euler('xyz')  # (N*K, 3)
    flipped_orientations = flipped_orientations_flat.reshape(N, K, 3)
    flipped_indices = [25, 26, 27, 28, 29, 21, 22, 23, 24, 15, 16, 17, 18, 19, 20]
    flipped_xela = trial_data.xela.copy()

    # Apply the swaps
    for i, j in enumerate(flipped_indices):
        flipped_xela[:, :, i] = trial_data.xela[:, :, j]
        flipped_xela[:, :, j] = trial_data.xela[:, :, i]
    flipped_xela_z = flipped_xela[:, :, :, 2]
    flipped_xela_norm = np.sqrt(np.sum(flipped_xela ** 2, axis=3))

    return TrialData(
        traj_press_forces=trial_data.traj_press_forces,
        traj_press_angles=trial_data.traj_press_angles,
        traj_press_dx=trial_data.traj_press_dx,
        traj_press_dy=trial_data.traj_press_dy,
        traj_names=trial_data.traj_names,
        xela=flipped_xela,
        xela_z=flipped_xela_z,
        xela_norm=flipped_xela_norm,
        ee_positions=flipped_positions,
        ee_rotations=flipped_orientations,
        traj_name_indices=trial_data.traj_name_indices,
        num_sensors=trial_data.num_sensors,
        num_trajectories=trial_data.num_trajectories,
        trajectory_length=trial_data.trajectory_length,
        lump_center=trial_data.lump_center
    )


def rotate_tensor(tensor: torch.Tensor, angle: float, fill: float) -> torch.Tensor:
    """
    Rotate a tensor by the given angle.

    Parameters
    ----------
    tensor : torch.Tensor
        Input image tensor. Shape (C, H, W).
    angle : float
        Rotation angle in degrees. Positive values mean counter-clockwise.
    fill : float
        Value to fill empty pixels after rotation

    Returns
    -------
    torch.Tensor
        Rotated tensor.
    """
    return TF.rotate(tensor.unsqueeze(0), angle, fill=fill).squeeze(0)


def hflip_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """
    Horizontally flip a tensor.

    Parameters
    ----------
    tensor : torch.Tensor
        Input image tensor. Shape (C, H, W).

    Returns
    -------
    torch.Tensor
        Flipped tensor.
    """
    return TF.hflip(tensor.unsqueeze(0)).squeeze(0)
