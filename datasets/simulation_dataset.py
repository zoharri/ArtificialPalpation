import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class SimulationDataset(Dataset):
    def __init__(self, hdf5_file, num_trajs=1, include_radius=False, close_to_lump_trajs=False, only_healthy=False,
                 only_sick=False, no_images=False, dont_permute_trajs=False, shift_trial_origin=True,
                 uniform_dist_traj_subsample=False, random_first_traj_index=False, orig_label=False, num_last_steps=-1):
        self.file_path = hdf5_file
        self.all_keys = None
        self.num_trajs = num_trajs
        self.include_radius = include_radius
        self.close_to_lump_trajs = close_to_lump_trajs
        self.only_healthy = only_healthy
        self.only_sick = only_sick
        self.no_images = no_images
        self.dont_permute_trajs = dont_permute_trajs
        self.shift_trial_origin = shift_trial_origin
        self.uniform_dist_traj_subsample = uniform_dist_traj_subsample
        self.random_first_traj_index = random_first_traj_index
        self.orig_label = orig_label
        self.num_last_steps = num_last_steps

        with h5py.File(self.file_path, 'r') as file:
            # assert 'version' in file.attrs, "Attribute 'version' not found in HDF5 file."
            # assert file.attrs['version'] == 3, "Invalid version. Expected version 3."
            all_num_trajs = [attr for attr in file[list(file.keys())[0]].attrs.keys() if "num_trajs" in attr]
            self.num_trials = len(all_num_trajs)
            self.all_keys = []
            for k, group in file.items():
                if group['label'][()] != 2 and self.only_sick or group['label'][()] != 1 and self.only_healthy:
                    continue
                for i in range(self.num_trials):
                    self.all_keys.append(f"{k}_{i}")

    def get_sequence_length(self):
        idx = 0
        totlen = 0
        with h5py.File(self.file_path, 'r') as file:
            group_key = self.all_keys[idx]
            trial_number = int(group_key.split('_')[-1])
            experiment_key = group_key[:-(len(str(trial_number)) + 1)]
            group = file[experiment_key]
            for i in range(self.num_trajs):
                if self.num_last_steps == -1:
                    totlen += group.attrs[f'num_images_{trial_number}_{i}']
                else:
                    totlen += min(group.attrs[f'num_images_{trial_number}_{i}'], self.num_last_steps)
        return totlen

    def __len__(self):
        return len(self.all_keys)

    def __getitem__(self, idx):
        with h5py.File(self.file_path, 'r') as file:
            if idx >= len(self.all_keys):
                raise IndexError("Index out of range")

            group_key = self.all_keys[idx]
            trial_number = int(group_key.split('_')[-1])
            experiment_key = group_key[:-(len(str(trial_number)) + 1)]
            group = file[experiment_key]
            # Load images and vectors; we load the first trajectory and then a random selection of the remaining trajectories
            if self.num_trajs > group.attrs[f'num_trajs_{trial_number}']:
                raise ValueError(
                    f"num_trajs ({self.num_trajs}) is greater than the number of trajectories in the group ({group.attrs['num_trajs']}).")

            if self.close_to_lump_trajs:
                # get lump center
                lump_center = np.array(group[f'lump_center_{trial_number}'][:])
                # get last location of each trajectory
                all_locations = np.mean(np.array([group[f'vectors_{trial_number}_{i}'][:][-1, :16] for i in
                                                  range(group.attrs[f'num_trajs_{trial_number}'])]), axis=1)
                all_distances = np.linalg.norm(all_locations - lump_center, axis=1)
                traj_indices = np.argsort(all_distances)[:self.num_trajs]
                # shuffle the indices
                if not self.dont_permute_trajs:
                    traj_indices = np.random.permutation(traj_indices).tolist()
            else:
                if self.uniform_dist_traj_subsample:
                    # determenistic sampling (uniform jumps)
                    start_index = np.random.randint(0, group.attrs[
                        f'num_trajs_{trial_number}'] - self.num_trajs) if self.random_first_traj_index else 0
                    jumps = group.attrs[f'num_trajs_{trial_number}'] / self.num_trajs
                    traj_indices = [start_index + int(i * jumps) for i in range(self.num_trajs)]
                else:
                    start_index = np.random.randint(0, group.attrs[
                        f'num_trajs_{trial_number}'] - self.num_trajs) if self.random_first_traj_index else 0
                    traj_indices = list(range(start_index, self.num_trajs))
                if not self.dont_permute_trajs:
                    traj_indices = np.random.permutation(traj_indices).tolist()

            vectors = [torch.tensor(group[f'vectors_{trial_number}_{traj_indices[i]}'][:]) for i in
                       range(self.num_trajs)]
            for i in range(len(vectors)):
                if self.num_last_steps != -1:
                    curr_num_vectors = min(vectors[i].shape[0], self.num_last_steps)
                    vectors[i] = vectors[i][-curr_num_vectors:]
            vectors = torch.cat(vectors, dim=0)
            locations = vectors[:, :16]
            if self.shift_trial_origin:
                # mean over dim 0, 1
                mean_location = torch.mean(locations, dim=(0, 1), keepdim=True)
                locations = locations - mean_location

            forces = vectors[:, 16:]
            if not self.no_images:
                images = []
                for i in range(self.num_trajs):
                    num_images = group.attrs[f'num_images_{trial_number}_{traj_indices[i]}']
                    traj_images = [torch.tensor(group[f'image_{trial_number}_{traj_indices[i]}_{j}'][:]) for j in
                                   range(num_images)]
                    if self.num_last_steps != -1:
                        curr_num_images = min(num_images, self.num_last_steps)
                        traj_images = traj_images[-curr_num_images:]
                    images.extend(traj_images)
                images = torch.stack(images)
            else:
                images = torch.zeros(1, 1, 1, 1)
            label = torch.tensor(group['label'][:])
            # turn label 2 into 1
            if not self.orig_label:
                label = torch.where(label == 2, torch.tensor(1), label)
            model_images = torch.tensor(group[f'model_imaging_{trial_number}'][:])
            model_images = self.semantic_classification_image(model_images)
            if self.include_radius:
                radius = torch.tensor(group['radius'][()])
            else:
                radius = torch.tensor([0])
        return images, locations, forces, label, model_images, radius, experiment_key

    def semantic_classification_image(self, batch_images):
        # Constants for classification
        # Assuming image pixel values are scaled between 0 and 1
        background_threshold = 250 / 256
        object_threshold = 0 / 256
        lump_threshold = 0 / 256

        # Assuming batch_images is in shape [B, C, H, W]
        # Extract R, G, B channels
        if len(batch_images.shape) == 3:
            R = batch_images[0, :, :]
            G = batch_images[1, :, :]
            B = batch_images[2, :, :]
        else:
            R = batch_images[:, 0, :, :]
            G = batch_images[:, 1, :, :]
            B = batch_images[:, 2, :, :]

        # Initialize classification tensor with zeros (background)
        classification = torch.zeros_like(R, dtype=torch.long)

        # Object classification (predominantly blue and above object_threshold)
        is_object = (B >= G) & (B > R) & (B > object_threshold)
        classification[is_object] = 1

        # Lump classification (predominantly green and above lump_threshold)
        is_lump = (G > B) & (G > R) & (G > lump_threshold)
        classification[is_lump] = 2

        # Background classification (high RGB values indicating white or near white)
        is_background = (R > background_threshold) & (G > background_threshold) & (B > background_threshold)
        classification[is_background] = 0

        return classification

    def get_split(self, split_type: str):
        if split_type == "random":
            train_size = int(0.8 * len(self))
            return list(range(train_size)), list(range(train_size, len(self)))
        else:
            raise ValueError(f"Unknown split type for sim dataset: {split_type}")
