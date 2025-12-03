import os
from typing import Dict

from torch.utils.data import Dataset

from real_data_utils import *


class RealDataset(Dataset):
    def __init__(self, hdf5_dir, num_trajs=1, close_to_lump_trajs=False, filter_data: Optional[DataFilter] = None,
                 max_traj_length=500, synthetic_mri_image=False, sensor_subsample=1, only_z=False, rotation_aug=False,
                 flip_aug=False,
                 no_angles=False, shift_trial_origin=True, shift_trial_orientation=False, include_s3=False,
                 shuffle_keys=False, randomly_choose_mri=False, merge01_image_class=False,
                 removec5=False, removed10=False, removed15=False, filter_hard_trials=None, num_last_steps=-1,
                 rel_forces=False, location_noise_std=0.0, angle_noise_std=0.0):
        self.dir_path = Path(hdf5_dir)
        self.all_keys = None
        self.num_trajs = num_trajs
        self.close_to_lump_trajs = close_to_lump_trajs
        self.filter_data = filter_data
        self.max_traj_length = max_traj_length
        self.synthetic_mri_image = synthetic_mri_image
        self.sensor_subsample = sensor_subsample
        self.subsampled_traj_len = max_traj_length // sensor_subsample
        self.only_z = only_z
        self.rotation_aug = rotation_aug
        self.flip_aug = flip_aug
        self.no_angles = no_angles
        self.shift_trial_origin = shift_trial_origin
        self.shift_trial_orientation = shift_trial_orientation
        self.include_s3 = include_s3
        self.shuffle_keys = shuffle_keys
        self.randomly_choose_mri = randomly_choose_mri
        self.merge01_image_class = merge01_image_class
        self.removec5 = removec5
        self.removed10 = removed10
        self.removed15 = removed15
        self.filter_hard_trials = filter_hard_trials
        self.num_last_steps = num_last_steps
        self.rel_forces = rel_forces
        self.location_noise_std = location_noise_std
        self.angle_noise_std = angle_noise_std

        mri_dir_name = "mri_images"
        self.mri_images_dir = os.path.join(hdf5_dir, mri_dir_name)

        # set all keys to the names of the hdf5 files in the directory
        self.all_keys = []
        for d in os.listdir(hdf5_dir):
            if os.path.isdir(os.path.join(hdf5_dir, d)):
                for f in os.listdir(os.path.join(hdf5_dir, d)):
                    if os.path.isfile(os.path.join(hdf5_dir, d, f)) and f.endswith('.hdf5'):
                        if os.path.join(d, f) in [""]:
                            continue
                        if "S3" in f and not self.include_s3:  # currently S3 doesn't look good
                            continue
                        if ("C5" in f or "c5" in f) and self.removec5:
                            continue
                        if ("D10" in f or "d10" in f) and self.removed10:
                            continue
                        if ("D15" in f or "d15" in f) and self.removed15:
                            continue

                        if self.filter_hard_trials is not None:
                            data = load_h5_data(self.dir_path / os.path.join(d, f),
                                                max_traj_length=self.max_traj_length)
                            if is_hard_trial(data, self.filter_hard_trials, 70):
                                continue

                        self.all_keys.append(os.path.join(d, f))
        # sort the keys
        self.all_keys.sort()

        if self.shuffle_keys:
            np.random.shuffle(self.all_keys)
            self.all_keys = list(self.all_keys)

    def split_first_trial(self):
        """
        We sampled many of the phantoms multiple times, train only on the first trial.
        """
        first_try_indices = []
        second_try_indices = []

        for i, key in enumerate(self.all_keys):
            p = Path(key)
            if len(p.parent.name.split("_")) < 3:
                first_try_indices.append(i)
            else:
                second_try_indices.append(i)
        return first_try_indices, second_try_indices

    def get_sequence_length(self):
        if self.num_last_steps == -1:
            return min(self.max_traj_length, self.num_last_steps) * self.num_trajs
        else:
            return self.num_last_steps * self.num_trajs

    def split_full_leaveout(self, trial_list: List[str]):
        """
        Leave entire trials (all rotations) in the test set.
        """
        train_indices = []
        test_indices = []
        for i, key in enumerate(self.all_keys):
            p = Path(key)
            if p.parent.name in trial_list:
                test_indices.append(i)
            else:
                train_indices.append(i)
        return train_indices, test_indices

    def split_full_leaveout_change_human_study(self):
        """
        Split for the human study comparison.
        """
        full_leaveout_trial_list = ["S1_St8C13_250331", "S1_St12C13_250504", "S1_St14C13_250506",
                                    "S1_St8D15", "S1_St12D15", "S1_St14D15"]
        return self.split_full_leaveout(full_leaveout_trial_list)

    def split_full_leaveout_change(self):
        full_leaveout_trial_list = ["S1_St8C13_250331", "S1_St12C13_250504", "S1_St12C13", "S1_St14C13_250506",
                                    "S1_St14C13",
                                    "S1_St12D15", "S1_St14D15", "S1_St14D15_250507", "S1_St12C5_250330",
                                    "S1_St8C5_250327"]
        return self.split_full_leaveout(full_leaveout_trial_list)

    def split_half_leaveout(self, trial_list: List[str], num_orientation: int = 5,
                            fix_angles_dict: Optional[Dict[str, int]] = None):
        """
        A split that can be used for change detection.
        For some lump locations (e.g C5, C13, D15, D10), leave *half* of the rotations in test for *all* sizes.
        If fix angles is True - take constant angles, else take random angles.
        """
        count_leaveout_trial = {trial: 0 for trial in trial_list}
        fix_angles = fix_angles_dict is not None
        train_indices = []
        test_indices = []
        for i, key in enumerate(self.all_keys):
            p = Path(key)
            name = p.parent.name
            model_name = os.path.basename(key).split(".")[0]
            orientation = int(model_name.split("_")[-1])
            correct_orientation = False
            for loc in fix_angles_dict:
                if loc in name:
                    correct_orientation = orientation in fix_angles_dict[loc]
                    break
            correct_orientation = correct_orientation or (
                    not fix_angles and name in count_leaveout_trial.keys() and count_leaveout_trial[
                name] < num_orientation)
            if name in count_leaveout_trial.keys() and correct_orientation:
                test_indices.append(i)
                count_leaveout_trial[p.parent.name] += 1
            else:
                train_indices.append(i)
        return train_indices, test_indices

    def split_half_leaveout_change_human_study(self, fix_angles=True):
        """
        Split for the human study comparison.
        """
        trial_list = ["S1_St8C13_250331", "S1_St12C13_250504", "S1_St12C13", "S1_St14C13_250506",
                      "S1_St14C13", "S1_St8D15",
                      "S1_St12D15", "S1_St14D15", "S1_St14D15_250507",
                      "S1_St12C5_250330", "S1_St8C5_250327", "S1_St12C5_250428",
                      "S1_St14C5_250330", "S1_St14C5_250428", "S1_St14C5_250505"]
        fix_angles_dict = {"C5": [0, 1, 2, 3, 5, 7], "C13": [1, 2, 3, 4, 6, 7], "D15": [0, 1, 3, 4, 5, 6],
                           "D10": [0, 2, 4, 5, 6, 7]}
        fix_angles_dict = fix_angles_dict if fix_angles else None
        return self.split_half_leaveout(trial_list, fix_angles_dict=fix_angles_dict)

    def split_half_leaveout_change(self, fix_angles=False):
        trial_list = ["S1_St8C13_250331", "S1_St12C13_250504", "S1_St12C13", "S1_St14C13_250506",
                      "S1_St14C13",
                      "S1_St12D15", "S1_St14D15", "S1_St14D15_250507",
                      "S1_St12C5_250330", "S1_St8C5_250327", "S1_St12C5_250428",
                      "S1_St14C5_250330", "S1_St14C5_250428", "S1_St14C5_250505"]
        fix_angles_dict = {"C5": [0, 1, 2, 3, 5, 7], "C13": [1, 2, 3, 4, 6, 7], "D15": [0, 1, 3, 4, 5, 6],
                           "D10": [0, 2, 4, 5, 6, 7]}

        fix_angles_dict = fix_angles_dict if fix_angles else None
        return self.split_half_leaveout(trial_list, fix_angles_dict=fix_angles_dict)

    def __len__(self):
        return len(self.all_keys)

    def __getitem__(self, idx):
        if idx >= len(self.all_keys):
            raise IndexError("Index out of range")
        file_path = self.dir_path / self.all_keys[idx]

        model_name = os.path.basename(file_path).split(".")[0]
        model_images = create_mri_image(model_name, self.synthetic_mri_image, Path(self.mri_images_dir),
                                        randomly_choose_mri=self.randomly_choose_mri,
                                        merge01_image_class=self.merge01_image_class)

        # load data from the file
        data = load_h5_data(file_path, max_traj_length=self.max_traj_length, data_filter=self.filter_data,
                            num_trajectories_to_keep=self.num_trajs, close_to_lump_trajs=self.close_to_lump_trajs,
                            shift_origin=self.shift_trial_origin, shift_orientation=self.shift_trial_orientation,
                            rel_forces=self.rel_forces)
        if self.rotation_aug:
            angle = np.random.uniform(0, 360)
            data = rotate_locations(data, angle)

            model_images = rotate_tensor(model_images, angle, fill=0)

        if self.flip_aug:
            flip = np.random.choice([True, False])
            if flip:
                data = flip_x_locations(data)

                model_images = hflip_tensor(model_images)

        # The arrays are big now so we need to use contiguous() on them
        model_images = torch.as_tensor(model_images, dtype=torch.long).contiguous()

        images = torch.zeros(1, 1, 1, 1)  # currently no image support
        locations = torch.tensor(data.ee_positions, dtype=torch.float32)
        locations = locations + self.location_noise_std * torch.randn_like(locations)
        if not self.no_angles:
            angles = torch.tensor(data.ee_rotations, dtype=torch.float32)
            angles = angles + self.angle_noise_std * torch.randn_like(angles)
            locations = torch.cat((locations, angles), dim=2).unsqueeze(2)
        else:
            locations = locations.unsqueeze(2)

        locations = locations[:, ::self.sensor_subsample]
        if self.num_last_steps != -1:
            curr_num_vectors = min(locations.shape[0], self.num_last_steps)
            locations = locations[-curr_num_vectors:]
        locations = locations.reshape(locations.shape[0] * locations.shape[1], 1, -1)
        forces = torch.tensor(data.xela, dtype=torch.float32)
        forces = forces[:, ::self.sensor_subsample]
        if self.num_last_steps != -1:
            curr_num_vectors = min(forces.shape[0], self.num_last_steps)
            forces = forces[-curr_num_vectors:]
        forces = forces.reshape(forces.shape[0] * forces.shape[1], forces.shape[2], -1)
        if self.only_z:
            forces = forces[:, :, 2:3]
        label = torch.tensor([0])

        radius = torch.tensor([0])

        model_images = model_images.long().contiguous()

        return images, locations, forces, label, model_images, radius, self.all_keys[idx]

    def get_split(self, split_type: str):
        if split_type == "split_second_try":
            return self.split_first_trial()
        elif split_type == "split_change_half":
            return self.split_half_leaveout_change(fix_angles=False)
        elif split_type == "split_change_half_fixed":
            return self.split_half_leaveout_change(fix_angles=True)
        elif split_type == "split_change_full":
            return self.split_full_leaveout_change()
        elif split_type == "split_change_half_human_study":
            return self.split_half_leaveout_change_human_study()
        elif split_type == "split_change_full_human_study":
            return self.split_full_leaveout_change_human_study()
        elif split_type == "random":
            train_size = int(0.8 * len(self))
            return list(range(train_size)), list(range(train_size, len(self)))
        else:
            raise ValueError(f"Unknown split type for real dataset: {split_type}")
