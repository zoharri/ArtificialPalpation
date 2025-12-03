from dataclasses import dataclass, field
from typing import Optional

from models.downstream.configs import DownstreamModelConfig
from models.representation_learning.configs import RepresentationModelConfig


@dataclass
class DatasetConfig:
    """Configuration for the dataset."""
    data_folder: str = './data'  # Path to the data folder
    h5_name: Optional[str] = None  # Name of the h5 file
    real_data: bool = False  # Use real data
    removec5: bool = False  # Remove c5 from real data
    removed10: bool = False  # Remove d10 from real data
    removed15: bool = False  # Remove d15 from real data
    include_s3: bool = True  # Include S3 in the dataset
    split_type: str = "random"  # How to split the dataset to train and test
    filter_hard_trials: Optional[int] = None  # Filter hard trials
    shuffle_keys: bool = True  # Shuffle keys in real data
    randomly_choose_mri: bool = True  # Randomly choose MRI image between 2, if not choose the first one
    use_rotation_aug: bool = False  # Use rotation augmentation
    use_flip_aug: bool = False  # Use flip augmentation
    dont_permute_trajs: bool = False  # Do not permute trajectories
    only_z: bool = False  # Use only z forces in real data
    synthetic_mri_image: bool = False  # Use synthetic MRI image in real data
    sensor_subsample: int = 1  # Change the frequency of the sensor data
    only_healthy: bool = False  # Only use healthy experiments
    num_training_trajs: int = 100  # Number of training trajectories
    uniform_dist_traj_subsample: bool = True  # Subsample trajectories uniformly
    random_first_traj_index: bool = False  # Randomly select the first trajectory index
    amount_of_data: Optional[int] = None  # Number of data points to use
    shift_trial_origin: bool = True  # Shift all trials to the same origin
    shift_trial_orientation: bool = True  # Shift all trials to the same orientation
    close_to_lump_trajs: bool = False  # Use trajs close to the lump
    no_angles: bool = False  # Do not use angles in the dataset
    no_image_input: bool = True  # Disable image input
    merge01_image_class: bool = False  # Only predict lump or background
    num_last_steps: int = -1  # Use only the last N steps of the trajectory, -1 to use all steps
    max_traj_length: int = 10  # Max trajectory length
    rel_forces: bool = True  # Use relative forces
    location_noise_std: float = 0.0  # Standard deviation of the noise added to the locations
    angle_noise_std: float = 0.0  # Standard deviation of the noise added to the angles


@dataclass
class DataLoaderConfig:
    """Configuration for the data loader."""
    batch_size: int = 4  # Batch size for the data loader
    num_workers: int = 20  # Number of workers for data loading


@dataclass
class OptimizerConfig:
    """Configuration for the optimizer."""
    name: str = "Adam"  # Name
    learning_rate: float = 0.0003  # Learning rate
    force_reconstruction_weight: float = 100.0  # Weight of the force reconstruction loss
    downstream_loss_weight: float = 1.0  # Weight of the image loss
    freeze_rep_learning_model: bool = False  # Freeze the representation learning model
    disable_downstream_model: bool = False  # Disable downstream model usage
    gradient_clipping: Optional[float] = None  # Gradient clipping value
    learning_rate_scheduler: str = "cosine"  # cosine/const/step
    lr_scheduler_step_size: int = 100  # Step size for the learning rate scheduler
    num_random_rep_for_downstream_training: int = -1  # Num random rep for downstream training, -1 to use last
    use_first_rep_for_downstream_training: bool = False  # Use first rep as well for downstream training


@dataclass
class DataProcessingConfig:
    """Configuration for the data processing."""
    data_stats_load_path: str = None  # Path to a npz file which includes the data statistics
    data_stats_save_path: str = None  # Path to save the data statistics


@dataclass
class ModelsConfig:
    rep_learning_model: RepresentationModelConfig = field(
        default_factory=RepresentationModelConfig)  # Configuration for the representation learning model
    rep_learning_model_checkpoint: str = ""  # Checkpoint for the representation learning model
    downstream_model: DownstreamModelConfig = field(
        default_factory=DownstreamModelConfig)  # Configuration for the downstream
    downstream_model_checkpoint: str = ""  # Checkpoint for the downstream model


@dataclass
class TrainConfig:
    """Configuration for training the model."""
    dataset: DatasetConfig = field(default_factory=DatasetConfig)  # Dataset configuration
    data_loader: DataLoaderConfig = field(default_factory=DataLoaderConfig)  # Data loader configuration
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)  # Optimizer configuration
    data_processing: DataProcessingConfig = field(default_factory=DataProcessingConfig)  # Data processing configuration
    models: ModelsConfig = field(default_factory=ModelsConfig)  # Models configuration
    num_epochs: int = 1000  # Number of epochs
    random_seed: int = 0  # Random seed
    dont_norm_locations: bool = False  # Do not normalize locations
    dont_detach_representation: bool = False  # Do not detach representation before downstream
    zero_representation: bool = False  # Zero the representation before downstream
    zero_location: bool = False  # Zero the locations
    zero_forces: bool = False  # Zero the forces
    constant_force_norm: float = 0  # Constant force norm
    shuffle_order: bool = False  # Shuffle vectors before input
    relative_locations: bool = False  # Use relative locations to the first location in each trajectory
    vis_log_interval: int = 50  # Interval for logging figures, -1 to disable
    model_log_interval: int = 100  # Interval for logging models, -1 to disable

    # post init
    def __post_init__(self):
        self.models.rep_learning_model.trajectory_length = self.dataset.max_traj_length // self.dataset.sensor_subsample
