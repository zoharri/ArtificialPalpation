"""
Vector-based reconstruction model for representation learning.

This module contains the ReconstructionModel, which is a configurable neural network
model for reconstructing forces from locations. It supports multiple architectures
including GRU and Transformer.

The model uses positional encoding to encode spatial information and can be configured
for different types of force reconstruction tasks.
"""

from typing import Tuple, Dict

import numpy as np
import torch
from torch import nn

from models.representation_learning.positional_encoding.encoding import FreqEncoder_torch
from models.representation_learning.representation_learning_model import RepresentationLearningModel


class VectorLocationEncoder(nn.Module):
    """
    Encodes vectors and locations into a combined representation using positional encoding.
    
    This module combines force vectors with location information using positional encoding
    to create rich representations that capture both the force magnitude and spatial context.
    
    Args:
        locations_size (int): Size of the locations vectors
        pe_max_freq_log2 (int): Maximum frequency for positional encoding
        vector_encoding_size (int): Size of the output encoding for vectors
        vector_size (int): Size of the input vectors to be encoded
    """
    """
    Encodes vectors and locations into a combined representation using positional encoding.
    Args:
    locations_size (int): Size of the locations vectors
    pe_max_freq_log2 (int): Maximum frequency for positional encoding.
    vector_encoding_size (int): Size of the output encoding for vectors.
    vector_size (int): Size of the input vectors to be encoded.
    """

    def __init__(self, locations_size: int, pe_max_freq_log2: int, vector_encoding_size: int, vector_size: int):
        """
        Initializes the VectorLocationEncoder.
        Args:
        locations_size (int): Size of the locations vectors.
        pe_max_freq_log2 (int): Maximum frequency for positional encoding.
        vector_encoding_size (int): Size of the output encoding for vectors.
        vector_size (int): Size of the input vectors to be encoded.
        Raises:
        ValueError: If vector_encoding_size is not divisible by locations_size * 2.
        """
        super(VectorLocationEncoder, self).__init__()
        assert vector_encoding_size % (locations_size * 2) == 0, ValueError(
            f"vector size ({vector_encoding_size}) must be divisible by locations_size * 2 ({locations_size * 2})")
        self.vector_encoding_size = vector_encoding_size
        self.pe = FreqEncoder_torch(input_dim=locations_size, max_freq_log2=pe_max_freq_log2,
                                    N_freqs=vector_encoding_size // (locations_size * 2),
                                    log_sampling=True, include_input=False)

        self.vector_encoder = nn.Linear(vector_size, vector_encoding_size)

    def forward(self, vectors: torch.Tensor, locations: torch.Tensor) -> torch.Tensor:
        locations_mean = locations.mean(dim=1)
        pe_out = self.pe(locations_mean)
        force_encodings = self.vector_encoder(vectors)
        return pe_out + force_encodings

    def output_size(self) -> int:
        return self.vector_encoding_size


class ReconstructionModel(RepresentationLearningModel):
    """
    A configurable model for reconstructing forces from locations.
    
    This model supports multiple architectures:
    - GRU: Gated Recurrent Unit for sequential processing
    - Transformer: Attention-based architecture for parallel processing
    
    The model takes force and location data as input and learns to reconstruct
    forces from the learned representations. It uses positional encoding to
    capture spatial information and can be configured for different types of
    force reconstruction tasks.
    
    This model replaces the old GRUReconstructionModel and provides a more
    flexible and extensible architecture.
    """

    def __init__(self, config_path: str):
        super(ReconstructionModel, self).__init__(config_path)

        self.force_location_encoder = VectorLocationEncoder(self.config.locations_size, self.config.pe_max_freq_log2,
                                                            self.config.gru_input_embed_dim,
                                                            self.config.force_size)
        self.latent_location_encoder = VectorLocationEncoder(self.config.locations_size, self.config.pe_max_freq_log2,
                                                             self.config.decoder_input_embed_dim,
                                                             self.config.representation_size)

        arch = self.config.arch.lower()

        if arch == "gru":
            self.encoder = nn.GRU(input_size=self.force_location_encoder.output_size(),
                                  hidden_size=self.config.representation_size,
                                  batch_first=True,
                                  dropout=self.config.dropout)
        elif arch == "transformer":
            self.input_encoder_layer = nn.Linear(self.force_location_encoder.output_size(),
                                                 self.config.representation_size)
            encoder_layer = nn.TransformerEncoderLayer(d_model=self.config.representation_size,
                                                       nhead=self.config.transformer_nhead,
                                                       dim_feedforward=self.config.transformer_ff_dim,
                                                       batch_first=True,
                                                       dropout=self.config.dropout)
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.config.transformer_num_layers)
        else:
            raise ValueError(f"Unknown arch: {arch}")

        self.force_predictor = nn.Sequential(
            nn.Linear(self.latent_location_encoder.output_size(), self.config.force_predictor_hidden),
            nn.ReLU(),
            nn.Linear(self.config.force_predictor_hidden, self.config.force_predictor_hidden // 2),
            nn.ReLU(),
            nn.Linear(self.config.force_predictor_hidden // 2, self.config.force_size)
        )

    def inference(self, forces: torch.Tensor, locations: torch.Tensor, **kwargs) -> Tuple[
        torch.Tensor, Dict[str, torch.Tensor]]:
        h_0 = kwargs.get("h_0", None)
        predict_all = kwargs.get("predict_all", False)
        input_forces, input_locations, output_locations, mask_input_indices, mask_output_indices = (
            self.get_masked_input_output(forces, locations))
        representations, h_n = self.encode(input_locations, input_forces, h_0=h_0)
        predicted_forces, input_steps, reconstruction_steps = self.decode(representations, output_locations,
                                                                          predict_all=predict_all)

        batch_size = mask_output_indices.size(0)
        input_steps = mask_input_indices[torch.arange(batch_size).unsqueeze(-1), input_steps]
        reconstruction_steps = mask_output_indices[
            torch.arange(batch_size).unsqueeze(-1).unsqueeze(-1), reconstruction_steps]

        return h_n[-1], {"predicted_forces": predicted_forces, "h_n": h_n, "input_steps": input_steps,
                         "reconstruction_steps": reconstruction_steps, "all_outputs": representations.clone()}

    def get_masked_input_output(self, forces: torch.Tensor, locations: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generates masked input and output tensors based on the configured mask percentage.
        """
        if self.config.mask_percentage > 0:
            batch_size, num_steps, _, _ = locations.shape
            if self.config.trajectory_level_mask:
                traj_len = self.config.trajectory_length
                num_trajs = num_steps // traj_len
                num_input_trajs = int(num_trajs * (1 - self.config.mask_percentage))

                rand = torch.rand(batch_size, num_trajs, device=self.device)
                permuted_trajs = rand.argsort(dim=1)

                input_trajs = permuted_trajs[:, :num_input_trajs].sort(dim=1)[0]
                output_trajs = permuted_trajs[:, num_input_trajs:].sort(dim=1)[0]

                base = torch.arange(traj_len, device=self.device).view(1, 1, traj_len)
                input_indices = (input_trajs.unsqueeze(-1) * traj_len + base).reshape(batch_size, -1)
                output_indices = (output_trajs.unsqueeze(-1) * traj_len + base).reshape(batch_size, -1)
            else:
                rand = torch.rand(batch_size, num_steps, device=self.device)
                permuted = rand.argsort(dim=1)

                num_input_steps = int(num_steps * self.config.mask_percentage)
                input_indices = permuted[:, :num_input_steps].sort(dim=1)[0]
                output_indices = permuted[:, num_input_steps:].sort(dim=1)[0]
            batch_indices = torch.arange(batch_size).unsqueeze(-1)
            input_locations = locations[batch_indices, input_indices, :]
            input_forces = forces[batch_indices, input_indices, :]
            output_locations = locations[batch_indices, output_indices, :]
        else:
            input_indices = torch.arange(locations.size(1)).unsqueeze(0).expand(locations.size(0), -1).to(self.device)
            output_indices = input_indices
            input_locations = locations
            input_forces = forces
            output_locations = locations

        return input_forces, input_locations, output_locations, input_indices, output_indices

    def encode(self, locations: torch.Tensor, forces: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        h_0 = kwargs.get("h_0", None)
        batch_size, seq_len, num_sensors, dim_location = locations.shape
        combined = self.force_location_encoder(forces.reshape(batch_size * seq_len, -1),
                                               locations.reshape(batch_size * seq_len, num_sensors, dim_location))
        combined = combined.reshape(batch_size, seq_len, -1)

        arch = self.config.arch.lower()
        if arch == "gru":
            if self.config.gru_tbptt_step_size is not None:
                output = []
                hidden_state = h_0
                for i in range(int(np.ceil(combined.shape[1] / self.gru_tbptt_step_size))):
                    curr_input = combined[:,
                                 i * self.gru_tbptt_step_size:i * self.gru_tbptt_step_size + self.gru_tbptt_step_size]
                    curr_output, hidden_state = self.encoder(curr_input, hidden_state)
                    output.append(curr_output)
                    hidden_state = hidden_state.detach()
                output = torch.cat(output, dim=1)
                h_n = hidden_state
            else:
                output, h_n = self.encoder(combined, h_0)
            return output, h_n

        elif arch == "transformer":
            combined = self.input_encoder_layer(combined)
            mask = torch.triu(torch.ones((seq_len, seq_len), device=self.device) * float('-inf'),
                              diagonal=1) if self.config.transformer_use_causal_mask else None
            output = self.encoder(combined, mask=mask, is_causal=self.config.transformer_use_causal_mask)
            h_n = output[:, -1, :].unsqueeze(0)
            return output, h_n

    def decode(self, representations: torch.Tensor, locations: torch.Tensor, predict_all: bool = False):
        locations_mean = locations.mean(dim=2)
        batch_size, loc_seq_len, _ = locations_mean.shape
        _, rep_seq_len, _ = representations.shape

        if self.config.input_num_random_samples != -1 and predict_all is False:
            input_steps = torch.randint(0, rep_seq_len, (batch_size, self.config.input_num_random_samples)).to(
                self.device)
        else:
            input_steps = torch.arange(rep_seq_len).unsqueeze(0).expand(batch_size, rep_seq_len).to(self.device)
        if self.config.reconstruction_num_random_samples != -1 and predict_all is False:
            reconstruction_steps = torch.randint(0, loc_seq_len, (
                batch_size, self.config.input_num_random_samples, self.config.reconstruction_num_random_samples)).to(
                self.device)
        else:
            if self.config.input_num_random_samples != -1 and predict_all is False:
                reconstruction_steps = (
                    torch.arange(0, loc_seq_len).unsqueeze(0).unsqueeze(0).expand(batch_size,
                                                                                  self.config.input_num_random_samples,
                                                                                  -1).to(self.device))
            else:
                reconstruction_steps = torch.arange(0, loc_seq_len).unsqueeze(0).unsqueeze(0).expand(batch_size,
                                                                                                     rep_seq_len,
                                                                                                     -1).to(self.device)

        batch_indices = torch.arange(batch_size).unsqueeze(-1)
        expanded_output = representations[batch_indices, input_steps].view(batch_size, input_steps.size(1), -1)
        expanded_output = expanded_output.unsqueeze(2).expand(-1, -1, reconstruction_steps.size(2), -1)

        expanded_locations = locations.unsqueeze(1).expand(-1, rep_seq_len, -1, -1, -1)
        expanded_locations = expanded_locations[
            batch_indices.unsqueeze(-1), input_steps.unsqueeze(-1), reconstruction_steps]

        combined_features = self.latent_location_encoder(expanded_output.reshape(-1, expanded_output.size(-1)),
                                                         expanded_locations.reshape(-1, expanded_locations.size(-2),
                                                                                    expanded_locations.size(-1)))
        combined_features = combined_features.view(batch_size, input_steps.size(1), reconstruction_steps.size(2), -1)

        predicted_forces = self.force_predictor(combined_features)
        return predicted_forces, input_steps, reconstruction_steps

    def compute_loss(self, inference_results: Dict[str, torch.Tensor], forces: torch.Tensor,
                     locations: torch.Tensor,
                     **kwargs) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        batch_indices = torch.arange(forces.size(0)).unsqueeze(-1).unsqueeze(-1)
        forces_target = forces.to(self.device)
        forces_target = forces_target.view(forces_target.size(0), forces_target.size(1), -1)
        forces_target = forces_target.unsqueeze(1).expand(-1, forces_target.size(1), -1,
                                                          -1)  # expand to match predicted shape
        forces_target = forces_target[
            batch_indices, inference_results["input_steps"].unsqueeze(-1), inference_results[
                "reconstruction_steps"]]
        predicted_vectors = inference_results["predicted_forces"]
        vector_prediction_criterion = nn.MSELoss()
        all_recon_loss = vector_prediction_criterion(predicted_vectors, forces_target)

        consistency_loss = torch.mean(
            torch.norm(inference_results["all_outputs"][:, 1:] - inference_results["all_outputs"][:, :-1], dim=-1))

        total_loss = all_recon_loss

        if self.config.consistency_reg_weight > 0:
            total_loss += self.config.consistency_reg_weight * consistency_loss

        return total_loss, {"all_forces_loss": all_recon_loss, "consistency_hidden_loss": consistency_loss}

    @property
    def representation_size(self) -> int:
        """
        Returns the size of the representation produced by the model.
        This should be overridden in subclasses to return the specific representation size.
        """
        return self.config.representation_size
