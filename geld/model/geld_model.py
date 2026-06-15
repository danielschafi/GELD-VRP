"""GELD neural TSP model (Global-view Encoder + Local-view Decoder)."""

from dataclasses import dataclass

import torch
import torch.nn as nn

from geld.model.geometry import (
    LARGE_INSTANCE_THRESHOLD,
    build_distance_matrix,
    compute_distance_matrix,
    map_coordinates_to_regions,
    normalize_coordinates,
)
from geld.model.global_encoder import GlobalEncoder
from geld.model.local_decoder import LocalDecoder

_LEGACY_DECODER_KEY_PREFIXES = {
    "decoder.embedding_first_node1.": "decoder.first_node_embedding.",
    "decoder.embedding_last_node1.": "decoder.last_node_embedding.",
    "decoder.Linear_final.": "decoder.final_projection.",
}


@dataclass
class DecodeStepOutput:
    """Outputs from one autoregressive MDP decode step."""

    teacher_action: torch.Tensor | None = None
    step_prob: torch.Tensor | None = None
    transition_probs: torch.Tensor | None = None
    predicted_action: torch.Tensor | None = None

    @property
    def action(self) -> torch.Tensor:
        """Return ground-truth action in training, else greedy prediction."""
        if self.teacher_action is not None:
            return self.teacher_action
        return self.predicted_action


class GeldModel(nn.Module):
    """GELD: lightweight GE plus heavyweight LD for autoregressive TSP solving."""

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.mode = model_params["mode"]
        self.encoder = GlobalEncoder(**model_params)
        self.decoder = LocalDecoder(**model_params)
        self.encoded_nodes = None
        self.data = None
        self.dis_matrix = None
        self.region = None

    @staticmethod
    def _remap_legacy_state_dict(state_dict):
        """Rename legacy LEHD checkpoint keys for decoder compatibility."""
        remapped = {}
        for key, value in state_dict.items():
            new_key = key
            for old_prefix, new_prefix in _LEGACY_DECODER_KEY_PREFIXES.items():
                if key.startswith(old_prefix):
                    new_key = new_prefix + key[len(old_prefix) :]
                    break
            remapped[new_key] = value
        return remapped

    def load_state_dict(self, state_dict, strict=True):
        """Load weights after remapping legacy decoder key names."""
        return super().load_state_dict(
            self._remap_legacy_state_dict(state_dict), strict=strict
        )

    def prepare_instance(self, coordinates, normalize: bool = True):
        """Normalize topology, build distance matrix, and assign RALA regions."""
        if normalize:
            self.data = normalize_coordinates(coordinates)
        else:
            self.data = coordinates

        if coordinates.size(1) > LARGE_INSTANCE_THRESHOLD:
            self.decoder.data = self.data
            self.dis_matrix = compute_distance_matrix(
                self.data, LARGE_INSTANCE_THRESHOLD
            )
        else:
            self.dis_matrix = build_distance_matrix(self.data)
        self.region = map_coordinates_to_regions(self.data)

    def forward(
        self,
        constructed_tour,
        label_tour,
        current_step,
        repair=False,
        beam_search=False,
        beam_size=16,
        reencode_each_step=False,
    ):
        """One MDP step: teacher forcing (SL), greedy, beam search, or RC repair."""
        batch_size = constructed_tour.size(0)

        if self.mode == "train" and self.training:
            encoded_nodes = self.encoder(self.data, self.region)
            probs = self.decoder(encoded_nodes, constructed_tour, self.dis_matrix)
            predicted = probs.argmax(dim=1)
            teacher = label_tour[:, current_step - 1]
            step_prob = probs[
                torch.arange(batch_size)[:, None], teacher[:, None]
            ].reshape(batch_size, 1)
            return DecodeStepOutput(
                teacher_action=teacher,
                step_prob=step_prob,
                predicted_action=predicted,
            )

        if self.mode == "test" or not self.training:
            if not repair:
                if current_step <= 1 and not reencode_each_step:
                    self.encoded_nodes = self.encoder(self.data, self.region)

                if not beam_search:
                    probs = self.decoder(
                        self.encoded_nodes, constructed_tour, self.dis_matrix
                    )
                    predicted = probs.argmax(dim=1)
                    return DecodeStepOutput(predicted_action=predicted)

                transition_probs = self.decoder(
                    self.encoded_nodes,
                    constructed_tour,
                    self.dis_matrix,
                    beam_search=True,
                    beam_size=beam_size,
                )
                return DecodeStepOutput(transition_probs=transition_probs)

            if current_step <= 2:
                self.encoded_nodes = self.encoder(self.data, self.region)
            probs = self.decoder(self.encoded_nodes, constructed_tour, self.dis_matrix)
            predicted = probs.argmax(dim=1)
            return DecodeStepOutput(predicted_action=predicted)
