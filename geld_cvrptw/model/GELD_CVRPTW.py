"""
GELD-VRP Model
Global Encorde + Local Decoder.
constraints satisfied by masking in CVRPTW.py env
"""

import torch
import torch.nn as nn

from dataclasses import dataclass

from geld_cvrptw.model.helpers import (
    LARGE_INSTANCE_THRESHOLD,
    build_distance_matrix,
    compute_distance_matrix,
    map_coordinates_to_regions,
    normalize_coordinates,
)
from geld_cvrptw.model.global_encoder import GlobalEncoder
from geld_cvrptw.model.local_decoder import LocalDecoder


@dataclass
class DecodeStepOutput:
    """
    Outputs from one autoregressive MDP decode step.
    """

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


class GeldCvrptwModel(nn.Module):
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

    def prepare_instance(self, coordinates, normalize: bool = True):
        """Normalize topology, build distance matrix, and assign RALA regions."""

        self.data = normalize_coordinates(coordinates)

        if coordinates.size(1) > LARGE_INSTANCE_THRESHOLD:
            self.decoder.data = self.data
            self.dis_matrix = compute_distance_matrix(self.data, LARGE_INSTANCE_THRESHOLD)
        else:
            self.dis_matrix = build_distance_matrix(self.data)

        self.region = map_coordinates_to_regions(self.data)
