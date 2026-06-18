"""
GELD-VRP Model
Global Encorde + Local Decoder.
constraints satisfied by masking in CVRPTW.py env
"""

import torch
import torch.nn as nn

from geld_cvrptw.env.CVRPTW import DynamicState, StaticState
from geld_cvrptw.model.helpers import (
    LARGE_INSTANCE_THRESHOLD,
    apply_feasibility_mask,
    build_distance_matrix,
    compute_distance_matrix,
    map_coordinates_to_regions,
    normalize_coordinates,
)
from geld_cvrptw.model.global_encoder import GlobalEncoder
from geld_cvrptw.model.local_decoder import LocalDecoder


class GeldCvrptwModel(nn.Module):
    """GELD: lightweight GE plus heavyweight LD for autoregressive TSP solving."""

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.encoder = GlobalEncoder(**model_params)
        self.decoder = LocalDecoder(**model_params)
        self.encoded_nodes = None
        self.data = None
        self.dis_matrix = None
        self.region = None

    def prepare_instance(self, static_state: StaticState) -> None:
        """Normalize topology, build distance matrix, assign RALA regions, and encode once."""
        node_coords = static_state.node_coords
        self.data = normalize_coordinates(node_coords)

        if node_coords.size(1) > LARGE_INSTANCE_THRESHOLD:
            self.decoder.data = self.data
            self.dis_matrix = compute_distance_matrix(self.data, LARGE_INSTANCE_THRESHOLD)
        else:
            self.dis_matrix = build_distance_matrix(self.data)

        self.region = map_coordinates_to_regions(self.data)
        self.encoded_nodes = self.encoder(self.data, self.region)

    def forward(self, dynamic_state: DynamicState) -> torch.Tensor:
        """Return masked next-node probabilities, shape (batch, num_nodes)."""
        probs = self.decoder(
            self.encoded_nodes,
            dynamic_state.constructed_tour,
            self.dis_matrix,
        )
        return apply_feasibility_mask(probs, dynamic_state.ninf_mask)
