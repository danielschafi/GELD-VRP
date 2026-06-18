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
    compute_distance_matrix,
    approximate_distance_matrix,
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
        self.normalized_coords = None
        self.dis_matrix = None
        self.node_to_region_map = None

    def embed_static_state_once(self, static_state: StaticState) -> None:
        """Normalize topology, build distance matrix, assign RALA regions, and encode once. Encoded features are reused."""
        node_coords = static_state.node_coords
        self.normalized_coords = normalize_coordinates(node_coords)

        if node_coords.size(1) > LARGE_INSTANCE_THRESHOLD:
            self.dis_matrix = approximate_distance_matrix(self.normalized_coords, LARGE_INSTANCE_THRESHOLD)
        else:
            self.dis_matrix = compute_distance_matrix(self.normalized_coords)

        self.node_to_region_map = map_coordinates_to_regions(self.normalized_coords)
        self.encoded_nodes = self.encoder(static_state, self.normalized_coords, self.node_to_region_map)

    def forward(self, dynamic_state: DynamicState) -> torch.Tensor:
        """Return masked next-node probabilities, shape (batch, num_nodes)."""
        if self.encoded_nodes is None:
            raise RuntimeError("Call embed_static_state_once before forward.")

        probs = self.decoder(
            self.encoded_nodes,
            dynamic_state,
            self.normalized_coords,
            self.dis_matrix,
        )
        return apply_feasibility_mask(probs, dynamic_state.ninf_mask)
