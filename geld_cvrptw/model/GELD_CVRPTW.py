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
    map_nodes_to_regions,
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
        self.static_state: StaticState | None = None
        self.time_horizon: torch.Tensor | None = None
        self.encoded_nodes = None
        self.normalized_coords = None
        self.dis_matrix = None
        self.node_to_region_map = None

    def embed_static_state_once(self, static_state: StaticState) -> None:
        """Normalize topology, build distance matrix, and assign RALA regions for one instance."""
        self.static_state = static_state
        self.time_horizon = static_state.node_tw_end[:, 0]
        node_coords = static_state.node_coords
        self.normalized_coords = normalize_coordinates(node_coords)

        if node_coords.size(1) > LARGE_INSTANCE_THRESHOLD:
            self.dis_matrix = approximate_distance_matrix(self.normalized_coords, LARGE_INSTANCE_THRESHOLD)
        else:
            self.dis_matrix = compute_distance_matrix(self.normalized_coords)

        self.node_to_region_map = map_nodes_to_regions(self.normalized_coords)
        self.encoded_nodes = None
        if not self.training:
            self.encoded_nodes = self.encoder(static_state, self.normalized_coords, self.node_to_region_map)

    def forward(self, dynamic_state: DynamicState, mask_feasibility: bool = True) -> torch.Tensor:
        """Return next-node probabilities, shape (batch, num_nodes)."""
        if self.static_state is None or self.normalized_coords is None or self.dis_matrix is None:
            raise RuntimeError("Call embed_static_state_once before forward.")

        if self.training:
            # encode each stepd  
            encoded_nodes = self.encoder(self.static_state, self.normalized_coords, self.node_to_region_map)
        else:
            # encode only once
            if self.encoded_nodes is None:
                self.encoded_nodes = self.encoder(
                    self.static_state, self.normalized_coords, self.node_to_region_map
                )
            encoded_nodes = self.encoded_nodes

        probs = self.decoder(
            encoded_nodes,
            dynamic_state,
            self.normalized_coords,
            self.dis_matrix,
            self.time_horizon,
        )
        if mask_feasibility:
            return apply_feasibility_mask(probs, dynamic_state.ninf_mask)
        return probs
