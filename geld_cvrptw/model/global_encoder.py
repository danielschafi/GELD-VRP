"""Global-view Encoder (GE) with RALA for CVRPTW."""

import torch
import torch.nn as nn

from geld_cvrptw.env.CVRPTW import StaticState
from geld.model.attention import FeedForwardModule, RegionAverageLinearAttention

NODE_FEATURE_DIM = 5


class GlobalEncoder(nn.Module):
    """Global-view Encoder: embed (x, y, demand, tw_start, tw_end) and run RALA."""

    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params["embedding_dim"]
        self.embedding = nn.Linear(NODE_FEATURE_DIM, embedding_dim, bias=True)
        self.layer = EncoderLayer(**model_params)

    @staticmethod
    def build_node_features(static_state: StaticState, normalized_coords: torch.Tensor) -> torch.Tensor:
        """Stack per-node inputs for Linear(5, d): coords + demand + time windows."""
        return torch.cat(
            (
                normalized_coords,
                static_state.node_demand.unsqueeze(-1),
                static_state.node_tw_start.unsqueeze(-1),
                static_state.node_tw_end.unsqueeze(-1),
            ),
            dim=-1,
        )

    def forward(
        self,
        static_state: StaticState,
        normalized_coords: torch.Tensor,
        region: torch.Tensor,
    ) -> torch.Tensor:
        node_features = self.build_node_features(static_state, normalized_coords)
        embedded_input = self.embedding(node_features)
        return self.layer(embedded_input, region)


class EncoderLayer(nn.Module):
    """Single GE layer: RALA followed by feed-forward."""

    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params["embedding_dim"]
        self.attentionlayer = RegionAverageLinearAttention(model_params=model_params)
        self.multi_head_combine = nn.Linear(embedding_dim, embedding_dim)
        self.feedForward = FeedForwardModule(**model_params)

    def forward(self, input_tensor: torch.Tensor, index_region: torch.Tensor) -> torch.Tensor:
        multi_head_out = self.multi_head_combine(self.attentionlayer(input_tensor, index_region))
        out1 = input_tensor + multi_head_out
        hidden_states = self.feedForward(out1)
        return out1 + hidden_states
