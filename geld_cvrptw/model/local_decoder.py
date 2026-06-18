"""CVRPTW local decoder (TSP LD re-used until depot/capacity/TW context is added)."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from geld_cvrptw.env.CVRPTW import DynamicState
from geld_cvrptw.model.helpers import LARGE_INSTANCE_THRESHOLD
from geld.model.attention import RMSNorm, FeedForwardModule, AttentionFusionModule

K_NEAREST_NEIGHBORS = 99


class LocalDecoder(nn.Module):
    """Local-view Decoder (LD): refined local selection over k-NN candidate set."""

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = model_params["embedding_dim"]
        decoder_layer_num = model_params["decoder_layer_num"]
        self.first_node_embedding = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.last_node_embedding = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.layers_global = nn.ModuleList([DecoderLayer(**model_params) for _ in range(decoder_layer_num)])
        self.final_projection = nn.Linear(embedding_dim, 1, bias=True)

    def forward(
        self,
        encoded_nodes: torch.Tensor,
        dynamic_state: DynamicState,
        normalized_coords: torch.Tensor,
        dis_matrix: torch.Tensor,
        
    ) -> torch.Tensor:
        pass


class DecoderLayer(nn.Module):
    """LD Layer with Attention Fusion Module and feed forward nn."""

    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params["embedding_dim"]
        head_num = model_params["head_num"]
        qkv_dim = model_params["qkv_dim"]

        self.input_layernorm = RMSNorm(embedding_dim)
        self.post_attention_layernorm = RMSNorm(embedding_dim)

        self.attention_fusion_layer = AttentionFusionModule(model_params=model_params)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim, bias=False)
        self.feedForward = FeedForwardModule(**model_params)

    def forward(self, embeddings, normalized_distance_matrix):
        """
        Apply pre-norm AFM and post-norm FFN with residuals

        Parameters:
        - input_tensor: [depot + kfeasible + current] node embeddings

        """
        x_1 = self.input_layernorm(embeddings)
        x_1 = self.attention_fusion_layer(x_1, normalized_distance_matrix)
        x_1 = self.multi_head_combine(x_1)

        # feed forward with residual connection
        x_1 = embeddings + x_1
        x_2 = self.post_attention_layernorm(x_1)
        x_2 = self.feedForward(x_2)

        return x_1 + x_2
