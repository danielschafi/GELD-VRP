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
        self.embedding_dim = model_params["embedding_dim"]
        decoder_layer_num = model_params["decoder_layer_num"]
        self.depot_node_embedding = nn.Linear(self.embedding_dim, self.embedding_dim, bias=True)
        self.current_node_embedding = nn.Linear(self.embedding_dim, self.embedding_dim, bias=True)
        self.layers_global = nn.ModuleList([DecoderLayer(**model_params) for _ in range(decoder_layer_num)])
        self.final_projection = nn.Linear(self.embedding_dim, 1, bias=True)

    def forward(
        self,
        encoded_nodes: torch.Tensor,
        dynamic_state: DynamicState,
        normalized_coords: torch.Tensor,
        dis_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """Predicts the next nodes from the candidate set."""
        batch_size =  encoded_nodes.shape[0]
        problem_size = encoded_nodes.shape[1]

        self.build_candidate_set(encoded_nodes, dynamic_state, dis_matrix)



    def build_candidate_set(
        self,
        encoded_nodes: torch.Tensor,
        dynamic_state: DynamicState,
        dis_matrix: torch.Tensor,
    ):
        batch_size = encoded_nodes.shape[0]
        problem_size = encoded_nodes.shape[1]
        device = encoded_nodes.device
        sample_idx = torch.arange(batch_size, device=device)

        current_node_idx = dynamic_state.current_node_idx
        if current_node_idx is None:
            current_node_idx = torch.zeros(batch_size, dtype=torch.long, device=device)

        # First and last node embeddings, for local decoding update their static embeddings
        depot_node_embedding = self.depot_node_embedding(encoded_nodes[sample_idx, 0])
        current_node_embedding = self.current_node_embedding(encoded_nodes[sample_idx, current_node_idx])

        # Temporarily set the infeasible / visited nodes in the distance matrix to float(inf)
        # then get the topk closest nodes (indices) from the distance matrix with masked out infeasible nodes.
        # Get the embeddings of these topk nodes. -> local candidates embeddings.
        distances = dis_matrix[sample_idx, current_node_idx]
        infeasible = dynamic_state.ninf_mask == float("-inf")
        distances = distances.masked_fill(infeasible, float("inf"))

        k = min(K_NEAREST_NEIGHBORS, problem_size)
        local_candidates_indexes = torch.topk(distances, k=k, dim=1, largest=False).indices

        local_candidates_embedding = encoded_nodes.gather(
            1,
            local_candidates_indexes.unsqueeze(-1).expand(-1, -1, self.embedding_dim),
        )

        return depot_node_embedding, current_node_embedding, local_candidates_embedding, local_candidates_indexes, current_node_idx

    def _local_distance_matrix(
        self,
        candidate_node_indices: torch.Tensor,
        dis_matrix: torch.Tensor,
        normalized_coords: torch.Tensor,
        problem_size: int,
    ) -> torch.Tensor:
        batch_size = candidate_node_indices.size(0)
        batch_idx = torch.arange(batch_size, dtype=torch.long, device=dis_matrix.device)

        if problem_size > LARGE_INSTANCE_THRESHOLD:
            index_un = candidate_node_indices.unsqueeze(2).expand(batch_size, -1, 2)
            coords = normalized_coords.gather(dim=1, index=index_un)
            local_dist = torch.cdist(coords, coords, p=2)
            local_dist.diagonal(dim1=-2, dim2=-1).zero_()
            return local_dist

        index_un = candidate_node_indices.unsqueeze(1)
        return dis_matrix[batch_idx.unsqueeze(1), index_un, index_un.transpose(1, 2)]


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
