"""CVRPTW local decoder (TSP LD re-used until depot/capacity/TW context is added)."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from geld_cvrptw.env.CVRPTW import DynamicState
from geld_cvrptw.model.helpers import LARGE_INSTANCE_THRESHOLD, normalize_time_for_model
from geld_cvrptw.model.attention import RMSNorm, FeedForwardModule, AttentionFusionModule

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
        self.context_embedding = nn.Linear(3, self.embedding_dim, bias=True)
        self.decoder_layers = nn.ModuleList([DecoderLayer(**model_params) for _ in range(decoder_layer_num)])
        self.final_projection = nn.Linear(self.embedding_dim, 1, bias=True)

    def forward(
        self,
        encoded_nodes: torch.Tensor,
        dynamic_state: DynamicState,
        normalized_coords: torch.Tensor,
        dis_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """Predicts probabilities for all nodes to be the next one in the tour."""
        batch_size = encoded_nodes.shape[0]
        problem_size = encoded_nodes.shape[1]
        device = encoded_nodes.device

        candidate_set, local_candidates_indexes = self.build_candidate_set(encoded_nodes, dynamic_state, dis_matrix)
        k = local_candidates_indexes.size(1)
        norm_local_distance_matrix = self.local_distance_matrix(
            dis_matrix,
            local_candidates_indexes,
            dynamic_state.current_node_idx,
            normalized_coords,
            problem_size,
        )
        for layer in self.decoder_layers:
            candidate_set = layer(candidate_set, norm_local_distance_matrix)

        logits = self.final_projection(candidate_set).squeeze(-1)[:, 1 : 1 + k]
        probs = F.softmax(logits, dim=-1)

        full_probs = torch.full((batch_size, problem_size), 1e-5, device=device, dtype=probs.dtype)
        full_probs.scatter_(1, local_candidates_indexes, probs)
        return full_probs

    def build_candidate_set(
        self,
        encoded_nodes: torch.Tensor,
        dynamic_state: DynamicState,
        dis_matrix: torch.Tensor,
    ):
        """Returns the candidate set of k nearest feasible nodes + dynamic context vector for input to the decoder layers.
        Also returns the indices of the nodes that are in the candidate set."""
        batch_size = encoded_nodes.shape[0]
        problem_size = encoded_nodes.shape[1]
        device = encoded_nodes.device
        sample_idx = torch.arange(batch_size, device=device)
        current_node_idx = dynamic_state.current_node_idx

        # First and last node embeddings, for local decoding update their static embeddings
        depot_node_embedding = self.depot_node_embedding(encoded_nodes[sample_idx, 0])
        current_node_embedding = self.current_node_embedding(encoded_nodes[sample_idx, current_node_idx])

        context = self.build_context_vector(dynamic_state, dis_matrix)
        context_embedding = self.context_embedding(context)

        # Mask out infeasible next nodes
        distances = dis_matrix[sample_idx, current_node_idx]
        infeasible = dynamic_state.ninf_mask == float("-inf")
        distances = distances.masked_fill(infeasible, float("inf"))

        k = min(K_NEAREST_NEIGHBORS, problem_size)
        local_candidates_indexes = torch.topk(distances, k=k, dim=1, largest=False).indices
        local_candidates_embedding = encoded_nodes.gather(
            1, local_candidates_indexes.unsqueeze(-1).expand(-1, -1, self.embedding_dim)
        )
        # Add a dim (size 1) expand it (to size embedding), that dim has the value repeated (node idx repeated d times)
        # We need to get each value from the embedding dim separately

        candidate_set = torch.cat(
            [
                depot_node_embedding.unsqueeze(1),
                local_candidates_embedding,
                current_node_embedding.unsqueeze(1),
                context_embedding.unsqueeze(1),
            ],
            dim=1,
        )
        return candidate_set, local_candidates_indexes

    def local_distance_matrix(
        self,
        dis_matrix: torch.Tensor,
        candidate_node_indices: torch.Tensor,
        current_node_idx: torch.Tensor,
        normalized_coords: torch.Tensor,
        problem_size: int,
    ) -> torch.Tensor:
        """Build normalized distances for [depot, k candidates, current, context]."""
        batch_size = candidate_node_indices.size(0)
        device = dis_matrix.device
        sample_idx = torch.arange(batch_size, dtype=torch.long, device=device)

        # indices for which we later need to get the distances for [depot, candidates, current]
        sequence_indices = torch.cat(
            [
                torch.zeros(batch_size, 1, dtype=torch.long, device=device),
                candidate_node_indices,
                current_node_idx.unsqueeze(1),
            ],
            dim=1,
        )
        seq_len = sequence_indices.size(1)

        if problem_size > LARGE_INSTANCE_THRESHOLD:
            coord_indices = sequence_indices.unsqueeze(2).expand(-1, -1, 2)
            coords = normalized_coords.gather(dim=1, index=coord_indices)
            local_dist = torch.cdist(coords, coords, p=2)
            local_dist.diagonal(dim1=-2, dim2=-1).zero_()
        else:
            index_un = sequence_indices.unsqueeze(1)
            local_dist = dis_matrix[sample_idx.unsqueeze(1).unsqueeze(2), index_un, index_un.transpose(1, 2)]

        # Context token has no coordinates; pad with zeros for AFM.
        # For AttentionFusionModule expect x and dis_matrix to be of same size. add a zero column and row to local_dist matrix
        zeros_row = torch.zeros(batch_size, 1, seq_len, device=device, dtype=local_dist.dtype)
        local_dist = torch.cat([local_dist, zeros_row], dim=1)
        zeros_col = torch.zeros(batch_size, seq_len + 1, 1, device=device, dtype=local_dist.dtype)
        return torch.cat([local_dist, zeros_col], dim=2)

    def build_context_vector(self, dynamic_state: DynamicState, dis_matrix: torch.Tensor) -> torch.Tensor:
        """Returns the dynamic state vector"""
        batch_size = dynamic_state.current_time.shape[0]
        sample_idx = torch.arange(batch_size, device=dynamic_state.current_time.device)

        time_norm = normalize_time_for_model(dynamic_state.current_time)
        capacity_norm = dynamic_state.remaining_capacity
        dist_to_depot = dis_matrix[sample_idx, dynamic_state.current_node_idx, 0]

        return torch.stack([time_norm, capacity_norm, dist_to_depot], dim=-1)  # (B,3)


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
