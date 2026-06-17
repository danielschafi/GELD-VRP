"""Local-view Decoder (LD) with k-NN local selection and AFM."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from geld.model.geometry import LARGE_INSTANCE_THRESHOLD
from geld.model.attention import RMSNorm, FeedForwardModule, AttentionFusionModule

K_NEAREST_NEIGHBORS = 99


class DecoderLayer(nn.Module):
    """Single LD layer: AFM with RMSNorm and feed-forward."""

    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params["embedding_dim"]
        head_num = model_params["head_num"]
        qkv_dim = model_params["qkv_dim"]
        self.input_layernorm = RMSNorm(embedding_dim)
        self.post_attention_layernorm = RMSNorm(embedding_dim)
        self.attentionlayer = AttentionFusionModule(model_params=model_params)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim, bias=False)
        self.feedForward = FeedForwardModule(**model_params)

    def forward(self, input_tensor, dis_matrix):
        """Apply pre-norm AFM and post-norm FFN with residuals."""
        hidden_states = self.input_layernorm(input_tensor)
        multi_head_out = self.multi_head_combine(self.attentionlayer(hidden_states, dis_matrix))
        out1 = input_tensor + multi_head_out
        hidden_states = self.post_attention_layernorm(out1)
        hidden_states = self.feedForward(hidden_states)
        return out1 + hidden_states


class LocalDecoder(nn.Module):
    """Local-view Decoder (LD): refined local selection over k-NN candidate set."""

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = model_params["embedding_dim"]
        decoder_layer_num = model_params["decoder_layer_num"]
        self.data = None
        self.first_node_embedding = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.last_node_embedding = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.layers_global = nn.ModuleList([DecoderLayer(**model_params) for _ in range(decoder_layer_num)])
        self.final_projection = nn.Linear(embedding_dim, 1, bias=True)

    def _select_local_candidates(
        self,
        data,
        selected_tour,
        problem_size,
        batch_size,
        dis_matrix,
        beam_search=False,
        beam_size=16,
    ):
        """Build Kset: all unvisited nodes at train time, k-NN at inference."""
        if beam_search:
            effective_batch = batch_size * beam_size
        else:
            effective_batch = batch_size

        remaining_mask = torch.arange(problem_size, device=data.device)[None, :].repeat(effective_batch, 1)
        selected_indices = selected_tour.type(torch.long)
        batch_indices = torch.arange(effective_batch, dtype=torch.long, device=data.device)[:, None].expand(
            effective_batch, selected_indices.shape[1]
        )
        remaining_mask[batch_indices, selected_indices] = -2
        num_remaining = problem_size - selected_tour.shape[1]

        if self.training:
            unselected = remaining_mask[torch.gt(remaining_mask, -1)].view(batch_size, num_remaining)
        else:
            if num_remaining <= K_NEAREST_NEIGHBORS:
                unselected = remaining_mask[torch.gt(remaining_mask, -1)].view(effective_batch, num_remaining)
            else:
                last_node = selected_tour[:, -1]
                if beam_search:
                    last_node = last_node.view(batch_size, beam_size, 1).expand(-1, -1, problem_size)
                    distance = dis_matrix.gather(1, last_node).view(batch_size * beam_size, -1)
                else:
                    distance = dis_matrix.gather(
                        1,
                        last_node.view(effective_batch, 1, 1).expand(-1, -1, problem_size),
                    ).squeeze(1)
                mask = torch.zeros_like(distance)
                mask[batch_indices, selected_tour] = 1e2
                unselected = torch.topk(distance + mask, dim=1, k=K_NEAREST_NEIGHBORS, largest=False).indices

        emb_dim = data.shape[-1]
        if beam_search:
            expanded_data = data.unsqueeze(1).expand(-1, beam_size, -1, -1)
            local_embeddings = torch.gather(
                expanded_data,
                2,
                unselected.view(batch_size, beam_size, -1, 1).expand(-1, -1, -1, emb_dim),
            )
            local_embeddings = local_embeddings.view(effective_batch, -1, emb_dim)
        else:
            local_embeddings = torch.gather(data, 1, unselected.unsqueeze(2).expand(-1, -1, emb_dim))
        return local_embeddings, unselected

    def _gather_node_embeddings(self, encoded_nodes, node_index, beam_search=False, beam_size=16):
        """Gather encoder embeddings for destination and previously selected nodes."""
        batch_size = node_index.size(0)
        num_selected = node_index.size(1)
        embedding_dim = encoded_nodes.size(2)
        if beam_search:
            encoded_nodes = encoded_nodes.unsqueeze(1).expand(batch_size // beam_size, beam_size, -1, embedding_dim)
            gathering_index = node_index.view(batch_size // beam_size, beam_size, num_selected, 1).expand(
                batch_size // beam_size, beam_size, num_selected, embedding_dim
            )
            picked = encoded_nodes.gather(dim=2, index=gathering_index)
            return picked.view(batch_size, num_selected, embedding_dim)
        gathering_index = node_index[:, :, None].expand(batch_size, num_selected, embedding_dim)
        return encoded_nodes.gather(dim=1, index=gathering_index)

    def _local_distance_matrix(
        self,
        unselected_nodes,
        dis_matrix,
        batch_size,
        beam_search,
        beam_size,
        problem_size,
    ):
        """Extract normalized subtopology distance matrix among local candidates."""
        if problem_size > LARGE_INSTANCE_THRESHOLD:
            if beam_search:
                index_un = unselected_nodes.view(batch_size, beam_size, -1, 1).expand(batch_size, beam_size, -1, 2)
                coords = (
                    self.data.view(batch_size, 1, -1, 2)
                    .expand(batch_size, beam_size, -1, 2)
                    .gather(dim=2, index=index_un)
                    .view(batch_size * beam_size, -1, 2)
                )
            else:
                index_un = unselected_nodes.unsqueeze(2).expand(batch_size, -1, 2)
                coords = self.data.gather(dim=1, index=index_un)
            local_dist = torch.cdist(coords, coords, p=2)
            local_dist.diagonal(dim1=-2, dim2=-1).zero_()
            return local_dist

        if beam_search:
            index_1 = torch.arange(batch_size, dtype=torch.long).unsqueeze(1).unsqueeze(2).unsqueeze(3)
            index_2 = torch.arange(beam_size, dtype=torch.long).unsqueeze(0).unsqueeze(2).unsqueeze(3)
            index_un = unselected_nodes.view(batch_size, beam_size, -1, 1)
            expanded = dis_matrix.unsqueeze(1).expand(batch_size, beam_size, problem_size, problem_size)
            return expanded[index_1, index_2, index_un, index_un.transpose(2, 3)].view(
                batch_size * beam_size,
                unselected_nodes.shape[-1],
                unselected_nodes.shape[-1],
            )

        index_1 = torch.arange(batch_size, dtype=torch.long).unsqueeze(1).unsqueeze(2)
        index_un = unselected_nodes.unsqueeze(1)
        return dis_matrix[index_1, index_un, index_un.transpose(1, 2)]

    def forward(self, data, selected_tour, dis_matrix, beam_search=False, beam_size=16):
        """Return per-node selection probabilities p_θ(a_t) at the current MDP step."""
        batch_size = data.shape[0]
        problem_size = data.shape[1]

        local_embeddings, unselected_nodes = self._select_local_candidates(
            data,
            selected_tour,
            problem_size,
            batch_size,
            dis_matrix,
            beam_search,
            beam_size,
        )
        first_and_last = self._gather_node_embeddings(
            data,
            selected_tour[:, [0, -1]],
            beam_search=beam_search,
            beam_size=beam_size,
        )
        embedded_first = self.first_node_embedding(first_and_last[:, 0])
        embedded_last = self.last_node_embedding(first_and_last[:, 1])
        decoder_input = torch.cat(
            (embedded_first.unsqueeze(1), local_embeddings, embedded_last.unsqueeze(1)),
            dim=1,
        )
        unselected_nodes = torch.cat(
            (
                selected_tour[:, 0].unsqueeze(1),
                unselected_nodes,
                selected_tour[:, -1].unsqueeze(1),
            ),
            dim=-1,
        ).type(torch.long)

        local_dist = self._local_distance_matrix(
            unselected_nodes,
            dis_matrix,
            batch_size,
            beam_search,
            beam_size,
            problem_size,
        )
        out = decoder_input
        for layer in self.layers_global:
            out = layer(out, local_dist)
        logits = self.final_projection(out).squeeze(-1)
        logits[:, [0, -1]] = logits[:, [0, -1]] + float("-inf")

        probs = F.softmax(logits, dim=-1)[:, 1:-1]
        small_mask = torch.le(probs, 1e-5)
        probs_fixed = probs.clone()
        probs_fixed[small_mask] = probs_fixed[small_mask] + torch.tensor(
            1e-7, dtype=probs_fixed[small_mask].dtype, device=probs.device
        )
        probs = probs_fixed

        if probs.isnan().any():
            nan_mask = torch.isnan(probs)
            row_indices = nan_mask.any(dim=1).nonzero(as_tuple=True)[0]
            probs[nan_mask] = 1e-5
            probs[row_indices, 0] = 1

        output_batch_size = batch_size * beam_size if beam_search else batch_size
        full_probs = torch.zeros(output_batch_size, problem_size, device=out.device) + 1e-5
        batch_indices = torch.arange(output_batch_size, dtype=torch.long, device=out.device)[:, None]
        full_probs[batch_indices, unselected_nodes[:, 1:-1]] = probs
        full_probs[batch_indices, selected_tour] = 1e-20
        return full_probs
