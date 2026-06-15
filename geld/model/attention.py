"""Attention blocks: RALA (encoder) and AFM (decoder)."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def reshape_by_heads(qkv: torch.Tensor, head_num: int) -> torch.Tensor:
    """Reshape projected embeddings into multi-head layout."""
    batch_size, num_nodes, _ = qkv.size()
    q_reshaped = qkv.reshape(batch_size, num_nodes, head_num, -1)
    return q_reshaped.transpose(1, 2)


class FeedForwardModule(nn.Module):
    """Position-wise feed-forward network after attention."""

    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params["embedding_dim"]
        ff_hidden_dim = model_params["ff_hidden_dim"]
        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Apply two-layer MLP with ReLU."""
        return self.W2(F.relu(self.W1(input_tensor)))


class RMSNorm(nn.Module):
    """Root-mean-square normalization used in LD attention."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        """Compute RMS normalization on the last dimension."""
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """Normalize and scale input."""
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class RegionAverageLinearAttention(nn.Module):
    """RALA: O(n) attention via regional proxies in GE."""

    def __init__(self, model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = model_params["embedding_dim"]
        self.Wq = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def forward(self, x, index_region):
        """Exchange global information through region-averaged query proxies."""
        head_num = self.model_params["head_num"]
        query_proj = self.Wq(x)
        q = reshape_by_heads(query_proj, head_num=head_num)
        k = reshape_by_heads(self.Wk(x), head_num=head_num)
        v = reshape_by_heads(self.Wv(x), head_num=head_num)
        batch_size = k.size(0)
        key_dim = k.size(3)
        region_mask = torch.zeros(
            batch_size, 9, requires_grad=False, dtype=torch.float, device=x.device
        )
        region_mask.scatter_add_(
            dim=1,
            index=index_region,
            src=torch.ones(index_region.shape, device=x.device),
        )

        region_mask = torch.where(
            region_mask == 0,
            torch.tensor(1, dtype=torch.float, device=region_mask.device),
            region_mask,
        )
        region_sums = torch.zeros(batch_size, 9, x.size(2), device=x.device)
        region_sums.scatter_add_(
            dim=1,
            index=index_region.unsqueeze(-1).expand(query_proj.shape),
            src=query_proj,
        )
        agent = reshape_by_heads(
            region_sums / region_mask.unsqueeze(-1), head_num=head_num
        )

        score = torch.matmul(q, agent.transpose(2, 3)) * (key_dim) ** (-0.5)
        attention1 = F.softmax(score, dim=-1)
        score_k = torch.matmul(agent, k.transpose(2, 3)) * (key_dim) ** (-0.5)
        attention2 = F.softmax(score_k, dim=-1)
        out = torch.matmul(attention2, v)
        out = torch.matmul(attention1, out)
        out_transposed = out.transpose(1, 2)
        return out_transposed.reshape(x.shape)


class AttentionFusionModule(nn.Module):
    """AFM: distance-weighted attention over local decoder inputs."""

    def __init__(self, model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = model_params["embedding_dim"]
        self.Wq = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.alpha = nn.Parameter(torch.tensor(1.0))

    def forward(self, x, dis_matrix):
        """Fuse node features using the normalized distance matrix."""
        head_num = self.model_params["head_num"]
        q = reshape_by_heads(self.Wq(x), head_num=head_num)
        k = reshape_by_heads(self.Wk(x), head_num=head_num)
        v = reshape_by_heads(self.Wv(x), head_num=head_num)

        dis_weight = torch.exp(-self.alpha * np.log2(dis_matrix.size(1)) * dis_matrix)
        k_weight = torch.exp(k)
        weight_1 = torch.einsum("bij, bhik->bhjk", dis_weight, torch.mul(k_weight, v))
        weight_2 = torch.einsum("bij, bhik->bhjk", dis_weight, k_weight)
        weight_3 = torch.div(weight_1, weight_2)
        q_weight = torch.sigmoid(q)
        out = torch.mul(q_weight, weight_3)
        out_transposed = out.transpose(1, 2)
        return out_transposed.reshape(x.shape)
