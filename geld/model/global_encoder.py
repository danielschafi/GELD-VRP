"""Global-view Encoder (GE) with RALA."""

import torch.nn as nn

from geld.model.attention import RegionAverageLinearAttention, FeedForwardModule


class GlobalEncoder(nn.Module):
    """Global-view Encoder (GE): embed nodes and extract global features via RALA."""

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = model_params["embedding_dim"]
        self.embedding = nn.Linear(2, embedding_dim, bias=True)
        self.layers_global = nn.ModuleList([EncoderLayer(**model_params)])

    def forward(self, data, region):
        """Project normalized coordinates and run broad global assessment."""
        embedded_input = self.embedding(data)
        out = embedded_input
        for layer in self.layers_global:
            out = layer(out, region)
        return out


class EncoderLayer(nn.Module):
    """Single GE layer: RALA followed by feed-forward."""

    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params["embedding_dim"]
        self.attentionlayer = RegionAverageLinearAttention(model_params=model_params)
        self.multi_head_combine = nn.Linear(embedding_dim, embedding_dim)
        self.feedForward = FeedForwardModule(**model_params)

    def forward(self, input_tensor, index_region):
        """Apply RALA and FFN with residual connections."""
        multi_head_out = self.multi_head_combine(
            self.attentionlayer(input_tensor, index_region)
        )
        out1 = input_tensor + multi_head_out
        hidden_states = self.feedForward(out1)
        return out1 + hidden_states
