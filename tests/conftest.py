import random

import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def fix_seeds():
    random.seed(2024)
    np.random.seed(2024)
    torch.manual_seed(2024)


@pytest.fixture
def tiny_coordinates():
    return torch.tensor(
        [
            [[0.1, 0.2], [0.5, 0.5], [0.9, 0.1], [0.2, 0.8]],
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        ],
        dtype=torch.float32,
    )


@pytest.fixture
def default_model_params():
    return {
        "mode": "test",
        "embedding_dim": 128,
        "decoder_layer_num": 6,
        "qkv_dim": 16,
        "head_num": 8,
        "ff_hidden_dim": 128,
    }
