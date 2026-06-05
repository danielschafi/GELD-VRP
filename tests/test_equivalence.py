from pathlib import Path

import pytest
import torch

from geld.env.base import StepState
from geld.model.geld_model import GeldModel
from geld.paths import project_root


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


def test_geld_model_loads_pretrained_checkpoint(default_model_params):
    checkpoint_path = project_root() / "result" / "pre_trained_model" / "checkpoint-49.pt"
    if not checkpoint_path.exists():
        pytest.skip("Pretrained checkpoint not available")

    model = GeldModel(**default_model_params)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    assert "encoder.embedding.weight" in checkpoint["model_state_dict"]


def test_geld_model_forward_greedy_step(default_model_params):
    model = GeldModel(**default_model_params)
    model.eval()
    batch_size = 2
    num_nodes = 20
    coords = torch.rand(batch_size, num_nodes, 2)
    state = StepState(data=coords)
    model.prepare_instance(state)
    selected = torch.zeros(batch_size, 1, dtype=torch.long)
    output = model(state, selected, None, current_step=1)
    assert output.predicted_action.shape == (batch_size,)
