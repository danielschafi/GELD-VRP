import pytest
import torch

from geld.model.geometry import tour_length


def test_tour_length_hand_computed():
    # Square with side 1; tour 0->1->2->3->0 has length 4
    problems = torch.tensor(
        [
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
            ]
        ]
    )
    tour = torch.tensor([[0, 1, 2, 3]])
    assert tour_length(problems, tour).item() == pytest.approx(4.0, rel=1e-5)
