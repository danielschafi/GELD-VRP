import torch

from geld.data.augmentations import extract_subpath_batch
from geld.search.prc import accept_repair_if_shorter


def test_extract_subpath_batch_shape():
    problems = torch.rand(2, 20, 2)
    solution = torch.stack([torch.randperm(20) for _ in range(2)])
    indices = torch.arange(0, 20, step=5)[:3]
    sub_problems, sub_rank = extract_subpath_batch(problems, solution, indices, 4)
    assert sub_problems.shape == (2, 3, 4, 2)
    assert sub_rank.shape == (2, 3, 4)


def test_accept_repair_keeps_shorter_tour():
    full_tour = torch.tensor([[0, 1, 2, 3, 4, 5]])
    indices = torch.tensor([0, 3])
    subpath_length = 2
    repaired = torch.tensor([[[1, 0], [1, 0]]])
    length_before = torch.tensor([10.0, 10.0])
    length_after = torch.tensor([8.0, 12.0])
    updated = accept_repair_if_shorter(
        repaired, length_before, length_after, indices, subpath_length, full_tour.clone()
    )
    assert updated.shape == full_tour.shape
    # First segment was improved; second was not (length increased)
    assert updated[0, 0].item() == 1
    assert updated[0, 1].item() == 0
