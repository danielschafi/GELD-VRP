import torch

from geld.model.geometry import (
    build_distance_matrix,
    map_coordinates_to_regions,
    normalize_coordinates,
    tour_length,
)


def test_normalize_coordinates_in_unit_square():
    coords = torch.tensor([[[0.0, 0.0], [2.0, 0.0], [2.0, 4.0]]])
    normalized = normalize_coordinates(coords)
    assert torch.allclose(normalized.min(), torch.tensor(0.0))
    assert torch.allclose(normalized.max(), torch.tensor(1.0))


def test_map_coordinates_to_regions():
    coords = torch.tensor([[[0.0, 0.0], [0.5, 0.5], [0.99, 0.99]]])
    regions = map_coordinates_to_regions(normalize_coordinates(coords))
    assert regions.shape == (1, 3)
    assert regions.min() >= 0
    assert regions.max() <= 8


def test_blocked_and_full_distance_matrix_close(tiny_coordinates):
    normalized = normalize_coordinates(tiny_coordinates)
    full = build_distance_matrix(normalized).float()
    assert full.shape == (2, 4, 4)
    assert torch.allclose(full.diagonal(dim1=-2, dim2=-1), torch.zeros(2, 4))


def test_tour_length_unit_square():
    coords = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]])
    tour = torch.tensor([[0, 1, 2, 3]])
    length = tour_length(coords, tour)
    assert torch.allclose(length, torch.tensor([4.0]))
