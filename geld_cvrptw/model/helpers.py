"""Helpers for the GELD-CVRPTW model."""

import torch


LARGE_INSTANCE_THRESHOLD = 10000


def build_distance_matrix(normalized_node_coords: torch.Tensor) -> torch.Tensor:
    """Full pairwise L2 distance matrix on normalized coordinates."""
    distance_matrix = torch.cdist(normalized_node_coords, normalized_node_coords, p=2)
    distance_matrix.diagonal(dim1=-2, dim2=-1).zero_()
    return distance_matrix


def compute_distance_matrix(node_coords: torch.Tensor, block_size: int = 5000) -> torch.Tensor:
    """Blocked pairwise L2 distance matrix for large-scale TSP instances."""
    _, num_nodes, _ = node_coords.size()
    device = node_coords.device
    distance_matrix = torch.empty(node_coords.size(0), num_nodes, num_nodes, dtype=torch.float16, device=device)
    for i in range(0, num_nodes, block_size):
        end_i = min(i + block_size, num_nodes)
        for j in range(0, num_nodes, block_size):
            end_j = min(j + block_size, num_nodes)
            block_i = node_coords[:, i:end_i]
            block_j = node_coords[:, j:end_j]
            block_distances = torch.cdist(block_i, block_j, p=2)
            block_distances.diagonal(dim1=-2, dim2=-1).zero_()
            distance_matrix[:, i:end_i, j:end_j] = block_distances.to(dtype=torch.float16)
    return distance_matrix


def map_coordinates_to_regions(normalized_node_coords: torch.Tensor, grid_size: int = 3) -> torch.Tensor:
    """Assign each node to a region for RALA (m = grid_size²)."""
    region_indices = torch.floor(normalized_node_coords * grid_size).long()
    region_indices = torch.clamp(region_indices, min=0, max=grid_size - 1)

    # for grid size 3 -> 9 cells. {0,...,8}
    # second row (x_cell=1), third column (y_cell=2) = cell 1*3 + 2 = 5 correct (cell nr 0 indexed )
    return region_indices[:, :, 0] * grid_size + region_indices[:, :, 1]


def normalize_coordinates(node_coords: torch.Tensor) -> torch.Tensor:
    """Min-max normalize node coordinates per TSP instance (Eq. 2)."""
    min_val, _ = torch.min(node_coords, dim=1, keepdim=True)
    max_val, _ = torch.max(node_coords, dim=1, keepdim=True)
    max_diff, _ = torch.max(max_val - min_val, dim=-1)
    return (node_coords - min_val) / max_diff.unsqueeze(2)


def tour_length(normalized_node_coords: torch.Tensor, tour: torch.Tensor) -> torch.Tensor:
    """Compute total tour length L(π) for batched Euclidean TSP solutions."""
    gathering_index = tour.unsqueeze(2).expand(normalized_node_coords.shape[0], normalized_node_coords.shape[1], 2)
    ordered = normalized_node_coords.gather(dim=1, index=gathering_index)
    rolled = ordered.roll(dims=1, shifts=-1)
    segment_lengths = ((ordered - rolled) ** 2).sum(2).sqrt()
    return segment_lengths.sum(1)
