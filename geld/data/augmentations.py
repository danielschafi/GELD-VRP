"""Coordinate and tour augmentations used in training and PRC."""

import torch


def apply_rotation(problems: torch.Tensor, rotation_id: int) -> torch.Tensor:
    """Apply one of eight coordinate augmentations (POMO-style diversifying inputs)."""
    x = problems[:, :, [0]]
    y = problems[:, :, [1]]
    transforms = {
        0: (x, y),
        1: (1 - x, y),
        2: (x, 1 - y),
        3: (1 - x, 1 - y),
        4: (y, x),
        5: (1 - y, x),
        6: (y, 1 - x),
        7: (1 - y, 1 - x),
    }
    x_new, y_new = transforms[rotation_id]
    return torch.cat((x_new, y_new), dim=2)


def maybe_reverse_tour(solution: torch.Tensor, probability: float = 0.5) -> torch.Tensor:
    """Randomly reverse tour direction for training diversity."""
    if torch.randint(low=0, high=100, size=[1])[0] < int(probability * 100):
        return torch.flip(solution, dims=[1])
    return solution


def sample_training_subpath(problems, solution, length_fix=False, mode="test", repair=False, low_index=4):
    """Extract a random sub-solution and reindexed sub-topology for SL/SIL."""
    problem_size = problems.shape[1]
    batch_size = problems.shape[0]
    embedding_size = problems.shape[2]
    first_node_index = torch.randint(low=0, high=problem_size, size=[1])[0]
    if mode == "test":
        subpath_length = torch.randint(low=low_index, high=problem_size + 1, size=[1])[0]
    elif length_fix:
        subpath_length = problem_size
    else:
        subpath_length = torch.randint(low=low_index, high=problem_size + 1, size=[1])[0]

    doubled_solution = torch.cat([solution, solution], dim=-1)
    sub_tour = doubled_solution[:, first_node_index : first_node_index + subpath_length]
    sub_tour_sorted, rank = torch.sort(sub_tour, dim=-1, descending=False)
    _, sub_tour_rank = torch.sort(rank, dim=-1, descending=False)
    node_order, _ = torch.cat((sub_tour_sorted, sub_tour_sorted), dim=1).type(torch.long).sort(
        dim=-1, descending=False
    )
    batch_indices = torch.arange(batch_size, dtype=torch.long)[:, None].expand(batch_size, node_order.shape[1])
    coord_indices = torch.arange(embedding_size, dtype=torch.long)[None, :].expand(batch_size, embedding_size)
    coord_indices = coord_indices.repeat([1, subpath_length])
    sub_problems = problems[batch_indices, node_order, coord_indices].view(batch_size, subpath_length, 2)

    if repair:
        return sub_problems, sub_tour_rank, first_node_index, subpath_length, doubled_solution
    return sub_problems, sub_tour_rank


def extract_subpath_batch(problems, solution, segment_indices, subpath_length):
    """Extract fixed sub-solutions at given segment starts for PRC."""
    batch_size = problems.shape[0]
    expanded_indices = segment_indices.unsqueeze(1) + torch.arange(subpath_length)
    sub_tour = solution[:, expanded_indices]
    sub_tour_sorted, rank = torch.sort(sub_tour, dim=-1, descending=False)
    _, sub_tour_rank = torch.sort(rank, dim=-1, descending=False)
    node_order, _ = sub_tour_sorted.type(torch.long).sort(dim=-1, descending=False)
    node_order = node_order.view(batch_size, -1)
    batch_indices = torch.arange(batch_size, dtype=torch.long)[:, None].expand(batch_size, node_order.shape[1])
    sub_problems = problems[batch_indices, node_order].view(batch_size, -1, subpath_length, 2)
    return sub_problems, sub_tour_rank
