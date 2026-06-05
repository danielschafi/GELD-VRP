"""Parallel Repair and Completion (PRC) helpers."""

import torch

from geld.data.augmentations import apply_rotation, extract_subpath_batch


def accept_repair_if_shorter(
    repaired_sub_tour,
    length_before,
    length_after,
    segment_indices,
    subpath_length,
    full_tour,
):
    """Replace sub-solution segments when RC yields a shorter tour."""
    expanded_indices = segment_indices.unsqueeze(1) + torch.arange(subpath_length)
    original_sub_tour = full_tour[:, expanded_indices]
    sorted_original, _ = torch.sort(original_sub_tour, dim=-1, descending=False)
    repaired_mapped = sorted_original.gather(2, repaired_sub_tour.view(sorted_original.shape))
    should_repair = length_before > length_after
    should_repair = should_repair.unsqueeze(1).view(sorted_original.shape[0], sorted_original.shape[1])
    updated = full_tour[:, expanded_indices].clone()
    updated[should_repair] = repaired_mapped[should_repair]
    full_tour[:, expanded_indices] = updated
    return full_tour


def run_repair_episode(model, env, subpath_length):
    """Greedy RC decode on a sub-topology to regenerate a sub-solution."""
    result = env.reset()
    model.prepare_instance(result.coordinates)
    current_step = 0
    while not result.done:
        if current_step == 0:
            teacher_node = env.label_tour[:, -1]
            predicted_node = env.label_tour[:, -1]
        elif current_step == 1:
            teacher_node = env.label_tour[:, 0]
            predicted_node = env.label_tour[:, 0]
        else:
            output = model(env.constructed_tour, env.label_tour, current_step, repair=True)
            teacher_node = output.action
            predicted_node = output.predicted_action
        current_step += 1
        result = env.step(teacher_node, predicted_node)
    return torch.roll(env.constructed_tour, shifts=-1, dims=1)


def apply_prc_iteration(
    model,
    env,
    origin_problem,
    current_tour,
    problem_size,
    num_segments,
    max_subpath_length,
    large_instance=False,
):
    """One PRC iteration: sample sub-solution, repair, accept if improved."""
    interval = problem_size // num_segments
    segment_indices = torch.arange(0, problem_size, step=interval, dtype=torch.long)[:num_segments]
    if large_instance:
        high_max = int(31622 / (num_segments ** 0.5))
        subpath_length = torch.randint(low=4, high=min(high_max, max_subpath_length) + 1, size=[1])[0]
    else:
        subpath_length = torch.randint(low=4, high=max_subpath_length + 1, size=[1])[0]

    new_problem, new_solution = extract_subpath_batch(
        origin_problem, current_tour, segment_indices, subpath_length
    )
    env.problems = new_problem.view(-1, subpath_length, 2)
    env.label_tour = new_solution.view(-1, subpath_length)
    env.batch_size = env.problems.size(0)
    length_before = env.compute_tour_length(env.problems, env.label_tour)
    repaired_sub = run_repair_episode(model, env, subpath_length)
    length_after = env.compute_tour_length(env.problems, env.model_tour)
    return accept_repair_if_shorter(
        repaired_sub,
        length_before,
        length_after,
        segment_indices,
        subpath_length,
        current_tour.clone(),
    )


def run_prc_loop(
    model,
    env,
    origin_problem,
    initial_tour,
    num_iterations,
    *,
    large_instance=False,
    lower_segment_bound=2,
):
    """Run PRC with diversified inputs: shifts, flips, and ×8 augmentations."""
    problem_size = origin_problem.shape[1]
    sample_max = problem_size // (4 if not large_instance else 10)
    if large_instance:
        lower_segment_bound = max(lower_segment_bound, problem_size // 10000)

    best_tour = initial_tour
    working_problem = origin_problem
    for step in range(num_iterations):
        num_segments = torch.randint(low=lower_segment_bound, high=sample_max + 1, size=[1])[0]
        max_subpath = problem_size // num_segments
        if step % 2 != 0:
            best_tour = torch.flip(best_tour, dims=[1])
        best_tour = best_tour.roll(
            dims=1, shifts=int(torch.randint(low=0, high=problem_size, size=[1])[0])
        )
        rotation_id = torch.randint(low=0, high=8, size=[1])[0].item()
        if rotation_id != 0:
            working_problem = apply_rotation(origin_problem, rotation_id)
        else:
            working_problem = origin_problem

        best_tour = apply_prc_iteration(
            model,
            env,
            working_problem,
            best_tour,
            problem_size,
            num_segments,
            max_subpath,
            large_instance=large_instance,
        )
    return best_tour
