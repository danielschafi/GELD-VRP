"""Re-Construction (RC) post-processor for CVRPTW."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from geld_cvrptw.data.augmentations import apply_random_rotation
from geld_cvrptw.data.loaders import TOUR_PAD_VALUE
from geld_cvrptw.env.CVRPTW import CVRPTWEnv, DynamicState, StaticState
from geld_cvrptw.inference.types import PostProcessor, SolveResult
from geld_cvrptw.model.GELD_CVRPTW import GeldCvrptwModel


@dataclass
class RollingTourState:
    """Depart time and remaining capacity immediately after each tour position is served."""

    depart_time: torch.Tensor  # (batch, tour_len)
    capacity_after: torch.Tensor  # (batch, tour_len)


@dataclass
class RepairWindowPlan:
    """
    Parallel repair windows laid out evenly along the tour.

    Window ``k`` covers positions ``window_starts[k] .. window_starts[k] + window_length - 1``.
    Both endpoint nodes stay fixed; the decoder may reorder the interior.
    """

    window_count: int
    window_starts: torch.Tensor  # (window_count,)
    window_length: int


@dataclass
class _EnvBatchSnapshot:
    batch_size: int
    batch_coords: torch.Tensor
    batch_demand: torch.Tensor
    batch_tw_start: torch.Tensor
    batch_tw_end: torch.Tensor
    batch_service_time: torch.Tensor


@dataclass
class _ModelInferenceSnapshot:
    static_state: StaticState | None
    encoded_nodes: torch.Tensor | None
    normalized_coords: torch.Tensor | None
    dis_matrix: torch.Tensor | None
    depot_tw_end: torch.Tensor | None
    node_to_region_map: torch.Tensor | None


class ReConstruction(PostProcessor):
    """
    Re-Construction post-processor for CVRPTW.

    Each iteration picks several evenly spaced windows on the tour, tries to improve
    them in parallel, and keeps changes that shorten distance while staying feasible.
    """

    def __init__(
        self,
        num_iterations: int = 1000,
        min_window_length: int = 4,
        min_window_count: int = 2,
        diversify_coords: bool = False,
    ) -> None:
        self.num_iterations = num_iterations
        self.min_window_length = min_window_length
        self.min_window_count = min_window_count
        self.diversify_coords = diversify_coords

    @torch.no_grad()
    def refine(self, model: GeldCvrptwModel, env: CVRPTWEnv, result: SolveResult) -> SolveResult:
        """Run RC for ``num_iterations`` and return the refined tour."""
        tour = result.tour.clone()
        model.embed_static_state_once(env._build_static_state())
        schedule = self.build_tour_schedule(env, tour)

        for _ in range(self.num_iterations):
            tour, schedule = self.run_one_iteration(model, env, tour, schedule)

        length = env.compute_tour_length(env.batch_coords, tour)
        return SolveResult(tour=tour, length_normalized=length)

    @torch.no_grad()
    def run_one_iteration(
        self,
        model: GeldCvrptwModel,
        env: CVRPTWEnv,
        tour: torch.Tensor,
        schedule: RollingTourState,
    ) -> tuple[torch.Tensor, RollingTourState]:
        """
        One RC iteration:

        1. Optionally rotate coordinates (off by default — each rotation re-embeds)
        2. Plan evenly spaced repair windows
        3. Try improving every window in parallel
        4. Merge windows that are shorter and still yield a feasible full tour
        """
        origin_coords = env.batch_coords
        if self.diversify_coords:
            rotation_id = self.get_random_rotation_id()
            env.batch_coords = (
                apply_random_rotation(origin_coords, rotation_id) if rotation_id != 0 else origin_coords
            )
            model.embed_static_state_once(env._build_static_state())
            schedule = self.build_tour_schedule(env, tour)

        tour_len = self._effective_tour_length(tour)
        window_count = self.sample_window_count(tour_len)
        start_offset = int(torch.randint(low=0, high=max(tour_len, 1), size=[1])[0])
        plan = self.plan_repair_windows(
            tour_len, window_count, start_offset=start_offset, device=tour.device
        )

        length_before, length_after, repaired_windows = self.attempt_window_repairs(
            model, env, tour, schedule, plan, origin_coords
        )
        positions = self.window_position_table(plan, tour_len)
        tour, schedule = self.merge_improved_windows(
            env,
            tour,
            schedule,
            positions,
            repaired_windows,
            length_before,
            length_after,
        )

        if self.diversify_coords:
            env.batch_coords = origin_coords
            model.embed_static_state_once(env._build_static_state())
        return tour, schedule

    def get_random_rotation_id(self) -> int:
        """
        Pick a coordinate rotation id for this iteration.

        CVRPTW tours are not cyclic (unlike TSP), so we do not roll the tour
        itself — window placement is diversified via ``start_offset`` instead.
        """
        return torch.randint(low=0, high=8, size=[1])[0].item()

    def sample_window_count(self, tour_len: int) -> int:
        """How many parallel repair windows to open this iteration."""
        if tour_len < self.min_window_length:
            return 1

        max_by_length = tour_len // self.min_window_length
        sample_cap = max(self.min_window_count, tour_len // 4)
        upper = min(max_by_length, sample_cap)
        if upper <= self.min_window_count:
            return max(1, upper)

        return int(torch.randint(low=self.min_window_count, high=upper + 1, size=[1])[0])

    def plan_repair_windows(
        self,
        tour_len: int,
        window_count: int,
        *,
        start_offset: int = 0,
        device: torch.device,
    ) -> RepairWindowPlan:
        """Evenly spaced window starts with a shared random window length."""
        spacing = tour_len // window_count
        base_starts = torch.arange(0, tour_len, step=spacing, dtype=torch.long, device=device)[
            :window_count
        ]
        window_starts = (base_starts + start_offset) % max(tour_len, 1)
        max_window_length = max(self.min_window_length, tour_len // window_count)
        window_length = int(
            torch.randint(low=self.min_window_length, high=max_window_length + 1, size=[1])[0]
        )
        return RepairWindowPlan(
            window_count=window_count,
            window_starts=window_starts,
            window_length=window_length,
        )

    def window_position_table(self, plan: RepairWindowPlan, tour_len: int) -> torch.Tensor:
        """Absolute tour indices per window; shape ``(window_count, window_length)``."""
        offsets = torch.arange(plan.window_length, device=plan.window_starts.device)
        positions = plan.window_starts.unsqueeze(1) + offsets.unsqueeze(0)
        return positions % max(tour_len, 1)

    def gather_windows(self, tour: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Slice node indices out of the tour; shape ``(batch, window_count, window_length)``."""
        batch_size = tour.size(0)
        window_count, window_length = positions.shape
        flat_positions = positions.reshape(-1)
        gathered = tour[:, flat_positions]
        return gathered.view(batch_size, window_count, window_length)

    def build_allowed_node_mask(
        self,
        window_nodes: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        """
        Per-window boolean mask of nodes the repair may still visit.

        ``window_nodes`` shape ``(batch, window_count, window_length)``.
        Depot (0) is always allowed so the decoder can insert returns if needed.
        """
        batch_size, window_count, window_length = window_nodes.shape
        device = window_nodes.device

        allowed = torch.zeros(batch_size, window_count, num_nodes, dtype=torch.bool, device=device)
        allowed[:, :, 0] = True
        src = torch.ones(batch_size, window_count, window_length, dtype=torch.bool, device=device)
        allowed.scatter_(2, window_nodes.clamp(min=0), src)
        return allowed

    def build_tour_schedule(self, env: CVRPTWEnv, tour: torch.Tensor) -> RollingTourState:
        """Record depart time and remaining capacity after every served tour position."""
        tour_lengths = self._tour_step_count_per_row(tour)
        depart_time, capacity_after = env.replay_tour_schedule(tour, tour_lengths)
        return RollingTourState(depart_time=depart_time, capacity_after=capacity_after)

    def build_warm_start_states(
        self,
        env: CVRPTWEnv,
        tour: torch.Tensor,
        schedule: RollingTourState,
        plan: RepairWindowPlan,
        window_nodes: torch.Tensor,
        allowed_nodes: torch.Tensor,
    ) -> DynamicState:
        """Batched dynamic state at the depart instant of each window's first node."""
        batch_size, window_count, window_length = window_nodes.shape
        flat_batch = batch_size * window_count
        device = tour.device
        num_nodes = env.num_nodes

        batch_ids = torch.arange(batch_size, device=device).repeat_interleave(window_count)
        start_positions = plan.window_starts.repeat(batch_size)
        start_nodes = window_nodes.view(flat_batch, window_length)[:, 0]

        depart_times = schedule.depart_time[batch_ids, start_positions]
        capacities = schedule.capacity_after[batch_ids, start_positions]
        coords = env.batch_coords[batch_ids, start_nodes]

        visited_flag = torch.zeros(flat_batch, num_nodes, device=device)
        allowed_flat = allowed_nodes.view(flat_batch, num_nodes)
        visited_flag[~allowed_flat] = float("-inf")
        visited_flag[torch.arange(flat_batch, device=device), start_nodes] = float("-inf")

        dynamic_state = DynamicState(
            num_completed_steps=1,
            current_node_idx=start_nodes,
            current_node_coord=coords,
            constructed_tour=start_nodes.unsqueeze(1),
            model_tour=start_nodes.unsqueeze(1),
            ninf_mask=visited_flag.clone(),
            visited_ninf_flag=visited_flag,
            remaining_capacity=capacities,
            current_time=depart_times,
            length=torch.zeros(flat_batch, device=device),
            done=torch.zeros(flat_batch, dtype=torch.bool, device=device),
        )
        env.batch_size = flat_batch
        env._apply_capacity_constraint(dynamic_state)
        env._apply_time_window_constraint(dynamic_state)
        return dynamic_state

    def run_repair_episode(
        self,
        model: GeldCvrptwModel,
        env: CVRPTWEnv,
        dynamic_state: DynamicState,
        end_nodes: torch.Tensor,
        window_length: int,
    ) -> torch.Tensor:
        """Greedy decode inside one window until each row reaches its fixed end node."""
        steps_remaining = window_length - 1
        for step in range(steps_remaining):
            if step == steps_remaining - 1:
                next_node = end_nodes
            else:
                next_node = model(dynamic_state).argmax(dim=1)
            dynamic_state = env.step(next_node, next_node, dynamic_state)
        return dynamic_state.model_tour

    def attempt_window_repairs(
        self,
        model: GeldCvrptwModel,
        env: CVRPTWEnv,
        tour: torch.Tensor,
        schedule: RollingTourState,
        plan: RepairWindowPlan,
        origin_coords: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Try to improve every planned window in parallel.

        Returns ``(length_before, length_after, repaired_windows)``.
        """
        batch_size = tour.size(0)
        positions = self.window_position_table(plan, self._effective_tour_length(tour))
        original_windows = self.gather_windows(tour, positions)

        length_before = self.measure_window_distances(origin_coords, original_windows, plan.window_count)
        allowed_nodes = self.build_allowed_node_mask(original_windows, env.num_nodes)

        env_snapshot = self._snapshot_env_batch(env)
        model_snapshot = self._snapshot_model_inference(model)
        self._repeat_env_batch(env, plan.window_count)
        self._expand_model_for_repair(model, plan.window_count)
        try:
            dynamic_state = self.build_warm_start_states(
                env, tour, schedule, plan, original_windows, allowed_nodes
            )
            end_nodes = original_windows[:, :, -1].reshape(-1)
            repaired_flat = self.run_repair_episode(
                model, env, dynamic_state, end_nodes, plan.window_length
            )
        finally:
            self._restore_env_batch(env, env_snapshot)
            self._restore_model_inference(model, model_snapshot)

        repaired_windows = repaired_flat.view(batch_size, plan.window_count, plan.window_length)
        coords_for_measure = origin_coords.repeat_interleave(plan.window_count, dim=0)
        length_after = self._leg_distance_sum(coords_for_measure, repaired_flat).view(
            batch_size, plan.window_count
        )

        return length_before, length_after, repaired_windows

    def measure_window_distances(
        self,
        coords: torch.Tensor,
        window_nodes: torch.Tensor,
        window_count: int,
    ) -> torch.Tensor:
        """Total Euclidean distance of each window; shape ``(batch, window_count)``."""
        batch_size, _, window_length = window_nodes.shape
        flat_nodes = window_nodes.reshape(batch_size * window_count, window_length)
        coords_expanded = coords.repeat_interleave(window_count, dim=0)
        lengths = self._leg_distance_sum(coords_expanded, flat_nodes)
        return lengths.view(batch_size, window_count)

    def merge_improved_windows(
        self,
        env: CVRPTWEnv,
        tour: torch.Tensor,
        schedule: RollingTourState,
        positions: torch.Tensor,
        repaired_windows: torch.Tensor,
        length_before: torch.Tensor,
        length_after: torch.Tensor,
    ) -> tuple[torch.Tensor, RollingTourState]:
        """Keep strictly shorter windows that still produce a feasible full tour."""
        if not (length_before > length_after).any():
            return tour, schedule

        tour = tour.clone()
        improved = length_before > length_after
        any_accepted = False
        window_count = positions.size(0)
        earliest_patch_idx: int | None = None

        for window_idx in range(window_count):
            if not improved[:, window_idx].any():
                continue

            window_positions = positions[window_idx]
            candidates_to_check = improved[:, window_idx]
            candidate = tour.clone()
            candidate[:, window_positions] = repaired_windows[:, window_idx]

            feasible = env.is_tour_feasible(candidate)
            accept = candidates_to_check & feasible
            if not accept.any():
                continue

            accepted_rows = accept.nonzero(as_tuple=True)[0]
            tour[accepted_rows[:, None], window_positions] = repaired_windows[accepted_rows, window_idx]
            any_accepted = True
            patch_idx = int(window_positions.min().item())
            earliest_patch_idx = patch_idx if earliest_patch_idx is None else min(earliest_patch_idx, patch_idx)

        if any_accepted and earliest_patch_idx is not None:
            schedule = self.patch_tour_schedule(env, tour, schedule, earliest_patch_idx)
        return tour, schedule

    def patch_tour_schedule(
        self,
        env: CVRPTWEnv,
        tour: torch.Tensor,
        schedule: RollingTourState,
        from_idx: int,
    ) -> RollingTourState:
        """Recompute depart times and capacities from ``from_idx`` forward."""
        tour_lengths = self._tour_step_count_per_row(tour)
        depart_time = schedule.depart_time.clone()
        capacity_after = schedule.capacity_after.clone()
        env.replay_tour_schedule(
            tour,
            tour_lengths,
            record_from=from_idx,
            depart_time_out=depart_time,
            capacity_after_out=capacity_after,
        )
        return RollingTourState(depart_time=depart_time, capacity_after=capacity_after)

    def _leg_distance_sum(self, coords: torch.Tensor, nodes: torch.Tensor) -> torch.Tensor:
        batch_idx = torch.arange(coords.size(0), device=coords.device)[:, None]
        ordered = coords[batch_idx, nodes.clamp(min=0)]
        leg_lengths = (ordered[:, 1:] - ordered[:, :-1]).norm(p=2, dim=-1)
        return leg_lengths.sum(dim=1)

    def _effective_tour_length(self, tour: torch.Tensor) -> int:
        if tour.numel() == 0:
            return 0
        return int(self._tour_step_count_per_row(tour).max().item())

    def _tour_step_count_per_row(self, tour: torch.Tensor) -> torch.Tensor:
        return (tour != TOUR_PAD_VALUE).sum(dim=1)

    def _snapshot_env_batch(self, env: CVRPTWEnv) -> _EnvBatchSnapshot:
        return _EnvBatchSnapshot(
            batch_size=env.batch_size,
            batch_coords=env.batch_coords,
            batch_demand=env.batch_demand,
            batch_tw_start=env.batch_tw_start,
            batch_tw_end=env.batch_tw_end,
            batch_service_time=env.batch_service_time,
        )

    def _repeat_env_batch(self, env: CVRPTWEnv, repeat: int) -> None:
        env.batch_coords = env.batch_coords.repeat_interleave(repeat, dim=0)
        env.batch_demand = env.batch_demand.repeat_interleave(repeat, dim=0)
        env.batch_tw_start = env.batch_tw_start.repeat_interleave(repeat, dim=0)
        env.batch_tw_end = env.batch_tw_end.repeat_interleave(repeat, dim=0)
        env.batch_service_time = env.batch_service_time.repeat_interleave(repeat, dim=0)
        env.batch_size = env.batch_coords.size(0)

    def _restore_env_batch(self, env: CVRPTWEnv, snapshot: _EnvBatchSnapshot) -> None:
        env.batch_size = snapshot.batch_size
        env.batch_coords = snapshot.batch_coords
        env.batch_demand = snapshot.batch_demand
        env.batch_tw_start = snapshot.batch_tw_start
        env.batch_tw_end = snapshot.batch_tw_end
        env.batch_service_time = snapshot.batch_service_time

    def _snapshot_model_inference(self, model: GeldCvrptwModel) -> _ModelInferenceSnapshot:
        return _ModelInferenceSnapshot(
            static_state=model.static_state,
            encoded_nodes=model.encoded_nodes,
            normalized_coords=model.normalized_coords,
            dis_matrix=model.dis_matrix,
            depot_tw_end=model.depot_tw_end,
            node_to_region_map=model.node_to_region_map,
        )

    def _expand_model_for_repair(self, model: GeldCvrptwModel, repeat: int) -> None:
        model.encoded_nodes = model.encoded_nodes.repeat_interleave(repeat, dim=0)
        model.normalized_coords = model.normalized_coords.repeat_interleave(repeat, dim=0)
        model.dis_matrix = model.dis_matrix.repeat_interleave(repeat, dim=0)
        model.depot_tw_end = model.depot_tw_end.repeat_interleave(repeat, dim=0)

    def _restore_model_inference(self, model: GeldCvrptwModel, snapshot: _ModelInferenceSnapshot) -> None:
        model.static_state = snapshot.static_state
        model.encoded_nodes = snapshot.encoded_nodes
        model.normalized_coords = snapshot.normalized_coords
        model.dis_matrix = snapshot.dis_matrix
        model.depot_tw_end = snapshot.depot_tw_end
        model.node_to_region_map = snapshot.node_to_region_map
