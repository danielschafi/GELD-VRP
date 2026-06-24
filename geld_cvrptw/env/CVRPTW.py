"""
Combining what we can use from MVMoE's VRPPTWEnv.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from geld_cvrptw.data.loaders import load_cvrptw_data_with_labels, TOUR_PAD_VALUE
from geld_cvrptw.data.augmentations import apply_rotation


@dataclass
class StaticState:
    """
    Static State information.
    Fixed one time, does not change.
    Global Encoder input
    """

    depot_coords: torch.Tensor
    node_coords: torch.Tensor
    node_demand: torch.Tensor
    node_service_time: torch.Tensor
    node_tw_start: torch.Tensor
    node_tw_end: torch.Tensor
    label_tour: torch.Tensor | None = None  # SL only


@dataclass
class DynamicState:
    """
    Dynamic State information.
    Changes every step
    Local Decoder input
    """

    num_completed_steps: int  # Number of construction steps completed (t).
    current_node_idx: torch.Tensor | None  # shape: (batch,) — last visited node index; None before the first step.
    current_node_coord: torch.Tensor  # shape: (batch, 2) — coordinates of current_node.

    constructed_tour: torch.Tensor  # shape: (batch, t) — teacher-forced prefix (decoder input).
    model_tour: (
        torch.Tensor
    )  # shape: (batch, t) — model's argmax prefix (SL quality tracking). Not necessarily valid tour

    ninf_mask: torch.Tensor  # shape: (batch, num_nodes) — -inf masks infeasible next nodes.
    visited_ninf_flag: torch.Tensor  # shape: (batch, num_nodes) — visited nodes mask only

    remaining_capacity: torch.Tensor  # shape: (batch,) — remaining vehicle capacity.
    current_time: torch.Tensor  # shape: (batch,) — clock after serving current_node.
    length: torch.Tensor  # shape: (batch,) — distance traveled on the current route segment.

    done: torch.Tensor  # shape: (batch,) bool — episode finished for each instance.

    def clone(self) -> DynamicState:
        return DynamicState(
            num_completed_steps=self.num_completed_steps,
            current_node_idx=self.current_node_idx.clone() if self.current_node_idx is not None else None,
            current_node_coord=self.current_node_coord.clone(),
            constructed_tour=self.constructed_tour.clone(),
            model_tour=self.model_tour.clone(),
            ninf_mask=self.ninf_mask.clone(),
            visited_ninf_flag=self.visited_ninf_flag.clone(),
            remaining_capacity=self.remaining_capacity.clone(),
            current_time=self.current_time.clone(),
            length=self.length.clone(),
            done=self.done.clone(),
        )


class CVRPTWEnv:
    def __init__(self, **env_params):
        self.env_params = env_params
        self.device = (
            torch.device("cuda", torch.cuda.current_device())
            if "device" not in env_params.keys()
            else env_params["device"]
        )

        # Full dataset — set by load_raw_data
        self.full_node_coords = None
        self.full_node_demand = None
        self.full_node_tw_start = None
        self.full_node_tw_end = None
        self.full_node_service_time = None
        self.full_label_tours = None
        self.full_label_costs = None

        # Active batch — set by load_problems
        self.batch_offset = None
        self.batch_size = None
        self.num_nodes = None
        self.batch_coords = None
        self.batch_demand = None
        self.batch_tw_start = None
        self.batch_tw_end = None
        self.batch_service_time = None
        self.batch_label_tours = None
        self.batch_label_costs = None

        # Episode constants
        self.speed = 1.0
        self.depot_tw_start = 0.0
        self.depot_tw_end = 3.0

    def load_all_data(self):
        """
        Currently only loads the train dataset for stage 1 supervised learning training
        """
        dataset = load_cvrptw_data_with_labels()

        self.full_node_coords = dataset["coords"].requires_grad_(False)
        self.full_node_demand = dataset["demand"].requires_grad_(False)
        self.full_node_tw_start = dataset["tw_start"].requires_grad_(False)
        self.full_node_tw_end = dataset["tw_end"].requires_grad_(False)
        self.full_node_service_time = dataset["service_time"].requires_grad_(False)
        self.full_label_tours = dataset["label_tours"].requires_grad_(False)
        self.full_label_costs = dataset["costs"].requires_grad_(False)

    def load_problem_tensors(
        self,
        coords: torch.Tensor,
        demand: torch.Tensor,
        tw_start: torch.Tensor,
        tw_end: torch.Tensor,
        service_time: torch.Tensor,
    ) -> None:
        """Load one batch tensors for inference/evaluation"""
        self.batch_coords = coords
        self.batch_demand = demand
        self.batch_tw_start = tw_start
        self.batch_tw_end = tw_end
        self.batch_service_time = service_time
        self.batch_label_tours = None
        self.batch_label_costs = None
        self.batch_size = coords.size(0)
        self.num_nodes = coords.size(1)
        self.sync_batch_to_device()

    def num_samples(self) -> int:
        """Number of loaded training instances."""
        return len(self.full_node_coords)

    def load_one_batch_of_problems(self, batch_offset: int, batch_size: int, train: bool = True):
        """Load one batch of samples."""
        self.batch_offset = batch_offset
        self.batch_size = batch_size

        # Load just one batch of problems
        self.batch_coords = self.full_node_coords[batch_offset : batch_offset + batch_size]  # was self.problems
        self.batch_demand = self.full_node_demand[batch_offset : batch_offset + batch_size]
        self.batch_tw_start = self.full_node_tw_start[batch_offset : batch_offset + batch_size]
        self.batch_tw_end = self.full_node_tw_end[batch_offset : batch_offset + batch_size]
        self.batch_service_time = self.full_node_service_time[batch_offset : batch_offset + batch_size]
        self.batch_label_tours = self.full_label_tours[batch_offset : batch_offset + batch_size]
        self.batch_label_costs = self.full_label_costs[batch_offset : batch_offset + batch_size]

        self.num_nodes = self.batch_coords.shape[1]

        # Reversal does not work because of tw constraints
        # self.batch_label_tours = maybe_reverse_tour(self.batch_label_tours)

        if train:
            rotation_id = torch.randint(low=0, high=8, size=[1])[0].item()
            self.batch_coords = apply_rotation(self.batch_coords, rotation_id)

        self.sync_batch_to_device()

    def reset(self, batch_size=None) -> tuple[StaticState, DynamicState]:
        """Start a new episode; return static and initial dynamic state."""
        if batch_size is not None:
            self.batch_size = batch_size

        static_state = self._build_static_state()
        dynamic_state = DynamicState(
            num_completed_steps=0,
            current_node_idx=None,
            current_node_coord=self.batch_coords[:, 0, :],
            # The t-1 nodes based on which the decoder will predict the t-th node. tracks tour under construction
            constructed_tour=torch.zeros((self.batch_size, 0), dtype=torch.long, device=self.device),
            # Training: Nodes that have been predicted by argmax over model pred.
            model_tour=torch.zeros((self.batch_size, 0), dtype=torch.long, device=self.device),
            ninf_mask=torch.zeros(self.batch_size, self.num_nodes, device=self.device),
            visited_ninf_flag=torch.zeros(self.batch_size, self.num_nodes, device=self.device),
            remaining_capacity=torch.ones(self.batch_size, device=self.device),
            current_time=torch.zeros(self.batch_size, device=self.device),
            length=torch.zeros(self.batch_size, device=self.device),
            done=torch.zeros(self.batch_size, dtype=torch.bool, device=self.device),
        )
        return static_state, dynamic_state

    def step(
        self,
        teacher_node_idx: torch.Tensor,
        predicted_node_idx: torch.Tensor,
        dynamic_state: DynamicState,
    ) -> DynamicState:
        """
        Append selected nodes to tours and return updated dynamic state.
        Masking is already applied for next step decoding.

        Clones the input state first so the caller's ``dynamic_state`` is left unchanged.
        """
        dynamic_state = dynamic_state.clone()

        # Transitioning to new state
        dynamic_state.num_completed_steps += 1
        dynamic_state.current_node_idx = teacher_node_idx
        at_the_depot = teacher_node_idx == 0

        # None adds a size 1 dim, so that they are matching and can be concatenated.
        dynamic_state.constructed_tour = torch.cat(
            (dynamic_state.constructed_tour, dynamic_state.current_node_idx[:, None]), dim=1
        )
        dynamic_state.model_tour = torch.cat((dynamic_state.model_tour, predicted_node_idx[:, None]), dim=1)

        # Tracking / Stats
        added_length = self._update_tour_length(dynamic_state, at_the_depot)
        self._update_remaining_capacity(dynamic_state, at_the_depot)
        self._update_current_time(dynamic_state, at_the_depot, added_length)
        # Update mask based on feasibility
        self._apply_visited_constraint(dynamic_state, at_the_depot)
        self._apply_capacity_constraint(dynamic_state)
        self._apply_time_window_constraint(dynamic_state)

        # check which ones are finished / have visited all nodes.
        new_done = (dynamic_state.visited_ninf_flag == float("-inf")).all(dim=-1)
        dynamic_state.done = dynamic_state.done | new_done  # depot feasibility switches multiple times, thats why not just + on top
        dynamic_state.ninf_mask[:, 0] = torch.where(
            dynamic_state.done, torch.zeros_like(dynamic_state.ninf_mask[:, 0]), dynamic_state.ninf_mask[:, 0]
        )

        return dynamic_state

    def _update_tour_length(self, dynamic_state: DynamicState, at_the_depot: torch.Tensor) -> torch.Tensor:
        """Updates tour length. Adds the distance between previous and current node to the length"""
        prev_node_coord = dynamic_state.current_node_coord
        sample_idx = torch.arange(self.batch_size, device=self.device)
        dynamic_state.current_node_coord = self.batch_coords[sample_idx, dynamic_state.current_node_idx]
        added_length = (dynamic_state.current_node_coord - prev_node_coord).norm(p=2, dim=-1)
        dynamic_state.length += added_length
        dynamic_state.length[at_the_depot] = 0  # reset at depot
        return added_length

    def _update_remaining_capacity(self, dynamic_state: DynamicState, at_the_depot: torch.Tensor) -> None:
        """subtracts the serviced nodes demand from the vehicles remaining capacity"""
        sample_idx = torch.arange(self.batch_size, device=self.device)
        selected_demand = self.batch_demand[sample_idx, dynamic_state.current_node_idx]
        dynamic_state.remaining_capacity -= selected_demand
        dynamic_state.remaining_capacity[at_the_depot] = 1  # Capacity refilled at the depot

    def _update_current_time(
        self,
        dynamic_state: DynamicState,
        at_the_depot: torch.Tensor,
        added_length: torch.Tensor,
    ) -> None:
        """
        Current time: end time of serving the current node / time where vehicle can move to next node
        prev node ──travel (added_length/speed)──► arrive ──maybe wait──► service ──► current_time

        Params:
        - added_length: the distance that has been travelled from previous to current node
        """
        sample_idx = torch.arange(self.batch_size, device=self.device)
        arrival_time = dynamic_state.current_time + added_length / self.speed
        earliest_possible_service_start = torch.max(
            arrival_time, self.batch_tw_start[sample_idx, dynamic_state.current_node_idx]
        )
        dynamic_state.current_time = (
            earliest_possible_service_start + self.batch_service_time[sample_idx, dynamic_state.current_node_idx]
        )
        dynamic_state.current_time[at_the_depot] = 0  # clock resets at depot ("new vehicle" starts at t=0)

    def _apply_visited_constraint(self, dynamic_state: DynamicState, at_the_depot: torch.Tensor) -> None:
        """Masks out the nodes already visited"""
        # Incrementally masking out where we have been each step
        sample_idx = torch.arange(self.batch_size, device=self.device)
        dynamic_state.visited_ninf_flag[sample_idx, dynamic_state.current_node_idx] = float("-inf")
        dynamic_state.visited_ninf_flag[:, 0] = torch.where(
            at_the_depot,
            dynamic_state.visited_ninf_flag[:, 0],
            torch.zeros_like(dynamic_state.visited_ninf_flag[:, 0]),
        )  # if not at depot allow visit depot
        dynamic_state.ninf_mask = (
            dynamic_state.visited_ninf_flag.clone()
        )  # ninf mask is the one used by decoder. but we need to maintain visited mask

    def _apply_capacity_constraint(self, dynamic_state: DynamicState) -> None:
        """Mask out nodes that would exceed vehicle capacity"""
        round_error_tol = 0.00001
        demand_exceeds_capacity = (
            dynamic_state.remaining_capacity[:, None] + round_error_tol < self.batch_demand
        )
        dynamic_state.ninf_mask[demand_exceeds_capacity] = float("-inf")

    def _apply_time_window_constraint(self, dynamic_state: DynamicState) -> None:
        """Mask out where we cant complete service within the time window"""
        round_error_tol = 0.00001

        # 1. Earliest possible service start time for all nodes max(tw start, currTime + travelTime)
        dist = (dynamic_state.current_node_coord.unsqueeze(1) - self.batch_coords).norm(p=2, dim=-1)
        travel_time = dist / self.speed
        earliest_possible_service_start = torch.max(
            dynamic_state.current_time.unsqueeze(1) + travel_time, self.batch_tw_start
        )

        # 2. service starts after tw end -> infeasible
        is_out_of_tw = earliest_possible_service_start > self.batch_tw_end + round_error_tol
        dynamic_state.ninf_mask[is_out_of_tw] = float("-inf")

        # 3. Return to depot would be after depot tw end -> infeasible
        depot_coords = self.batch_coords[:, 0:1, :]
        dist_next_node_to_depot = (depot_coords - self.batch_coords).norm(p=2, dim=-1)
        travel_time_to_depot = dist_next_node_to_depot / self.speed
        return_is_out_of_depot_tw = (
            earliest_possible_service_start + self.batch_service_time + travel_time_to_depot
            > self.batch_tw_end[:, 0:1] + round_error_tol
        )
        dynamic_state.ninf_mask[return_is_out_of_depot_tw] = float("-inf")

    def compute_tour_length(
        self,
        coords: torch.Tensor,
        tour: torch.Tensor,
        tour_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Total Euclidean distance along a node-index tour.
        Sums consecutive leg lengths; ignores padded tail when tour_lengths is given.
        """
        batch_size = coords.size(0)
        if tour_lengths is None:
            tour_lengths = (tour != TOUR_PAD_VALUE).sum(dim=1)

        batch_idx = torch.arange(batch_size, device=coords.device)
        step_idx = torch.arange(tour.size(1), device=coords.device)
        safe_tour = tour.clamp(min=0)

        ordered = coords[batch_idx[:, None], safe_tour]
        rolled = ordered.roll(shifts=-1, dims=1)
        segment_lengths = (ordered - rolled).norm(p=2, dim=-1)

        valid_segments = (step_idx[None, :] + 1) < tour_lengths[:, None]
        return (segment_lengths * valid_segments).sum(dim=1)

    # COMPUTE DEVICE MANAGEMENT

    def set_device(self, device: torch.device):
        """
        Move the complete raw data tensors to the specified device.
        and sets the env's device
        """
        self.device = device

        if isinstance(self.full_node_coords, torch.Tensor):
            self.full_node_coords = self.full_node_coords.to(device)
        if isinstance(self.full_node_demand, torch.Tensor):
            self.full_node_demand = self.full_node_demand.to(device)
        if isinstance(self.full_node_tw_start, torch.Tensor):
            self.full_node_tw_start = self.full_node_tw_start.to(device)
        if isinstance(self.full_node_tw_end, torch.Tensor):
            self.full_node_tw_end = self.full_node_tw_end.to(device)
        if isinstance(self.full_node_service_time, torch.Tensor):
            self.full_node_service_time = self.full_node_service_time.to(device)
        if isinstance(self.full_label_tours, torch.Tensor):
            self.full_label_tours = self.full_label_tours.to(device)
        if isinstance(self.full_label_costs, torch.Tensor):
            self.full_label_costs = self.full_label_costs.to(device)

    def sync_batch_to_device(self):
        """Move only the active batch tensors to the environment's device."""
        if isinstance(self.batch_coords, torch.Tensor) and self.batch_coords is not None:
            self.batch_coords = self.batch_coords.to(self.device)
        if isinstance(self.batch_demand, torch.Tensor):
            self.batch_demand = self.batch_demand.to(self.device)
        if isinstance(self.batch_tw_start, torch.Tensor):
            self.batch_tw_start = self.batch_tw_start.to(self.device)
        if isinstance(self.batch_tw_end, torch.Tensor):
            self.batch_tw_end = self.batch_tw_end.to(self.device)
        if isinstance(self.batch_service_time, torch.Tensor):
            self.batch_service_time = self.batch_service_time.to(self.device)
        if isinstance(self.batch_label_tours, torch.Tensor):
            self.batch_label_tours = self.batch_label_tours.to(self.device)
        if isinstance(self.batch_label_costs, torch.Tensor):
            self.batch_label_costs = self.batch_label_costs.to(self.device)

    def shuffle_full_data(self):
        """Shuffle stored training instances."""
        index = torch.randperm(len(self.full_node_coords)).long()

        self.full_node_coords = self.full_node_coords[index]
        self.full_node_demand = self.full_node_demand[index]
        self.full_node_tw_start = self.full_node_tw_start[index]
        self.full_node_tw_end = self.full_node_tw_end[index]
        self.full_node_service_time = self.full_node_service_time[index]
        self.full_label_tours = self.full_label_tours[index]
        self.full_label_costs = self.full_label_costs[index]

    def _build_static_state(self) -> StaticState:
        return StaticState(
            depot_coords=self.batch_coords[:, 0],
            node_coords=self.batch_coords,
            node_demand=self.batch_demand,
            node_tw_start=self.batch_tw_start,
            node_tw_end=self.batch_tw_end,
            node_service_time=self.batch_service_time,
            label_tour=self.batch_label_tours,
        )
