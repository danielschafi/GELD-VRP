"""
Combining what we can use from MVMoE's VRPPTWEnv.
"""

from dataclasses import dataclass

import torch
from geld_cvrptw.data.loaders import load_cvrptw_data_with_labels
from geld_cvrptw.data.augmentations import apply_rotation, maybe_reverse_tour


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
    label_tour: torch.Tensor | None = None   # SL only

@dataclass
class DynamicState:
    """
    Dynamic State information.
    Changes every step
    Local Decoder input
    """

    num_completed_steps: int # Number of construction steps completed (t).
    current_node_idx: torch.Tensor | None # shape: (batch,) — last visited node index; None before the first step.
    current_node_coord: torch.Tensor # shape: (batch, 2) — coordinates of current_node.
    
    constructed_tour: torch.Tensor # shape: (batch, t) — teacher-forced prefix (decoder input).
    model_tour: torch.Tensor # shape: (batch, t) — model's argmax prefix (SL quality tracking). Not necessarily valid tour
    
    ninf_mask: torch.Tensor # shape: (batch, problem+1) — -inf masks infeasible next nodes.
    remaining_capacity: torch.Tensor # shape: (batch,) — remaining vehicle capacity.
    current_time: torch.Tensor # shape: (batch,) — clock after serving current_node.
    length: torch.Tensor # shape: (batch,) — distance traveled on the current route segment.

    done: torch.Tensor # shape: (batch,) bool — episode finished for each instance.


class CVRPTWEnv:
    def __init__(self, **env_params):
        self.data_path = env_params.get("data_path")
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
        self.problem_size = None
        self.batch_coords = None
        self.batch_demand = None
        self.batch_tw_start = None
        self.batch_tw_end = None
        self.batch_service_time = None
        self.batch_label_tours = None
        self.batch_label_costs = None

        self.depot_node_xy = None

        # Episode constants
        self.speed = 1.0
        self.depot_tw_start =  0.0
        self.depot_tw_end = 3.0

        # Dynamic episode state — set by reset(), updated by step()
        self.num_completed_steps = None
        self.current_node_idx = None
        self.current_node_coord = None
        self.constructed_tour = None
        self.model_tour = None
        self.at_the_depot = None
        self.remaining_capacity = None
        self.visited_ninf_flag = None
        self.ninf_mask = None
        self.current_time = None
        self.length = None
        self.done = None

        self.static_state: StaticState | None = None
        self.dynamic_state: DynamicState | None = None


    def load_raw_data(self):
        """
        Currently only loads the train dataset for stage 1 supervised learning training
        """
        dataset = load_cvrptw_data_with_labels()

        self.full_node_coords = dataset.coords.requires_grad_(False)
        self.full_node_demand = dataset.demand.requires_grad_(False)
        self.full_node_tw_start = dataset.tw_start.requires_grad_(False)
        self.full_node_tw_end = dataset.tw_end.requires_grad_(False)
        self.full_node_service_time = dataset.service_time.requires_grad_(False)
        self.full_label_tours = dataset.label_tours.requires_grad_(False)
        self.full_label_costs = dataset.costs.requires_grad_(False)

    def load_problems(self, batch_offset: int, batch_size: int, train: bool = True):
        """¨
        Load one batch of samples. could be combined with reset step for clarity. # TODO check after we do inference etc. if we can do that.
        """
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

        self.problem_size = self.batch_coords.shape[1]

        self.batch_label_tours = maybe_reverse_tour(self.batch_label_tours)

        if train:
            rotation_id = torch.randint(low=0, high=8, size=[1])[0].item()
            self.batch_coords = apply_rotation(self.batch_coords, rotation_id)

        self.sync_batch_to_device()

    def reset(self, batch_size=None) -> tuple[StaticState, DynamicState]:
        """Start a new episode; return static and initial dynamic state."""
        if batch_size is not None:
            self.batch_size = batch_size

        # The t-1 nodes based on which the decoder will predict the t-th node. tracks tour under construction
        self.constructed_tour = torch.zeros((self.batch_size, 0), dtype=torch.long, device=self.device)
        # Training: Nodes that have been predicted by argmax over model pred.
        self.model_tour = torch.zeros((self.batch_size, 0), dtype=torch.long, device=self.device)

        self.num_completed_steps = 0
        self.current_node_idx = None
        self.current_node_coord = self.batch_coords[:, 0, :]

        self.at_the_depot = torch.ones(self.batch_size, dtype=torch.bool, device=self.device)
        self.remaining_capacity = torch.ones(self.batch_size, device=self.device)
        self.visited_ninf_flag = torch.zeros(
            self.batch_size, self.problem_size + 1, device=self.device
        )
        self.ninf_mask = torch.zeros(self.batch_size, self.problem_size + 1, device=self.device)
        self.done = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)
        self.current_time = torch.zeros(self.batch_size, device=self.device)
        self.length = torch.zeros(self.batch_size, device=self.device)

        self.static_state = self._build_static_state()
        self.dynamic_state = self._build_dynamic_state()
        return self.static_state, self.dynamic_state

    def step(
        self,
        teacher_node_idx: torch.Tensor,
        predicted_node_idx: torch.Tensor,
    ) -> DynamicState:
        """
        Append selected nodes to tours and return updated dynamic state.
        Masking is already applied for next step decoding. 
        """

        # Transitioning to new state
        self.num_completed_steps += 1
        self.current_node_idx = teacher_node_idx
        self.at_the_depot = teacher_node_idx == 0

        # None adds a size 1 dim, so that they are matching and can be concatenated.
        self.constructed_tour = torch.cat((self.constructed_tour, self.current_node_idx[:, None]), dim=1)
        self.model_tour = torch.cat((self.model_tour, predicted_node_idx[:, None]), dim=1)

        # Tracking / Stats
        added_length = self.update_tour_length()
        self.update_remaining_capacity()
        self.update_current_time(added_length)
        # Update mask based on feasibility
        self.apply_visited_constraint()
        self.apply_capacity_constraint()
        self.apply_time_window_constraint()

        # check which ones are finished / have visited all nodes.
        new_done = (self.visited_ninf_flag == float("-inf")).all(dim=-1)
        self.done = self.done | new_done # depot feasibility switches multiple times, thats why not just + on top
        
        # always allow depot choice if done, so there is a legal move for decoder TODO: Check if needed
        self.ninf_mask[:, 0][self.done] = 0

        self.dynamic_state = self._build_dynamic_state()
        return self.dynamic_state


    def update_tour_length(self):
        """Updates tour length. Adds the distance between previous and current node to the length"""
        prev_node_coord = self.current_node_coord
        sample_idx = torch.arange(self.batch_size, device=self.device)
        self.current_node_coord = self.batch_coords[sample_idx, self.current_node_idx]
        added_length = (self.current_node_coord - prev_node_coord).norm(p=2, dim=-1)
        self.length += added_length
        self.length[self.at_the_depot] = 0 # reset at depot
        return added_length

    def update_remaining_capacity(self):
        """subtracts the serviced nodes demand from the vehicles remaining capacity"""
        sample_idx = torch.arange(self.batch_size, device=self.device)
        selected_demand = self.batch_demand[sample_idx, self.current_node_idx]
        self.remaining_capacity -= selected_demand
        self.remaining_capacity[self.at_the_depot] = 1  # Capacity refilled at the depot

    def update_current_time(self, added_length):
        """
        Current time: end time of serving the current node / time where vehicle can move to next node
        prev node ──travel (added_length/speed)──► arrive ──maybe wait──► service ──► current_time

        Params:
        - added_length: the distance that has been travelled from previous to current node
        """
        sample_idx = torch.arange(self.batch_size, device=self.device)
        arrival_time = self.current_time + added_length / self.speed
        earliest_possible_service_start = torch.max(arrival_time, self.batch_tw_start[sample_idx, self.current_node_idx])
        self.current_time = earliest_possible_service_start + self.batch_service_time[sample_idx, self.current_node_idx]
        self.current_time[self.at_the_depot] = 0   # clock resets at depot ("new vehicle" starts at t=0)

    def apply_visited_constraint(self):
        """Masks out the nodes already visited"""
        # Incrementally masking out where we have been each step
        sample_idx = torch.arange(self.batch_size, device=self.device)
        self.visited_ninf_flag[sample_idx, self.current_node_idx] = float("-inf")
        self.visited_ninf_flag[:, 0][~self.at_the_depot] = 0 # if not at depot allow visit depot    
        self.ninf_mask = self.visited_ninf_flag.clone() # ninf mask is the one used by decoder. but we need to maintain visited mask

    def apply_capacity_constraint(self):
        """Mask out nodes that would exceet vehicle capacity"""
        round_error_tol = 0.00001
        demand_exceeds_capacity = self.remaining_capacity[:, None] + round_error_tol < self.batch_demand
        self.ninf_mask[demand_exceeds_capacity] = float("-inf")

    def apply_time_window_constraint(self):
        """Mask out where we cant complete service within the time window"""
        round_error_tol = 0.00001

        # 1. Earliest possible service start time for all nodes max(tw start, currTime + travelTime)
        dist = (self.current_node_coord - self.batch_coords).norm(p=2, dim=-1)
        travel_time =  dist / self.speed
        earliest_possible_service_start = torch.max(self.current_time + travel_time, self.batch_tw_start)

        # 2. service starts after tw end -> infeasible
        is_out_of_tw = earliest_possible_service_start > self.batch_tw_end + round_error_tol
        self.ninf_mask[is_out_of_tw] = float("-inf")

        # 3. Return to depot would be after depot tw end -> infeasible
        depot_coords = self.batch_coords[:,0:1,:]
        dist_next_node_to_depot = (depot_coords - self.batch_coords).norm(p=2, dim=-1)
        travel_time_to_depot = dist_next_node_to_depot / self.speed
        return_is_out_of_depot_tw = (
            earliest_possible_service_start 
            + self.batch_service_time 
            + travel_time_to_depot 
            > self.depot_tw_end + round_error_tol 
        )
        self.ninf_mask[return_is_out_of_depot_tw] = float("-inf")





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


    def shuffle_data(self):
        """Shuffle stored training instances."""
        # TODO: Maybe just do it in the load raw data method? then user does not have to think about it in trainer loop
        index = torch.randperm(len(self.full_node_coords)).long()

        self.full_node_coords = self.full_node_coords[index]
        self.full_node_demand = self.full_node_demand[index]
        self.full_node_tw_start = self.full_node_tw_start[index]
        self.full_node_tw_end = self.full_node_tw_end[index]
        self.full_node_service_time = self.full_node_service_time[index]
        self.full_label_tours = self.full_label_tours[index]
        self.full_label_costs = self.full_label_costs[index]

    def _build_dynamic_state(self) -> DynamicState:
        return DynamicState(
            num_completed_steps=self.num_completed_steps,
            current_node_idx=self.current_node_idx,
            current_node_coord=self.current_node_coord,
            constructed_tour=self.constructed_tour,
            model_tour=self.model_tour,
            ninf_mask=self.ninf_mask,
            remaining_capacity=self.remaining_capacity,
            current_time=self.current_time,
            length=self.length,
            done=self.done,
        )

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