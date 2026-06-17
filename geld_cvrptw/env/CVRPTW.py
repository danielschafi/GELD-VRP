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
    load: torch.Tensor # shape: (batch,) — remaining vehicle capacity.
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
        self.load = None
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
            load=self.load,
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
        self.load = torch.ones(self.batch_size, device=self.device)
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
        """Append selected nodes to tours and return updated dynamic state."""
        self.constructed_tour = torch.cat(
            (self.constructed_tour, teacher_node_idx[:, None]), dim=1
        )
        self.model_tour = torch.cat(
            (self.model_tour, predicted_node_idx[:, None]), dim=1
        )
        self.num_completed_steps += 1
        self.current_node_idx = teacher_node_idx
        self.at_the_depot = teacher_node_idx == 0
        batch_idx = torch.arange(self.batch_size, device=self.device)
        self.current_node_coord = self.batch_coords[batch_idx, teacher_node_idx]

        self.dynamic_state = self._build_dynamic_state()
        return self.dynamic_state


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
