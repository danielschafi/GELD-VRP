"""
Combining what we can use from MVMoE's VRPPTWEnv.

"""

from dataclasses import dataclass

import torch
from geld.model.geometry import tour_length
from geld_cvrptw.data.loaders import load_cvrptw_data_with_labels
from geld_cvrptw.data.augmentations import apply_rotation, maybe_reverse_tour

# Reminder this is used in geld rn
# @dataclass
# class StepResult:
#     """Result of reset() or step() during autoregressive tour construction."""

#     coordinates: torch.Tensor # static
#     reference_length: torch.Tensor | float | None = None # dynamic
#     predicted_length: torch.Tensor | float | None = None # dynamic
#     done: bool = False #dynamic


@dataclass
class StaticState:
    """
    Static State information.
    usually result of reset()
    fixed one time
    """

    depot_xy: torch.Tensor = None
    # shape: (batch, 1, 2)
    node_xy: torch.Tensor = None  # was coordinates
    # shape: (batch, problem, 2)
    node_demand: torch.Tensor = None
    # shape: (batch, problem)
    node_service_time: torch.Tensor = None
    # shape: (batch, problem)
    node_tw_start: torch.Tensor = None
    # shape: (batch, problem)
    node_tw_end: torch.Tensor = None
    # shape: (batch, problem)
    prob_emb: torch.Tensor = None
    # shape: (num_training_prob)


@dataclass
class DynamicState:
    """
    Dynamic State information.
    Changes every step
    usually result of step()
    """

    # TODO: Verify if ALL attributes are needed.
    BATCH_IDX: torch.Tensor = None
    # POMO_IDX: torch.Tensor = None
    START_NODE: torch.Tensor = None
    # shape: (batch, pomo)
    selected_count: int = None
    current_node: torch.Tensor = None
    # shape: (batch, pomo)
    ninf_mask: torch.Tensor = None
    # shape: (batch, pomo, problem+1)
    done: torch.Tensor | bool = None  # was finished in MVMoE
    # shape: (batch, pomo)
    load: torch.Tensor = None
    # shape: (batch, pomo)
    current_time: torch.Tensor = None
    # shape: (batch, pomo)
    length: torch.Tensor = None
    # shape: (batch, pomo)
    open: torch.Tensor = None
    # shape: (batch, pomo)
    current_coord: torch.Tensor = None
    # shape: (batch, pomo, 2)

    # need these for supervised learning training
    reference_length: torch.Tensor | float | None = None
    predicted_length: torch.Tensor | float | None = None


class CVRPTWEnv:
    def __init__(self, **env_params):
        self.data_path = env_params.get("data_path")

        # Const @INIT
        ####################################
        self.env_params = env_params

        # self.pomo_size = env_params['pomo_size'] # nr of fist possible customer to visit to try. 1 for us
        # self.loc_scaler = env_params['loc_scaler'] if 'loc_scaler' in env_params.keys() else None
        self.device = (
            torch.device("cuda", torch.cuda.current_device())
            if "device" not in env_params.keys()
            else env_params["device"]
        )

        # Const @Load_Problem
        ####################################

        self.problem_size = None

        self.batch_size = None
        self.BATCH_IDX = None
        # self.POMO_IDX = None
        self.START_NODE = None
        # IDX.shape: (batch, pomo)
        self.depot_node_xy = None
        # shape: (batch, problem+1, 2)
        self.depot_node_demand = None
        # shape: (batch, problem+1)
        self.depot_node_service_time = None
        # shape: (batch, problem+1)
        self.depot_node_tw_start = None
        # shape: (batch, problem+1)
        self.depot_node_tw_end = None
        # shape: (batch, problem+1)
        self.speed = 1.0
        self.depot_start, self.depot_end = 0.0, 3.0  # tw for depot [0, 3]

        # Dynamic-1
        ####################################
        self.selected_count = None
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = None
        # shape: (batch, pomo, 0~)

        # Dynamic-2
        ####################################
        self.at_the_depot = None
        # shape: (batch, pomo)
        self.load = None
        # shape: (batch, pomo)
        self.visited_ninf_flag = None
        # shape: (batch, pomo, problem+1)
        self.ninf_mask = None
        # shape: (batch, pomo, problem+1)
        self.done = None
        # shape: (batch, pomo)
        self.current_time = None
        # shape: (batch, pomo)
        self.length = None
        # shape: (batch, pomo)
        self.open = None
        # shape: (batch, pomo)
        self.current_coord = None
        # shape: (batch, pomo, 2)

        self.static_state = StaticState()
        self.dynamic_state = DynamicState()

        self.problems = None  # this is now more complex Static State attribs?
        self.batch_label_tours = None
        self.selected_count = None
        self.constructed_tour = None
        self.model_tour = None
        self.batch_offset = None
        self.device = torch.device("cpu")

    def load_raw_data(self):
        """
        Currently only loads the train dataset for stage 1 supervised learning training
        """
        dataset = load_cvrptw_data_with_labels()

        self.raw_nodes = dataset.coords.requires_grad_(False)
        self.raw_demand = dataset.demand.requires_grad_(False)
        self.raw_tw_start = dataset.tw_start.requires_grad_(False)
        self.raw_tw_end = dataset.tw_end.requires_grad_(False)
        self.raw_service_time = dataset.service_time.requires_grad_(False)
        self.raw_label_tours = dataset.label_tours.requires_grad_(False)
        self.raw_costs = dataset.costs.requires_grad_(False)

    def load_problems(self, batch_offset: int, batch_size: int, train: bool = True):
        """¨
        Load one batch of samples. could be combined with reset step for clarity. # TODO check after we do inference etc. if we can do that.
        """
        self.batch_offset = batch_offset
        self.batch_size = batch_size

        # Load just one batch of problems
        self.batch_coords = self.raw_nodes[batch_offset : batch_offset + batch_size]  # was self.problems
        self.batch_demand = self.raw_demand[batch_offset : batch_offset + batch_size]
        self.batch_tw_start = self.raw_tw_start[batch_offset : batch_offset + batch_size]
        self.batch_tw_end = self.raw_tw_end[batch_offset : batch_offset + batch_size]
        self.batch_service_time = self.raw_service_time[batch_offset : batch_offset + batch_size]
        self.batch_label_tours = self.raw_label_tours[batch_offset : batch_offset + batch_size]
        self.batch_costs = self.raw_costs[batch_offset : batch_offset + batch_size]

        self.problem_size = self.batch_coords.shape[1]

        self.batch_label_tours = maybe_reverse_tour(self.batch_label_tours)

        if train:
            rotation_id = torch.randint(low=0, high=8, size=[1])[0].item()
            self.raw_nodes = apply_rotation(self.raw_nodes, rotation_id)

        self.sync_batch_to_device()

    def shuffle_data(self):
        """Shuffle stored training instances."""
        # TODO: Maybe just do it in the load raw data method? then user does not have to think about it in trainer loop
        index = torch.randperm(len(self.raw_nodes)).long()

        self.raw_nodes = self.raw_nodes[index]
        self.raw_demand = self.raw_demand[index]
        self.raw_tw_start = self.raw_tw_start[index]
        self.raw_tw_end = self.raw_tw_end[index]
        self.raw_service_time = self.raw_service_time[index]
        self.raw_label_tours = self.raw_label_tours[index]
        self.raw_costs = self.raw_costs[index]

    def reset(self, batch_size=None) -> StaticState:
        """
        Start a new tour-construction episode and return the initial coordinates.


        containers for

        - self.constructed_tour: decoder input  / ground truth path (t-1 steps of it) / autoregressively built tour
        - self.model_tour: tour of model argmax predictions at each step
        - self.nodes_selected: nr of constuction steps completed
        - label_tour: complete ground truth reference tour

        Returns:
        - StepResult with self.problems=coordinates and done=false
        """

        #############################################
        # Init env internal static state information
        # Tracking of tour
        #############################################

        if batch_size is not None:
            self.batch_size = batch_size

        # The t-1 nodes based on which the decoder will predict the t-th node. tracks tour under construction
        self.constructed_tour = torch.zeros((self.batch_size, 0), dtype=torch.long, device=self.device)
        # Training: Nodes that have been predicted by argmax over model pred.
        self.model_tour = torch.zeros((self.batch_size, 0), dtype=torch.long, device=self.device)

        ############################################
        # Init containers for masking etc
        ############################################

        # Num nodes already predicted (t-1)
        self.selected_count = 0
        self.current_node = None
        # shape: (batch)

        # self.selected_node_list = torch.zeros((self.batch_size, 0), dtype=torch.long).to(self.device) -> is constructed tour here
        # shape: (batch, 0~)

        self.at_the_depot = torch.ones(size=(self.batch_size), dtype=torch.bool).to(self.device)
        # shape: (batch)
        self.load = torch.ones(size=(self.batch_size)).to(self.device)
        # shape: (batch)
        self.visited_ninf_flag = torch.zeros(size=(self.batch_size, self.problem_size + 1)).to(self.device)
        # shape: (batch, problem+1)
        self.ninf_mask = torch.zeros(size=(self.batch_size, self.problem_size + 1)).to(self.device)
        # shape: (batch, problem+1)
        self.done = torch.zeros(size=(self.batch_size), dtype=torch.bool).to(self.device)  # was finished in MVMoE
        # shape: (batch)
        self.current_time = torch.zeros(size=(self.batch_size)).to(self.device)
        # shape: (batch)
        self.length = torch.zeros(size=(self.batch_size)).to(self.device)
        # shape: (batch)
        self.current_coord = self.depot_node_xy[:, :1, :]  # depot
        # shape: (batch, 2)

        # Return Static Problem Definition
        return StaticState(
            depot_xy=self.batch_coords[:, 0],  # first is depot, maybe not explicitly needed and let model learn that.
            node_xy=self.batch_coords,
            node_demand=self.batch_demand,
            node_tw_start=self.batch_tw_start,
            node_tw_end=self.batch_tw_end,
            node_service_time=self.batch_service_time,
        )

    def step(self) -> DynamicState:
        # Init env internal dynamic state information

        # Return what is needed outsied the env (for model prediction)

        return DynamicState()

    # COMPUTE DEVICE MANAGEMENT

    def set_device(self, device: torch.device):
        """
        Move the complete raw data tensors to the specified device.
        and sets the env's device
        """
        self.device = device

        if isinstance(self.raw_nodes, torch.Tensor):
            self.raw_nodes = self.raw_nodes.to(device)
        if isinstance(self.raw_demand, torch.Tensor):
            self.raw_demand = self.raw_demand.to(device)
        if isinstance(self.raw_tw_start, torch.Tensor):
            self.raw_tw_start = self.raw_tw_start.to(device)
        if isinstance(self.raw_tw_end, torch.Tensor):
            self.raw_tw_end = self.raw_tw_end.to(device)
        if isinstance(self.raw_service_time, torch.Tensor):
            self.raw_service_time = self.raw_service_time.to(device)
        if isinstance(self.raw_label_tours, torch.Tensor):
            self.raw_label_tours = self.raw_label_tours.to(device)
        if isinstance(self.raw_costs, torch.Tensor):
            self.raw_costs = self.raw_costs.to(device)

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
        if isinstance(self.batch_costs, torch.Tensor):
            self.batch_costs = self.batch_costs.to(self.device)
