"""Shared TSP construction environment logic."""

from dataclasses import dataclass

import torch

from geld.model.geometry import tour_length


@dataclass
class StepResult:
    """Result of reset() or step() during autoregressive tour construction."""

    coordinates: torch.Tensor
    reference_length: torch.Tensor | float | None = None
    predicted_length: torch.Tensor | float | None = None
    done: bool = False


class TSPEnvironmentBase:
    """Autoregressive TSP construction environment for tour building."""

    def __init__(self, **env_params):
        self.env_params = env_params
        self.problem_size = None
        self.data_path = env_params.get("data_path")
        self.use_subpath_augmentation = env_params.get(
            "use_subpath_augmentation", False
        )
        self.eval_tsplib = env_params.get("eval_tsplib", False)
        self.batch_size = None
        self.problems = None
        self.label_tour = None
        self.nodes_selected = None
        self.constructed_tour = None
        self.model_tour = None
        self.batch_offset = None
        self.tsplib_cost = None
        self.tsplib_name = None
        self.device = torch.device("cpu")

    def set_device(self, device: torch.device):
        """Move cached tensors to the given device."""
        self.device = device
        tensor_attrs = (
            "problems",
            "label_tour",
            "raw_data_nodes",
            "raw_data_tours",
            "raw_data_nodes_100",
            "raw_data_tours_100",
        )
        for attr in tensor_attrs:
            value = getattr(self, attr, None)
            if isinstance(value, torch.Tensor):
                setattr(self, attr, value.to(device))
        if isinstance(self.tsplib_cost, torch.Tensor):
            self.tsplib_cost = self.tsplib_cost.to(device)
        return self

    def sync_batch_to_device(self):
        """Move the active batch tensors to the environment device."""
        if self.problems is not None:
            self.problems = self.problems.to(self.device)
        if isinstance(self.label_tour, torch.Tensor):
            self.label_tour = self.label_tour.to(self.device)
        return self

    def reset(self, batch_size=None) -> StepResult:
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
        if batch_size is not None:
            self.batch_size = batch_size
        self.constructed_tour = torch.zeros(
            (self.batch_size, 0), dtype=torch.long, device=self.problems.device
        )
        self.model_tour = torch.zeros(
            (self.batch_size, 0), dtype=torch.long, device=self.problems.device
        )
        self.nodes_selected = 0
        return StepResult(coordinates=self.problems, done=False)

    def step(self, teacher_node, predicted_node) -> StepResult:
        """Append selected nodes and compute tour lengths when the tour completes."""
        self.nodes_selected += 1
        self.constructed_tour = torch.cat(
            (self.constructed_tour, teacher_node[:, None]), dim=1
        )
        self.model_tour = torch.cat((self.model_tour, predicted_node[:, None]), dim=1)
        done = self.nodes_selected == self.problems.shape[1]
        if done:
            reference_length = self.compute_tour_length(
                self.problems, self.constructed_tour
            )
            predicted_length = self.compute_tour_length(self.problems, self.model_tour)
            return StepResult(
                coordinates=self.problems,
                reference_length=reference_length,
                predicted_length=predicted_length,
                done=done,
            )
        return StepResult(coordinates=self.problems, done=done)

    def step_beam(self, selected_node, beam=16) -> StepResult:
        """Advance beam-expanded tours and return lengths when done."""
        self.nodes_selected += 1
        self.constructed_tour = torch.cat(
            (self.constructed_tour, selected_node[:, None]), dim=1
        )
        done = self.nodes_selected == self.problems.shape[1]
        if done:
            expanded = torch.repeat_interleave(self.problems, beam, 0)
            tour_lengths = self.compute_tour_length(expanded, self.constructed_tour)
            return StepResult(
                coordinates=self.problems, reference_length=tour_lengths, done=done
            )
        return StepResult(coordinates=self.problems, done=done)

    def compute_tour_length(self, problems, tour, return_known_optimal: bool = False):
        """Compute L(π); return known optimal length for TSPLIB when requested."""
        if self.eval_tsplib and return_known_optimal:
            return self.tsplib_cost, self.tsplib_name
        if self.eval_tsplib and self.label_tour is None and not return_known_optimal:
            problems = problems.clone().detach()
        return tour_length(problems, tour)

    def label_and_model_length(self):
        """Return label (optimal/teacher) and model tour lengths."""
        if self.eval_tsplib:
            reference = self.tsplib_cost
        elif self.label_tour is not None:
            reference = tour_length(self.problems, self.label_tour)
        else:
            reference = 0
        predicted = tour_length(self.problems, self.model_tour)
        return reference, predicted
