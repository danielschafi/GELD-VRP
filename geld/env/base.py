"""Shared TSP construction environment logic."""

from dataclasses import dataclass

import torch

from geld.model.geometry import tour_length


@dataclass
class ResetState:
    """Environment state returned after reset."""

    problems: torch.Tensor


@dataclass
class StepState:
    """Environment state at the current MDP step."""

    data: torch.Tensor


class TSPEnvironmentBase:
    """Autoregressive TSP construction environment for tour building."""

    def __init__(self, **env_params):
        self.env_params = env_params
        self.problem_size = None
        self.data_path = env_params.get("data_path")
        self.sub_path = env_params.get("sub_path", False)
        self.test_in_tsplib = env_params.get("test_in_tsplib", False)
        self.batch_size = None
        self.problems = None
        self.solution = None
        self.selected_count = None
        self.reference_tour = None
        self.predicted_tour = None
        self.episode = None
        self.step_state = None
        self.tsplib_cost = None
        self.tsplib_name = None
        self.device = torch.device("cpu")

    def set_device(self, device: torch.device):
        """Move cached tensors to the given device."""
        self.device = device
        tensor_attrs = (
            "problems",
            "solution",
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
        if isinstance(self.solution, torch.Tensor):
            self.solution = self.solution.to(self.device)
        return self

    def reset(self, batch_size=None):
        """Start a new tour-construction episode."""
        if batch_size is not None:
            self.batch_size = batch_size
        self.reference_tour = torch.zeros(
            (self.batch_size, 0), dtype=torch.long, device=self.problems.device
        )
        self.predicted_tour = torch.zeros(
            (self.batch_size, 0), dtype=torch.long, device=self.problems.device
        )
        self.selected_count = 0
        self.step_state = StepState(data=self.problems)
        return ResetState(self.problems), None, False

    def pre_step(self):
        """Return current step state before the model selects the next node."""
        return self.step_state, None, None, False

    def step(self, reference_action, predicted_action):
        """Append selected nodes and compute tour lengths when the tour completes."""
        self.selected_count += 1
        self.reference_tour = torch.cat((self.reference_tour, reference_action[:, None]), dim=1)
        self.predicted_tour = torch.cat((self.predicted_tour, predicted_action[:, None]), dim=1)
        done = self.selected_count == self.problems.shape[1]
        if done:
            reference_length = self.compute_tour_length(self.problems, self.reference_tour)
            predicted_length = self.compute_tour_length(self.problems, self.predicted_tour)
            return self.step_state, reference_length, predicted_length, done
        return self.step_state, None, None, done

    def step_beam(self, selected, beam=16):
        """Advance beam-expanded tours and return lengths when done."""
        self.selected_count += 1
        self.reference_tour = torch.cat((self.reference_tour, selected[:, None]), dim=1)
        done = self.selected_count == self.problems.shape[1]
        if done:
            expanded = torch.repeat_interleave(self.problems, beam, 0)
            tour_lengths = self.compute_tour_length(expanded, self.reference_tour)
            return self.step_state, tour_lengths, done
        return self.step_state, None, done

    def compute_tour_length(self, problems, solution, need_optimal: bool = False):
        """Compute L(π); return known optimal length for TSPLIB when requested."""
        if self.test_in_tsplib and need_optimal:
            return self.tsplib_cost, self.tsplib_name
        if self.test_in_tsplib and self.solution is None and not need_optimal:
            problems = problems.clone().detach()
        return tour_length(problems, solution)

    def reference_and_predicted_length(self):
        """Return reference (optimal/label) and predicted tour lengths."""
        if self.test_in_tsplib:
            reference = self.tsplib_cost
        elif self.solution is not None:
            reference = tour_length(self.problems, self.solution)
        else:
            reference = 0
        predicted = tour_length(self.problems, self.predicted_tour)
        return reference, predicted
