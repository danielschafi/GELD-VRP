"""Greedy, beam search, and PRC inference pipeline."""

from dataclasses import dataclass

import torch

from geld.search.beam_search import BeamSearch
from geld.search.prc import run_prc_loop
from geld.utils.device import float_dtype, long_dtype


@dataclass
class SolveResult:
    """Greedy/beam/PRC inference result for one batch."""

    tour: torch.Tensor
    tour_length: torch.Tensor
    baseline_length: torch.Tensor | None = None


class InferenceSolver:
    """Inference pipeline: greedy → beam search → PRC post-processing."""

    def __init__(self, model, env, device, use_beam=True, use_prc=True, beam_size=16, prc_iterations=1000):
        self.model = model
        self.env = env
        self.device = device
        self.use_beam = use_beam
        self.use_prc = use_prc
        self.beam_size = beam_size
        self.prc_iterations = prc_iterations
        self.float_dtype = float_dtype(device)
        self.long_dtype = long_dtype(device)

    @torch.no_grad()
    def solve_batch(
        self,
        batch_offset,
        batch_size,
        *,
        reencode_each_step=False,
        skip_greedy=False,
        skip_beam=False,
        initial_tour=None,
        large_instance_prc=False,
        load_kwargs=None,
    ) -> SolveResult:
        """Solve a batch via greedy, optional BS, and optional PRC."""
        load_kwargs = load_kwargs or {}
        self.model.eval()
        self.env.load_problems(batch_offset, batch_size, **load_kwargs)
        origin_problem = self.env.problems
        if self.env.label_tour is not None:
            baseline_length = self.env.compute_tour_length(origin_problem, self.env.label_tour)
        elif self.env.tsplib_cost is not None:
            baseline_length = self.env.tsplib_cost
        else:
            baseline_length = None

        problem_size = origin_problem.shape[1]

        if initial_tour is not None:
            best_tour = initial_tour
        elif not skip_greedy:
            best_tour = self._run_greedy(batch_size, reencode_each_step)
        else:
            raise ValueError("Either provide initial_tour or run greedy decoding.")

        current_length = self.env.compute_tour_length(origin_problem, best_tour)

        if self.use_beam and not skip_beam:
            beam_tour, beam_length = self._run_beam(batch_size, problem_size, reencode_each_step)
            if beam_length.mean() < current_length.mean():
                best_tour = beam_tour
                current_length = beam_length
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        if self.use_prc:
            best_tour = run_prc_loop(
                self.model,
                self.env,
                origin_problem,
                best_tour,
                self.prc_iterations,
                large_instance=large_instance_prc,
            )
            current_length = self.env.compute_tour_length(origin_problem, best_tour)

        return SolveResult(
            tour=best_tour,
            tour_length=current_length,
            baseline_length=baseline_length,
        )

    def _run_greedy(self, batch_size, reencode_each_step=False):
        """Autoregressive greedy tour construction (G)."""
        result = self.env.reset()
        self.model.prepare_instance(result.coordinates)
        current_step = 0
        while not result.done:
            if current_step == 0:
                teacher_node = torch.zeros(batch_size, dtype=torch.long, device=self.device)
                predicted_node = teacher_node
            else:
                output = self.model(
                    self.env.constructed_tour,
                    self.env.label_tour,
                    current_step,
                    reencode_each_step=reencode_each_step,
                )
                teacher_node = output.action
                predicted_node = output.predicted_action
            current_step += 1
            result = self.env.step(teacher_node, predicted_node)
        return self.env.constructed_tour

    def _run_beam(self, batch_size, problem_size, reencode_each_step=False):
        """Beam search decode; return best sub-tour per instance."""
        beam = BeamSearch(
            self.beam_size,
            batch_size,
            problem_size,
            self.float_dtype,
            self.long_dtype,
            probs_type="logits",
            random_start=False,
            device=self.device,
        )
        result = self.env.reset(batch_size * self.beam_size)
        self.model.prepare_instance(result.coordinates)
        current_step = 0
        while not result.done:
            if current_step == 0:
                selected_node = torch.zeros(batch_size * self.beam_size, dtype=torch.long, device=self.device)
            else:
                output = self.model(
                    self.env.constructed_tour,
                    self.env.label_tour,
                    current_step,
                    beam_search=True,
                    beam_size=self.beam_size,
                    reencode_each_step=reencode_each_step,
                )
                probs = torch.log(output.transition_probs.view(batch_size, self.beam_size, -1))
                probs[probs.isnan()] = 0
                self.env.constructed_tour = beam.advance(probs, self.env.constructed_tour)
                selected_node = beam.next_nodes[-1].view(-1)
            result = self.env.step_beam(selected_node, beam=self.beam_size)
            current_step += 1

        tour_lengths = result.reference_length.view(batch_size, self.beam_size)
        beam_tours = self.env.constructed_tour.view(batch_size, self.beam_size, -1)
        best_lengths, min_idx = tour_lengths.min(1)
        batch_indices = torch.arange(batch_size, dtype=torch.long, device=self.device)
        best_tour = beam_tours[batch_indices, min_idx]
        return best_tour, best_lengths

    @torch.no_grad()
    def run_beam_on_coordinates(self, batch_offset, batch_size, problem_size):
        """Beam search on validation coordinates for SIL pseudo-label generation."""
        beam = BeamSearch(
            self.beam_size,
            batch_size,
            problem_size,
            self.float_dtype,
            self.long_dtype,
            probs_type="logits",
            random_start=False,
            device=self.device,
        )
        self.env.load_problems_val(batch_offset, batch_size)
        result = self.env.reset(batch_size * self.beam_size)
        self.model.prepare_instance(result.coordinates)
        current_step = 0
        while not result.done:
            if current_step == 0:
                selected_node = torch.zeros(batch_size * self.beam_size, dtype=torch.long, device=self.device)
            else:
                output = self.model(
                    self.env.constructed_tour,
                    self.env.label_tour,
                    current_step,
                    beam_search=True,
                    beam_size=self.beam_size,
                )
                probs = torch.log(output.transition_probs.view(batch_size, self.beam_size, -1))
                probs[probs.isnan()] = 0
                self.env.constructed_tour = beam.advance(probs, self.env.constructed_tour)
                selected_node = beam.next_nodes[-1].view(-1)
            result = self.env.step_beam(selected_node, beam=self.beam_size)
            current_step += 1

        tour_lengths = result.reference_length.view(batch_size, self.beam_size)
        beam_tours = self.env.constructed_tour.view(batch_size, self.beam_size, -1)
        best_lengths, min_idx = tour_lengths.min(1)
        batch_indices = torch.arange(batch_size, dtype=torch.long, device=self.device)
        return best_lengths, beam_tours[batch_indices, min_idx]

    @torch.no_grad()
    def run_greedy_on_coordinates(self, batch_offset, batch_size):
        """Greedy decode on validation coordinates for SIL pseudo-label generation."""
        self.env.load_problems_val(batch_offset, batch_size)
        tour = self._run_greedy(batch_size)
        lengths = self.env.compute_tour_length(self.env.problems, tour)
        return lengths, tour
