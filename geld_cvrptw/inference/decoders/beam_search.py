"""Beam search autoregressive decoder for CVRPTW."""

from __future__ import annotations

import torch

from geld_cvrptw.env.CVRPTW import CVRPTWEnv, DynamicState
from geld_cvrptw.inference.decoders.bootstrap import apply_bootstrap
from geld_cvrptw.inference.types import Decoder, SolveResult
from geld_cvrptw.model.GELD_CVRPTW import GeldCvrptwModel


def copy_parent_beam_states(beam_dynamic_states: list[DynamicState], parent_idx: torch.Tensor) -> DynamicState:
    """
    Gather one parent dynamic state per batch row. parent_idx shape: (batch,).
    We need to copy the topB winning beams entire history and not only append the topB best nodes.
    For each we need to: 
    1. Copy the parent beams ful history and dynamic state
    2. Append the next node with env.step
    """
    batch_size = parent_idx.size(0)
    batch_idx = torch.arange(batch_size, device=parent_idx.device)

    def gather(field_name: str) -> torch.Tensor:
        stacked = torch.stack([getattr(state, field_name) for state in beam_dynamic_states], dim=1)
        return stacked[batch_idx, parent_idx]

    current_node_idx = gather("current_node_idx")
    return DynamicState(
        num_completed_steps=beam_dynamic_states[0].num_completed_steps,
        current_node_idx=current_node_idx,
        current_node_coord=gather("current_node_coord"),
        constructed_tour=gather("constructed_tour"),
        model_tour=gather("model_tour"),
        ninf_mask=gather("ninf_mask"),
        visited_ninf_flag=gather("visited_ninf_flag"),
        remaining_capacity=gather("remaining_capacity"),
        current_time=gather("current_time"),
        length=gather("length"),
        done=gather("done"),
    )


class BeamSearchDecoder(Decoder):
    """
    Decodes a tour using BeamSearch on the models probabilities for next nodes. 
    Beam search keeps the B best partial tours ranked by cumulative log-probability, then selects the final tour by actual Euclidean tour length
    """
    def __init__(
        self,
        bootstrap_start_node: int = 1,
        max_steps_factor: int = 4,
        beam_size: int = 16,
    ):
        """
        we need two 
        """
        self.bootstrap_start_node = bootstrap_start_node
        # because of return to depot trips we need more steps than tour length. this is for safety only we stop if all are done
        self.max_steps_factor = max_steps_factor
        self.beam_size = beam_size

    @torch.no_grad()
    def decode(self, model: GeldCvrptwModel, env: CVRPTWEnv) -> SolveResult:
        static_state, dynamic_state = env.reset()
        model.embed_static_state_once(static_state)
        dynamic_state = apply_bootstrap(env, dynamic_state, start_node=self.bootstrap_start_node)


        # Initialize state
        problem_size = static_state.node_coords.size(1)
        batch_size = static_state.depot_coords.size(0)
        intermediate_size = batch_size * self.beam_size

        # For each beam we need a separate dynamic State
        beam_dynamic_states = [dynamic_state.clone()  for _ in range(self.beam_size)] # (beamSize)
        beam_scores = torch.zeros((batch_size, self.beam_size), device=env.device)
        # first step outside loop
        is_first_expansion = True

        max_steps = env.num_nodes * self.max_steps_factor
        for t in range(max_steps):
            # list with all the cumulative log probs (batch, beam, nodes)
            cum_scores  = torch.full((batch_size, self.beam_size, problem_size), fill_value=float("-inf"), device=env.device)

            for beam_idx, dyn_state in enumerate(beam_dynamic_states):
                probs = model(dyn_state)  # (batch, nodes)
                cum_scores[:, beam_idx, :] =  beam_scores[:, beam_idx].unsqueeze(1) + torch.log(probs.clamp(min=1e-10))

            # TSP first expansion: all beams start identical, so only beam 0 may contribute candidates.
            if is_first_expansion:
                cum_scores[:, 1:, :] = float("-inf")
                
            # Over all beams get the top B nodes with the highest cumulative probability
            # (batch,beam,nodes) -> (batch, beam * nodes) 
            flat_scores = cum_scores.view(batch_size, self.beam_size * problem_size)
            topB = torch.topk(flat_scores, self.beam_size, dim=1)  # values/indices: (batch, beam)

            origin_beam_indices = topB.indices // problem_size # (batch, beam)
            next_node_indices = topB.indices % problem_size  # (batch, beam)
            new_scores = topB.values # (batch, beam)

            new_beam_dynamic_states = []
            for j  in range(self.beam_size):
                # For each next node in the top-B (across batches)
                parent_idx = origin_beam_indices[:, j]   # (batch) 
                next_nodes = next_node_indices[:, j]     # (batch) 
                # Get their parent beams history
                parent_state = copy_parent_beam_states(beam_dynamic_states, parent_idx)
                next_nodes = torch.where(parent_state.done, torch.zeros_like(next_nodes), next_nodes)
                # With the correct state, make a step in the env
                new_state = env.step(next_nodes, next_nodes, parent_state)
                new_beam_dynamic_states.append(new_state)
            
            beam_dynamic_states = new_beam_dynamic_states
            beam_scores = new_scores
            is_first_expansion = False

            if all(s.done.all() for s in beam_dynamic_states):
                break

        # Get the tour with the best tour length
        lengths = torch.stack(
            [env.compute_tour_length(env.batch_coords, state.model_tour) for state in beam_dynamic_states],
            dim=1,
        )  # (batch, beam)
        best_beam = lengths.argmin(dim=1)
        batch_idx = torch.arange(batch_size, device=env.device)
        tours = torch.stack([state.model_tour for state in beam_dynamic_states], dim=1)
        best_tour = tours[batch_idx, best_beam]
        
        return SolveResult(tour=best_tour, length_normalized=lengths[batch_idx, best_beam])
