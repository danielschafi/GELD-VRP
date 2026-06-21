"""Greedy autoregressive decoder for CVRPTW."""

from __future__ import annotations

import torch

from geld_cvrptw.inference.decoders.bootstrap import apply_bootstrap
from geld_cvrptw.inference.types import Decoder, SolveResult


class GreedyDecoder(Decoder):
    """Greedily constructs a tour by always taking the next node with the highest probability by the model.
    argmax over model probabilities. """

    def __init__(self, bootstrap_start_node: int = 1, max_steps_factor: int = 4):
        """
        we need two 
        """
        self.bootstrap_start_node = bootstrap_start_node
        # because of return to depot trips we need more steps than tour length. this is for safety only we stop if all are done
        self.max_steps_factor = max_steps_factor 

    @torch.no_grad()
    def decode(self, model, env) -> SolveResult:
        static_state, dynamic_state = env.reset()
        model.embed_static_state_once(static_state)
        dynamic_state = apply_bootstrap(env, start_node=self.bootstrap_start_node)

        max_steps = env.num_nodes * self.max_steps_factor
        for _ in range(max_steps):
            if dynamic_state.done.all():
                break
            probs = model(dynamic_state)
            next_node = probs.argmax(dim=1)
            next_node = torch.where(dynamic_state.done, torch.zeros_like(next_node), next_node)
            dynamic_state = env.step(next_node, next_node)

        length_normalized = env.compute_tour_length(env.batch_coords, dynamic_state.model_tour)
        return SolveResult(tour=dynamic_state.model_tour, length_normalized=length_normalized)
