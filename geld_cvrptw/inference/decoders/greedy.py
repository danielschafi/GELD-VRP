"""Greedy autoregressive decoder for CVRPTW."""

from __future__ import annotations

import torch

from geld_cvrptw.inference.types import Decoder, SolveResult


class GreedyDecoder(Decoder):
    """Greedily constructs a tour by always taking the next node with the highest probability by the model.
    argmax over model probabilities. """

    def __init__(self, horizon_factor: int = 4):
        # Because of return-to-depot trips we need more steps than tour length; stop early when all are done.
        self.horizon_factor = horizon_factor

    @torch.no_grad()
    def decode(self, model, env) -> SolveResult:
        static_state, dynamic_state = env.reset()
        model.embed_static_state_once(static_state)

        max_steps = env.num_nodes * self.horizon_factor
        for _ in range(max_steps):
            if dynamic_state.done.all():
                break
            probs = model(dynamic_state)
            next_node = probs.argmax(dim=1)
            next_node = torch.where(dynamic_state.done, torch.zeros_like(next_node), next_node)
            dynamic_state = env.step(next_node, next_node, dynamic_state)

        length_normalized = env.compute_decoded_tour_length(env.batch_coords, dynamic_state.model_tour)
        return SolveResult(tour=dynamic_state.model_tour, length_normalized=length_normalized)
