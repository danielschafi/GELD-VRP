"""Construction bootstrap shared by greedy decoding and future decoders."""

from __future__ import annotations

import torch


def apply_bootstrap(env, start_node: int = 1):
    """Two fixed env steps before the model takes over (matches stage-1 training)."""
    # TODO: There should be a better solution to this.
    batch_size = env.batch_size
    device = env.device

    start = torch.full((batch_size,), start_node, dtype=torch.long, device=device)
    dynamic_state = env.step(start, start)

    depot = torch.zeros(batch_size, dtype=torch.long, device=device)
    dynamic_state = env.step(depot, depot)
    return dynamic_state
