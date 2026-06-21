"""Compose decoders and optional post-processors."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from geld_cvrptw.inference.decoders.greedy import GreedyDecoder
from geld_cvrptw.inference.types import Decoder, PostProcessor, SolveResult


class InferencePipeline:
    """Runs decode → optional post-process chain."""

    def __init__(
        self,
        decoder: Decoder,
        postprocessors: Sequence[PostProcessor] | None = None,
    ):
        self.decoder = decoder
        self.postprocessors = list(postprocessors or ())

    @torch.no_grad()
    def run(self, model, env) -> SolveResult:
        result = self.decoder.decode(model, env)
        for postprocessor in self.postprocessors:
            result = postprocessor.refine(model, env, result)
        return result


def build_pipeline(eval_params: dict) -> InferencePipeline:
    """Construct an inference pipeline from evaluation config."""
    decoder_cfg = eval_params["decoder"]
    decoder_name = decoder_cfg["name"]
    if decoder_name == "greedy":
        decoder = GreedyDecoder(bootstrap_start_node=decoder_cfg["bootstrap_start_node"])
    else:
        raise ValueError(f"Unsupported decoder: {decoder_name}")

    if eval_params.get("postprocessors"):
        raise NotImplementedError("Post-processors are not implemented yet.")

    return InferencePipeline(decoder=decoder)
