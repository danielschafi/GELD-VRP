"""Compose decoders and optional post-processors."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from geld_cvrptw.inference.decoders.beam_search import BeamSearchDecoder
from geld_cvrptw.inference.decoders.greedy import GreedyDecoder
from geld_cvrptw.inference.postprocess.re_construction import ReConstruction
from geld_cvrptw.inference.types import Decoder, PostProcessor, SolveResult


@dataclass
class TimedSolveResult:
    """Decode result with wall-clock timing for the decode phase only."""

    result: SolveResult
    decode_time_sec: float


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


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

    @torch.no_grad()
    def run_timed(self, model, env) -> TimedSolveResult:
        """Run decode only and return wall-clock time (post-processors excluded)."""
        _sync_device(env.device)
        start = time.perf_counter()
        result = self.decoder.decode(model, env)
        _sync_device(env.device)
        return TimedSolveResult(result=result, decode_time_sec=time.perf_counter() - start)


def build_postprocessors(eval_params: dict) -> list[PostProcessor]:
    """Construct post-processors from evaluation config."""
    postprocessors: list[PostProcessor] = []

    reconstruction_cfg = eval_params.get("reconstruction", {})
    if reconstruction_cfg.get("enabled", False):
        postprocessors.append(
            ReConstruction(
                num_iterations=reconstruction_cfg.get("num_iterations", 100),
                min_window_length=reconstruction_cfg.get("min_window_length", 4),
                min_window_count=reconstruction_cfg.get("min_window_count", 2),
                diversify_coords=reconstruction_cfg.get("diversify_coords", False),
            )
        )

    return postprocessors


def build_pipeline(eval_params: dict) -> InferencePipeline:
    """Construct an inference pipeline from evaluation config."""
    decoder_cfg = eval_params["decoder"]
    decoder_name = decoder_cfg["name"]
    bootstrap_start_node = decoder_cfg["bootstrap_start_node"]
    if decoder_name == "greedy":
        decoder = GreedyDecoder(bootstrap_start_node=bootstrap_start_node)
    elif decoder_name == "beam_search":
        decoder = BeamSearchDecoder(
            bootstrap_start_node=bootstrap_start_node,
            max_steps_factor=decoder_cfg.get("max_steps_factor", 4),
            beam_size=decoder_cfg.get("beam_size", 16),
        )
    else:
        raise ValueError(f"Unsupported decoder: {decoder_name}")

    return InferencePipeline(decoder=decoder, postprocessors=build_postprocessors(eval_params))
