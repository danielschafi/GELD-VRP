"""Synthetic beam-search scaling benchmark orchestrator."""

from __future__ import annotations

import math
from logging import getLogger

import numpy as np
import torch

from geld_cvrptw.config.defaults_params import default_cvrptw_env_params, default_model_params
from geld_cvrptw.data.benchmark_loaders import BenchmarkInstance, benchmark_batch_to_device, records_to_benchmark_instances
from geld_cvrptw.data.generator import generate_instances
from geld_cvrptw.data.loaders import iter_instance_batches
from geld_cvrptw.env.CVRPTW import CVRPTWEnv
from geld_cvrptw.inference.pipeline import InferencePipeline, build_pipeline
from geld_cvrptw.model.GELD_CVRPTW import GeldCvrptwModel
from geld_cvrptw.utils.device import setup_device
from geld_cvrptw.utils.experiment_tracker import (
    ExperimentTracker,
    ScalingInstanceResult,
    ScalingSizeSummary,
    ScalingSummary,
)
from geld_cvrptw.utils.metrics import TimeEstimator


def default_episodes_for_size(problem_size: int) -> int:
    if problem_size <= 500:
        return 50
    if problem_size <= 2000:
        return 20
    if problem_size <= 10000:
        return 10
    return 5


def default_batch_size_for_size(problem_size: int) -> int:
    if problem_size <= 100:
        return 50
    if problem_size <= 500:
        return 10
    if problem_size <= 1000:
        return 4
    return 1


def default_generation_batch_size(problem_size: int, episodes: int) -> int:
    """Cap in-memory generation batch size for very large instances."""
    memory_budget = max(1, 50_000 // problem_size)
    return min(512, episodes, memory_budget)


def fit_scaling_exponent(
    size_summaries: list[ScalingSizeSummary],
) -> tuple[float | None, float | None]:
    """Fit time ~ prefactor * n^alpha via log-log linear regression."""
    if len(size_summaries) < 2:
        return None, None

    sizes = np.array([item.problem_size for item in size_summaries], dtype=np.float64)
    times = np.array([item.decode_time_mean_sec for item in size_summaries], dtype=np.float64)
    if np.any(sizes <= 0) or np.any(times <= 0):
        return None, None

    log_sizes = np.log(sizes)
    log_times = np.log(times)
    slope, intercept = np.polyfit(log_sizes, log_times, 1)
    prefactor = float(math.exp(intercept))
    return float(slope), prefactor


class ScalingBenchmark:
    """Measures beam-search decode time across synthetic problem sizes."""

    def __init__(
        self,
        benchmark_params: dict,
        tracker: ExperimentTracker | None = None,
    ):
        self.benchmark_params = benchmark_params
        self.tracker = tracker
        self.logger = getLogger(name="scaling_benchmark")
        self.device = setup_device(benchmark_params["use_cuda"], benchmark_params["cuda_device_num"])

        env_params = default_cvrptw_env_params()
        env_params["device"] = self.device
        self.env = CVRPTWEnv(**env_params)

        self.model = GeldCvrptwModel(**default_model_params(mode="test")).to(self.device)
        checkpoint_path = (
            f"{benchmark_params['model_load']['path']}/checkpoint-{benchmark_params['model_load']['epoch']}.pt"
        )
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.pipeline: InferencePipeline = build_pipeline(benchmark_params)
        self.time_estimator = TimeEstimator()

    def run(self) -> ScalingSummary:
        sizes = list(self.benchmark_params["n_customers_values"])
        decoder_cfg = self.benchmark_params["decoder"]
        beam_size = decoder_cfg["beam_size"]
        horizon_factor = decoder_cfg.get("horizon_factor", 4)
        seed = self.benchmark_params.get("seed", 2024)
        alpha = self.benchmark_params.get("alpha", 1.0)

        all_instances: list[ScalingInstanceResult] = []
        size_summaries: list[ScalingSizeSummary] = []
        max_successful_size = 0

        for n_customers in sizes:
            num_instances = self.benchmark_params.get("num_instances") or default_episodes_for_size(n_customers)
            decode_batch_size = (
                self.benchmark_params.get("decode_batch_size") or default_batch_size_for_size(n_customers)
            )

            self.logger.info(
                f"Scaling benchmark: n_customers={n_customers}, "
                f"num_instances={num_instances}, decode_batch_size={decode_batch_size}"
            )

            try:
                size_summary, instance_results = self._run_size(
                    n_customers=n_customers,
                    num_instances=num_instances,
                    decode_batch_size=decode_batch_size,
                    seed=seed + n_customers,
                    alpha=alpha,
                )
            except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
                if self._is_oom_error(exc):
                    self.logger.error(
                        f"OOM or memory error at n_customers={n_customers}: {exc}. Stopping size sweep."
                    )
                    break
                raise
            finally:
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()

            size_summaries.append(size_summary)
            all_instances.extend(instance_results)
            max_successful_size = n_customers

        scaling_exponent, scaling_prefactor = fit_scaling_exponent(size_summaries)
        summary = ScalingSummary(
            beam_size=beam_size,
            horizon_factor=horizon_factor,
            n_customers_attempted=sizes,
            max_successful_size=max_successful_size,
            scaling_exponent=scaling_exponent,
            scaling_prefactor=scaling_prefactor,
            size_summaries=size_summaries,
            instances=all_instances,
        )

        if scaling_exponent is not None:
            self.logger.info(
                f"Scaling fit: time ~ {scaling_prefactor:.4e} * n^{scaling_exponent:.2f}"
            )
        self.logger.info(f"Max successful size: {max_successful_size}")

        if self.tracker is not None:
            self.tracker.save_scaling_results(summary)
        return summary

    def _run_size(
        self,
        *,
        n_customers: int,
        num_instances: int,
        decode_batch_size: int,
        seed: int,
        alpha: float,
    ) -> tuple[ScalingSizeSummary, list[ScalingInstanceResult]]:
        records = generate_instances(
            num_instances,
            problem_size=n_customers,
            alpha=alpha,
            seed=seed,
            batch_size=default_generation_batch_size(n_customers, num_instances),
        )
        benchmark_instances = records_to_benchmark_instances(
            records,
            name_prefix=f"scaling_n{n_customers}",
        )

        decode_times: list[float] = []
        tour_lengths: list[float] = []
        instance_results: list[ScalingInstanceResult] = []

        total = len(benchmark_instances)
        num_processed = 0
        self.time_estimator.reset()

        for batch in iter_instance_batches(benchmark_instances, decode_batch_size):
            per_instance_times, per_instance_lengths = self._decode_batch(batch)
            decode_times.extend(per_instance_times)
            tour_lengths.extend(per_instance_lengths)

            for index, instance in enumerate(batch):
                instance_results.append(
                    ScalingInstanceResult(
                        name=instance.name,
                        problem_size=instance.num_customers,
                        decode_time_sec=per_instance_times[index],
                        tour_length=per_instance_lengths[index],
                    )
                )

            num_processed += len(batch)
            elapsed, remain = self.time_estimator.get_est_string(num_processed, total)
            self.logger.info(
                f"n_customers={n_customers}: {num_processed}/{total}, Elapsed[{elapsed}], Remain[{remain}]"
            )

        decode_times_arr = np.asarray(decode_times, dtype=np.float64)
        tour_lengths_arr = np.asarray(tour_lengths, dtype=np.float64)
        mean_time = float(decode_times_arr.mean())
        std_time = float(decode_times_arr.std(ddof=0))
        p95_time = float(np.percentile(decode_times_arr, 95))
        mean_length = float(tour_lengths_arr.mean())
        instances_per_sec = 1.0 / mean_time if mean_time > 0 else 0.0

        self.logger.info(
            f"n_customers={n_customers}: mean decode={mean_time:.4f}s, p95={p95_time:.4f}s, "
            f"throughput={instances_per_sec:.2f} inst/s"
        )

        size_summary = ScalingSizeSummary(
            problem_size=n_customers,
            num_instances=total,
            decode_time_mean_sec=mean_time,
            decode_time_std_sec=std_time,
            decode_time_p95_sec=p95_time,
            tour_length_mean=mean_length,
            instances_per_sec=instances_per_sec,
        )
        return size_summary, instance_results

    def _decode_batch(
        self,
        batch: list[BenchmarkInstance],
    ) -> tuple[list[float], list[float]]:
        coords, demand, tw_start, tw_end, service_time = benchmark_batch_to_device(batch, self.device)
        self.env.set_batch(coords, demand, tw_start, tw_end, service_time)
        timed_result = self.pipeline.run_timed(self.model, self.env)

        batch_len = len(batch)
        per_instance_time = timed_result.decode_time_sec / batch_len

        lengths = (
            timed_result.result.length_normalized
            * torch.tensor(
                [instance.scaler for instance in batch],
                device=self.device,
                dtype=torch.float32,
            )
        ).detach().cpu().numpy()

        return [per_instance_time] * batch_len, [float(length) for length in lengths]

    @staticmethod
    def _is_oom_error(exc: BaseException) -> bool:
        message = str(exc).lower()
        return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in message
