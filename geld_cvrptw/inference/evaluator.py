"""CVRPTW benchmark evaluation orchestrator."""

from __future__ import annotations

from enum import Enum
from logging import getLogger
from typing import Literal

import numpy as np
import torch

from geld_cvrptw.config.defaults_params import default_cvrptw_env_params, default_model_params
from geld_cvrptw.data.benchmark_loaders import (
    BenchmarkInstance,
    benchmark_batch_to_device,
    default_synthetic_paths,
    group_by_size,
    iter_homberger_instances,
    iter_solomon_instances,
    load_synthetic_pkl,
)
from geld_cvrptw.data.loaders import iter_instance_batches
from geld_cvrptw.env.CVRPTW import CVRPTWEnv
from geld_cvrptw.inference.pipeline import InferencePipeline, build_pipeline
from geld_cvrptw.model.GELD_CVRPTW import GeldCvrptwModel
from geld_cvrptw.utils.device import setup_device
from geld_cvrptw.utils.experiment_tracker import EvalInstanceResult, EvalSummary, ExperimentTracker
from geld_cvrptw.utils.metrics import AverageMeter, TimeEstimator


class EvalBenchmark(str, Enum):
    SYNTHETIC = "synthetic"
    SOLOMON = "solomon"
    HOMBERGER = "homberger"
    ALL = "all"


def _size_bucket(num_customers: int) -> str:
    return str(num_customers)


class CvrptwEvaluator:
    """Runs CVRPTW benchmarks and reports optimality gaps."""

    def __init__(
        self,
        eval_params: dict,
        tracker: ExperimentTracker | None = None,
    ):
        self.eval_params = eval_params
        self.tracker = tracker
        self.logger = getLogger(name="evaluator")
        self.device = setup_device(eval_params["use_cuda"], eval_params["cuda_device_num"])

        env_params = default_cvrptw_env_params()
        env_params["device"] = self.device
        self.env = CVRPTWEnv(**env_params)

        self.model = GeldCvrptwModel(**default_model_params(mode="test")).to(self.device)
        checkpoint_path = (
            f"{eval_params['model_load']['path']}/checkpoint-{eval_params['model_load']['epoch']}.pt"
        )
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.pipeline: InferencePipeline = build_pipeline(eval_params)
        self.time_estimator = TimeEstimator()

    def run(self, benchmark: Literal["synthetic", "solomon", "homberger", "all"]) -> list[EvalSummary]:
        """Run one or all benchmark suites."""
        targets = {
            "synthetic": [EvalBenchmark.SYNTHETIC],
            "solomon": [EvalBenchmark.SOLOMON],
            "homberger": [EvalBenchmark.HOMBERGER],
            "all": [EvalBenchmark.SYNTHETIC, EvalBenchmark.SOLOMON, EvalBenchmark.HOMBERGER],
        }[benchmark]

        summaries: list[EvalSummary] = []
        for target in targets:
            if target is EvalBenchmark.SYNTHETIC:
                summaries.append(self._run_synthetic())
            elif target is EvalBenchmark.SOLOMON:
                summaries.append(self._run_solomon())
            elif target is EvalBenchmark.HOMBERGER:
                summaries.append(self._run_homberger())
        return summaries

    def _evaluate_instances(
        self,
        instances: list[BenchmarkInstance],
        mode: str,
        batch_size: int,
        size: int | None = None,
        distribution: str | None = None,
        save_results: bool = True,
    ) -> EvalSummary:
        baseline_meter = AverageMeter()
        predicted_meter = AverageMeter()
        gap_buckets: dict[str, list[float]] = {}
        results: list[EvalInstanceResult] = []

        total = len(instances)
        num_processed = 0
        self.time_estimator.reset()
        for batch in iter_instance_batches(instances, batch_size):
            coords, demand, tw_start, tw_end, service_time = benchmark_batch_to_device(batch, self.device)
            self.env.set_batch(coords, demand, tw_start, tw_end, service_time)
            solve_result = self.pipeline.run(self.model, self.env)

            predicted_lengths = (solve_result.length_normalized * torch.tensor(
                [instance.scaler for instance in batch],
                device=self.device,
                dtype=torch.float32,
            )).detach().cpu().numpy()
            baseline_lengths = np.array([instance.optimal_cost for instance in batch], dtype=np.float64)

            for index, instance in enumerate(batch):
                baseline = float(baseline_lengths[index])
                predicted = float(predicted_lengths[index])
                gap = (predicted - baseline) / baseline * 100.0
                bucket = _size_bucket(instance.num_customers)
                gap_buckets.setdefault(bucket, []).append(gap)
                results.append(
                    EvalInstanceResult(
                        name=instance.name,
                        problem_size=instance.num_customers,
                        baseline_length=baseline,
                        predicted_length=predicted,
                        gap_percent=gap,
                        gap_bucket=bucket,
                    )
                )
                baseline_meter.update(baseline, 1)
                predicted_meter.update(predicted, 1)

            num_processed += len(batch)
            elapsed, remain = self.time_estimator.get_est_string(num_processed, total)
            self.logger.info(
                f"{mode}: {num_processed}/{total}, Elapsed[{elapsed}], Remain[{remain}], "
                f"avg gap so far: {(predicted_meter.avg - baseline_meter.avg) / baseline_meter.avg * 100:.4f}%"
            )

        bucket_means = {
            bucket: float(np.mean(gaps)) for bucket, gaps in gap_buckets.items() if gaps
        }
        bucket_counts = {bucket: len(gaps) for bucket, gaps in gap_buckets.items() if gaps}
        average_gap = (predicted_meter.avg - baseline_meter.avg) / baseline_meter.avg * 100.0

        self.logger.info(f"*** {mode} evaluation done ***")
        self.logger.info(f"Average gap: {average_gap:.4f}%")
        for bucket, mean_gap in bucket_means.items():
            self.logger.info(f"  size {bucket}: {mean_gap:.4f}% ({bucket_counts[bucket]} instances)")

        summary = EvalSummary(
            mode=mode,
            average_gap_percent=float(average_gap),
            baseline_length_avg=baseline_meter.avg,
            predicted_length_avg=predicted_meter.avg,
            num_instances=total,
            bucket_means=bucket_means,
            bucket_counts=bucket_counts,
            size=size,
            distribution=distribution,
            instances=results,
        )
        if self.tracker is not None and save_results:
            self.tracker.save_eval_results(summary)
        return summary

    def _run_synthetic(self) -> EvalSummary:
        cfg = self.eval_params["synthetic"]
        problem_path, solution_path = default_synthetic_paths(cfg["n_customers"])
        instances = load_synthetic_pkl(problem_path, solution_path, max_instances=cfg["num_instances"])
        return self._evaluate_instances(
            instances,
            mode=EvalBenchmark.SYNTHETIC.value,
            batch_size=cfg["batch_size"],
            size=cfg["n_customers"],
            distribution="uniform",
        )

    def _run_solomon(self) -> EvalSummary:
        instances = list(iter_solomon_instances())
        return self._evaluate_instances(
            instances,
            mode=EvalBenchmark.SOLOMON.value,
            batch_size=min(56, len(instances)),
            size=100,
        )

    def _run_homberger(self) -> EvalSummary:
        instances = list(iter_homberger_instances())
        grouped = group_by_size(instances)
        all_results: list[EvalInstanceResult] = []
        baseline_meter = AverageMeter()
        predicted_meter = AverageMeter()
        gap_buckets: dict[str, list[float]] = {}

        for size, size_instances in grouped.items():
            self.logger.info(f"Homberger: evaluating size bucket n={size} ({len(size_instances)} instances)")
            summary = self._evaluate_instances(
                size_instances,
                mode=EvalBenchmark.HOMBERGER.value,
                batch_size=min(32, len(size_instances)),
                size=size,
                save_results=False,
            )
            all_results.extend(summary.instances)
            baseline_meter.update(summary.baseline_length_avg, summary.num_instances)
            predicted_meter.update(summary.predicted_length_avg, summary.num_instances)
            for instance in summary.instances:
                gap_buckets.setdefault(instance.gap_bucket, []).append(instance.gap_percent)

        bucket_means = {bucket: float(np.mean(gaps)) for bucket, gaps in gap_buckets.items()}
        bucket_counts = {bucket: len(gaps) for bucket, gaps in gap_buckets.items()}
        average_gap = (predicted_meter.avg - baseline_meter.avg) / baseline_meter.avg * 100.0

        summary = EvalSummary(
            mode=EvalBenchmark.HOMBERGER.value,
            average_gap_percent=float(average_gap),
            baseline_length_avg=baseline_meter.avg,
            predicted_length_avg=predicted_meter.avg,
            num_instances=len(all_results),
            bucket_means=bucket_means,
            bucket_counts=bucket_counts,
            instances=all_results,
        )
        if self.tracker is not None:
            self.tracker.save_eval_results(summary)
        return summary
