"""Unified inference evaluator for synthetic, TSPLIB, and post-processing modes."""

from enum import Enum
from logging import getLogger

import numpy as np
import torch

from geld.data.collections import NATIONAL_TSP_OPTIMAL_LENGTHS, TSPLIB_OPTIMAL_LENGTHS
from geld.env.synthetic import SyntheticEnvironment
from geld.env.tsplib import TSPLIBEnvironment
from geld.model.geld_model import GeldModel
from geld.paths import baseline_solutions_dir
from geld.search.solver import InferenceSolver
from geld.utils.device import setup_device
from geld.utils.experiment_tracker import EvalInstanceResult, EvalSummary, ExperimentTracker
from geld.utils.logging import get_result_folder
from geld.utils.metrics import AverageMeter, TimeEstimator

GAP_SIZE_BUCKETS = ["100", "500", "1k", "5k", "10k", "10k+"]


def _gap_bucket_key(problem_size: int) -> str:
    """Map TSP-n size to a gap reporting bucket."""
    if problem_size <= 100:
        return "100"
    if problem_size <= 500:
        return "500"
    if problem_size <= 1000:
        return "1k"
    if problem_size <= 5000:
        return "5k"
    if problem_size <= 10000:
        return "10k"
    return "10k+"


def _new_gap_buckets() -> dict[str, list[float]]:
    """Create empty gap lists per scale bucket."""
    return {name: [] for name in GAP_SIZE_BUCKETS}


def _record_gap(gap_buckets: dict[str, list[float]], problem_size: int, gap: float) -> None:
    """Append gap (Eq. 12) to the matching scale bucket."""
    gap_buckets[_gap_bucket_key(problem_size)].append(gap)


def _gap_bucket_means(gap_buckets: dict[str, list[float]]) -> tuple[dict[str, float], dict[str, int]]:
    """Compute mean gap and count per TSP scale bucket."""
    means = {}
    counts = {}
    for label, gaps in gap_buckets.items():
        if gaps:
            means[label] = float(np.mean(gaps) * 100.0)
            counts[label] = len(gaps)
    return means, counts


def _log_gap_bucket_means(logger, gap_buckets: dict[str, list[float]]) -> tuple[dict[str, float], dict[str, int]]:
    """Log mean gap per TSP scale bucket."""
    means, counts = _gap_bucket_means(gap_buckets)
    for label, mean_gap in means.items():
        logger.info(f"problems_{label} mean gap: {mean_gap:.4f}% ({counts[label]} instances)")
    return means, counts


class EvalMode(Enum):
    """Evaluation target: synthetic benchmarks, TSPLIB, or PRC post-processing."""

    SYNTHETIC = "synthetic"
    TSPLIB = "tsplib"
    POSTPROCESS = "postprocess"


class InferenceEvaluator:
    """Unified evaluator for standalone solving and PRC post-processing."""

    def __init__(
        self,
        env_params,
        model_params,
        eval_params,
        mode: EvalMode = EvalMode.SYNTHETIC,
        tracker: ExperimentTracker | None = None,
    ):
        self.env_params = env_params
        self.model_params = model_params
        self.eval_params = eval_params
        self.mode = mode

        self.logger = getLogger(name="evaluator")
        self.result_folder = get_result_folder()
        self.tracker = tracker
        self.device = setup_device(eval_params["use_cuda"], eval_params["cuda_device_num"])

        if mode == EvalMode.SYNTHETIC:
            self.env = SyntheticEnvironment(**env_params)
        else:
            self.env = TSPLIBEnvironment(**env_params)
        self.env.set_device(self.device)

        self.model = GeldModel(**model_params).to(self.device)
        checkpoint = torch.load(
            f"{eval_params['model_load']['path']}/checkpoint-{eval_params['model_load']['epoch']}.pt",
            map_location=self.device,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])

        self.solver = InferenceSolver(
            self.model,
            self.env,
            self.device,
            use_beam=eval_params.get("beam", False),
            use_prc=eval_params.get("PRC", False),
            beam_size=eval_params.get("beam_size", 16),
            prc_iterations=eval_params.get("num_PRC", 1000),
        )
        self.time_estimator = TimeEstimator()
        self.time_estimator_batch = TimeEstimator()

    def run(self, size=None, distribution=None):
        """Dispatch evaluation by mode."""
        self.time_estimator.reset()
        self.time_estimator_batch.reset()

        if self.mode == EvalMode.SYNTHETIC:
            return self._run_synthetic(size, distribution)
        if self.mode == EvalMode.POSTPROCESS:
            return self._run_postprocess()
        return self.run_tsplib(use_tsplib_dir=False)

    def run_tsplib(self, use_tsplib_dir: bool = False) -> EvalSummary:
        """Evaluate all TSPLIB or National TSP instances."""
        collections = TSPLIB_OPTIMAL_LENGTHS if use_tsplib_dir else NATIONAL_TSP_OPTIMAL_LENGTHS
        gap_buckets = _new_gap_buckets()
        all_gaps = []
        instances: list[EvalInstanceResult] = []
        baseline_length_meter = AverageMeter()
        predicted_length_meter = AverageMeter()

        for name, opt_len in collections.items():
            instance_name = name if use_tsplib_dir else name.lower()
            result = self.solver.solve_batch(
                0,
                1,
                reencode_each_step=use_tsplib_dir,
                load_kwargs={
                    "name": instance_name,
                    "opt_len": torch.tensor(float(opt_len)),
                    "use_tsplib_dir": use_tsplib_dir,
                },
            )
            baseline = float(result.baseline_length)
            predicted = float(result.tour_length.mean())
            gap = (predicted - baseline) / baseline
            all_gaps.append(gap)
            baseline_length_meter.update(baseline, 1)
            predicted_length_meter.update(predicted, 1)
            bucket = _gap_bucket_key(self.env.problem_size)
            _record_gap(gap_buckets, self.env.problem_size, gap)
            instances.append(
                EvalInstanceResult(
                    name=name,
                    problem_size=self.env.problem_size,
                    baseline_length=baseline,
                    predicted_length=predicted,
                    gap_percent=gap * 100.0,
                    gap_bucket=bucket,
                )
            )

            self.logger.info(
                f"PRC, name:{name}, gap:{gap * 100:5f} %, "
                f"predicted:{predicted:5f}, optimal:{baseline:5f}"
            )

        bucket_means, bucket_counts = _log_gap_bucket_means(self.logger, gap_buckets)

        average_gap = float(np.mean(all_gaps) * 100.0)
        self.logger.info(" *** Test Done *** ")
        self.logger.info(f" Average Gap: {average_gap:.4f}%")
        summary = EvalSummary(
            mode="tsplib" if use_tsplib_dir else "national",
            average_gap_percent=average_gap,
            baseline_length_avg=baseline_length_meter.avg,
            predicted_length_avg=predicted_length_meter.avg,
            num_instances=len(instances),
            bucket_means=bucket_means,
            bucket_counts=bucket_counts,
            instances=instances,
        )
        if self.tracker is not None:
            self.tracker.save_eval_results(summary)
        return summary

    def _run_synthetic(self, size, distribution) -> EvalSummary:
        """Evaluate synthetic TSP-n benchmarks across distributions."""
        self.env.load_raw_data(
            self.eval_params["test_episodes"],
            load_eval_data=True,
            load_synthetic_benchmark=True,
            size=size,
            distribution=distribution,
        )
        self.env.set_device(self.device)
        baseline_length_meter = AverageMeter()
        predicted_length_meter = AverageMeter()
        episode = 0
        test_num_episode = self.eval_params["test_episodes"]

        while episode < test_num_episode:
            remaining = test_num_episode - episode
            batch_size = min(self.eval_params["test_batch_size"], remaining)
            baseline, predicted, _ = self._evaluate_batch(episode, batch_size)
            baseline_length_meter.update(baseline, batch_size)
            predicted_length_meter.update(predicted, batch_size)
            episode += batch_size
            elapsed, remain = self.time_estimator.get_est_string(episode, test_num_episode)
            self.logger.info(
                f"episode {episode:3d}/{test_num_episode:3d}, Elapsed[{elapsed}], Remain[{remain}], "
                f"Baseline length:{baseline:.4f}, Predicted length: {predicted:.4f}"
            )

        gap = (predicted_length_meter.avg - baseline_length_meter.avg) / baseline_length_meter.avg * 100
        self.logger.info(" *** Test Done *** ")
        self.logger.info(f" Baseline length: {baseline_length_meter.avg:.4f} ")
        self.logger.info(f" Predicted length: {predicted_length_meter.avg:.4f} ")
        self.logger.info(f" Gap: {gap:.4f}%")
        summary = EvalSummary(
            mode="synthetic",
            average_gap_percent=float(gap),
            baseline_length_avg=baseline_length_meter.avg,
            predicted_length_avg=predicted_length_meter.avg,
            num_instances=test_num_episode,
            size=size,
            distribution=distribution,
        )
        if self.tracker is not None:
            self.tracker.save_eval_results(summary)
        return summary

    def _run_postprocess(self) -> EvalSummary:
        """PRC-only refinement of baseline neural solver tours."""
        baseline_solutions = np.load(
            baseline_solutions_dir() / "INV_so.npy", allow_pickle=True
        ).item()
        gap_buckets = _new_gap_buckets()
        all_gaps = []
        instances: list[EvalInstanceResult] = []
        baseline_length_meter = AverageMeter()
        predicted_length_meter = AverageMeter()

        for name, solution in baseline_solutions.items():
            opt_len = NATIONAL_TSP_OPTIMAL_LENGTHS[name]
            self.env.load_problems(
                0,
                1,
                name=name.lower(),
                opt_len=torch.tensor(float(opt_len)),
            )
            initial_tour = torch.tensor(solution, device=self.device).unsqueeze(0)
            result = self.solver.solve_batch(
                0,
                1,
                skip_greedy=True,
                skip_beam=True,
                initial_tour=initial_tour,
                large_instance_prc=True,
                load_kwargs={"name": name.lower(), "opt_len": torch.tensor(float(opt_len))},
            )
            baseline = float(opt_len)
            predicted = float(result.tour_length.mean())
            gap = (predicted - baseline) / baseline
            all_gaps.append(gap)
            baseline_length_meter.update(baseline, 1)
            predicted_length_meter.update(predicted, 1)
            bucket = _gap_bucket_key(self.env.problem_size)
            _record_gap(gap_buckets, self.env.problem_size, gap)
            instances.append(
                EvalInstanceResult(
                    name=name,
                    problem_size=self.env.problem_size,
                    baseline_length=baseline,
                    predicted_length=predicted,
                    gap_percent=gap * 100.0,
                    gap_bucket=bucket,
                )
            )

            self.logger.info(
                f"PRC postprocess, name:{name}, gap:{gap * 100:5f} %, "
                f"predicted:{predicted:5f}, optimal:{baseline:5f}"
            )

        bucket_means, bucket_counts = _log_gap_bucket_means(self.logger, gap_buckets)

        average_gap = float(np.mean(all_gaps) * 100.0)
        self.logger.info(" *** Test Done *** ")
        self.logger.info(f" Average Gap: {average_gap:.4f}%")
        summary = EvalSummary(
            mode="postprocess",
            average_gap_percent=average_gap,
            baseline_length_avg=baseline_length_meter.avg,
            predicted_length_avg=predicted_length_meter.avg,
            num_instances=len(instances),
            bucket_means=bucket_means,
            bucket_counts=bucket_counts,
            instances=instances,
        )
        if self.tracker is not None:
            self.tracker.save_eval_results(summary)
        return summary

    def _evaluate_batch(self, episode, batch_size):
        """Evaluate one synthetic batch and log gap to baseline."""
        result = self.solver.solve_batch(batch_offset=episode, batch_size=batch_size)
        baseline = float(result.baseline_length.mean())
        predicted = float(result.tour_length.mean())
        gap = (predicted - baseline) / baseline * 100
        elapsed_time, _ = self.time_estimator_batch.get_est_string(1, 1)
        self.logger.info(
            f"batch gap:{gap:4f} %, Elapsed[{elapsed_time}], "
            f"predicted:{predicted:4f}, baseline:{baseline:4f}"
        )
        return baseline, predicted, self.env.problem_size
