"""CLI entry point for CVRPTW beam-search scaling benchmark."""

from __future__ import annotations

import argparse
import logging
import random

import numpy as np
import torch

from geld_cvrptw.config.defaults_params import default_scaling_benchmark_params
from geld_cvrptw.inference.scaling_benchmark import ScalingBenchmark
from geld_cvrptw.utils.experiment_tracker import ExperimentTracker
from geld_cvrptw.utils.logging import create_logger, get_result_folder


def seed_everything(seed: int = 2024) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_parser() -> argparse.ArgumentParser:
    defaults = default_scaling_benchmark_params()
    parser = argparse.ArgumentParser(
        description="Benchmark GELD-CVRPTW beam-search decode scaling on synthetic instances",
    )
    parser.add_argument("--checkpoint-path", type=str, default=defaults["model_load"]["path"])
    parser.add_argument("--checkpoint-epoch", type=int, default=defaults["model_load"]["epoch"])
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=defaults["sizes"],
        help="Problem sizes (customer counts) to benchmark",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Instances per size (default: 50 for n<=500, 20 for n<=2000, 10 for n<=10000, 5 above)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override adaptive batch size per size bucket",
    )
    parser.add_argument("--beam-size", type=int, default=defaults["decoder"]["beam_size"])
    parser.add_argument("--seed", type=int, default=defaults["seed"])
    parser.add_argument("--alpha", type=float, default=defaults["alpha"], help="TW tightness for generation")
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="geld-vrp")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    create_logger(log_file={"prefix": "eval", "desc": "benchmark_scaling", "filename": "log.txt"})
    seed_everything(args.seed)

    benchmark_params = default_scaling_benchmark_params(
        use_cuda=not args.no_cuda,
        cuda_device_num=args.cuda_device,
    )
    benchmark_params["model_load"] = {
        "path": args.checkpoint_path,
        "epoch": args.checkpoint_epoch,
    }
    benchmark_params["sizes"] = args.sizes
    benchmark_params["episodes"] = args.episodes
    benchmark_params["batch_size"] = args.batch_size
    benchmark_params["seed"] = args.seed
    benchmark_params["alpha"] = args.alpha
    benchmark_params["decoder"]["beam_size"] = args.beam_size

    tracker = ExperimentTracker(
        get_result_folder(),
        run_type="benchmark_scaling",
        wandb_enabled=args.wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        wandb_config={"benchmark_params": benchmark_params},
        run_params={
            "benchmark_params": benchmark_params,
            "cli_args": vars(args),
        },
    )

    summary = ScalingBenchmark(benchmark_params, tracker=tracker).run()

    logger = logging.getLogger("root")
    for size_summary in summary.size_summaries:
        logger.info(
            f"n={size_summary.problem_size}: "
            f"mean={size_summary.decode_time_mean_sec:.4f}s, "
            f"p95={size_summary.decode_time_p95_sec:.4f}s, "
            f"{size_summary.instances_per_sec:.2f} inst/s"
        )
    if summary.scaling_exponent is not None:
        logger.info(
            f"Scaling exponent alpha={summary.scaling_exponent:.2f}, "
            f"prefactor={summary.scaling_prefactor:.4e}"
        )
    logger.info(f"Max successful size: {summary.max_successful_size}")
    tracker.finish()


if __name__ == "__main__":
    main()
