"""CLI entry point for CVRPTW benchmark evaluation."""

from __future__ import annotations

import argparse
import logging
import random

import numpy as np
import torch

from geld_cvrptw.config.defaults_params import default_cvrptw_eval_params
from geld_cvrptw.inference.evaluator import CvrptwEvaluator
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
    defaults = default_cvrptw_eval_params()
    parser = argparse.ArgumentParser(description="Evaluate GELD-CVRPTW on benchmark instances")
    parser.add_argument("--checkpoint-path", type=str, default=defaults["model_load"]["path"])
    parser.add_argument("--checkpoint-epoch", type=int, default=defaults["model_load"]["epoch"])
    parser.add_argument(
        "--benchmark",
        type=str,
        default="all",
        choices=["synthetic", "solomon", "homberger", "all"],
    )
    parser.add_argument("--synthetic-episodes", type=int, default=defaults["synthetic"]["episodes"])
    parser.add_argument("--synthetic-batch-size", type=int, default=defaults["synthetic"]["batch_size"])
    parser.add_argument("--bootstrap-start-node", type=int, default=defaults["decoder"]["bootstrap_start_node"])
    parser.add_argument("--no-beam", action="store_true", help="Use greedy decoding instead of beam search")
    parser.add_argument("--beam-size", type=int, default=defaults["decoder"]["beam_size"])
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="geld-vrp")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    create_logger(log_file={"desc": "eval_cvrptw", "filename": "log.txt"})
    seed_everything(2024)

    eval_params = default_cvrptw_eval_params(use_cuda=not args.no_cuda, cuda_device_num=args.cuda_device)
    eval_params["model_load"] = {
        "path": args.checkpoint_path,
        "epoch": args.checkpoint_epoch,
    }
    eval_params["synthetic"]["episodes"] = args.synthetic_episodes
    eval_params["synthetic"]["batch_size"] = args.synthetic_batch_size
    eval_params["decoder"]["bootstrap_start_node"] = args.bootstrap_start_node
    eval_params["decoder"]["name"] = "greedy" if args.no_beam else "beam_search"
    eval_params["decoder"]["beam_size"] = args.beam_size

    tracker = ExperimentTracker(
        get_result_folder(),
        run_type="eval_cvrptw",
        wandb_enabled=args.wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        wandb_config={"eval_params": eval_params},
    )

    evaluator = CvrptwEvaluator(eval_params, tracker=tracker)
    summaries = evaluator.run(args.benchmark)

    logger = logging.getLogger("root")
    for summary in summaries:
        logger.info(
            f"{summary.mode}: gap={summary.average_gap_percent:.4f}% "
            f"({summary.num_instances} instances)"
        )
    tracker.finish()


if __name__ == "__main__":
    main()
