"""CLI entry point for TSPLIB / National TSP evaluation."""

import argparse
import logging
import random

import numpy as np
import torch

from geld.config.defaults import (
    default_env_params,
    default_eval_params,
    default_model_params,
)
from geld.inference.evaluator import EvalMode, InferenceEvaluator
from geld.utils.experiment_tracker import ExperimentTracker
from geld.utils.logging import create_logger, get_result_folder


def seed_everything(seed=2024):
    """Set random seeds for reproducible evaluation."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser for TSPLIB/National TSP evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate GELD on real-world TSP instances"
    )
    parser.add_argument(
        "--checkpoint-path", type=str, default="result/pre_trained_model"
    )
    parser.add_argument("--checkpoint-epoch", type=int, default=49)
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument(
        "--postprocess",
        action="store_true",
        help="Run PRC post-processing on baseline tours",
    )
    parser.add_argument(
        "--tsplib", action="store_true", help="Use TSPLIB instead of National TSP"
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Log evaluation metrics to Weights & Biases",
    )
    parser.add_argument("--wandb-project", type=str, default="geld")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    return parser


def main():
    """Evaluate GELD on real-world instances or run PRC post-processing."""
    args = build_parser().parse_args()
    create_logger(log_file={"desc": "eval_tsplib", "filename": "log.txt"})
    seed_everything(2024)

    env_params = default_env_params(mode="test", use_subpath_augmentation=False)
    env_params["eval_tsplib"] = True
    model_params = default_model_params(mode="test")
    eval_params = default_eval_params(
        use_cuda=not args.no_cuda, cuda_device_num=args.cuda_device
    )
    eval_params["model_load"] = {
        "path": args.checkpoint_path,
        "epoch": args.checkpoint_epoch,
    }

    mode = EvalMode.POSTPROCESS if args.postprocess else EvalMode.TSPLIB
    tracker = ExperimentTracker(
        get_result_folder(),
        run_type="eval_tsplib",
        wandb_enabled=args.wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        wandb_config={"eval_params": eval_params, "mode": mode.value},
    )
    evaluator = InferenceEvaluator(
        env_params, model_params, eval_params, mode=mode, tracker=tracker
    )

    if args.tsplib and not args.postprocess:
        summary = evaluator.run_tsplib(use_tsplib_dir=True)
    else:
        summary = evaluator.run()

    logging.getLogger("root").info(
        f"Evaluation complete. Average gap: {summary.average_gap_percent:.4f}%"
    )
    tracker.finish()


if __name__ == "__main__":
    main()
