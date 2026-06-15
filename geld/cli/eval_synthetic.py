"""CLI entry point for synthetic benchmark evaluation."""

import argparse
import logging
import random

import numpy as np
import torch

from geld.config.defaults import default_env_params, default_eval_params, default_model_params
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
    """Build argument parser for synthetic benchmark evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate GELD on synthetic benchmarks")
    parser.add_argument("--checkpoint-path", type=str, default="result/pre_trained_model")
    parser.add_argument("--checkpoint-epoch", type=int, default=49)
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--no-beam", action="store_true")
    parser.add_argument("--no-prc", action="store_true")
    parser.add_argument("--wandb", action="store_true", help="Log evaluation metrics to Weights & Biases")
    parser.add_argument("--wandb-project", type=str, default="geld")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    return parser


def main():
    """Evaluate GELD on synthetic TSP-n benchmarks (four distributions)."""
    args = build_parser().parse_args()
    create_logger(log_file={"desc": "eval_synthetic", "filename": "log.txt"})
    seed_everything(2024)

    env_params = default_env_params(mode="test", use_subpath_augmentation=False)
    model_params = default_model_params(mode="test")
    eval_params = default_eval_params(use_cuda=not args.no_cuda, cuda_device_num=args.cuda_device)
    eval_params["model_load"] = {"path": args.checkpoint_path, "epoch": args.checkpoint_epoch}
    eval_params["beam"] = not args.no_beam
    eval_params["PRC"] = not args.no_prc

    distributions = ["uniform", "clustered", "explosion", "implosion"]
    sizes = [100, 500, 1000, 5000, 10000]
    result_folder = get_result_folder()
    tracker = ExperimentTracker(
        result_folder,
        run_type="eval_synthetic",
        wandb_enabled=args.wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        wandb_config={"eval_params": eval_params},
    )

    for distribution in distributions:
        for size in sizes:
            if size == 100:
                eval_params["test_episodes"] = 200
                eval_params["test_batch_size"] = 200
            elif size == 500:
                eval_params["test_episodes"] = 200
                eval_params["test_batch_size"] = 100
            elif size == 1000:
                eval_params["test_episodes"] = 200
                eval_params["test_batch_size"] = 200
            elif size in (5000, 10000):
                eval_params["test_episodes"] = 20
                eval_params["test_batch_size"] = 20

            logging.getLogger("root").info(
                f"Evaluating size={size}, distribution={distribution}"
            )
            evaluator = InferenceEvaluator(
                env_params,
                model_params,
                eval_params,
                mode=EvalMode.SYNTHETIC,
                tracker=tracker,
            )
            summary = evaluator.run(size=size, distribution=distribution)
            logging.getLogger("root").info(
                f"Summary size={size} distribution={distribution} gap={summary.average_gap_percent:.4f}%"
            )

    tracker.finish()


if __name__ == "__main__":
    main()
