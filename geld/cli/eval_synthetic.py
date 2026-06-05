"""CLI entry point for synthetic benchmark evaluation."""

import argparse
import logging
import random

import numpy as np
import torch

from geld.config.defaults import default_env_params, default_eval_params, default_model_params
from geld.inference.evaluator import EvalMode, InferenceEvaluator
from geld.utils.logging import copy_all_src, create_logger


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
    return parser


def main():
    """Evaluate GELD on synthetic TSP-n benchmarks (four distributions)."""
    args = build_parser().parse_args()
    create_logger(log_file={"desc": "eval_synthetic", "filename": "log.txt"})
    seed_everything(2024)

    env_params = default_env_params(mode="test", sub_path=False)
    model_params = default_model_params(mode="test")
    eval_params = default_eval_params(use_cuda=not args.no_cuda, cuda_device_num=args.cuda_device)
    eval_params["model_load"] = {"path": args.checkpoint_path, "epoch": args.checkpoint_epoch}
    eval_params["beam"] = not args.no_beam
    eval_params["PRC"] = not args.no_prc

    distributions = ["uniform", "clustered", "explosion", "implosion"]
    sizes = [100, 500, 1000, 5000, 10000]

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
            evaluator = InferenceEvaluator(env_params, model_params, eval_params, mode=EvalMode.SYNTHETIC)
            copy_all_src(evaluator.result_folder)
            evaluator.run(size=size, distribution=distribution)


if __name__ == "__main__":
    main()
