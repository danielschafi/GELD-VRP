"""CLI entry point for stage-2 curriculum training."""

import argparse
import logging
import random

import numpy as np
import torch

from geld.config.defaults import (
    default_env_params,
    default_model_params,
    default_stage2_optimizer_params,
    default_stage2_trainer_params,
)
from geld.training.stage2_trainer import CurriculumTrainer
from geld.utils.logging import copy_all_src, create_logger


def seed_everything(seed=2024):
    """Set random seeds for reproducible training and evaluation."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser for stage-2 SIL curriculum training."""
    parser = argparse.ArgumentParser(description="GELD stage-2 curriculum training")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--model-load-path", type=str, default="result/Here")
    parser.add_argument("--model-load-epoch", type=int, default=1)
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser


def main():
    """Run stage-2 self-improvement learning with BS/PRC pseudo-labels."""
    args = build_parser().parse_args()
    create_logger(log_file={"desc": "train_stage2", "filename": "log.txt"})
    seed_everything(2024)

    env_params = default_env_params(mode="train")
    model_params = default_model_params(mode="train")
    optimizer_params = default_stage2_optimizer_params()
    trainer_params = default_stage2_trainer_params(
        use_cuda=not args.no_cuda,
        cuda_device_num=args.cuda_device,
        model_load_path=args.model_load_path,
        model_load_epoch=args.model_load_epoch,
    )
    trainer_params["epochs"] = args.epochs

    if args.debug:
        trainer_params["epochs"] = 2
        trainer_params["train_episodes"] = 8
        trainer_params["train_batch_size"] = 4

    logger = logging.getLogger("root")
    logger.info(f"Starting stage-2 training with params: {trainer_params}")

    trainer = CurriculumTrainer(env_params, model_params, optimizer_params, trainer_params)
    copy_all_src(trainer.result_folder)
    trainer.run()


if __name__ == "__main__":
    main()
