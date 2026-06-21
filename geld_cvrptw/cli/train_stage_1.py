"""Kicks off training stage 1 for GELD - CVRPTW. CLI entrypoint"""

import argparse
import logging

from geld_cvrptw.config.defaults_params import (
    default_env_params,
    default_model_params,
    default_training_stage_1_params,
    default_training_stage_1_optimizer_params,
)

from geld_cvrptw.training.stage_1_trainer import Stage1Trainer
from geld_cvrptw.utils.experiment_tracker import ExperimentTracker
from geld_cvrptw.utils.logging import create_logger, get_result_folder


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser for stage-1 training."""
    stage_1_defaults = default_training_stage_1_params()
    parser = argparse.ArgumentParser(description="GELD stage-1 supervised training")
    parser.add_argument("--epochs", type=int, default=stage_1_defaults["epochs"])
    parser.add_argument("--train-episodes", type=int, default=1000000) # this is a cap, is set to dataset size (samples per epoch)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--wandb", action="store_true", help="Log metrics to Weights & Biases")
    parser.add_argument("--wandb-project", type=str, default="geld-vrp")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument(
        "--batch-log-interval",
        type=int,
        default=None,
        help="Log every N batches (default: 50)",
    )
    return parser


def main():
    """Run stage-1 supervised learning on small-scale TSP instances."""
    args = build_parser().parse_args()
    create_logger(log_file={"desc": "train_stage_1", "filename": "log.txt"})

    env_params = default_env_params(mode="train")
    model_params = default_model_params(mode="train")
    optimizer_params = default_training_stage_1_optimizer_params()
    trainer_params = default_training_stage_1_params(use_cuda=not args.no_cuda, cuda_device_num=args.cuda_device)

    if args.debug:
        trainer_params["epochs"] = 2
        trainer_params["train_episodes"] = 8
        trainer_params["train_batch_size"] = 4
    else:
        trainer_params["epochs"] = args.epochs
        trainer_params["train_episodes"] = args.train_episodes
        trainer_params["train_batch_size"] = args.batch_size
    if args.batch_log_interval is not None:
        trainer_params["logging"]["batch_log_interval"] = args.batch_log_interval

    logger = logging.getLogger("root")
    logger.info(f"Starting stage-1 training with params: {trainer_params}")

    tracker = ExperimentTracker(
        get_result_folder(),
        run_type="train_stage_1",
        wandb_enabled=args.wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        wandb_config={
            "env_params": env_params,
            "model_params": model_params,
            "optimizer_params": optimizer_params,
            "trainer_params": trainer_params,
        },
    )
    trainer = Stage1Trainer(env_params, model_params, optimizer_params, trainer_params, tracker=tracker)
    trainer.run_training()


if __name__ == "__main__":
    main()
