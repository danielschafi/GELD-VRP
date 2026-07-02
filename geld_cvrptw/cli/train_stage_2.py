"""Kicks off training stage 2 for GELD - CVRPTW. CLI entrypoint"""

import argparse
import logging

from geld_cvrptw.config.defaults_params import (
    default_env_params,
    default_model_params,
    default_training_stage_2_params,
    default_training_stage_2_optimizer_params,
)

from geld_cvrptw.training.stage_2_trainer import Stage2Trainer
from geld_cvrptw.utils.experiment_tracker import ExperimentTracker
from geld_cvrptw.utils.logging import create_logger, get_result_folder, set_result_folder


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser for stage-2 training."""
    stage_2_defaults = default_training_stage_2_params()
    parser = argparse.ArgumentParser(description="GELD stage-2 supervised training")
    parser.add_argument("--epochs", type=int, default=stage_2_defaults["epochs"])
    parser.add_argument("--train-episodes", type=int, default=stage_2_defaults["train_episodes"])
    parser.add_argument("--batch-size", type=int, default=stage_2_defaults["train_batch_size"])
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--stage1-checkpoint-path",
        type=str,
        required=True,
        help="Path to stage-1 result folder containing the pretrained checkpoint",
    )
    parser.add_argument(
        "--stage1-checkpoint-epoch",
        type=int,
        default=49,
        help="Stage-1 checkpoint epoch to initialize from",
    )
    parser.add_argument(
        "--model-load-path",
        type=str,
        default=None,
        help="Resume stage-2 training from checkpoint directory",
    )
    parser.add_argument("--model-load-epoch", type=int, default=1, help="Checkpoint epoch to load when resuming")
    parser.add_argument(
        "--result-folder",
        type=str,
        default=None,
        help="Override result output directory (defaults to --model-load-path when resuming)",
    )
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
    """Run stage-2 curriculum / self-improvement training on increasing instance sizes."""
    args = build_parser().parse_args()
    create_logger(log_file={"prefix": "train", "desc": "stage_2", "filename": "log.txt"})

    result_folder = args.result_folder
    if result_folder is None and args.model_load_path is not None:
        result_folder = args.model_load_path
    if result_folder is not None:
        set_result_folder(result_folder)

    env_params = default_env_params(mode="train")
    model_params = default_model_params(mode="train")
    optimizer_params = default_training_stage_2_optimizer_params()
    trainer_params = default_training_stage_2_params(
        use_cuda=not args.no_cuda,
        cuda_device_num=args.cuda_device,
        model_load_path=args.stage1_checkpoint_path,
        model_load_epoch=args.stage1_checkpoint_epoch,
    )

    if args.debug:
        trainer_params["epochs"] = 2
        trainer_params["train_episodes"] = 8
        trainer_params["train_batch_size"] = 4
        trainer_params["problem_size_min"] = 20
        trainer_params["problem_size_max"] = 30
    else:
        trainer_params["epochs"] = args.epochs
        trainer_params["train_episodes"] = args.train_episodes
        trainer_params["train_batch_size"] = args.batch_size
    if args.batch_log_interval is not None:
        trainer_params["logging"]["batch_log_interval"] = args.batch_log_interval
    if args.model_load_path is not None:
        trainer_params["model_load"] = {
            "enable": True,
            "path": args.model_load_path,
            "epoch": args.model_load_epoch,
        }

    logger = logging.getLogger("root")
    logger.info(f"Starting stage-2 training with params: {trainer_params}")

    tracker = ExperimentTracker(
        get_result_folder(),
        run_type="train_stage_2",
        wandb_enabled=args.wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        wandb_config={
            "env_params": env_params,
            "model_params": model_params,
            "optimizer_params": optimizer_params,
            "trainer_params": trainer_params,
        },
        run_params={
            "env_params": env_params,
            "model_params": model_params,
            "optimizer_params": optimizer_params,
            "trainer_params": trainer_params,
            "cli_args": vars(args),
        },
    )
    trainer = Stage2Trainer(env_params, model_params, optimizer_params, trainer_params, tracker=tracker)
    trainer.run_training()


if __name__ == "__main__":
    main()
