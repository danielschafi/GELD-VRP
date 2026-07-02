"""CLI entrypoint for stage-2 curriculum / self-improvement training."""

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
    parser.add_argument(
        "--instances-per-epoch",
        type=int,
        default=stage_2_defaults["instances_per_epoch"],
        help="Generated large instances per epoch",
    )
    parser.add_argument("--batch-size", type=int, default=stage_2_defaults["batch_size"])
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--pretrained-dir",
        type=str,
        required=True,
        help="Stage-1 result folder containing the pretrained checkpoint",
    )
    parser.add_argument(
        "--pretrained-epoch",
        type=int,
        default=49,
        help="Stage-1 checkpoint epoch to initialize from",
    )
    parser.add_argument(
        "--resume-dir",
        type=str,
        default=None,
        help="Resume stage-2 training from checkpoint directory",
    )
    parser.add_argument(
        "--resume-epoch",
        type=int,
        default=1,
        help="Checkpoint epoch to load when resuming stage 2",
    )
    parser.add_argument(
        "--result-folder",
        type=str,
        default=None,
        help="Override result output directory (defaults to --resume-dir when resuming)",
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
    if result_folder is None and args.resume_dir is not None:
        result_folder = args.resume_dir
    if result_folder is not None:
        set_result_folder(result_folder)

    env_params = default_env_params(mode="train")
    model_params = default_model_params(mode="train")
    optimizer_params = default_training_stage_2_optimizer_params()
    trainer_params = default_training_stage_2_params(
        use_cuda=not args.no_cuda,
        cuda_device_num=args.cuda_device,
        pretrained_dir=args.pretrained_dir,
        pretrained_epoch=args.pretrained_epoch,
    )

    if args.debug:
        trainer_params["epochs"] = 2
        trainer_params["instances_per_epoch"] = 8
        trainer_params["batch_size"] = 4
        trainer_params["n_customers_min"] = 20
        trainer_params["n_customers_max"] = 30
    else:
        trainer_params["epochs"] = args.epochs
        trainer_params["instances_per_epoch"] = args.instances_per_epoch
        trainer_params["batch_size"] = args.batch_size
    if args.batch_log_interval is not None:
        trainer_params["logging"]["batch_log_interval"] = args.batch_log_interval
    if args.resume_dir is not None:
        trainer_params["resume_checkpoint"] = {
            "enable": True,
            "path": args.resume_dir,
            "epoch": args.resume_epoch,
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
