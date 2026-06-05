"""End-to-end smoke pipeline: stage-1 SL, stage-2 SIL, synthetic eval."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

from geld.config.defaults import default_env_params, default_eval_params, default_model_params
from geld.inference.evaluator import EvalMode, InferenceEvaluator
from geld.paths import project_root, result_dir
from geld.utils.experiment_tracker import ExperimentTracker
from geld.utils.logging import create_logger, get_result_folder, set_result_folder

from tests.support.lehd_fixture import ensure_minimal_lehd_data

logger = logging.getLogger(__name__)

SMOKE_MARKER = "smoke_pipeline_complete.json"


def latest_result_with_checkpoint(
    checkpoint_epoch: int = 2,
    *,
    exclude: set[Path] | None = None,
) -> Path:
    """Return the newest result directory containing the requested checkpoint."""
    exclude = exclude or set()
    ignored = {"pre_trained_model", "_smoke_test_ckpt"}
    candidates = [
        path
        for path in result_dir().iterdir()
        if path.is_dir()
        and path.name not in ignored
        and path not in exclude
        and (path / f"checkpoint-{checkpoint_epoch}.pt").exists()
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No result directories found with checkpoint-{checkpoint_epoch}.pt"
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def run_smoke_eval(
    checkpoint_path: Path,
    checkpoint_epoch: int,
    *,
    use_cuda: bool = True,
    cuda_device: int = 0,
) -> dict:
    """Run a tiny synthetic evaluation on TSP-100 uniform."""
    create_logger(log_file={"desc": "eval_smoke", "filename": "log.txt"})
    result_folder = get_result_folder()

    env_params = default_env_params(mode="test", use_subpath_augmentation=False)
    model_params = default_model_params(mode="test")
    eval_params = default_eval_params(use_cuda=use_cuda, cuda_device_num=cuda_device)
    eval_params["model_load"] = {"path": str(checkpoint_path), "epoch": checkpoint_epoch}
    eval_params["test_episodes"] = 2
    eval_params["test_batch_size"] = 2
    eval_params["beam"] = False
    eval_params["PRC"] = False

    tracker = ExperimentTracker(result_folder, run_type="eval_smoke")
    evaluator = InferenceEvaluator(
        env_params,
        model_params,
        eval_params,
        mode=EvalMode.SYNTHETIC,
        tracker=tracker,
    )
    summary = evaluator.run(size=100, distribution="uniform")
    tracker.finish()

    payload = {
        "mode": summary.mode,
        "size": summary.size,
        "distribution": summary.distribution,
        "num_instances": summary.num_instances,
        "average_gap_percent": summary.average_gap_percent,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_epoch": checkpoint_epoch,
        "eval_result_folder": str(result_folder),
    }
    return payload


def write_completion_marker(marker_path: Path, payload: dict) -> None:
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def run_cli_step(command: list[str]) -> None:
    logger.info("Running: %s", " ".join(command))
    subprocess.run(command, check=True, cwd=project_root())


def run_pipeline(*, use_cuda: bool = True, cuda_device: int = 0) -> dict:
    """Run stage-1, stage-2, and minimal synthetic eval."""
    ensure_minimal_lehd_data()

    cuda_args: list[str] = []
    if not use_cuda:
        cuda_args.append("--no-cuda")
    else:
        cuda_args.extend(["--cuda-device", str(cuda_device)])

    run_cli_step(["uv", "run", "geld-train-sl", "--debug", *cuda_args])
    sl_run = latest_result_with_checkpoint(checkpoint_epoch=2)
    sl_checkpoint = sl_run / "checkpoint-2.pt"
    if not sl_checkpoint.exists():
        raise FileNotFoundError(f"Stage-1 checkpoint not found: {sl_checkpoint}")

    run_cli_step(
        [
            "uv",
            "run",
            "geld-train-stage2",
            "--debug",
            "--model-load-path",
            str(sl_run),
            "--model-load-epoch",
            "2",
            *cuda_args,
        ]
    )
    stage2_run = latest_result_with_checkpoint(checkpoint_epoch=2, exclude={sl_run})
    stage2_checkpoint = stage2_run / "checkpoint-2.pt"
    if not stage2_checkpoint.exists():
        raise FileNotFoundError(f"Stage-2 checkpoint not found: {stage2_checkpoint}")

    eval_summary = run_smoke_eval(
        stage2_checkpoint.parent,
        checkpoint_epoch=2,
        use_cuda=use_cuda,
        cuda_device=cuda_device,
    )

    marker_path = result_dir() / SMOKE_MARKER
    payload = {
        "status": "ok",
        "stage1_run": str(sl_run),
        "stage2_run": str(stage2_run),
        "eval": eval_summary,
    }
    write_completion_marker(marker_path, payload)
    logger.info("Smoke pipeline completed successfully.")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run GELD smoke pipeline")
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument(
        "--result-folder",
        type=str,
        default=None,
        help="Optional fixed result root for eval logging",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    if args.result_folder:
        set_result_folder(Path(args.result_folder))
    run_pipeline(use_cuda=not args.no_cuda, cuda_device=args.cuda_device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
