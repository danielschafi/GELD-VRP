"""Structured experiment metrics: CSV/JSON export, plots, optional wandb."""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from geld.utils.logging import util_save_log_image_with_label
from geld.utils.metrics import LogData

logger = logging.getLogger(__name__)

METRICS_CSV = "metrics.csv"
METRICS_JSON = "metrics.json"
EVAL_INSTANCES_CSV = "eval_instances.csv"
EVAL_SUMMARY_JSON = "eval_summary.json"
EVAL_SYNTHETIC_SUMMARY_CSV = "eval_synthetic_summary.csv"


@dataclass
class EvalInstanceResult:
    """Per-instance evaluation record."""

    name: str
    problem_size: int
    baseline_length: float
    predicted_length: float
    gap_percent: float
    gap_bucket: str


@dataclass
class EvalSummary:
    """Aggregated evaluation result."""

    mode: str
    average_gap_percent: float
    baseline_length_avg: float
    predicted_length_avg: float
    num_instances: int
    bucket_means: dict[str, float] = field(default_factory=dict)
    bucket_counts: dict[str, int] = field(default_factory=dict)
    size: int | None = None
    distribution: str | None = None
    instances: list[EvalInstanceResult] = field(default_factory=list)


def save_metrics_csv(result_log: LogData, path: Path) -> None:
    """Write epoch-keyed training metrics to a pandas-friendly CSV."""
    records = result_log.to_epoch_records()
    if not records:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for record in records for key in record})
    if "epoch" in fieldnames:
        fieldnames.remove("epoch")
        fieldnames = ["epoch", *fieldnames]

    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def save_metrics_json(
    result_log: LogData,
    path: Path,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Write metrics plus optional run metadata to JSON."""
    payload = {
        "metadata": metadata or {},
        "epochs": result_log.to_epoch_records(),
        "series": {key: result_log.get(key) for key in sorted(result_log.get_keys())},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as json_file:
        json.dump(payload, json_file, indent=2)


def save_training_plots(result_folder: Path, result_log: LogData, logging_config: dict | None = None) -> None:
    """Save matplotlib PNG curves and legacy JPG plots when configured."""
    records = result_log.to_epoch_records()
    if not records:
        return

    plots_dir = result_folder / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    epochs = [record["epoch"] for record in records]
    metric_keys = sorted(key for key in records[0] if key != "epoch")

    for metric_key in metric_keys:
        values = [record[metric_key] for record in records]
        fig, axis = plt.subplots(figsize=(8, 4))
        axis.plot(epochs, values, marker="o", linewidth=1.5, markersize=3)
        axis.set_xlabel("epoch")
        axis.set_ylabel(metric_key)
        axis.set_title(metric_key)
        axis.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(plots_dir / f"{metric_key}.png", dpi=150)
        plt.close(fig)

    length_keys = [key for key in metric_keys if "length" in key]
    if len(length_keys) >= 2:
        fig, axis = plt.subplots(figsize=(8, 4))
        for metric_key in length_keys:
            values = [record[metric_key] for record in records]
            axis.plot(epochs, values, marker="o", linewidth=1.5, markersize=3, label=metric_key)
        axis.set_xlabel("epoch")
        axis.set_ylabel("tour length")
        axis.set_title("tour lengths")
        axis.legend()
        axis.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(plots_dir / "tour_lengths.png", dpi=150)
        plt.close(fig)

    if logging_config:
        image_prefix = str(result_folder / "latest")
        if "log_image_params_1" in logging_config:
            util_save_log_image_with_label(
                image_prefix,
                logging_config["log_image_params_1"],
                result_log,
                labels=["train_reference_length"],
            )
        if "log_image_params_2" in logging_config:
            util_save_log_image_with_label(
                image_prefix,
                logging_config["log_image_params_2"],
                result_log,
                labels=["train_loss"],
            )


def save_eval_instances_csv(instances: list[EvalInstanceResult], path: Path) -> None:
    """Write per-instance evaluation rows to CSV."""
    if not instances:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(instances[0]).keys())
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for instance in instances:
            writer.writerow(asdict(instance))


def save_eval_summary(summary: EvalSummary, path: Path) -> None:
    """Write aggregated evaluation summary to JSON."""
    payload = asdict(summary)
    payload.pop("instances", None)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as json_file:
        json.dump(payload, json_file, indent=2)


def append_eval_synthetic_row(summary: EvalSummary, path: Path) -> None:
    """Append one synthetic benchmark row to a session summary CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "size": summary.size,
        "distribution": summary.distribution,
        "baseline_length_avg": summary.baseline_length_avg,
        "predicted_length_avg": summary.predicted_length_avg,
        "gap_percent": summary.average_gap_percent,
        "num_instances": summary.num_instances,
    }
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


class ExperimentTracker:
    """Persist structured metrics, plots, and optional wandb logging."""

    def __init__(
        self,
        result_folder: Path,
        run_type: str,
        wandb_enabled: bool = False,
        wandb_project: str = "geld",
        wandb_run_name: str | None = None,
        wandb_config: dict[str, Any] | None = None,
    ):
        self.result_folder = Path(result_folder)
        self.run_type = run_type
        self.wandb_enabled = wandb_enabled
        self._wandb_run = None

        if wandb_enabled:
            try:
                import wandb
            except ImportError as exc:
                raise ImportError(
                    "wandb is not installed. Run `uv sync --extra wandb` or pass --no-wandb."
                ) from exc

            self._wandb_run = wandb.init(
                project=wandb_project,
                name=wandb_run_name,
                config=wandb_config or {},
                dir=str(self.result_folder),
            )

    def log_epoch(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log one epoch of scalar metrics to wandb when enabled."""
        if not self._wandb_run:
            return
        import wandb

        epoch = int(step if step is not None else metrics.get("epoch", 0))
        wandb.log(metrics, step=epoch)

    def save_training_progress(
        self,
        result_log: LogData,
        logging_config: dict | None = None,
        metadata: dict[str, Any] | None = None,
        save_plots: bool = True,
    ) -> None:
        """Write metrics.csv/json and refresh training plots."""
        save_metrics_csv(result_log, self.result_folder / METRICS_CSV)
        save_metrics_json(result_log, self.result_folder / METRICS_JSON, metadata=metadata)
        if save_plots:
            save_training_plots(self.result_folder, result_log, logging_config)

    def save_eval_results(self, summary: EvalSummary) -> None:
        """Persist evaluation CSV/JSON artifacts."""
        save_eval_instances_csv(summary.instances, self.result_folder / EVAL_INSTANCES_CSV)
        save_eval_summary(summary, self.result_folder / EVAL_SUMMARY_JSON)

        if summary.mode == "synthetic":
            append_eval_synthetic_row(summary, self.result_folder / EVAL_SYNTHETIC_SUMMARY_CSV)

        if self._wandb_run:
            import wandb

            log_payload = {
                "eval/average_gap_percent": summary.average_gap_percent,
                "eval/baseline_length_avg": summary.baseline_length_avg,
                "eval/predicted_length_avg": summary.predicted_length_avg,
                "eval/num_instances": summary.num_instances,
            }
            if summary.size is not None:
                log_payload["eval/size"] = summary.size
            if summary.distribution is not None:
                log_payload["eval/distribution"] = summary.distribution
            for bucket, mean_gap in summary.bucket_means.items():
                log_payload[f"eval/bucket_{bucket}_gap_percent"] = mean_gap
            wandb.log(log_payload)

    def finish(self) -> None:
        """Close optional wandb run."""
        if self._wandb_run is not None:
            import wandb

            wandb.finish()
            self._wandb_run = None


def should_log_batch(episode: int, total: int, interval: int) -> bool:
    """Return whether to emit a batch progress log line."""
    if interval <= 0:
        return False
    if episode >= total:
        return True
    if episode % interval == 0:
        return True
    if total <= 20:
        return True
    progress_pct = 100.0 * episode / total
    prev_pct = 100.0 * max(episode - interval, 0) / total
    for milestone in range(10, 100, 10):
        if prev_pct < milestone <= progress_pct:
            return True
    return False
