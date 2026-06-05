"""Tests for structured experiment metrics export."""

import csv
import json
from pathlib import Path

from geld.utils.experiment_tracker import (
    EvalInstanceResult,
    EvalSummary,
    append_eval_synthetic_row,
    save_eval_instances_csv,
    save_eval_summary,
    save_metrics_csv,
    save_metrics_json,
    should_log_batch,
)
from geld.utils.metrics import LogData


def test_log_data_to_epoch_records():
    log = LogData()
    log.append("train_loss", 1, 0.5)
    log.append("train_loss", 2, 0.3)
    log.append("train_reference_length", 1, 10.0)
    log.append("train_reference_length", 2, 9.5)

    records = log.to_epoch_records()
    assert records == [
        {"epoch": 1, "train_loss": 0.5, "train_reference_length": 10.0},
        {"epoch": 2, "train_loss": 0.3, "train_reference_length": 9.5},
    ]


def test_save_metrics_csv_and_json(tmp_path: Path):
    log = LogData()
    log.append("train_loss", 1, 0.5)
    log.append("train_loss", 2, 0.3)

    csv_path = tmp_path / "metrics.csv"
    json_path = tmp_path / "metrics.json"
    save_metrics_csv(log, csv_path)
    save_metrics_json(log, json_path, metadata={"run_type": "train_sl"})

    with csv_path.open(encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert rows == [
        {"epoch": "1", "train_loss": "0.5"},
        {"epoch": "2", "train_loss": "0.3"},
    ]

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["metadata"]["run_type"] == "train_sl"
    assert payload["epochs"][0]["train_loss"] == 0.5


def test_eval_artifacts(tmp_path: Path):
    instances = [
        EvalInstanceResult(
            name="eil101",
            problem_size=101,
            baseline_length=100.0,
            predicted_length=105.0,
            gap_percent=5.0,
            gap_bucket="100",
        )
    ]
    summary = EvalSummary(
        mode="national",
        average_gap_percent=5.0,
        baseline_length_avg=100.0,
        predicted_length_avg=105.0,
        num_instances=1,
        bucket_means={"100": 5.0},
        bucket_counts={"100": 1},
        instances=instances,
    )

    save_eval_instances_csv(instances, tmp_path / "eval_instances.csv")
    save_eval_summary(summary, tmp_path / "eval_summary.json")

    with (tmp_path / "eval_instances.csv").open(encoding="utf-8") as csv_file:
        row = next(csv.DictReader(csv_file))
    assert row["name"] == "eil101"
    assert float(row["gap_percent"]) == 5.0

    payload = json.loads((tmp_path / "eval_summary.json").read_text(encoding="utf-8"))
    assert payload["average_gap_percent"] == 5.0
    assert payload["bucket_means"]["100"] == 5.0


def test_append_eval_synthetic_summary(tmp_path: Path):
    summary = EvalSummary(
        mode="synthetic",
        average_gap_percent=1.2,
        baseline_length_avg=10.0,
        predicted_length_avg=10.12,
        num_instances=200,
        size=100,
        distribution="uniform",
    )
    path = tmp_path / "eval_synthetic_summary.csv"
    append_eval_synthetic_row(summary, path)
    append_eval_synthetic_row(summary, path)

    with path.open(encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert len(rows) == 2
    assert rows[0]["distribution"] == "uniform"
    assert float(rows[0]["gap_percent"]) == 1.2


def test_should_log_batch_milestones():
    assert should_log_batch(100, 100, 50) is True
    assert should_log_batch(50, 1000, 50) is True
    assert should_log_batch(51, 1000, 50) is False
    assert should_log_batch(100, 1000, 50) is True
