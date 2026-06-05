"""Minimal LEHD-format training data for smoke tests."""

from __future__ import annotations

from pathlib import Path

from geld.data.loaders import load_tsp_instances_with_baselines
from geld.paths import benchmark_data_dir, sl_training_data_dir

DEFAULT_LEHD_FILENAME = "train_TSP100_n100w-001.txt"
DEFAULT_NUM_INSTANCES = 64


def instance_to_lehd_line(nodes, tour) -> str:
    """Serialize one TSP-100 instance into LEHD text format."""
    coord_parts: list[str] = []
    for x, y in nodes:
        coord_parts.extend([str(float(x)), str(float(y))])
    tour_parts = [str(int(node) + 1) for node in tour.tolist()]
    if tour_parts:
        tour_parts.append(tour_parts[0])
    return " ".join(coord_parts) + " output " + " ".join(tour_parts)


def write_minimal_lehd_training_file(
    output_path: Path | None = None,
    *,
    num_instances: int = DEFAULT_NUM_INSTANCES,
) -> Path:
    """Write a small LEHD training file from bundled synthetic TSP-100 data."""
    if output_path is None:
        output_path = sl_training_data_dir() / DEFAULT_LEHD_FILENAME

    instances, tours, _ = load_tsp_instances_with_baselines(
        benchmark_data_dir(),
        size=100,
        distribution="uniform",
    )
    if instances.shape[0] < num_instances:
        raise ValueError(
            f"Need at least {num_instances} synthetic instances, found {instances.shape[0]}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        instance_to_lehd_line(instances[idx], tours[idx])
        for idx in range(num_instances)
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def ensure_minimal_lehd_data(
    *,
    num_instances: int = DEFAULT_NUM_INSTANCES,
    force: bool = False,
) -> Path:
    """Create LEHD training data when missing (or when force=True)."""
    output_path = sl_training_data_dir() / DEFAULT_LEHD_FILENAME
    if output_path.exists() and not force:
        return output_path
    return write_minimal_lehd_training_file(output_path, num_instances=num_instances)
