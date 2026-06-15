"""Project path helpers."""

from pathlib import Path


def project_root() -> Path:
    """Return the repository root directory."""
    return Path(__file__).resolve().parents[1]


def data_dir() -> Path:
    """Path to data directory."""
    return project_root() / "data"


def benchmark_data_dir() -> Path:
    """Path to synthetic and real-world test datasets."""
    return data_dir() / "benchmark"


def training_stage_1_data_dir() -> Path:
    """Path to stage-1 SL training data (LEHD TSP-100)."""
    return data_dir() / "training_stage_1"


def baseline_solutions_dir() -> Path:
    """Path to baseline solver tours for PRC post-processing."""
    return data_dir() / "baseline_solutions"


def result_dir() -> Path:
    """Path to training and evaluation result outputs."""
    return data_dir() / "result"
