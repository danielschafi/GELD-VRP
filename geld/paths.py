"""Project path helpers."""

from pathlib import Path


def project_root() -> Path:
    """Return the repository root directory."""
    return Path(__file__).resolve().parents[1]


def benchmark_data_dir() -> Path:
    """Path to synthetic and real-world test datasets."""
    return project_root() / "Test_data"


def sl_training_data_dir() -> Path:
    """Path to stage-1 SL training data (LEHD TSP-100)."""
    return project_root() / "SL_training_data"


def baseline_solutions_dir() -> Path:
    """Path to baseline solver tours for PRC post-processing."""
    return project_root() / "baseline_solutions"


def result_dir() -> Path:
    """Path to training and evaluation result outputs."""
    return project_root() / "result"
