"""SLURM-backed end-to-end smoke test for training and evaluation."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

from geld.paths import project_root, result_dir
from tests.support.lehd_fixture import ensure_minimal_lehd_data
from tests.support.smoke_pipeline import SMOKE_MARKER, run_pipeline, run_smoke_eval

pytestmark = pytest.mark.smoke


def _slurm_available() -> bool:
    return shutil.which("sbatch") is not None


def _wait_for_slurm_job(job_id: str, marker_path: Path, timeout_seconds: int = 1800) -> None:
    """Wait until the smoke marker appears or the SLURM job leaves the queue."""
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if marker_path.exists():
            return

        queue = subprocess.run(
            ["squeue", "-j", job_id, "-h"],
            capture_output=True,
            text=True,
            check=False,
        )
        if queue.returncode != 0 or not queue.stdout.strip():
            if marker_path.exists():
                return
            err_logs = sorted(result_dir().glob(f"smoke_{job_id}.err"))
            if err_logs and "CalledProcessError" in err_logs[0].read_text(encoding="utf-8"):
                raise RuntimeError(f"SLURM job {job_id} failed; see {err_logs[0]}")
            time.sleep(5)
            if marker_path.exists():
                return
            raise RuntimeError(
                f"SLURM job {job_id} finished without writing {marker_path}. "
                f"Check result/smoke_{job_id}.out"
            )
        time.sleep(10)

    raise TimeoutError(f"SLURM job {job_id} did not finish within {timeout_seconds}s")


@pytest.fixture(scope="module")
def minimal_lehd_data() -> Path:
    return ensure_minimal_lehd_data(force=True)


def test_lehd_fixture_writes_training_file(minimal_lehd_data: Path):
    assert minimal_lehd_data.exists()
    lines = minimal_lehd_data.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 8
    assert " output " in lines[0]


@pytest.mark.skipif(
    os.environ.get("GELD_RUN_SMOKE") != "1",
    reason="Set GELD_RUN_SMOKE=1 to run the full local GPU smoke pipeline",
)
def test_local_smoke_pipeline(minimal_lehd_data: Path):
    payload = run_pipeline(use_cuda=True)
    assert payload["status"] == "ok"
    assert Path(payload["stage1_run"]).joinpath("checkpoint-2.pt").exists()
    assert Path(payload["stage2_run"]).joinpath("checkpoint-2.pt").exists()
    assert payload["eval"]["num_instances"] == 2


@pytest.mark.skipif(not _slurm_available(), reason="sbatch not available")
def test_slurm_smoke_pipeline(minimal_lehd_data: Path):
    """Submit the smoke pipeline via SLURM and verify all stages complete."""
    marker_path = result_dir() / SMOKE_MARKER
    if marker_path.exists():
        marker_path.unlink()

    submit = subprocess.run(
        ["sbatch", "scripts/smoke_pipeline.slurm"],
        cwd=project_root(),
        capture_output=True,
        text=True,
        check=True,
    )
    job_id = submit.stdout.strip().split()[-1]

    _wait_for_slurm_job(job_id, marker_path)

    assert marker_path.exists(), f"Missing completion marker: {marker_path}"
    payload = json.loads(marker_path.read_text(encoding="utf-8"))
    assert payload["status"] == "ok"
    assert Path(payload["stage1_run"]).joinpath("checkpoint-2.pt").exists()
    assert Path(payload["stage2_run"]).joinpath("checkpoint-2.pt").exists()
    assert payload["eval"]["num_instances"] == 2
    assert payload["eval"]["size"] == 100
    assert payload["eval"]["distribution"] == "uniform"


def test_smoke_eval_from_random_checkpoint(minimal_lehd_data: Path, tmp_path: Path):
    """Fast CPU check that synthetic eval wiring works without beam/PRC."""
    import torch

    from geld.model.geld_model import GeldModel

    ckpt_dir = tmp_path / "checkpoint"
    ckpt_dir.mkdir()
    model = GeldModel(**{"mode": "test", "embedding_dim": 128, "decoder_layer_num": 6, "qkv_dim": 16, "head_num": 8, "ff_hidden_dim": 128})
    torch.save({"model_state_dict": model.state_dict()}, ckpt_dir / "checkpoint-1.pt")

    summary = run_smoke_eval(ckpt_dir, checkpoint_epoch=1, use_cuda=False)
    assert summary["num_instances"] == 2
    assert summary["size"] == 100
