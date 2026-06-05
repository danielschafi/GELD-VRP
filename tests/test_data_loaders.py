from pathlib import Path

import pytest
import torch

from geld.data.loaders import read_tsplib_file, read_tsp_instances_from_file
from geld.paths import benchmark_data_dir


def test_read_tsplib_file():
    path = benchmark_data_dir() / "tsplib" / "eil51.tsp"
    nodes, name = read_tsplib_file(path)
    assert name == "eil51"
    assert len(nodes) == 51
    assert all(len(node) == 2 for node in nodes)


def test_read_synthetic_instances():
    path = benchmark_data_dir() / "data_farm" / "tsp100" / "tsp100_uniform.txt"
    if not path.exists():
        pytest.skip(f"Synthetic benchmark file not found: {path}")
    instances = read_tsp_instances_from_file(path)
    assert instances.ndim == 3
    assert instances.shape[1] == 100
