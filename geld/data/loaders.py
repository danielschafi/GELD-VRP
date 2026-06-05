"""Dataset I/O for synthetic benchmarks, LEHD training data, and TSPLIB."""

from pathlib import Path

import torch


def read_solutions_from_file(file_path):
    """Parse baseline tour file (tour, length, time per line)."""
    tour_storage = []
    tour_len_storage = []
    elapsed_time_storage = []
    with open(file_path, "r", encoding="utf8") as read_file:
        line_text = read_file.readline()
        while line_text:
            tour_text, tour_len_text, elapsed_time_text = line_text.strip().split(" ")
            tour_storage.append([int(val) for val in tour_text.split(",")])
            tour_len_storage.append(float(tour_len_text))
            elapsed_time_storage.append(float(elapsed_time_text))
            line_text = read_file.readline()

    tours = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(x) for x in tour_storage], batch_first=True, padding_value=0
    )
    return tours, torch.tensor(tour_len_storage), torch.tensor(elapsed_time_storage)


def read_tsp_instances_from_file(file_path):
    """Parse synthetic benchmark file into coordinate tensors."""
    tsp_instances = []
    with open(file_path, "r", encoding="utf8") as read_file:
        line_text = read_file.readline()
        while line_text:
            tsp_instance = []
            for node_text in line_text.strip().split(" "):
                tsp_instance.append([float(val) for val in node_text.split(",")])
            tsp_instances.append(tsp_instance)
            line_text = read_file.readline()
    return torch.tensor(tsp_instances)


def read_tour_file(file_path):
    """Parse TSPLIB TOUR_SECTION into a 0-indexed tour tensor."""
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
    tour = []
    reading_tour = False
    for line in lines:
        if line.startswith("TOUR_SECTION"):
            reading_tour = True
            continue
        if line.startswith("EOF"):
            break
        if reading_tour:
            tour.append(int(line.strip()))
    tour_tensor = torch.tensor(tour, dtype=torch.int64) - 1
    return tour_tensor[:-1]


def load_tsp_instances_with_baselines(root, size, distribution):
    """Load synthetic TSP-n instances with LKH3 baseline tours."""
    baseline = "LKH3_runs10" if size <= 1000 else "LKH3_runs1"
    instance_root = Path(root)
    instance_file = instance_root / f"data_farm/tsp{size}/tsp{size}_{distribution}.txt"
    tsp_instances = read_tsp_instances_from_file(instance_file)
    solution_file = instance_root / f"solution_farm/tsp{size}_{distribution}/{baseline}.txt"
    baseline_tours, baseline_lens, _ = read_solutions_from_file(solution_file)
    return tsp_instances, baseline_tours, baseline_lens


def read_tsplib_file(file_path):
    """Parse a .tsp file into node coordinates and instance name."""
    properties = {}
    reading_properties = True
    nodes = []
    with open(file_path, "r", encoding="utf8") as read_file:
        line = read_file.readline()
        while line.strip():
            if reading_properties:
                if ":" in line:
                    key, val = [part.strip() for part in line.split(":")]
                    properties[key] = val
                else:
                    reading_properties = False
            else:
                if not line.startswith("NODE_COORD_SECTION") and not line.startswith("EOF"):
                    parts = [part.strip() for part in line.split(" ") if part.strip()]
                    _, x, y = parts
                    nodes.append([float(x), float(y)])
            line = read_file.readline()
    return nodes, properties["NAME"]


def load_tsplib_file(root, tsplib_name, use_tsplib_dir: bool = False):
    """Load one TSPLIB or National TSP instance as a coordinate tensor."""
    tsplib_dir = "tsplib" if use_tsplib_dir else "National_TSP"
    file_path = Path(root).joinpath(tsplib_dir).joinpath(f"{tsplib_name}.tsp")
    instance, name = read_tsplib_file(file_path)
    return torch.tensor(instance), name


def load_lehd_line(line: str):
    """Parse one LEHD text line into node coordinates and tour indices."""
    parts = line.split(" ")
    num_nodes = int(parts.index("output") // 2)
    nodes = [[float(parts[idx]), float(parts[idx + 1])] for idx in range(0, 2 * num_nodes, 2)]
    tour_nodes = [int(node) - 1 for node in parts[parts.index("output") + 1 : -1]]
    return nodes, tour_nodes
