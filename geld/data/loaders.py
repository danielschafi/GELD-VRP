"""Dataset I/O for synthetic benchmarks, LEHD training data, and TSPLIB."""

from pathlib import Path

import torch
from geld.paths import training_stage_1_data_dir
import pandas as pd
import numpy as np
import pickle


# MVMoE / RCT synthetic CVRPTW defaults (not stored in pickle files).
DEPOT_TW_START = 0.0
DEPOT_TW_END = 3.0
TOUR_PAD_VALUE = -1

class CvrptwDataset(TypedDict):
    coords: torch.Tensor
    demand: torch.Tensor
    tw_start: torch.Tensor
    tw_end: torch.Tensor
    service_time: torch.Tensor
    label_tours: torch.Tensor
    costs: torch.Tensor


def problem_files() -> list[Path]:
    """Get all problem data files. (Not HGS solutions)"""
    return sorted(
        path for path in training_stage_1_data_dir.glob("*.pkl") if not path.stem.startswith("hgs_")
    )


def load_cvrptw_data_with_labels() -> tuple[torch.Tensor, torch.Tensor]:
    """Load all CVRPTW data with labels into memory."""

    data_dir = training_stage_1_data_dir()

    coords_list: list[torch.Tensor] = []
    demand_list: list[torch.Tensor] = []
    tw_start_list: list[torch.Tensor] = []
    tw_end_list: list[torch.Tensor] = []
    service_time_list: list[torch.Tensor] = []
    tour_list: list[torch.Tensor] = []
    costs: list[float] = []

    # Go over problem data files
    for problem_file in problem_files():
        solution_file = "hgs_" + problem_file.stem + ".pkl"

        if not problem_file.exists():
            raise FileNotFoundError(f"Problem file {problem_file} not found")
        if not solution_file.exists():
            raise FileNotFoundError(f"Solution file {solution_file} not found")


        with open(problem_file, "rb") as f:
            problem_records = pickle.load(f)
        with open(solution_file, "rb") as f:
            solution_records = pickle.load(f)

        if len(problem_records) != len(solution_records):
            raise ValueError(f"Problem and solution data have different lengths for file {problem_file}")

        # Extract records
        for record, (cost, raw_tour) in zip(problem_records, solution_records):
            coords, demand, tw_start, tw_end, service_time = prepare_cvrptw_instance(record)
            
            action_tour = format_hgs_tour(raw_tour)

            coords_list.append(torch.tensor(coords, dtype=torch.float64))
            demand_list.append(torch.tensor(demand, dtype=torch.float64))
            tw_start_list.append(torch.tensor(tw_start, dtype=torch.float64))
            tw_end_list.append(torch.tensor(tw_end, dtype=torch.float64))
            service_time_list.append(torch.tensor(service_time, dtype=torch.float64))
            tour_list.append(torch.tensor(action_tour, dtype=torch.long))
            costs.append(float(cost))


    return CvrptwDataset(
        coords=torch.stack(coords_list),
        demand=torch.stack(demand_list),
        tw_start=torch.stack(tw_start_list),
        tw_end=torch.stack(tw_end_list),
        service_time=torch.stack(service_time_list),
        label_tours=torch.nn.utils.rnn.pad_sequence(
            tour_list, batch_first=True, padding_value=TOUR_PAD_VALUE
        ),
        costs=torch.tensor(costs, dtype=torch.float64),
    )


def prepare_cvrptw_instance(record: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare one CVRPTW instance from a record. Setting datatypes, and preprending depot at index 0"""
    depot_xy, node_xy, demand, capacity, service_time, tw_start, tw_end = record
    capacity = float(capacity)

    depot_xy = np.asarray(depot_xy, dtype=np.float64).reshape(1, 2) # need to reshape to (1, 2) for vstack, insert depot at index 0
    node_xy = np.asarray(node_xy, dtype=np.float64)
    demand_norm = np.asarray(demand, dtype=np.float64) / capacity
    service = np.asarray(service_time, dtype=np.float64)
    tw_start_arr = np.asarray(tw_start, dtype=np.float64)
    tw_end_arr = np.asarray(tw_end, dtype=np.float64)
    
    coords = np.vstack((depot_xy, node_xy))
    demand_full = np.concatenate(([0.0], demand_norm))
    service_full = np.concatenate(([0.0], service))
    tw_start_full = np.concatenate(([DEPOT_TW_START], tw_start_arr))
    tw_end_full = np.concatenate(([DEPOT_TW_END], tw_end_arr))
    return coords, demand_full, tw_start_full, tw_end_full, service_full

def format_hgs_tour(raw_tour: list[int]) -> np.ndarray:
    """Convert HGS zero-separated tour to depot-inclusive tour."""
    tour = np.asarray(raw_tour, dtype=np.int64)
    return np.concatenate(([0], tour))



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
