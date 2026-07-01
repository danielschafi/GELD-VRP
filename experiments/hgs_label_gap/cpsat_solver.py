"""Exact CVRPTW solver via OR-Tools CP-SAT (proven optimality when status is OPTIMAL)."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

from ortools.sat.python import cp_model

from geld_cvrptw.data.generator import DEPOT_TW_END, DEPOT_TW_START
from geld_cvrptw.data.hgs_solver import record_to_arrays

SCALE = 1_000_000


@dataclass(frozen=True)
class ExactSolveResult:
    optimal_cost: float | None
    optimality_proven: bool
    solve_time_seconds: float
    solver_status: str
    solver_status_message: str
    lower_bound: float | None


def _scale(value: float) -> int:
    return int(round(float(value) * SCALE))


def _unscale(value: int) -> float:
    return float(value) / SCALE


def _build_distance_matrix(depot: Any, locations: Any) -> list[list[int]]:
    points = [(float(depot[0]), float(depot[1]))] + [
        (float(point[0]), float(point[1])) for point in locations
    ]
    num_nodes = len(points)
    distances = [[0] * num_nodes for _ in range(num_nodes)]
    for left in range(num_nodes):
        x_i, y_i = points[left]
        for right in range(num_nodes):
            if left == right:
                continue
            x_j, y_j = points[right]
            distances[left][right] = _scale(math.hypot(x_i - x_j, y_i - y_j))
    return distances


def _hgs_tour_to_arcs(tour: list[int]) -> set[tuple[int, int]]:
    """Convert HGS zero-separated multi-route tour to directed arcs (0-indexed nodes)."""
    arcs: set[tuple[int, int]] = set()
    current = 0
    for node in tour:
        if node == 0:
            if current != 0:
                arcs.add((current, 0))
            current = 0
            continue
        arcs.add((current, node))
        current = node
    if current != 0:
        arcs.add((current, 0))
    return arcs


def _apply_arc_hints(
    model: cp_model.CpModel,
    arc_vars: dict[tuple[int, int], cp_model.IntVar],
    hinted_arcs: set[tuple[int, int]],
) -> None:
    for (left, right), arc in arc_vars.items():
        model.add_hint(arc, 1 if (left, right) in hinted_arcs else 0)


def solve_exact(
    record: tuple[Any, ...],
    *,
    time_limit: float,
    upper_bound: float | None = None,
    max_vehicles: int | None = None,
    hgs_tour: list[int] | None = None,
) -> ExactSolveResult:
    """
    Solve RCT-format CVRPTW with CP-SAT and accept only OPTIMAL certificates.

    Uses AddMultipleCircuit for an unlimited fleet with depot as route separator.
    Time-window semantics match the RCT environment: service must start by tw_end.
    """
    del max_vehicles  # unlimited fleet is encoded directly by multiple-circuit.

    arrays = record_to_arrays(record)
    num_customers = len(arrays["loc"])
    num_nodes = num_customers + 1
    capacity = int(arrays["capacity"])

    service = [0] + [_scale(value) for value in arrays["service_time"]]
    demand = [0] + [int(value) for value in arrays["demand"]]
    tw_start = [_scale(DEPOT_TW_START)] + [_scale(value) for value in arrays["tw_start"]]
    tw_end = [_scale(DEPOT_TW_END)] + [_scale(value) for value in arrays["tw_end"]]
    depot_tw_end = _scale(DEPOT_TW_END)
    distances = _build_distance_matrix(arrays["depot"], arrays["loc"])

    model = cp_model.CpModel()
    arc_vars: dict[tuple[int, int], cp_model.IntVar] = {}
    circuit_arcs: list[tuple[int, int, cp_model.IntVar]] = []

    for left in range(num_nodes):
        for right in range(num_nodes):
            if left == right:
                continue
            arc = model.new_bool_var(f"arc_{left}_{right}")
            arc_vars[(left, right)] = arc
            circuit_arcs.append((left, right, arc))

    model.add_multiple_circuit(circuit_arcs)

    for customer in range(1, num_nodes):
        model.add(sum(arc_vars[(left, customer)] for left in range(num_nodes) if left != customer) == 1)
        model.add(sum(arc_vars[(customer, right)] for right in range(num_nodes) if right != customer) == 1)

    horizon = depot_tw_end + _scale(1.0)
    service_start = {
        node: model.new_int_var(0, horizon, f"service_start_{node}") for node in range(1, num_nodes)
    }
    load_after = {
        node: model.new_int_var(0, capacity, f"load_after_{node}") for node in range(1, num_nodes)
    }

    for customer in range(1, num_nodes):
        model.add(service_start[customer] >= tw_start[customer])
        model.add(service_start[customer] <= tw_end[customer])
        model.add(load_after[customer] <= capacity)

    for left in range(num_nodes):
        for right in range(num_nodes):
            if left == right:
                continue
            arc = arc_vars[(left, right)]
            travel = distances[left][right]

            if right == 0 and left > 0:
                model.add(service_start[left] + service[left] + travel <= depot_tw_end).only_enforce_if(arc)
                continue

            if right == 0:
                continue

            if left == 0:
                model.add(service_start[right] >= tw_start[right]).only_enforce_if(arc)
                model.add(service_start[right] >= travel).only_enforce_if(arc)
                model.add(load_after[right] == demand[right]).only_enforce_if(arc)
                continue

            arrival = model.new_int_var(0, horizon, f"arrival_{left}_{right}")
            model.add(arrival == service_start[left] + service[left] + travel).only_enforce_if(arc)
            model.add(service_start[right] >= arrival).only_enforce_if(arc)
            model.add(service_start[right] >= tw_start[right]).only_enforce_if(arc)
            model.add(load_after[right] == load_after[left] + demand[right]).only_enforce_if(arc)

    objective_terms = [
        distances[left][right] * arc_vars[(left, right)]
        for left in range(num_nodes)
        for right in range(num_nodes)
        if left != right
    ]
    model.minimize(sum(objective_terms))

    if upper_bound is not None:
        model.add(sum(objective_terms) <= math.ceil(float(upper_bound) * SCALE + 1.0))

    if hgs_tour is not None:
        _apply_arc_hints(model, arc_vars, _hgs_tour_to_arcs(hgs_tour))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit)
    solver.parameters.num_search_workers = 8

    started = time.perf_counter()
    status = solver.solve(model)
    elapsed = time.perf_counter() - started

    status_name = solver.status_name(status)
    proven = status == cp_model.OPTIMAL
    optimal_cost = None
    lower_bound = None
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        optimal_cost = _unscale(int(solver.objective_value))
    if proven:
        lower_bound = optimal_cost
    elif solver.best_objective_bound not in (None, float("inf"), float("-inf")):
        lower_bound = _unscale(int(solver.best_objective_bound))

    return ExactSolveResult(
        optimal_cost=optimal_cost,
        optimality_proven=proven,
        solve_time_seconds=elapsed,
        solver_status=status_name,
        solver_status_message=status_name,
        lower_bound=lower_bound,
    )
