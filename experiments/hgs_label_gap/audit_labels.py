#!/usr/bin/env python3
"""
Audit stage-1 HGS label quality against provably optimal VRPTW solutions.

Generate a small RCT sample (same procedure as training), label with HGS at 2000
iterations, and compare to OR-Tools CP-SAT exact solutions. The headline
gap statistic uses only instances solved to proven optimality within the time limit.

Interpretation (proven-optimal instances only):
  < 0.5%  — HGS-2000 labels are excellent teachers
  0.5–2%  — acceptable for supervised learning; document the gap
  > 2%    — labels may cap stage-1 quality; consider higher HGS iterations
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parents[1]
for path in (PROJECT_ROOT, EXPERIMENT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from geld_cvrptw.data.generator import ProblemRecord, generate_instances, save_problem_records
from geld_cvrptw.data.hgs_solver import (
    SolutionRecord,
    calc_vrp_cost,
    get_hgs_vrptw_executable,
    load_solution_records,
    save_solution_records,
    solve_instance,
)

from vrpsolver_adapter import ExactSolveResult, solve_exact

DEFAULT_DATA_DIR = EXPERIMENT_DIR / "data"
DEFAULT_RESULTS_DIR = EXPERIMENT_DIR / "results"
PROBLEMS_FILE = "problems.pkl"
LABELS_FILE = "labels_hgs_2000.pkl"
MANIFEST_FILE = "manifest.json"
GAP_VERIFY_TOLERANCE_PERCENT = 0.01


@dataclass(frozen=True)
class InstanceAuditResult:
    instance_id: int
    seed: int
    alpha: float
    hgs_cost: float
    optimal_cost: float | None
    gap_percent: float | None
    optimality_proven: bool
    solve_time_seconds: float
    solver_status: str
    solver_status_message: str
    lower_bound: float | None
    reference_verified: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit HGS label quality vs exact VRPTW solutions.")
    parser.add_argument("--num-instances", type=int, default=10)
    parser.add_argument("--problem-size", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--label-iters", type=int, default=2000)
    parser.add_argument("--time-limit", type=float, default=1800.0, help="Per-instance exact solver cap (seconds).")
    parser.add_argument("--global-budget", type=float, default=14400.0, help="Total wall-clock cap (seconds).")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--regenerate", action="store_true", help="Force regeneration of problems and HGS labels.")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Quick validation: 2 instances at n=20 with a 300s exact solver cap.",
    )
    parser.add_argument("--skip-exact", action="store_true", help="Only generate instances and HGS labels.")
    return parser.parse_args()


def configure_smoke_defaults(args: argparse.Namespace) -> argparse.Namespace:
    if not args.smoke:
        return args
    args.num_instances = 2
    args.problem_size = 20
    args.time_limit = 300.0
    args.global_budget = 1200.0
    return args


def manifest_path(data_dir: Path) -> Path:
    return data_dir / MANIFEST_FILE


def problems_path(data_dir: Path) -> Path:
    return data_dir / PROBLEMS_FILE


def labels_path(data_dir: Path) -> Path:
    return data_dir / LABELS_FILE


def load_manifest(data_dir: Path) -> dict | None:
    path = manifest_path(data_dir)
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_manifest(data_dir: Path, payload: dict) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    with manifest_path(data_dir).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def manifest_matches(args: argparse.Namespace, manifest: dict | None) -> bool:
    if manifest is None:
        return False
    return (
        manifest.get("num_instances") == args.num_instances
        and manifest.get("problem_size") == args.problem_size
        and manifest.get("alpha") == args.alpha
        and manifest.get("seed") == args.seed
        and manifest.get("label_iters") == args.label_iters
    )


def ensure_dataset(args: argparse.Namespace) -> tuple[list[ProblemRecord], list[SolutionRecord], dict]:
    data_dir = args.data_dir
    manifest = load_manifest(data_dir)
    problem_file = problems_path(data_dir)
    label_file = labels_path(data_dir)

    if (
        not args.regenerate
        and manifest_matches(args, manifest)
        and problem_file.is_file()
        and label_file.is_file()
    ):
        with problem_file.open("rb") as handle:
            problems = pickle.load(handle)
        labels = load_solution_records(label_file)
        if len(problems) == len(labels) == args.num_instances:
            print(f"Loaded existing dataset from {data_dir}")
            return problems, labels, manifest

    print(
        f"Generating {args.num_instances} RCT instances "
        f"(n={args.problem_size}, alpha={args.alpha}, seed={args.seed})"
    )
    problems = generate_instances(
        args.num_instances,
        problem_size=args.problem_size,
        alpha=args.alpha,
        seed=args.seed,
    )

    executable = get_hgs_vrptw_executable()
    cache_dir = data_dir / ".hgs_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    labels: list[SolutionRecord] = []
    for index, record in enumerate(problems):
        print(f"  HGS labeling instance {index + 1}/{len(problems)} at {args.label_iters} iterations")
        cost, tour = solve_instance(
            executable,
            cache_dir,
            f"audit_{index:04d}",
            record,
            max_iteration=args.label_iters,
            seed=args.seed + index,
        )
        labels.append((cost, tour))

    save_problem_records(problem_file, problems)
    save_solution_records(label_file, labels)
    manifest = {
        "num_instances": args.num_instances,
        "problem_size": args.problem_size,
        "alpha": args.alpha,
        "seed": args.seed,
        "label_iters": args.label_iters,
        "reference_algorithm": "OR-Tools CP-SAT (exact when OPTIMAL)",
    }
    save_manifest(data_dir, manifest)
    print(f"Wrote dataset to {data_dir}")
    return problems, labels, manifest


def compute_gap_percent(hgs_cost: float, optimal_cost: float) -> float:
    if optimal_cost <= 0.0:
        raise ValueError(f"Invalid optimal cost for gap computation: {optimal_cost}")
    return (hgs_cost - optimal_cost) / optimal_cost * 100.0


def audit_instances(
    problems: list[ProblemRecord],
    labels: list[SolutionRecord],
    *,
    seed: int,
    alpha: float,
    time_limit: float,
    global_budget: float,
) -> list[InstanceAuditResult]:
    results: list[InstanceAuditResult] = []
    budget_started = time.perf_counter()

    for index, (record, (hgs_cost, hgs_tour)) in enumerate(zip(problems, labels)):
        elapsed_budget = time.perf_counter() - budget_started
        if elapsed_budget >= global_budget:
            print(f"Global budget exhausted before instance {index}; skipping remaining instances.")
            break

        remaining_budget = max(global_budget - elapsed_budget, 1.0)
        instance_time_limit = min(time_limit, remaining_budget)
        print(
            f"Exact solve instance {index + 1}/{len(problems)} "
            f"(HGS cost={hgs_cost:.6f}, limit={instance_time_limit:.0f}s)"
        )
        exact_result = solve_exact(
            record,
            time_limit=instance_time_limit,
            upper_bound=hgs_cost,
            hgs_tour=hgs_tour,
        )
        gap_percent = None
        reference_verified = False
        if exact_result.optimal_cost is not None:
            gap_percent = compute_gap_percent(hgs_cost, exact_result.optimal_cost)
            if exact_result.optimality_proven:
                reference_verified = True
            elif gap_percent is not None and abs(gap_percent) <= GAP_VERIFY_TOLERANCE_PERCENT:
                reference_verified = True

        results.append(
            InstanceAuditResult(
                instance_id=index,
                seed=seed,
                alpha=alpha,
                hgs_cost=float(hgs_cost),
                optimal_cost=exact_result.optimal_cost,
                gap_percent=gap_percent,
                optimality_proven=exact_result.optimality_proven,
                solve_time_seconds=exact_result.solve_time_seconds,
                solver_status=exact_result.solver_status,
                solver_status_message=exact_result.solver_status_message,
                lower_bound=exact_result.lower_bound,
                reference_verified=reference_verified,
            )
        )
        status = "proven optimal" if exact_result.optimality_proven else (
            "verified match" if reference_verified else "no optimality certificate"
        )
        optimal_text = (
            f"{exact_result.optimal_cost:.6f}"
            if exact_result.optimal_cost is not None
            else "n/a"
        )
        gap_text = f", gap={gap_percent:.4f}%" if gap_percent is not None else ""
        print(
            f"  -> {status}, optimal={optimal_text}, "
            f"time={exact_result.solve_time_seconds:.1f}s{gap_text}"
        )

    return results


def summarize_results(
    results: list[InstanceAuditResult],
    *,
    manifest: dict,
    label_iters: int,
    time_limit: float,
    total_wall_time_seconds: float,
) -> dict:
    proven = [row for row in results if row.optimality_proven and row.gap_percent is not None]
    verified = [row for row in results if row.reference_verified and row.gap_percent is not None]
    gaps = np.array([row.gap_percent for row in proven], dtype=np.float64)
    verified_gaps = np.array([row.gap_percent for row in verified], dtype=np.float64)

    summary = {
        "reference_algorithm": manifest.get("reference_algorithm", "OR-Tools CP-SAT (exact when OPTIMAL)"),
        "num_instances_total": len(results),
        "num_proven_optimal": len(proven),
        "num_reference_verified": len(verified),
        "num_timed_out": len(results) - len(verified),
        "label_iterations": label_iters,
        "reference_time_limit_seconds": time_limit,
        "seed": manifest.get("seed"),
        "alpha": manifest.get("alpha"),
        "problem_size": manifest.get("problem_size"),
        "total_wall_time_seconds": total_wall_time_seconds,
        "mean_gap_percent": float(gaps.mean()) if len(gaps) else None,
        "median_gap_percent": float(np.median(gaps)) if len(gaps) else None,
        "max_gap_percent": float(gaps.max()) if len(gaps) else None,
        "std_gap_percent": float(gaps.std(ddof=0)) if len(gaps) else None,
        "verified_mean_gap_percent": float(verified_gaps.mean()) if len(verified_gaps) else None,
        "verified_median_gap_percent": float(np.median(verified_gaps)) if len(verified_gaps) else None,
        "gap_verify_tolerance_percent": GAP_VERIFY_TOLERANCE_PERCENT,
    }

    if len(proven):
        summary["claim"] = (
            f"On n={len(proven)} synthetic CVRPTW-{manifest.get('problem_size')} instances "
            f"(RCT generator, alpha={manifest.get('alpha')}, seed={manifest.get('seed')}), "
            f"all solved to proven optimality by OR-Tools CP-SAT within "
            f"{int(time_limit)}s per instance, stage-1 HGS labels at {label_iters} iterations are on "
            f"average {summary['mean_gap_percent']:.4f}% worse than optimal "
            f"(median {summary['median_gap_percent']:.4f}%, max {summary['max_gap_percent']:.4f}%). "
            f"{summary['num_timed_out']} additional instance(s) lacked an optimality certificate."
        )
    elif len(verified):
        summary["claim"] = (
            f"On n={len(verified)} synthetic CVRPTW-{manifest.get('problem_size')} instances "
            f"(RCT generator, alpha={manifest.get('alpha')}, seed={manifest.get('seed')}), "
            f"OR-Tools CP-SAT found a reference solution within {int(time_limit)}s per instance matching "
            f"HGS-2000 labels within {GAP_VERIFY_TOLERANCE_PERCENT}% (optimality not proven at this size). "
            f"Mean gap vs reference: {summary['verified_mean_gap_percent']:.4f}%. "
            f"{len(results) - len(verified)} instance(s) had no reference match."
        )
    else:
        summary["claim"] = (
            f"No instances were solved to proven optimality within the configured limits "
            f"(time_limit={int(time_limit)}s, global_budget used={total_wall_time_seconds:.0f}s). "
            "Increase --time-limit or reduce --problem-size before drawing conclusions."
        )

    return summary


def write_results(results_dir: Path, results: list[InstanceAuditResult], summary: dict) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)

    instances_csv = results_dir / "audit_instances.csv"
    with instances_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "instance_id",
                "seed",
                "alpha",
                "hgs_cost",
                "optimal_cost",
                "gap_percent",
                "optimality_proven",
                "solve_time_seconds",
                "solver_status",
                "solver_status_message",
                "lower_bound",
                "reference_verified",
            ],
        )
        writer.writeheader()
        for row in results:
            writer.writerow(asdict(row))

    summary_json = results_dir / "audit_summary.json"
    payload = {
        **summary,
        "instances": [asdict(row) for row in results],
    }
    with summary_json.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(f"Wrote {instances_csv}")
    print(f"Wrote {summary_json}")
    print()
    print(summary["claim"])


def verify_hgs_costs(problems: list[ProblemRecord], labels: list[SolutionRecord]) -> None:
    for index, (record, (stored_cost, tour)) in enumerate(zip(problems, labels)):
        arrays = record
        depot_xy, node_xy, *_rest = arrays
        recomputed = calc_vrp_cost(np.asarray(depot_xy), np.asarray(node_xy), tour)
        if abs(recomputed - stored_cost) > 1e-5:
            raise ValueError(
                f"HGS stored cost mismatch on instance {index}: stored={stored_cost}, recomputed={recomputed}"
            )


def main() -> None:
    args = configure_smoke_defaults(parse_args())
    started = time.perf_counter()

    problems, labels, manifest = ensure_dataset(args)
    verify_hgs_costs(problems, labels)

    if args.skip_exact:
        print("Skipping exact solves (--skip-exact).")
        return

    results = audit_instances(
        problems,
        labels,
        seed=args.seed,
        alpha=args.alpha,
        time_limit=args.time_limit,
        global_budget=args.global_budget,
    )
    total_wall_time = time.perf_counter() - started
    summary = summarize_results(
        results,
        manifest=manifest,
        label_iters=args.label_iters,
        time_limit=args.time_limit,
        total_wall_time_seconds=total_wall_time,
    )
    write_results(args.results_dir, results, summary)


if __name__ == "__main__":
    main()
