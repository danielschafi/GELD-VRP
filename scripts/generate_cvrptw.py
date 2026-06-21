#!/usr/bin/env python3
"""Generate RCT-format CVRPTW-100 instances and HGS reference labels."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from geld.paths import project_root, training_stage_1_data_dir
from geld_cvrptw.data.generator import (
    DEFAULT_SERVICE_TIME,
    DEFAULT_SPEED,
    allocate_problem_output_paths,
    generate_instances,
    label_path,
    labels_complete,
    load_problem_records,
    problem_path,
    save_problem_records,
)
from geld_cvrptw.data.hgs_solver import (
    get_hgs_vrptw_executable,
    save_solution_records,
    solve_instance_task,
    validate_solution,
)

DEFAULT_SCALES = (0.2, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0)


def parse_scales(raw: str) -> list[float]:
    return [float(value.strip()) for value in raw.split(",") if value.strip()]


def progress_path(label_file: Path) -> Path:
    return label_file.with_suffix(".progress.json")


def load_progress(path: Path) -> dict[int, list[float | list[int]]]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    solved = payload.get("solved", {})
    return {int(key): value for key, value in solved.items()}


def save_progress(path: Path, solved: dict[int, tuple[float, list[int]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {str(index): [cost, tour] for index, (cost, tour) in solved.items()}
    with path.open("w", encoding="utf-8") as handle:
        json.dump({"solved": serializable}, handle)


def assemble_solution_records(
    num_samples: int,
    solved: dict[int, tuple[float, list[int]]],
) -> list[tuple[float, list[int]]]:
    missing = [index for index in range(num_samples) if index not in solved]
    if missing:
        raise RuntimeError(f"Missing {len(missing)} solution(s), e.g. indices {missing[:5]}")
    return [solved[index] for index in range(num_samples)]


def solve_records_parallel(
    records: list[tuple],
    workers: int,
    max_iteration: int,
    seed: int,
    resume: bool,
    label_file: Path,
    cache_dir: Path,
) -> list[tuple[float, list[int]]]:
    executable = get_hgs_vrptw_executable()
    progress_file = progress_path(label_file)
    solved: dict[int, tuple[float, list[int]]] = {}
    if resume:
        raw_progress = load_progress(progress_file)
        solved = {index: (float(value[0]), list(value[1])) for index, value in raw_progress.items()}

    pending = [index for index in range(len(records)) if index not in solved]
    if not pending:
        return assemble_solution_records(len(records), solved)

    cache_dir.mkdir(parents=True, exist_ok=True)
    tasks = [
        (
            str(executable),
            str(cache_dir / f"worker_{index % workers}"),
            index,
            records[index],
            max_iteration,
            seed + index,
            10000,
        )
        for index in pending
    ]

    failures: list[int] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(solve_instance_task, task): task[2] for task in tasks}
        for future in tqdm(as_completed(futures), total=len(futures), desc="HGS solve"):
            index = futures[future]
            try:
                solved_index, solution = future.result()
                if not validate_solution(records[solved_index], solution[1]):
                    raise ValueError("solution failed validation")
                solved[solved_index] = solution
                save_progress(progress_file, solved)
            except Exception:
                failures.append(index)

    if failures:
        retry_tasks = [
            (
                str(executable),
                str(cache_dir / f"retry_{index}"),
                index,
                records[index],
                max_iteration,
                seed + index + 1_000_000,
                10000,
            )
            for index in failures
        ]
        for task in retry_tasks:
            index = task[2]
            try:
                solved_index, solution = solve_instance_task(task)
                if validate_solution(records[solved_index], solution[1]):
                    solved[solved_index] = solution
                    save_progress(progress_file, solved)
            except Exception as exc:
                raise RuntimeError(f"Failed to solve instance {index}") from exc

    return assemble_solution_records(len(records), solved)


def process_scale(
    scale: float,
    num_samples: int,
    output_dir: Path,
    workers: int,
    hgs_iterations: int,
    seed: int,
    batch_size: int,
    skip_solve: bool,
    solve_only: bool,
    resume: bool,
    overwrite: bool,
) -> None:
    if solve_only:
        problem_file = problem_path(
            output_dir,
            num_samples=num_samples,
            scale=scale,
            service_time=DEFAULT_SERVICE_TIME,
            speed=DEFAULT_SPEED,
        )
        label_file = label_path(problem_file)
        if not problem_file.is_file():
            raise FileNotFoundError(f"--solve-only requires existing problem file: {problem_file}")
        run_index = 0
    else:
        problem_file, label_file, run_index = allocate_problem_output_paths(
            output_dir,
            num_samples=num_samples,
            scale=scale,
            service_time=DEFAULT_SERVICE_TIME,
            speed=DEFAULT_SPEED,
            overwrite=overwrite,
        )

    effective_seed = seed + run_index

    if not solve_only:
        if problem_file.is_file() and not overwrite:
            print(f"Reusing existing problem file: {problem_file}")
        else:
            print(
                f"Generating {num_samples} instances for scale={scale} "
                f"(run_index={run_index}, seed={effective_seed})"
            )
            records = generate_instances(
                num_samples,
                alpha=scale,
                seed=effective_seed,
                batch_size=batch_size,
            )
            save_problem_records(problem_file, records)
            print(f"Saved problems to {problem_file}")

    if skip_solve:
        return

    if labels_complete(problem_file, label_file, num_samples):
        print(f"Labels already complete, skipping HGS: {label_file}")
        return

    records = load_problem_records(problem_file)
    if len(records) != num_samples:
        raise ValueError(f"Problem file {problem_file} has {len(records)} records, expected {num_samples}")

    cache_dir = output_dir / ".hgs_cache" / problem_file.stem
    solutions = solve_records_parallel(
        records,
        workers=workers,
        max_iteration=hgs_iterations,
        seed=effective_seed,
        resume=resume and not overwrite,
        label_file=label_file,
        cache_dir=cache_dir,
    )
    save_solution_records(label_file, solutions)
    progress_file = progress_path(label_file)
    if progress_file.is_file():
        progress_file.unlink()
    if cache_dir.is_dir():
        shutil.rmtree(cache_dir, ignore_errors=True)
    print(f"Saved labels to {label_file}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-samples", type=int, default=10_000)
    parser.add_argument("--scales", type=str, default=",".join(str(scale) for scale in DEFAULT_SCALES))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--hgs-iterations", type=int, default=2000) # matching the rest of the data https://github.com/CIAM-Group/Rethinking_Constraint_Tightness/blob/main/CVRPTW/Generate_data/generate_data.sh
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--gen-batch-size", type=int, default=512)
    parser.add_argument("--skip-solve", action="store_true")
    parser.add_argument("--solve-only", action="store_true")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--overwrite", action="store_true")
    parser.set_defaults(resume=True)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.skip_solve and args.solve_only:
        raise ValueError("Use only one of --skip-solve or --solve-only")

    output_dir = args.output_dir or training_stage_1_data_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    scales = parse_scales(args.scales)

    print(f"Output directory: {output_dir}")
    print(f"Project root: {project_root()}")
    print(f"Scales: {scales}")
    print(f"Workers: {args.workers}")
    print(f"HGS iterations: {args.hgs_iterations}")

    for scale in scales:
        process_scale(
            scale,
            num_samples=args.num_samples,
            output_dir=output_dir,
            workers=args.workers,
            hgs_iterations=args.hgs_iterations,
            seed=args.seed,
            batch_size=args.gen_batch_size,
            skip_solve=args.skip_solve,
            solve_only=args.solve_only,
            resume=args.resume,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
