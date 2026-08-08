"""Command-line interface for the proposed HCORAP research package."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from .cpsat import (
    solve_cpsat_epsilon_constraint,
    solve_cpsat_lexicographic,
    solve_cpsat_weighted,
)
from .crosscheck import crosscheck_cpp_instance
from .experiment import run_experiment_config
from .generator import (
    LOAD_PROFILES,
    generate_benchmark_batch,
    generate_nested_family,
    write_generated_instance,
)
from .io import read_instance
from .metrics import verify_assignments
from .model import Assignment
from .solvers import (
    solve_epsilon_constraint,
    solve_lexicographic,
    solve_weighted,
)


def _write_json(payload: Any, output: Path | None) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if output is None:
        sys.stdout.write(rendered)
    else:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")


def _coverage_flag(arguments: argparse.Namespace) -> bool:
    return not arguments.soft_coverage


def _solve(arguments: argparse.Namespace) -> int:
    instance = read_instance(arguments.instance)
    common = {
        "require_full_coverage": _coverage_flag(arguments),
        "timeout_seconds": arguments.timeout,
    }
    if arguments.method == "weighted":
        result = solve_weighted(
            instance,
            continuity_weight=arguments.continuity_weight,
            overtime_weight=arguments.overtime_weight,
            sat_solver=arguments.sat_solver,
            maxsat_algorithm=arguments.maxsat_algorithm,
            **common,
        )
    elif arguments.method == "lexicographic":
        result = solve_lexicographic(
            instance,
            policy=arguments.policy,
            sat_solver=arguments.sat_solver,
            maxsat_algorithm=arguments.maxsat_algorithm,
            **common,
        )
    elif arguments.method == "epsilon-constraint":
        result = solve_epsilon_constraint(
            instance,
            delta=arguments.delta,
            sat_solver=arguments.sat_solver,
            maxsat_algorithm=arguments.maxsat_algorithm,
            **common,
        )
    elif arguments.method == "cpsat-weighted":
        result = solve_cpsat_weighted(
            instance,
            continuity_weight=arguments.continuity_weight,
            overtime_weight=arguments.overtime_weight,
            workers=arguments.workers,
            random_seed=arguments.random_seed,
            **common,
        )
    elif arguments.method == "cpsat-lexicographic":
        result = solve_cpsat_lexicographic(
            instance,
            policy=arguments.policy,
            workers=arguments.workers,
            random_seed=arguments.random_seed,
            **common,
        )
    else:
        result = solve_cpsat_epsilon_constraint(
            instance,
            delta=arguments.delta,
            workers=arguments.workers,
            random_seed=arguments.random_seed,
            **common,
        )
    _write_json(result.as_dict(), arguments.output)
    return 0 if result.status in {"OPTIMUM", "TIMEOUT_FEASIBLE"} else 2


def _verify(arguments: argparse.Namespace) -> int:
    instance = read_instance(arguments.instance)
    payload = json.loads(Path(arguments.solution).read_text(encoding="utf-8"))
    raw_assignments = payload.get("assignments", payload)
    assignments = tuple(
        Assignment(
            agent=int(item["agent"]),
            service=int(item["service"]),
            time_slot=int(item["time_slot"]),
        )
        for item in raw_assignments
    )
    verification = verify_assignments(
        instance,
        assignments,
        require_full_coverage=not arguments.soft_coverage,
    )
    _write_json(
        {
            "valid": verification.valid,
            "violations": list(verification.violations),
            "metrics": verification.metrics.as_dict(),
        },
        arguments.output,
    )
    return 0 if verification.valid else 3


def _generate(arguments: argparse.Namespace) -> int:
    family = generate_nested_family(
        users=arguments.users,
        agent_counts=arguments.agents,
        services_per_user_counts=arguments.visits,
        seed=arguments.seed,
        days=arguments.days,
        slots_per_day=arguments.slots_per_day,
        normal_hour_cap=arguments.normal_hour_cap,
        overtime_penalty=arguments.overtime_penalty,
    )
    output_dir = Path(arguments.output_dir)
    written = []
    for (agents, visits), instance in family.items():
        name = f"instance_{arguments.users}_{agents}_{visits}_seed{arguments.seed}.txt"
        text_path, metadata_path = write_generated_instance(instance, output_dir / name)
        written.append(
            {
                "agents": agents,
                "visits": visits,
                "instance": str(text_path.resolve()),
                "metadata": str(metadata_path.resolve()),
                "summary": instance.to_summary(),
            }
        )
    _write_json({"generated": written}, None)
    return 0


def _crosscheck(arguments: argparse.Namespace) -> int:
    result = crosscheck_cpp_instance(
        arguments.instance,
        binary=arguments.binary,
        timeout_seconds=arguments.timeout,
        sat_solver=arguments.sat_solver,
    )
    _write_json(result, arguments.output)
    return 0 if result.get("match") is True else 4


def _generate_benchmark(arguments: argparse.Namespace) -> int:
    result = generate_benchmark_batch(
        users=arguments.users,
        agent_counts=arguments.agents,
        services_per_user_counts=arguments.visits,
        calibration_seeds=arguments.calibration_seeds,
        evaluation_seeds=arguments.evaluation_seeds,
        load_profiles=arguments.load_profiles,
        normal_fraction=arguments.normal_fraction,
        output_dir=arguments.output_dir,
        days=arguments.days,
        slots_per_day=arguments.slots_per_day,
        overtime_penalty=arguments.overtime_penalty,
    )
    _write_json(result, None)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hcorap", description="Proposed reproducible HCORAP methods"
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subcommands.add_parser("inspect", help="validate and summarize an instance")
    inspect_parser.add_argument("instance", type=Path)
    inspect_parser.set_defaults(
        handler=lambda args: (_write_json(read_instance(args.instance).to_summary(), None) or 0)
    )

    solve = subcommands.add_parser("solve", help="solve one HCORAP instance")
    solve.add_argument("instance", type=Path)
    solve.add_argument(
        "--method",
        choices=(
            "weighted",
            "lexicographic",
            "epsilon-constraint",
            "cpsat-weighted",
            "cpsat-lexicographic",
            "cpsat-epsilon-constraint",
        ),
        default="weighted",
    )
    solve.add_argument("--continuity-weight", type=int, default=1)
    solve.add_argument("--overtime-weight", type=int, default=1)
    solve.add_argument(
        "--policy",
        choices=(
            "continuity-priority",
            "continuity-overtime-similarity",
            "overtime-priority",
        ),
        default="continuity-priority",
    )
    solve.add_argument("--delta", default="0.05")
    solve.add_argument("--soft-coverage", action="store_true")
    solve.add_argument("--timeout", type=float)
    solve.add_argument("--sat-solver", default="g4")
    solve.add_argument(
        "--maxsat-algorithm",
        choices=("rc2", "rc2-stratified"),
        default="rc2-stratified",
    )
    solve.add_argument("--workers", type=int, default=1)
    solve.add_argument("--random-seed", type=int, default=0)
    solve.add_argument("--output", type=Path)
    solve.set_defaults(handler=_solve)

    verify = subcommands.add_parser("verify", help="independently verify a solution JSON")
    verify.add_argument("instance", type=Path)
    verify.add_argument("solution", type=Path)
    verify.add_argument("--soft-coverage", action="store_true")
    verify.add_argument("--output", type=Path)
    verify.set_defaults(handler=_verify)

    generate = subcommands.add_parser("generate", help="generate a corrected nested family")
    generate.add_argument("--users", type=int, required=True)
    generate.add_argument("--agents", type=int, nargs="+", required=True)
    generate.add_argument("--visits", type=int, nargs="+", required=True)
    generate.add_argument("--seed", type=int, required=True)
    generate.add_argument("--days", type=int, default=5)
    generate.add_argument("--slots-per-day", type=int, default=12)
    generate.add_argument("--normal-hour-cap", type=int, default=35)
    generate.add_argument("--overtime-penalty", type=int, default=-1)
    generate.add_argument("--output-dir", type=Path, required=True)
    generate.set_defaults(handler=_generate)

    benchmark = subcommands.add_parser(
        "generate-benchmark",
        help="generate a frozen corrected-v2 calibration/evaluation batch",
    )
    benchmark.add_argument("--users", type=int, nargs="+", required=True)
    benchmark.add_argument("--agents", type=int, nargs="+", required=True)
    benchmark.add_argument("--visits", type=int, nargs="+", required=True)
    benchmark.add_argument("--calibration-seeds", type=int, nargs="*", default=())
    benchmark.add_argument("--evaluation-seeds", type=int, nargs="*", default=())
    benchmark.add_argument(
        "--load-profiles",
        nargs="+",
        choices=tuple(LOAD_PROFILES),
        default=tuple(LOAD_PROFILES),
    )
    benchmark.add_argument("--normal-fraction", type=float, default=0.85)
    benchmark.add_argument("--days", type=int, default=5)
    benchmark.add_argument("--slots-per-day", type=int, default=12)
    benchmark.add_argument("--overtime-penalty", type=int, default=-1)
    benchmark.add_argument("--output-dir", type=Path, required=True)
    benchmark.set_defaults(handler=_generate_benchmark)

    experiment = subcommands.add_parser("experiment", help="run a JSON experiment grid")
    experiment.add_argument("config", type=Path)
    experiment.add_argument("--output", type=Path)
    experiment.add_argument("--resume", action="store_true")
    experiment.set_defaults(
        handler=lambda args: (
            _write_json(
                run_experiment_config(
                    args.config, output_path=args.output, resume=args.resume
                ),
                None,
            )
            or 0
        )
    )

    crosscheck = subcommands.add_parser(
        "crosscheck-cpp", help="compare weighted objective with the authors' C++ encoder"
    )
    crosscheck.add_argument("instance", type=Path)
    crosscheck.add_argument(
        "--binary", type=Path, default=Path("bin/release/hcorap2sat")
    )
    crosscheck.add_argument("--timeout", type=float, default=60.0)
    crosscheck.add_argument("--sat-solver", default="g4")
    crosscheck.add_argument("--output", type=Path)
    crosscheck.set_defaults(handler=_crosscheck)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        return int(arguments.handler(arguments))
    except (ValueError, RuntimeError, OSError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
