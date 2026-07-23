#!/usr/bin/env python3
"""Flatten one C++ campaign into Excel-ready raw and summary CSV files."""

from __future__ import annotations

import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


RAW_COLUMNS = (
    "run_id",
    "sha256",
    "instance",
    "cardinality_encoding",
    "implied_constraints",
    "symmetry_breaking",
    "method",
    "delta",
    "objective_mode",
    "objective_policy",
    "status",
    "exit_code",
    "elapsed_seconds",
    "timeout_seconds",
    "full_coverage",
    "continuity_weight",
    "overtime_weight",
    "coverage",
    "similarity",
    "continuity",
    "overtime",
    "overtime_cost",
    "verified",
    "solver_calls",
    "stage_count",
    "encode_seconds_sum",
    "solve_seconds_sum",
    "variables_max",
    "hard_clauses_max",
    "soft_clauses_max",
    "stage_objectives",
    "stage_optima",
    "similarity_reference_optimum",
    "similarity_lower_bound",
    "similarity_realized_loss_absolute",
    "similarity_realized_loss_fraction",
    "solver",
    "result_file",
    "error",
)

GROUP_COLUMNS = (
    "cardinality_encoding",
    "implied_constraints",
    "symmetry_breaking",
    "method",
    "delta",
)

UNSAT_STATUSES = {"UNSAT", "UNSATISFIABLE"}
SUCCESS_STATUSES = {"OPTIMUM", *UNSAT_STATUSES}

SUMMARY_COLUMNS = GROUP_COLUMNS + (
    "runs",
    "optimum_runs",
    "unsat_runs",
    "timeout_runs",
    "error_runs",
    "verified_runs",
    "mean_elapsed_seconds",
    "median_elapsed_seconds",
    "par2_seconds",
    "mean_encode_seconds",
    "mean_solve_seconds",
    "mean_variables_max",
    "mean_hard_clauses_max",
    "mean_soft_clauses_max",
    "mean_coverage",
    "mean_similarity",
    "mean_continuity",
    "mean_overtime",
    "mean_similarity_reference_optimum",
    "mean_similarity_lower_bound",
    "mean_similarity_realized_loss_absolute",
)


def _number(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _integer(value: Any) -> int | None:
    number = _number(value)
    return int(number) if number is not None else None


def _mean(rows: Iterable[dict[str, Any]], column: str) -> float | None:
    values = [value for row in rows if (value := _number(row.get(column))) is not None]
    return statistics.fmean(values) if values else None


def _read_result(path: Path) -> tuple[dict[str, Any], str]:
    try:
        return json.loads(path.read_text(encoding="utf-8")), ""
    except (OSError, json.JSONDecodeError) as exc:
        return {}, str(exc)


def _read_environment(path: Path) -> dict[str, str]:
    values = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return values
    for line in lines:
        key, separator, value = line.partition("=")
        if separator:
            values[key] = value
    return values


def flatten_campaign(result_dir: Path) -> list[dict[str, Any]]:
    manifest_path = result_dir / "manifest.tsv"
    with manifest_path.open(newline="", encoding="utf-8") as stream:
        manifest = list(csv.DictReader(stream, delimiter="\t"))
    environment = _read_environment(result_dir / "environment.txt")
    default_timeout = _number(environment.get("timeout"))

    rows = []
    for entry in manifest:
        result_path = Path(entry["result"])
        if not result_path.is_absolute():
            result_path = Path.cwd() / result_path
        result, read_error = _read_result(result_path)
        metrics = result.get("metrics") or {}
        stages = result.get("stages") or []
        row = {
            "run_id": entry["run_id"],
            "sha256": entry["sha256"],
            "instance": entry["instance"],
            "cardinality_encoding": result.get(
                "cardinality_encoding", entry["cardinality_encoding"]
            ),
            "implied_constraints": result.get(
                "implied_constraints", entry["implied_constraints"]
            ),
            "symmetry_breaking": result.get(
                "symmetry_breaking", entry["symmetry_breaking"]
            ),
            "method": result.get("method", entry["method"]),
            "delta": (
                result.get("delta", entry["delta"])
                if entry["method"] == "epsilon"
                else entry["delta"]
            ),
            "objective_mode": result.get("objective_mode"),
            "objective_policy": result.get("objective_policy"),
            "status": result.get("status", "MISSING_RESULT"),
            "exit_code": _integer(entry["exit_code"]),
            "elapsed_seconds": _number(result.get("elapsed_seconds")),
            "timeout_seconds": _number(
                result.get("timeout_seconds", default_timeout)
            ),
            "full_coverage": result.get("full_coverage"),
            "continuity_weight": _integer(result.get("continuity_weight")),
            "overtime_weight": _integer(result.get("overtime_weight")),
            "coverage": _integer(metrics.get("coverage")),
            "similarity": _integer(metrics.get("similarity")),
            "continuity": _integer(metrics.get("continuity")),
            "overtime": _integer(metrics.get("overtime")),
            "overtime_cost": _integer(metrics.get("overtime_cost")),
            "verified": metrics.get("verified"),
            "solver_calls": _integer(result.get("solver_calls")),
            "stage_count": len(stages),
            "encode_seconds_sum": sum(
                _number(stage.get("encode_seconds")) or 0 for stage in stages
            ),
            "solve_seconds_sum": sum(
                _number(stage.get("solve_seconds")) or 0 for stage in stages
            ),
            "variables_max": max(
                (_integer(stage.get("variables")) or 0 for stage in stages),
                default=0,
            ),
            "hard_clauses_max": max(
                (_integer(stage.get("hard_clauses")) or 0 for stage in stages),
                default=0,
            ),
            "soft_clauses_max": max(
                (_integer(stage.get("soft_clauses")) or 0 for stage in stages),
                default=0,
            ),
            "stage_objectives": " | ".join(
                str(stage.get("objective", "")) for stage in stages
            ),
            "stage_optima": " | ".join(
                str(stage.get("optimum", "")) for stage in stages
            ),
            "similarity_reference_optimum": _integer(
                result.get("similarity_reference_optimum")
            ),
            "similarity_lower_bound": _integer(
                result.get("similarity_lower_bound")
            ),
            "similarity_realized_loss_absolute": _integer(
                result.get("similarity_realized_loss_absolute")
            ),
            "similarity_realized_loss_fraction": result.get(
                "similarity_realized_loss_fraction"
            ),
            "solver": result.get("solver", ""),
            "result_file": str(result_path),
            "error": result.get("error") or read_error,
        }
        rows.append(row)
    return rows


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[column] for column in GROUP_COLUMNS)].append(row)

    summaries = []
    for key in sorted(groups):
        group = groups[key]
        statuses = [row["status"] for row in group]
        elapsed = [
            value
            for row in group
            if (value := _number(row.get("elapsed_seconds"))) is not None
        ]
        par2_values = []
        for row in group:
            run_elapsed = _number(row.get("elapsed_seconds")) or 0.0
            timeout = _number(row.get("timeout_seconds")) or run_elapsed
            par2_values.append(
                run_elapsed
                if row["status"] in SUCCESS_STATUSES
                else 2.0 * timeout
            )
        summary = dict(zip(GROUP_COLUMNS, key))
        summary.update(
            {
                "runs": len(group),
                "optimum_runs": statuses.count("OPTIMUM"),
                "unsat_runs": sum(status in UNSAT_STATUSES for status in statuses),
                "timeout_runs": statuses.count("TIMEOUT"),
                "error_runs": sum(
                    status not in SUCCESS_STATUSES | {"TIMEOUT"}
                    for status in statuses
                ),
                "verified_runs": sum(row.get("verified") is True for row in group),
                "mean_elapsed_seconds": statistics.fmean(elapsed) if elapsed else None,
                "median_elapsed_seconds": statistics.median(elapsed) if elapsed else None,
                "par2_seconds": statistics.fmean(par2_values) if par2_values else None,
                "mean_encode_seconds": _mean(group, "encode_seconds_sum"),
                "mean_solve_seconds": _mean(group, "solve_seconds_sum"),
                "mean_variables_max": _mean(group, "variables_max"),
                "mean_hard_clauses_max": _mean(group, "hard_clauses_max"),
                "mean_soft_clauses_max": _mean(group, "soft_clauses_max"),
                "mean_coverage": _mean(group, "coverage"),
                "mean_similarity": _mean(group, "similarity"),
                "mean_continuity": _mean(group, "continuity"),
                "mean_overtime": _mean(group, "overtime"),
                "mean_similarity_reference_optimum": _mean(
                    group, "similarity_reference_optimum"
                ),
                "mean_similarity_lower_bound": _mean(
                    group, "similarity_lower_bound"
                ),
                "mean_similarity_realized_loss_absolute": _mean(
                    group, "similarity_realized_loss_absolute"
                ),
            }
        )
        summaries.append(summary)
    return summaries


def _write_csv(path: Path, columns: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    if len(sys.argv) != 2:
        print(f"Usage: {Path(sys.argv[0]).name} RESULT_DIR", file=sys.stderr)
        return 2
    result_dir = Path(sys.argv[1])
    if not (result_dir / "manifest.tsv").is_file():
        print(f"Missing manifest: {result_dir / 'manifest.tsv'}", file=sys.stderr)
        return 2
    rows = flatten_campaign(result_dir)
    _write_csv(result_dir / "runs.csv", RAW_COLUMNS, rows)
    _write_csv(
        result_dir / "configuration_summary.csv",
        SUMMARY_COLUMNS,
        summarize(rows),
    )
    print(f"Raw runs: {result_dir / 'runs.csv'}")
    print(f"Summary: {result_dir / 'configuration_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
