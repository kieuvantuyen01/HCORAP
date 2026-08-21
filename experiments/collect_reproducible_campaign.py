#!/usr/bin/env python3
"""Flatten and validate outputs from run_reproducible_campaign.py."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
INSTANCE_PATTERN = re.compile(
    r"(?:instance_)?u?(?P<users>\d+)_a?(?P<agents>\d+)_v?(?P<visits>\d+)"
    r"(?:_seed(?P<seed>\d+))?(?:_(?P<load>relaxed|critical|saturated))?"
)
SUCCESS = {"OPTIMUM", "UNSAT", "UNSATISFIABLE"}

RAW_COLUMNS = (
    "run_id", "instance", "instance_sha256", "users", "agents", "visits",
    "seed", "load_profile", "rho", "method", "objective_mode",
    "objective_policy", "delta", "wc", "wo", "soft_coverage", "cardinality", "implied",
    "symmetry", "status", "exit_code", "hard_timeout", "validation_errors",
    "elapsed_seconds", "wall_seconds", "timeout_seconds", "peak_rss_mb",
    "coverage", "similarity", "continuity", "overtime", "overtime_cost",
    "weighted_reference_score", "verified", "assignment_count",
    "assignment_sha256", "solver_calls", "stage_count",
    "encode_seconds_sum", "solve_seconds_sum", "variables_max",
    "hard_clauses_max", "soft_clauses_max", "stage_objectives", "stage_optima",
    "similarity_reference_optimum", "similarity_lower_bound",
    "similarity_realized_loss_absolute", "pareto_nondominated", "result",
    "stderr_log",
)

METHOD_GROUP = (
    "method", "objective_policy", "delta", "wc", "wo", "soft_coverage", "cardinality",
    "implied", "symmetry", "load_profile",
)
CLASS_GROUP = METHOD_GROUP + ("users", "agents", "visits")
EPSILON_POINT_COLUMNS = RAW_COLUMNS + (
    "delta_count", "deltas", "minimum_delta", "maximum_delta",
)

SUMMARY_METRICS = (
    "elapsed_seconds", "wall_seconds", "peak_rss_mb", "similarity",
    "continuity", "overtime", "weighted_reference_score", "variables_max",
    "hard_clauses_max", "soft_clauses_max",
)


def _number(value: Any) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _relocated_path(path: Path, result_dir: Path, run_id: str) -> Path:
    """Resolve paths after a GCP result directory has been relocated."""
    if path.is_file():
        return path
    local_raw = result_dir / "raw" / f"{run_id}.json"
    return local_raw if local_raw.is_file() else path


def _relocated_instance(path: Path) -> Path:
    if path.is_file():
        return path
    text = path.as_posix()
    for marker in ("/instances/", "/tests/"):
        if marker in text:
            candidate = ROOT / marker.strip("/") / text.split(marker, 1)[1]
            if candidate.is_file():
                return candidate
    return path


def _instance_fields(path: Path) -> dict[str, Any]:
    match = INSTANCE_PATTERN.search(path.stem)
    fields: dict[str, Any] = {
        "users": None, "agents": None, "visits": None,
        "seed": None, "load_profile": None, "rho": None,
    }
    if match:
        fields.update(match.groupdict())
    sidecar = _read_json(path.with_suffix(path.suffix + ".json"))
    summary = sidecar.get("instance") or {}
    calibration = (sidecar.get("metadata") or {}).get("capacity_calibration") or {}
    fields.update(
        {
            "users": summary.get("users", fields["users"]),
            "agents": summary.get("agents", fields["agents"]),
            "visits": (
                summary.get("services", 0) // summary.get("users", 1)
                if summary.get("users")
                else fields["visits"]
            ),
            "load_profile": calibration.get("load_profile", fields["load_profile"]),
            "rho": summary.get("rho"),
        }
    )
    return fields


def flatten(result_dir: Path) -> list[dict[str, Any]]:
    manifest = result_dir / "manifest.jsonl"
    records = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = []
    for record in records:
        result_path = _relocated_path(
            Path(record["result"]), result_dir, record["run_id"]
        )
        payload = _read_json(result_path)
        specification = record["specification"]
        metrics = payload.get("metrics") or {}
        stages = payload.get("stages") or []
        assignments = sorted(payload.get("assignments") or [])
        assignment_sha256 = (
            hashlib.sha256(
                json.dumps(assignments, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            if assignments
            else None
        )
        instance_path = _relocated_instance(Path(record["instance"]))
        row = {
            "run_id": record["run_id"],
            "instance": str(instance_path),
            "instance_sha256": record["instance_sha256"],
            **_instance_fields(instance_path),
            "method": specification["method"],
            "objective_mode": payload.get("objective_mode"),
            "objective_policy": payload.get("objective_policy"),
            "delta": specification.get("delta", "-"),
            "wc": specification.get("wc", 1),
            "wo": specification.get("wo", 1),
            "soft_coverage": specification.get("soft_coverage", False),
            "cardinality": specification["cardinality"],
            "implied": specification["implied"],
            "symmetry": specification["symmetry"],
            "status": payload.get("status", record.get("result_status", "MISSING")),
            "exit_code": record.get("exit_code"),
            "hard_timeout": record.get("hard_timeout"),
            "validation_errors": " | ".join(record.get("validation_errors") or []),
            "elapsed_seconds": payload.get("elapsed_seconds"),
            "wall_seconds": record.get("wall_seconds"),
            "timeout_seconds": payload.get("timeout_seconds"),
            "peak_rss_mb": (
                record["peak_rss_bytes"] / (1024 * 1024)
                if record.get("peak_rss_bytes") is not None
                else None
            ),
            "coverage": metrics.get("coverage"),
            "similarity": metrics.get("similarity"),
            "continuity": metrics.get("continuity"),
            "overtime": metrics.get("overtime"),
            "overtime_cost": metrics.get("overtime_cost"),
            "weighted_reference_score": metrics.get("weighted_reference_score"),
            "verified": metrics.get("verified"),
            "assignment_count": len(assignments) if assignments else None,
            "assignment_sha256": assignment_sha256,
            "solver_calls": payload.get("solver_calls"),
            "stage_count": len(stages),
            "encode_seconds_sum": sum(_number(stage.get("encode_seconds")) or 0 for stage in stages),
            "solve_seconds_sum": sum(_number(stage.get("solve_seconds")) or 0 for stage in stages),
            "variables_max": max((stage.get("variables", 0) for stage in stages), default=0),
            "hard_clauses_max": max((stage.get("hard_clauses", 0) for stage in stages), default=0),
            "soft_clauses_max": max((stage.get("soft_clauses", 0) for stage in stages), default=0),
            "stage_objectives": " | ".join(str(stage.get("objective", "")) for stage in stages),
            "stage_optima": " | ".join(str(stage.get("optimum", "")) for stage in stages),
            "similarity_reference_optimum": payload.get("similarity_reference_optimum"),
            "similarity_lower_bound": payload.get("similarity_lower_bound"),
            "similarity_realized_loss_absolute": payload.get("similarity_realized_loss_absolute"),
            "pareto_nondominated": None,
            "result": str(result_path),
            "stderr_log": record.get("stderr_log"),
        }
        rows.append(row)
    _annotate_epsilon_pareto(rows)
    return rows


def _dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
    values_left = tuple(_number(left.get(key)) for key in ("coverage", "similarity", "continuity", "overtime"))
    values_right = tuple(_number(right.get(key)) for key in ("coverage", "similarity", "continuity", "overtime"))
    if None in values_left or None in values_right:
        return False
    weak = (
        values_left[0] >= values_right[0]
        and values_left[1] >= values_right[1]
        and values_left[2] <= values_right[2]
        and values_left[3] <= values_right[3]
    )
    return weak and values_left != values_right


def _annotate_epsilon_pareto(rows: list[dict[str, Any]]) -> None:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["method"] == "epsilon" and row["status"] == "OPTIMUM":
            key = tuple(
                row[column]
                for column in (
                    "instance_sha256", "soft_coverage", "cardinality",
                    "implied", "symmetry",
                )
            )
            groups[key].append(row)
    for group in groups.values():
        for row in group:
            row["pareto_nondominated"] = not any(
                other is not row and _dominates(other, row) for other in group
            )


def epsilon_unique_points(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse repeated delta outcomes before reporting a Pareto frontier."""

    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    identity = (
        "instance_sha256", "soft_coverage", "cardinality", "implied", "symmetry",
        "coverage", "similarity", "continuity", "overtime",
    )
    for row in rows:
        if row["method"] == "epsilon" and row["status"] == "OPTIMUM":
            groups[tuple(row.get(column) for column in identity)].append(row)
    points = []
    for group in groups.values():
        deltas = sorted({str(row["delta"]) for row in group}, key=float)
        point = dict(group[0])
        point.update(
            {
                "delta": " | ".join(deltas),
                "delta_count": len(deltas),
                "deltas": " | ".join(deltas),
                "minimum_delta": deltas[0],
                "maximum_delta": deltas[-1],
                "elapsed_seconds": statistics.fmean(
                    value
                    for row in group
                    if (value := _number(row.get("elapsed_seconds"))) is not None
                ),
            }
        )
        points.append(point)
    _annotate_epsilon_pareto(points)
    return sorted(
        points,
        key=lambda row: (
            str(row["instance"]),
            str(row["cardinality"]),
            float(row["minimum_delta"]),
        ),
    )


def _mean(rows: Iterable[dict[str, Any]], column: str) -> float | None:
    values = [_number(row.get(column)) for row in rows]
    present = [value for value in values if value is not None]
    return statistics.fmean(present) if present else None


def summarize(rows: list[dict[str, Any]], group_columns: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(column) for column in group_columns)].append(row)
    summaries = []
    for key, group in sorted(groups.items(), key=lambda item: tuple(str(value) for value in item[0])):
        statuses = [row["status"] for row in group]
        par2 = []
        for row in group:
            elapsed = _number(row.get("elapsed_seconds")) or 0.0
            timeout = _number(row.get("timeout_seconds")) or elapsed
            par2.append(elapsed if row["status"] in SUCCESS else 2 * timeout)
        result = dict(zip(group_columns, key))
        result.update(
            {
                "runs": len(group),
                "optimum_runs": statuses.count("OPTIMUM"),
                "unsat_runs": sum(status in {"UNSAT", "UNSATISFIABLE"} for status in statuses),
                "timeout_runs": sum(status.startswith("TIMEOUT") for status in statuses),
                "error_runs": sum(status not in SUCCESS and not status.startswith("TIMEOUT") for status in statuses),
                "verified_optimum_runs": sum(row["status"] == "OPTIMUM" and row["verified"] is True for row in group),
                "par2_seconds": statistics.fmean(par2) if par2 else None,
            }
        )
        for metric in SUMMARY_METRICS:
            result[f"mean_{metric}"] = _mean(group, metric)
            if metric in {"elapsed_seconds", "wall_seconds"}:
                values = sorted(value for row in group if (value := _number(row.get(metric))) is not None)
                result[f"median_{metric}"] = statistics.median(values) if values else None
        summaries.append(result)
    return summaries


def _write(path: Path, rows: list[dict[str, Any]], columns: Iterable[str] | None = None) -> None:
    fieldnames = list(columns or (rows[0].keys() if rows else []))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def collect(result_dir: Path) -> dict[str, Any]:
    result_dir = Path(result_dir)
    rows = flatten(result_dir)
    method_summary = summarize(rows, METHOD_GROUP)
    class_summary = summarize(rows, CLASS_GROUP)
    epsilon_points = epsilon_unique_points(rows)
    _write(result_dir / "runs.csv", rows, RAW_COLUMNS)
    _write(result_dir / "summary_by_method_config.csv", method_summary)
    _write(result_dir / "summary_by_class.csv", class_summary)
    _write(result_dir / "epsilon_unique_points.csv", epsilon_points, EPSILON_POINT_COLUMNS)
    _write(
        result_dir / "epsilon_pareto_frontier.csv",
        [row for row in epsilon_points if row["pareto_nondominated"] is True],
        EPSILON_POINT_COLUMNS,
    )
    return {
        "runs": len(rows),
        "method_groups": len(method_summary),
        "class_groups": len(class_summary),
        "optimum_runs": sum(row["status"] == "OPTIMUM" for row in rows),
        "timeout_runs": sum(str(row["status"]).startswith("TIMEOUT") for row in rows),
        "epsilon_unique_points": len(epsilon_points),
        "epsilon_pareto_points": sum(
            row["pareto_nondominated"] is True for row in epsilon_points
        ),
        "error_runs": sum(
            row["status"] not in SUCCESS and not str(row["status"]).startswith("TIMEOUT")
            for row in rows
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    arguments = parser.parse_args()
    try:
        result = collect(arguments.result_dir)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
