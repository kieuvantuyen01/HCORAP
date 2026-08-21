#!/usr/bin/env python3
"""Create manuscript-ready raw, summary, and cross-backend commercial tables."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from run_reproducible_campaign import _instance_dimensions  # noqa: E402


SUCCESS = {"OPTIMUM", "INFEASIBLE"}
RAW_COLUMNS = (
    "run_id", "instance", "instance_sha256", "users", "agents", "visits",
    "seed_instance", "load_profile", "backend", "formulation", "solver_version",
    "method", "objective_mode", "objective_policy", "delta", "wc", "wo", "soft_coverage",
    "status", "exit_code", "hard_timeout", "validation_errors", "elapsed_seconds",
    "wall_seconds", "timeout_seconds", "peak_rss_mb", "threads", "solver_seed",
    "mip_gap", "absolute_mip_gap", "coverage", "similarity", "continuity",
    "overtime", "overtime_cost", "weighted_reference_score", "verified",
    "assignment_count", "assignment_sha256", "solver_calls", "stage_count",
    "build_seconds_sum", "solve_seconds_sum", "verification_seconds_sum",
    "variables_max", "constraints_max", "search_nodes_or_branches_sum",
    "relative_gap_max", "stage_names", "stage_incumbents",
    "similarity_reference_optimum", "similarity_lower_bound", "result",
    "native_log", "stderr_log",
)
GROUP_COLUMNS = (
    "backend", "formulation", "method", "objective_policy", "delta", "soft_coverage",
    "users", "agents", "visits", "load_profile",
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
    """Resolve raw payloads after a GCP artifact has been copied elsewhere."""
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
            candidate = SCRIPT_DIR.parent / marker.strip("/") / text.split(marker, 1)[1]
            if candidate.is_file():
                return candidate
    return path


def flatten(result_dir: Path) -> list[dict[str, Any]]:
    records = [
        json.loads(line)
        for line in (result_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = []
    for record in records:
        result_path = _relocated_path(
            Path(record["result"]), result_dir, record["run_id"]
        )
        payload = _read_json(result_path)
        metrics = payload.get("metrics") or {}
        stages = payload.get("stages") or []
        assignments = sorted(payload.get("assignments") or [])
        assignment_hash = (
            hashlib.sha256(
                json.dumps(assignments, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            if assignments
            else None
        )
        try:
            dimensions = _instance_dimensions(
                _relocated_instance(Path(record["instance"]))
            )
        except ValueError:
            dimensions = {
                "users": None, "agents": None, "visits": None,
                "seed": None, "load": None,
            }
        specification = record["specification"]
        relative_gaps = [
            value for stage in stages if (value := _number(stage.get("relative_gap"))) is not None
        ]
        rows.append(
            {
                "run_id": record["run_id"],
                "instance": record["instance"],
                "instance_sha256": record["instance_sha256"],
                "users": dimensions["users"],
                "agents": dimensions["agents"],
                "visits": dimensions["visits"],
                "seed_instance": dimensions["seed"],
                "load_profile": dimensions.get("load"),
                "backend": specification["backend"],
                "formulation": specification["formulation"],
                "solver_version": payload.get("solver_version"),
                "method": specification["method"],
                "objective_mode": payload.get("objective_mode"),
                "objective_policy": payload.get("objective_policy"),
                "delta": specification.get("delta", "-"),
                "wc": specification.get("wc", 1),
                "wo": specification.get("wo", 1),
                "soft_coverage": specification.get("soft_coverage", False),
                "status": payload.get("status", record.get("result_status", "MISSING")),
                "exit_code": record.get("exit_code"),
                "hard_timeout": record.get("hard_timeout"),
                "validation_errors": " | ".join(record.get("validation_errors") or []),
                "elapsed_seconds": payload.get("elapsed_seconds"),
                "wall_seconds": record.get("wall_seconds"),
                "timeout_seconds": payload.get("timeout_seconds"),
                "peak_rss_mb": (
                    record["peak_rss_bytes"] / (1024 * 1024)
                    if record.get("peak_rss_bytes") is not None else None
                ),
                "threads": payload.get("threads"),
                "solver_seed": payload.get("seed"),
                "mip_gap": payload.get("mip_gap"),
                "absolute_mip_gap": payload.get("absolute_mip_gap"),
                "coverage": metrics.get("coverage"),
                "similarity": metrics.get("similarity"),
                "continuity": metrics.get("continuity"),
                "overtime": metrics.get("overtime"),
                "overtime_cost": metrics.get("overtime_cost"),
                "weighted_reference_score": metrics.get("weighted_reference_score"),
                "verified": metrics.get("verified"),
                "assignment_count": len(assignments) if assignments else None,
                "assignment_sha256": assignment_hash,
                "solver_calls": payload.get("solver_calls"),
                "stage_count": len(stages),
                "build_seconds_sum": sum(_number(stage.get("build_seconds")) or 0 for stage in stages),
                "solve_seconds_sum": sum(_number(stage.get("solve_seconds")) or 0 for stage in stages),
                "verification_seconds_sum": sum(_number(stage.get("verification_seconds")) or 0 for stage in stages),
                "variables_max": max((stage.get("variables", 0) for stage in stages), default=0),
                "constraints_max": max((stage.get("constraints", 0) for stage in stages), default=0),
                "search_nodes_or_branches_sum": sum(
                    int(stage.get("search_nodes_or_branches") or 0) for stage in stages
                ),
                "relative_gap_max": max(relative_gaps) if relative_gaps else None,
                "stage_names": " | ".join(str(stage.get("name", "")) for stage in stages),
                "stage_incumbents": " | ".join(str(stage.get("incumbent", "")) for stage in stages),
                "similarity_reference_optimum": payload.get("similarity_reference_optimum"),
                "similarity_lower_bound": payload.get("similarity_lower_bound"),
                "result": str(result_path),
                "native_log": record.get("native_log"),
                "stderr_log": record.get("stderr_log"),
            }
        )
    return rows


def _mean(rows: Iterable[dict[str, Any]], key: str) -> float | None:
    values = [value for row in rows if (value := _number(row.get(key))) is not None]
    return statistics.fmean(values) if values else None


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(column) for column in GROUP_COLUMNS)].append(row)
    output = []
    for key, group in sorted(groups.items(), key=lambda item: tuple(str(value) for value in item[0])):
        statuses = [row["status"] for row in group]
        par2 = []
        for row in group:
            elapsed = _number(row.get("elapsed_seconds")) or 0
            timeout = _number(row.get("timeout_seconds")) or elapsed
            par2.append(elapsed if row["status"] in SUCCESS else 2 * timeout)
        item = dict(zip(GROUP_COLUMNS, key))
        elapsed_values = [
            value for row in group if (value := _number(row.get("elapsed_seconds"))) is not None
        ]
        item.update(
            {
                "runs": len(group),
                "optimum_runs": statuses.count("OPTIMUM"),
                "infeasible_runs": statuses.count("INFEASIBLE"),
                "timeout_feasible_runs": statuses.count("TIMEOUT_FEASIBLE"),
                "timeout_runs": statuses.count("TIMEOUT"),
                "error_runs": sum(status not in SUCCESS | {"TIMEOUT", "TIMEOUT_FEASIBLE"} for status in statuses),
                "verified_incumbent_runs": sum(row["verified"] is True for row in group),
                "mean_elapsed_seconds": _mean(group, "elapsed_seconds"),
                "median_elapsed_seconds": statistics.median(elapsed_values) if elapsed_values else None,
                "par2_seconds": statistics.fmean(par2),
                "mean_peak_rss_mb": _mean(group, "peak_rss_mb"),
                "mean_similarity": _mean(group, "similarity"),
                "mean_continuity": _mean(group, "continuity"),
                "mean_overtime": _mean(group, "overtime"),
                "mean_relative_gap": _mean(group, "relative_gap_max"),
            }
        )
        output.append(item)
    return output


def backend_agreement(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["status"] == "OPTIMUM":
            key = (
                row["instance_sha256"], row["method"], row["delta"],
                row["wc"], row["wo"], row["soft_coverage"],
            )
            groups[key].append(row)
    output = []
    for key, group in groups.items():
        if len({row["backend"] for row in group}) < 2:
            continue
        vectors = {
            (row["coverage"], row["similarity"], row["continuity"], row["overtime"])
            for row in group
        }
        scores = {row["weighted_reference_score"] for row in group}
        output.append(
            {
                "instance_sha256": key[0],
                "instance": group[0]["instance"],
                "method": key[1],
                "delta": key[2],
                "wc": key[3],
                "wo": key[4],
                "soft_coverage": key[5],
                "backends": " | ".join(sorted({row["backend"] for row in group})),
                "objective_vector_agreement": len(vectors) == 1,
                "weighted_score_agreement": len(scores) == 1,
                "vectors": " | ".join(str(item) for item in sorted(vectors, key=str)),
            }
        )
    return sorted(output, key=lambda item: (item["instance"], item["method"], item["delta"]))


def _write(path: Path, rows: list[dict[str, Any]], columns: Iterable[str] | None = None) -> None:
    fieldnames = list(columns or (rows[0].keys() if rows else []))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def collect(result_dir: Path) -> dict[str, Any]:
    result_dir = Path(result_dir)
    rows = flatten(result_dir)
    summaries = summarize(rows)
    agreements = backend_agreement(rows)
    _write(result_dir / "runs.csv", rows, RAW_COLUMNS)
    _write(result_dir / "summary_by_backend_class.csv", summaries)
    _write(result_dir / "backend_agreement.csv", agreements)
    result = {
        "runs": len(rows),
        "summary_groups": len(summaries),
        "paired_backend_rows": len(agreements),
        "objective_vector_disagreements": sum(
            item["objective_vector_agreement"] is False for item in agreements
        ),
    }
    (result_dir / "collection_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    arguments = parser.parse_args()
    try:
        result = collect(arguments.result_dir)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
