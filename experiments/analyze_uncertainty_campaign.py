#!/usr/bin/env python3
"""Compare fixed nominal schedules with full re-optimization under absences."""

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

from hcorap.io import read_instance
from hcorap.metrics import verify_assignments
from hcorap.model import Assignment


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from run_reproducible_campaign import _instance_dimensions  # noqa: E402


RAW_COLUMNS = (
    "scenario_run_id", "base_instance", "base_instance_sha256",
    "scenario_instance", "scenario_instance_sha256", "users", "agents",
    "visits", "load_profile", "method", "cardinality", "implied", "symmetry",
    "absence_probability", "scenario_seed", "absent_agent_days",
    "removed_available_slots", "nominal_status", "nominal_coverage",
    "nominal_similarity", "nominal_continuity", "nominal_overtime",
    "fixed_coverage", "fixed_similarity", "fixed_continuity", "fixed_overtime",
    "fixed_coverage_ratio", "reoptimized_status", "reoptimized_coverage",
    "reoptimized_similarity", "reoptimized_continuity", "reoptimized_overtime",
    "recovered_services", "reoptimized_elapsed_seconds", "assignment_changed",
    "nominal_assignment_sha256", "reoptimized_assignment_sha256",
)
GROUP_COLUMNS = (
    "method", "absence_probability", "users", "agents", "visits", "load_profile"
)
EXCLUSION_COLUMNS = (
    "scenario_run_id", "base_instance_sha256", "scenario_instance",
    "method", "nominal_status", "reason",
)


def _hash_assignments(assignments: Iterable[Iterable[int]]) -> str:
    normalized = sorted([list(map(int, item)) for item in assignments])
    return hashlib.sha256(
        json.dumps(normalized, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _records(result_dir: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (result_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _payload(record: dict[str, Any]) -> dict[str, Any]:
    return json.loads(Path(record["result"]).read_text(encoding="utf-8"))


def _key(base_hash: str, specification: dict[str, Any]) -> tuple[Any, ...]:
    return (
        base_hash,
        specification["method"],
        specification["cardinality"],
        specification["implied"],
        specification["symmetry"],
    )


def analyze(
    nominal_dir: Path, scenario_dir: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    nominal: dict[tuple[Any, ...], tuple[dict[str, Any], dict[str, Any]]] = {}
    for record in _records(nominal_dir):
        payload = _payload(record)
        key = _key(record["instance_sha256"], record["specification"])
        if key in nominal:
            raise ValueError(f"duplicate nominal solution key: {key}")
        nominal[key] = (record, payload)

    rows = []
    exclusions = []
    for record in _records(scenario_dir):
        scenario_path = Path(record["instance"])
        sidecar = json.loads(
            scenario_path.with_suffix(".txt.json").read_text(encoding="utf-8")
        )
        uncertainty = sidecar["metadata"]["uncertainty"]
        base_hash = uncertainty["base_instance_sha256"]
        key = _key(base_hash, record["specification"])
        if key not in nominal:
            raise ValueError(
                f"no matching nominal assignment for scenario run {record['run_id']}"
            )
        nominal_record, nominal_payload = nominal[key]
        nominal_status = nominal_payload.get("status")
        nominal_assignments_payload = nominal_payload.get("assignments") or []
        nominal_verified = (nominal_payload.get("metrics") or {}).get("verified")
        if (
            nominal_status != "OPTIMUM"
            or not nominal_assignments_payload
            or nominal_verified is not True
        ):
            reasons = []
            if nominal_status != "OPTIMUM":
                reasons.append("nominal schedule is not proven optimal")
            if not nominal_assignments_payload:
                reasons.append("nominal assignment is missing")
            if nominal_verified is not True:
                reasons.append("nominal assignment is not verifier-approved")
            exclusions.append(
                {
                    "scenario_run_id": record["run_id"],
                    "base_instance_sha256": base_hash,
                    "scenario_instance": str(scenario_path),
                    "method": record["specification"]["method"],
                    "nominal_status": nominal_status,
                    "reason": "; ".join(reasons),
                }
            )
            continue
        scenario_payload = _payload(record)
        scenario = read_instance(scenario_path)
        nominal_assignments = [
            Assignment(agent=int(item[0]), service=int(item[1]), time_slot=int(item[2]))
            for item in nominal_assignments_payload
        ]
        surviving = tuple(
            item
            for item in nominal_assignments
            if scenario.agent_availability[item.agent][item.time_slot]
            and scenario.service_availability[item.service][item.time_slot]
            and scenario.rewards[item.agent][item.service] > 0
        )
        fixed = verify_assignments(
            scenario, surviving, require_full_coverage=False
        )
        if not fixed.valid:
            raise ValueError(
                f"surviving fixed schedule is invalid for {record['run_id']}: "
                f"{fixed.violations}"
            )
        nominal_metrics = nominal_payload["metrics"]
        reoptimized_metrics = scenario_payload.get("metrics") or {}
        dimensions = _instance_dimensions(scenario_path)
        reoptimized_assignments = scenario_payload.get("assignments") or []
        rows.append(
            {
                "scenario_run_id": record["run_id"],
                "base_instance": nominal_record["instance"],
                "base_instance_sha256": base_hash,
                "scenario_instance": str(scenario_path),
                "scenario_instance_sha256": record["instance_sha256"],
                "users": dimensions["users"],
                "agents": dimensions["agents"],
                "visits": dimensions["visits"],
                "load_profile": dimensions["load"],
                "method": record["specification"]["method"],
                "cardinality": record["specification"]["cardinality"],
                "implied": record["specification"]["implied"],
                "symmetry": record["specification"]["symmetry"],
                "absence_probability": uncertainty["absence_probability"],
                "scenario_seed": uncertainty["scenario_seed"],
                "absent_agent_days": len(uncertainty["absent_agent_days"]),
                "removed_available_slots": uncertainty["removed_available_slots"],
                "nominal_status": nominal_status,
                "nominal_coverage": nominal_metrics["coverage"],
                "nominal_similarity": nominal_metrics["similarity"],
                "nominal_continuity": nominal_metrics["continuity"],
                "nominal_overtime": nominal_metrics["overtime"],
                "fixed_coverage": fixed.metrics.coverage,
                "fixed_similarity": fixed.metrics.similarity,
                "fixed_continuity": fixed.metrics.continuity_penalty,
                "fixed_overtime": fixed.metrics.overtime,
                "fixed_coverage_ratio": fixed.metrics.coverage / scenario.services,
                "reoptimized_status": scenario_payload.get("status"),
                "reoptimized_coverage": reoptimized_metrics.get("coverage"),
                "reoptimized_similarity": reoptimized_metrics.get("similarity"),
                "reoptimized_continuity": reoptimized_metrics.get("continuity"),
                "reoptimized_overtime": reoptimized_metrics.get("overtime"),
                "recovered_services": (
                    reoptimized_metrics["coverage"] - fixed.metrics.coverage
                    if reoptimized_metrics.get("coverage") is not None else None
                ),
                "reoptimized_elapsed_seconds": scenario_payload.get("elapsed_seconds"),
                "assignment_changed": (
                    _hash_assignments(nominal_assignments_payload)
                    != _hash_assignments(reoptimized_assignments)
                    if reoptimized_assignments else None
                ),
                "nominal_assignment_sha256": _hash_assignments(nominal_assignments_payload),
                "reoptimized_assignment_sha256": (
                    _hash_assignments(reoptimized_assignments)
                    if reoptimized_assignments else None
                ),
            }
        )
    return rows, exclusions


def _number(value: Any) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[column] for column in GROUP_COLUMNS)].append(row)
    output = []
    for key, group in sorted(groups.items(), key=lambda item: tuple(str(value) for value in item[0])):
        recovered = [
            value for row in group if (value := _number(row["recovered_services"])) is not None
        ]
        elapsed = [
            value for row in group if (value := _number(row["reoptimized_elapsed_seconds"])) is not None
        ]
        item = dict(zip(GROUP_COLUMNS, key))
        item.update(
            {
                "scenarios": len(group),
                "scenarios_with_disruption": sum(row["fixed_coverage"] < row["nominal_coverage"] for row in group),
                "reoptimized_optimum": sum(row["reoptimized_status"] == "OPTIMUM" for row in group),
                "reoptimized_timeout_feasible": sum(row["reoptimized_status"] == "TIMEOUT_FEASIBLE" for row in group),
                "mean_fixed_coverage_ratio": statistics.fmean(row["fixed_coverage_ratio"] for row in group),
                "mean_recovered_services": statistics.fmean(recovered) if recovered else None,
                "mean_reoptimized_elapsed_seconds": statistics.fmean(elapsed) if elapsed else None,
            }
        )
        output.append(item)
    return output


def _write(path: Path, rows: list[dict[str, Any]], columns: Iterable[str] | None = None) -> None:
    fieldnames = list(columns or (rows[0].keys() if rows else []))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def collect(nominal_dir: Path, scenario_dir: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, exclusions = analyze(nominal_dir, scenario_dir)
    summaries = summarize(rows)
    _write(output_dir / "uncertainty_runs.csv", rows, RAW_COLUMNS)
    _write(output_dir / "uncertainty_summary.csv", summaries)
    _write(output_dir / "uncertainty_exclusions.csv", exclusions, EXCLUSION_COLUMNS)
    expected = len(_records(scenario_dir))
    result = {
        "scenario_runs": len(rows),
        "excluded_scenario_runs": len(exclusions),
        "accounted_scenario_runs": len(rows) + len(exclusions),
        "expected_scenario_runs": expected,
        "complete_analysis": len(rows) + len(exclusions) == expected and not exclusions,
        "reoptimized_optimum_runs": sum(
            row["reoptimized_status"] == "OPTIMUM" for row in rows
        ),
        "reoptimized_non_optimum_runs": sum(
            row["reoptimized_status"] != "OPTIMUM" for row in rows
        ),
        "summary_groups": len(summaries),
        "nominal_result_dir": str(nominal_dir),
        "scenario_result_dir": str(scenario_dir),
    }
    (output_dir / "analysis.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nominal-results", type=Path, required=True)
    parser.add_argument("--scenario-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        result = collect(
            arguments.nominal_results,
            arguments.scenario_results,
            arguments.output_dir,
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["complete_analysis"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
