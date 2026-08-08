#!/usr/bin/env python3
"""Summarize epsilon-to-point mappings and nondominated HCORAP outcomes."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


ALLOWED = {"OPTIMUM", "UNSAT", "UNSATISFIABLE", "TIMEOUT", "TIMEOUT_FEASIBLE"}
VECTOR_KEYS = ("coverage", "similarity", "continuity", "overtime")
GROUP_KEYS = ("instance_sha256", "cardinality", "implied", "symmetry")


def _read(result_dir: Path) -> list[dict[str, str]]:
    validation = json.loads(
        (result_dir / "validation.json").read_text(encoding="utf-8")
    )
    if not validation.get("complete"):
        raise ValueError(f"incomplete campaign: {result_dir}")
    with (result_dir / "runs.csv").open(newline="", encoding="utf-8") as stream:
        return [row for row in csv.DictReader(stream) if row["method"] == "epsilon"]


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _vector(row: dict[str, str]) -> tuple[int, int, int, int]:
    return tuple(int(float(row[key])) for key in VECTOR_KEYS)  # type: ignore[return-value]


def _dominates(left: tuple[int, ...], right: tuple[int, ...]) -> bool:
    weak = (
        left[0] >= right[0] and left[1] >= right[1]
        and left[2] <= right[2] and left[3] <= right[3]
    )
    return weak and left != right


def analyze(result_dir: Path, output_dir: Path) -> dict[str, Any]:
    rows = _read(result_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in GROUP_KEYS)].append(row)

    points = []
    mappings = []
    summaries = []
    for key, group in sorted(groups.items()):
        optimum = [row for row in group if row["status"] == "OPTIMUM"]
        by_vector: dict[tuple[int, int, int, int], list[dict[str, str]]] = defaultdict(list)
        for row in optimum:
            by_vector[_vector(row)].append(row)
        nondominated = {
            vector: not any(
                other != vector and _dominates(other, vector)
                for other in by_vector
            )
            for vector in by_vector
        }
        for vector, vector_rows in sorted(by_vector.items(), reverse=True):
            deltas = sorted({float(row["delta"]) for row in vector_rows})
            first = vector_rows[0]
            points.append(
                {
                    "instance": first["instance"],
                    **dict(zip(GROUP_KEYS, key)),
                    "coverage": vector[0],
                    "similarity": vector[1],
                    "continuity": vector[2],
                    "overtime": vector[3],
                    "delta_count": len(deltas),
                    "deltas": " | ".join(f"{value:g}" for value in deltas),
                    "minimum_delta": min(deltas),
                    "maximum_delta": max(deltas),
                    "nondominated": nondominated[vector],
                }
            )
        for row in sorted(group, key=lambda item: float(item["delta"])):
            mappings.append(
                {
                    "instance": row["instance"],
                    **dict(zip(GROUP_KEYS, key)),
                    "delta": row["delta"],
                    "status": row["status"],
                    "similarity_reference_optimum": row["similarity_reference_optimum"],
                    "similarity_lower_bound": row["similarity_lower_bound"],
                    "similarity_realized_loss_absolute": row["similarity_realized_loss_absolute"],
                    "vector": "/".join(map(str, _vector(row)))
                    if row["status"] == "OPTIMUM" else None,
                }
            )
        first = group[0]
        summaries.append(
            {
                "instance": first["instance"],
                **dict(zip(GROUP_KEYS, key)),
                "requested_deltas": len(group),
                "optimum_deltas": len(optimum),
                "timeout_deltas": sum(row["status"].startswith("TIMEOUT") for row in group),
                "unique_points": len(by_vector),
                "nondominated_points": sum(nondominated.values()),
                "redundant_optimum_deltas": len(optimum) - len(by_vector),
            }
        )

    _write(output_dir / "pareto_delta_mapping.csv", mappings)
    _write(output_dir / "pareto_unique_points.csv", points)
    _write(output_dir / "pareto_instance_summary.csv", summaries)
    unique_counts = [row["unique_points"] for row in summaries]
    result = {
        "epsilon_runs": len(rows),
        "instances": len(groups),
        "optimum_runs": sum(row["status"] == "OPTIMUM" for row in rows),
        "timeout_runs": sum(row["status"].startswith("TIMEOUT") for row in rows),
        "hard_errors": sum(row["status"] not in ALLOWED for row in rows),
        "unverified_optimum": sum(
            row["status"] == "OPTIMUM" and row["verified"] != "True" for row in rows
        ),
        "unique_points": len(points),
        "nondominated_points": sum(bool(row["nondominated"]) for row in points),
        "instances_with_multiple_points": sum(value >= 2 for value in unique_counts),
        "median_unique_points_per_instance": statistics.median(unique_counts)
        if unique_counts else None,
        "hypervolume_reported": False,
        "hypervolume_reason": "No decision-relevant reference point was predeclared.",
    }
    result["valid"] = result["hard_errors"] == 0 and result["unverified_optimum"] == 0
    (output_dir / "analysis.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        result = analyze(arguments.results, arguments.output_dir)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
