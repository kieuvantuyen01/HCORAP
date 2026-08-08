#!/usr/bin/env python3
"""Quantify objective-vector, assignment, and scale stability over weight grids."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "proposed"))
from hcorap.io import read_instance  # noqa: E402


ALLOWED = {"OPTIMUM", "UNSAT", "UNSATISFIABLE", "TIMEOUT", "TIMEOUT_FEASIBLE"}
VECTOR_KEYS = ("coverage", "similarity", "continuity", "overtime")


def _read(result_dir: Path) -> list[dict[str, str]]:
    validation = json.loads(
        (result_dir / "validation.json").read_text(encoding="utf-8")
    )
    if not validation.get("complete"):
        raise ValueError(f"incomplete campaign: {result_dir}")
    with (result_dir / "runs.csv").open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _vector(row: dict[str, str]) -> tuple[int, int, int, int]:
    return tuple(int(float(row[key])) for key in VECTOR_KEYS)  # type: ignore[return-value]


def analyze(result_dir: Path, output_dir: Path) -> dict[str, Any]:
    rows = _read(result_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    penalties: dict[str, int] = {}
    optimum = [row for row in rows if row["status"] == "OPTIMUM"]
    enriched = []
    for row in optimum:
        instance_hash = row["instance_sha256"]
        if instance_hash not in penalties:
            penalties[instance_hash] = read_instance(Path(row["instance"])).penalty
        penalty = penalties[instance_hash]
        wc, wo = int(row["wc"]), int(row["wo"])
        divisor = math.gcd(wc, wo)
        normalized = (wc // divisor, wo // divisor) if divisor else (wc, wo)
        vector = _vector(row)
        enriched.append(
            {
                "instance": row["instance"],
                "instance_sha256": instance_hash,
                "users": row["users"],
                "agents": row["agents"],
                "visits": row["visits"],
                "load_profile": row["load_profile"],
                "wc": wc,
                "wo": wo,
                "normalized_wc": normalized[0],
                "normalized_wo": normalized[1],
                "overtime_penalty": penalty,
                "effective_continuity_to_overtime_ratio": (
                    wc / (wo * penalty) if wo and penalty else None
                ),
                "coverage": vector[0],
                "similarity": vector[1],
                "continuity": vector[2],
                "overtime": vector[3],
                "vector": "/".join(map(str, vector)),
                "assignment_sha256": row["assignment_sha256"],
            }
        )

    by_instance: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in enriched:
        by_instance[row["instance_sha256"]].append(row)
    instance_summary = []
    scale_groups_total = 0
    scale_groups_single_vector = 0
    for instance_hash, group in sorted(by_instance.items()):
        vectors = {row["vector"] for row in group}
        assignments = {
            row["assignment_sha256"] for row in group if row["assignment_sha256"]
        }
        by_scale: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
        for row in group:
            by_scale[(row["normalized_wc"], row["normalized_wo"])].append(row)
        repeated_scale_groups = [items for items in by_scale.values() if len(items) >= 2]
        stable_scale_groups = sum(
            len({item["vector"] for item in items}) == 1
            for items in repeated_scale_groups
        )
        scale_groups_total += len(repeated_scale_groups)
        scale_groups_single_vector += stable_scale_groups
        ordered = sorted(
            group,
            key=lambda item: (
                item["effective_continuity_to_overtime_ratio"],
                item["wc"], item["wo"],
            ),
        )
        transitions = sum(
            left["vector"] != right["vector"]
            for left, right in zip(ordered, ordered[1:])
        )
        first = group[0]
        instance_summary.append(
            {
                "instance": first["instance"],
                "instance_sha256": instance_hash,
                "users": first["users"],
                "agents": first["agents"],
                "visits": first["visits"],
                "load_profile": first["load_profile"],
                "requested_runs": sum(row["instance_sha256"] == instance_hash for row in rows),
                "optimum_runs": len(group),
                "unique_objective_vectors": len(vectors),
                "unique_assignments": len(assignments),
                "weight_order_transitions": transitions,
                "repeated_scale_groups": len(repeated_scale_groups),
                "scale_groups_with_one_vector": stable_scale_groups,
            }
        )

    by_weight: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in enriched:
        by_weight[(row["wc"], row["wo"])].append(row)
    weight_summary = []
    for (wc, wo), group in sorted(by_weight.items()):
        counts = Counter(row["vector"] for row in group)
        weight_summary.append(
            {
                "wc": wc,
                "wo": wo,
                "optimum_runs": len(group),
                "unique_objective_vectors_across_instances": len(counts),
                "most_common_vector": counts.most_common(1)[0][0] if counts else None,
                "most_common_vector_runs": counts.most_common(1)[0][1] if counts else 0,
                "median_effective_continuity_to_overtime_ratio": statistics.median(
                    row["effective_continuity_to_overtime_ratio"] for row in group
                ) if group else None,
            }
        )

    _write(output_dir / "weight_optimum_runs.csv", enriched)
    _write(output_dir / "weight_instance_stability.csv", instance_summary)
    _write(output_dir / "weight_summary.csv", weight_summary)
    result = {
        "runs": len(rows),
        "optimum_runs": len(optimum),
        "timeout_runs": sum(row["status"].startswith("TIMEOUT") for row in rows),
        "hard_errors": sum(row["status"] not in ALLOWED for row in rows),
        "unverified_optimum": sum(
            row["status"] == "OPTIMUM" and row["verified"] != "True" for row in rows
        ),
        "instances": len(by_instance),
        "instances_with_multiple_vectors": sum(
            row["unique_objective_vectors"] >= 2 for row in instance_summary
        ),
        "maximum_vectors_per_instance": max(
            (row["unique_objective_vectors"] for row in instance_summary), default=0
        ),
        "repeated_scale_groups": scale_groups_total,
        "scale_groups_with_one_vector": scale_groups_single_vector,
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
