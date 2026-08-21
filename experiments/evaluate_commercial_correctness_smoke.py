#!/usr/bin/env python3
"""Validate three-backend commercial correctness smoke results."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


BACKENDS = {"gurobi-mip", "cplex-mip", "reference-enumerator"}
METHODS = {"weighted", "lex-cos"}
EXACT = {"OPTIMUM", "INFEASIBLE"}


def _normalized(value: Any) -> Any:
    try:
        return Decimal(str(value)).normalize()
    except (InvalidOperation, ValueError):
        return value


def _true(value: Any) -> bool:
    return str(value).lower() == "true"


def evaluate(result_dir: Path) -> dict[str, Any]:
    validation = json.loads(
        (result_dir / "validation.json").read_text(encoding="utf-8")
    )
    with (result_dir / "runs.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["instance_sha256"], row["method"])].append(row)

    incomplete_groups = 0
    status_disagreements = 0
    objective_disagreements = 0
    unverified_optima = 0
    for (_instance, method), group in groups.items():
        if (
            method not in METHODS
            or len(group) != 3
            or {row["backend"] for row in group} != BACKENDS
        ):
            incomplete_groups += 1
            continue
        statuses = {row["status"] for row in group}
        if len(statuses) != 1:
            status_disagreements += 1
            continue
        if statuses == {"OPTIMUM"}:
            unverified_optima += sum(not _true(row.get("verified")) for row in group)
            fields = (
                ("coverage", "weighted_reference_score")
                if method == "weighted"
                else ("coverage", "similarity", "continuity", "overtime")
            )
            vectors = {
                tuple(_normalized(row[field]) for field in fields) for row in group
            }
            objective_disagreements += len(vectors) != 1

    technical_errors = sum(row["status"] not in EXACT for row in rows)
    checks = {
        "collector_complete": validation.get("complete") is True,
        "runs": len(rows) == 18,
        "groups": len(groups) == 6 and incomplete_groups == 0,
        "status_agreement": status_disagreements == 0,
        "objective_agreement": objective_disagreements == 0,
        "technical_errors": technical_errors == 0,
        "verified_optima": unverified_optima == 0,
    }
    return {
        "scope": "commercial-correctness-smoke",
        "measured": False,
        "runs": len(rows),
        "groups": len(groups),
        "status_counts": dict(sorted(Counter(row["status"] for row in rows).items())),
        "incomplete_groups": incomplete_groups,
        "status_disagreements": status_disagreements,
        "objective_disagreements": objective_disagreements,
        "technical_errors": technical_errors,
        "unverified_optima": unverified_optima,
        "checks": checks,
        "pass": all(checks.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("experiments/results/gcp_commercial_correctness_smoke"),
    )
    arguments = parser.parse_args()
    try:
        result = evaluate(arguments.results)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        parser.error(str(error))
    output = arguments.results / "smoke_decision.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
