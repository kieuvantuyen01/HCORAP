#!/usr/bin/env python3
"""Gate the non-measured corrected-v2 commercial-solver calibration.

This gate prevents a 192-run measured campaign from starting unless both exact
solvers handle all three objective rules on at least six of eight calibration
instances and agree wherever both prove optimality.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


NONTECHNICAL = {"OPTIMUM", "INFEASIBLE", "TIMEOUT", "TIMEOUT_FEASIBLE"}


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _normalized(value: Any) -> Any:
    try:
        return Decimal(str(value)).normalize()
    except (InvalidOperation, ValueError):
        return value


def _true(value: Any) -> bool:
    return str(value).lower() == "true"


def evaluate(result_dir: Path, gates_path: Path) -> dict[str, Any]:
    validation = _json(result_dir / "validation.json")
    with (result_dir / "runs.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    gates = _json(gates_path)
    methods = set(gates["expected_methods"])
    backends = set(gates["expected_backends"])

    indexed: dict[tuple[str, str], dict[str, dict[str, str]]] = defaultdict(dict)
    duplicate_keys = 0
    for row in rows:
        key = (row["backend"], row["instance_sha256"])
        duplicate_keys += row["method"] in indexed[key]
        indexed[key][row["method"]] = row

    all_policy_optimum = Counter()
    matrix_errors = 0
    for (backend, _instance), group in indexed.items():
        matrix_errors += backend not in backends or set(group) != methods
        if set(group) == methods and all(
            group[method]["status"] == "OPTIMUM" for method in methods
        ):
            all_policy_optimum[backend] += 1

    status_disagreements = 0
    objective_disagreements = 0
    by_instance_method: dict[tuple[str, str], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        by_instance_method[(row["instance_sha256"], row["method"])][
            row["backend"]
        ] = row
    for (_instance, method), group in by_instance_method.items():
        if set(group) != backends:
            continue
        left, right = (group[backend] for backend in sorted(backends))
        status_disagreements += left["status"] != right["status"]
        if left["status"] == right["status"] == "OPTIMUM":
            fields = (
                ("coverage", "weighted_reference_score")
                if method == "weighted"
                else ("coverage", "similarity", "continuity", "overtime")
            )
            objective_disagreements += any(
                _normalized(left[field]) != _normalized(right[field])
                for field in fields
            )

    technical_errors = sum(row["status"] not in NONTECHNICAL for row in rows)
    unverified_optima = sum(
        row["status"] == "OPTIMUM" and not _true(row.get("verified"))
        for row in rows
    )
    checks = {
        "collector_complete": validation.get("complete") is True,
        "expected_runs": len(rows) == int(gates["expected_runs"]),
        "expected_instances": len({row["instance_sha256"] for row in rows})
        == int(gates["expected_instances"]),
        "expected_backends": {row["backend"] for row in rows} == backends,
        "matrix_complete": matrix_errors == 0 and duplicate_keys == 0,
        "all_policy_optimum_per_backend": all(
            all_policy_optimum[backend]
            >= int(gates["minimum_all_policy_optimum_instances_per_backend"])
            for backend in backends
        ),
        "status_agreement": status_disagreements
        <= int(gates["maximum_status_disagreements"]),
        "objective_agreement": objective_disagreements
        <= int(gates["maximum_objective_disagreements"]),
        "technical_errors": technical_errors
        <= int(gates["maximum_technical_errors"]),
        "unverified_optima": unverified_optima
        <= int(gates["maximum_unverified_optima"]),
    }
    return {
        "scope": "corrected-v2-commercial-calibration",
        "measured": False,
        "runs": len(rows),
        "instances": len({row["instance_sha256"] for row in rows}),
        "status_counts": dict(sorted(Counter(row["status"] for row in rows).items())),
        "all_policy_optimum_instances_by_backend": dict(sorted(all_policy_optimum.items())),
        "duplicate_keys": duplicate_keys,
        "matrix_errors": matrix_errors,
        "status_disagreements": status_disagreements,
        "objective_disagreements": objective_disagreements,
        "technical_errors": technical_errors,
        "unverified_optima": unverified_optima,
        "checks": checks,
        "gates": gates,
        "pass": all(checks.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("experiments/results/gcp_commercial_corrected_calibration"),
    )
    parser.add_argument(
        "--gates",
        type=Path,
        default=Path(
            "experiments/configs/corrected_commercial_calibration_gates.json"
        ),
    )
    arguments = parser.parse_args()
    try:
        result = evaluate(arguments.results, arguments.gates)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        parser.error(str(error))
    output = arguments.results / "calibration_decision.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
