#!/usr/bin/env python3
"""Compare MaxSAT with exact Gurobi/CPLEX objectives on the commercial subset."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


REFERENCE = ("totalizer", "both", "slot-service")
TECHNICAL_OK = {"OPTIMUM", "TIMEOUT", "TIMEOUT_FEASIBLE"}


def _read_rows(result_dir: Path) -> list[dict[str, str]]:
    validation = json.loads(
        (result_dir / "validation.json").read_text(encoding="utf-8")
    )
    if not validation.get("complete"):
        raise ValueError(f"incomplete campaign: {result_dir}")
    with (result_dir / "runs.csv").open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _true(value: Any) -> bool:
    return str(value).lower() == "true"


def _normalized(value: Any) -> Any:
    """Make numerically equal CSV renderings (for example 4 and 4.0) equal."""
    try:
        return Decimal(str(value)).normalize()
    except (InvalidOperation, ValueError):
        return value


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def analyze(arguments: argparse.Namespace) -> dict[str, Any]:
    commercial = _read_rows(arguments.commercial_results)
    commercial_instances = {row["instance_sha256"] for row in commercial}
    weighted = [
        row
        for row in _read_rows(arguments.weighted_maxsat_results)
        if row["instance_sha256"] in commercial_instances
        and (row["cardinality"], row["implied"], row["symmetry"]) == REFERENCE
        and row["method"] == "weighted"
    ]
    lex: list[dict[str, str]] = []
    if arguments.scope == "full":
        lex = [
            row
            for row in _read_rows(arguments.lex_maxsat_results)
            if row["instance_sha256"] in commercial_instances
            and (row["cardinality"], row["implied"], row["symmetry"]) == REFERENCE
            and row["method"] == "lex-cos"
        ]

    maxsat = {(row["instance_sha256"], row["method"]): row for row in weighted + lex}
    mip: dict[tuple[str, str], dict[str, dict[str, str]]] = {}
    for row in commercial:
        key = (row["instance_sha256"], row["method"])
        mip.setdefault(key, {})[row["backend"]] = row

    methods = ("weighted", "lex-cos") if arguments.scope == "full" else ("weighted",)
    output = []
    missing_groups = 0
    for instance in sorted(commercial_instances):
        for method in methods:
            key = (instance, method)
            maxsat_row = maxsat.get(key)
            backends = mip.get(key, {})
            gurobi = backends.get("gurobi-mip")
            cplex = backends.get("cplex-mip")
            if maxsat_row is None or gurobi is None or cplex is None:
                missing_groups += 1
                continue
            rows = (maxsat_row, gurobi, cplex)
            all_exact = all(row["status"] == "OPTIMUM" for row in rows)
            if method == "weighted":
                keys = ("coverage", "weighted_reference_score")
            else:
                keys = ("coverage", "similarity", "continuity", "overtime")
            vectors = {
                tuple(_normalized(row[field]) for field in keys) for row in rows
            }
            agreement = len(vectors) == 1 if all_exact else None
            output.append(
                {
                    "instance_sha256": instance,
                    "instance": maxsat_row["instance"],
                    "method": method,
                    "maxsat_status": maxsat_row["status"],
                    "gurobi_status": gurobi["status"],
                    "cplex_status": cplex["status"],
                    "all_exact_optimum": all_exact,
                    "objective_agreement": agreement,
                    "comparison_fields": "|".join(keys),
                    "maxsat_objective": str(tuple(maxsat_row[field] for field in keys)),
                    "gurobi_objective": str(tuple(gurobi[field] for field in keys)),
                    "cplex_objective": str(tuple(cplex[field] for field in keys)),
                }
            )

    source_rows = [*weighted, *lex, *commercial]
    expected_groups = len(commercial_instances) * len(methods)
    status_counts = Counter(row["status"] for row in source_rows)
    result = {
        "scope": arguments.scope,
        "commercial_instances": len(commercial_instances),
        "expected_groups": expected_groups,
        "complete_groups": len(output),
        "missing_groups": missing_groups,
        "all_exact_groups": sum(_true(row["all_exact_optimum"]) for row in output),
        "agreement_groups": sum(_true(row["objective_agreement"]) for row in output),
        "objective_disagreements": sum(
            str(row["objective_agreement"]).lower() == "false" for row in output
        ),
        "technical_errors": sum(
            count for status, count in status_counts.items() if status not in TECHNICAL_OK
        ),
        "unverified_optima": sum(
            row["status"] == "OPTIMUM" and not _true(row.get("verified"))
            for row in source_rows
        ),
        "status_counts": dict(sorted(status_counts.items())),
    }
    result["valid"] = (
        result["commercial_instances"] == 20
        and result["complete_groups"] == expected_groups
        and result["missing_groups"] == 0
        and result["technical_errors"] == 0
        and result["unverified_optima"] == 0
        and result["objective_disagreements"] == 0
    )
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    _write(arguments.output_dir / "cross_paradigm_agreement.csv", output)
    (arguments.output_dir / "cross_paradigm_validation.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weighted-maxsat-results",
        type=Path,
        default=Path("experiments/results/gcp_maxsat_commercial_validation"),
    )
    parser.add_argument(
        "--lex-maxsat-results",
        type=Path,
        default=Path("experiments/results/gcp_maxsat_commercial_validation"),
    )
    parser.add_argument(
        "--commercial-results",
        type=Path,
        default=Path("experiments/results/gcp_commercial_original"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/results/gcp_cross_paradigm_analysis"),
    )
    parser.add_argument("--scope", choices=("full", "weighted-only"), default="full")
    arguments = parser.parse_args()
    try:
        result = analyze(arguments)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
