#!/usr/bin/env python3
"""Evaluate whether Totalizer-only transfers to corrected-v2 lexicographic runs."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


TOTALIZER_ONLY = ("totalizer", "none", "none")
FULL = ("totalizer", "both", "slot-service")
CONFIG_KEYS = ("cardinality", "implied", "symmetry")
ALLOWED = {"OPTIMUM", "UNSAT", "UNSATISFIABLE", "TIMEOUT", "TIMEOUT_FEASIBLE"}
PROVED = {"OPTIMUM", "UNSAT", "UNSATISFIABLE"}


def _truth(value: Any) -> bool:
    return str(value).lower() == "true"


def _number(value: Any) -> float:
    if value in (None, ""):
        raise ValueError(f"expected a numeric value, got {value!r}")
    return float(value)


def _configuration(row: dict[str, str]) -> tuple[str, str, str]:
    return tuple(row[key] for key in CONFIG_KEYS)  # type: ignore[return-value]


def _par2(row: dict[str, str]) -> float:
    if row["status"] in PROVED:
        return _number(row["elapsed_seconds"])
    return 2 * _number(row["timeout_seconds"])


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write an empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(result_dir: Path, output_dir: Path, expected_instances: int) -> dict[str, Any]:
    validation = json.loads((result_dir / "validation.json").read_text(encoding="utf-8"))
    if validation.get("complete") is not True:
        raise ValueError(f"campaign is incomplete: {result_dir}")
    with (result_dir / "runs.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    indexed: dict[tuple[str, tuple[str, str, str]], dict[str, str]] = {}
    duplicates = 0
    for row in rows:
        key = (row["instance_sha256"], _configuration(row))
        duplicates += key in indexed
        indexed[key] = row

    instances = sorted({row["instance_sha256"] for row in rows})
    pair_rows: list[dict[str, Any]] = []
    missing_pairs = 0
    for instance in instances:
        totalizer = indexed.get((instance, TOTALIZER_ONLY))
        full = indexed.get((instance, FULL))
        if totalizer is None or full is None:
            missing_pairs += 1
            continue
        totalizer_stage = int(float(totalizer.get("stage_count") or 0))
        full_stage = int(float(full.get("stage_count") or 0))
        totalizer_par2 = _par2(totalizer)
        full_par2 = _par2(full)
        pair_rows.append(
            {
                "instance_sha256": instance,
                "instance": totalizer["instance"],
                "users": totalizer.get("users"),
                "agents": totalizer.get("agents"),
                "visits": totalizer.get("visits"),
                "seed": totalizer.get("seed"),
                "totalizer_only_status": totalizer["status"],
                "full_status": full["status"],
                "totalizer_only_stage_count": totalizer_stage,
                "full_stage_count": full_stage,
                "stage_advantage": totalizer_stage - full_stage,
                "totalizer_only_par2_seconds": totalizer_par2,
                "full_par2_seconds": full_par2,
                "par2_difference_seconds": totalizer_par2 - full_par2,
            }
        )

    if not pair_rows:
        raise ValueError("campaign contains no complete configuration pairs")

    totalizer_optima = sum(row["totalizer_only_status"] == "OPTIMUM" for row in pair_rows)
    full_optima = sum(row["full_status"] == "OPTIMUM" for row in pair_rows)
    net_extra_optima = totalizer_optima - full_optima
    stage_wins = sum(int(row["stage_advantage"]) > 0 for row in pair_rows)
    totalizer_par2 = statistics.fmean(
        float(row["totalizer_only_par2_seconds"]) for row in pair_rows
    )
    full_par2 = statistics.fmean(float(row["full_par2_seconds"]) for row in pair_rows)
    par2_improvement_pct = 100 * (full_par2 - totalizer_par2) / full_par2

    gates = {
        "at_least_two_net_extra_optima": net_extra_optima >= 2,
        "extra_completed_stage_on_four_pairs": stage_wins >= 4,
        "par2_improvement_at_least_ten_percent": par2_improvement_pct >= 10,
    }
    expected_runs = 2 * expected_instances
    structural_checks = {
        "complete": validation.get("complete") is True,
        "run_count": len(rows) == expected_runs,
        "instance_count": len(instances) == expected_instances,
        "pair_count": len(pair_rows) == expected_instances,
        "missing_pairs": missing_pairs == 0,
        "duplicate_keys": duplicates == 0,
        "method": all(row.get("method") == "lex-cos" for row in rows),
        "configurations": {_configuration(row) for row in rows}
        == {TOTALIZER_ONLY, FULL},
        "statuses": all(row.get("status") in ALLOWED for row in rows),
        "verified_optima": all(
            row["status"] != "OPTIMUM" or _truth(row.get("verified")) for row in rows
        ),
        "timeout": all(_number(row.get("timeout_seconds")) == 300 for row in rows),
    }
    result = {
        "scope": "corrected-v2-lex-encoding-transfer",
        "expected_instances": expected_instances,
        "runs": len(rows),
        "instances": len(instances),
        "totalizer_only_optima": totalizer_optima,
        "full_optima": full_optima,
        "net_extra_optima": net_extra_optima,
        "stage_wins": stage_wins,
        "totalizer_only_par2_seconds": totalizer_par2,
        "full_par2_seconds": full_par2,
        "par2_improvement_pct": par2_improvement_pct,
        "gates": gates,
        "decision": "GO" if any(gates.values()) else "STOP",
        "structural_checks": structural_checks,
        "structurally_valid": all(structural_checks.values()),
    }
    if not result["structurally_valid"]:
        result["decision"] = "INVALID"

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "lex_encoding_transfer_pairs.csv", pair_rows)
    _write_csv(
        output_dir / "lex_encoding_transfer_summary.csv",
        [
            {
                key: result[key]
                for key in (
                    "runs",
                    "instances",
                    "totalizer_only_optima",
                    "full_optima",
                    "net_extra_optima",
                    "stage_wins",
                    "totalizer_only_par2_seconds",
                    "full_par2_seconds",
                    "par2_improvement_pct",
                    "decision",
                )
            }
        ],
    )
    (output_dir / "lex_encoding_transfer_validation.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-instances", type=int, choices=(16, 48), required=True)
    arguments = parser.parse_args()
    try:
        result = analyze(arguments.results, arguments.output_dir, arguments.expected_instances)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["structurally_valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
