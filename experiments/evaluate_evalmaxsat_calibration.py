#!/usr/bin/env python3
"""Gate the four non-measured EvalMaxSAT lexicographic calibration rows."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


ALLOWED = {"OPTIMUM", "TIMEOUT", "TIMEOUT_FEASIBLE"}


def evaluate(result_dir: Path) -> dict[str, Any]:
    validation = json.loads(
        (result_dir / "validation.json").read_text(encoding="utf-8")
    )
    with (result_dir / "runs.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    optimum = sum(row["status"] == "OPTIMUM" for row in rows)
    technical_errors = sum(row["status"] not in ALLOWED for row in rows)
    unverified_optima = sum(
        row["status"] == "OPTIMUM" and row.get("verified") != "True" for row in rows
    )
    checks = {
        "collector_complete": validation.get("complete") is True,
        "runs": len(rows) == 4,
        "minimum_optimum": optimum >= 2,
        "technical_errors": technical_errors == 0,
        "verified_optima": unverified_optima == 0,
    }
    return {
        "scope": "evalmaxsat-lexicographic-calibration",
        "measured": False,
        "runs": len(rows),
        "optimum_runs": optimum,
        "status_counts": dict(sorted(Counter(row["status"] for row in rows).items())),
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
        default=Path("experiments/results/gcp_evalmaxsat_lex_calibration"),
    )
    arguments = parser.parse_args()
    try:
        result = evaluate(arguments.results)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        parser.error(str(error))
    output = arguments.results / "calibration_decision.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
