#!/usr/bin/env python3
"""Analyze corrected-v2 objective and priority-order results."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any, Iterable


REFERENCE = ("totalizer", "both", "slot-service")
PROVED = {"OPTIMUM", "UNSAT", "UNSATISFIABLE"}
ALLOWED = PROVED | {"TIMEOUT", "TIMEOUT_FEASIBLE"}
METRICS = ("similarity", "continuity", "overtime")


def _number(value: Any) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _true(value: Any) -> bool:
    return str(value).lower() == "true"


def _median(rows: Iterable[dict[str, Any]], key: str) -> float | None:
    values = [value for row in rows if (value := _number(row.get(key))) is not None]
    return statistics.median(values) if values else None


def _par2(rows: Iterable[dict[str, Any]]) -> float | None:
    values = []
    for row in rows:
        elapsed = _number(row.get("elapsed_seconds"))
        timeout = _number(row.get("timeout_seconds"))
        if row.get("status") in PROVED and elapsed is not None:
            values.append(elapsed)
        elif timeout is not None:
            values.append(2 * timeout)
    return statistics.fmean(values) if values else None


def _read(result_dir: Path) -> list[dict[str, str]]:
    validation = json.loads(
        (result_dir / "validation.json").read_text(encoding="utf-8")
    )
    if validation.get("complete") is not True:
        raise ValueError(f"incomplete corrected-v2 campaign: {result_dir}")
    with (result_dir / "runs.csv").open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def analyze(result_dir: Path, output_dir: Path) -> dict[str, Any]:
    rows = _read(result_dir)
    indexed: dict[str, dict[str, dict[str, str]]] = {}
    configuration_errors = 0
    load_profile_errors = 0
    duplicate_logical_keys = 0
    for row in rows:
        configuration = (row["cardinality"], row["implied"], row["symmetry"])
        configuration_errors += configuration != REFERENCE
        load_profile_errors += row.get("load_profile") != "critical"
        methods = indexed.setdefault(row["instance_sha256"], {})
        duplicate_logical_keys += row["method"] in methods
        methods[row["method"]] = row

    pairs = []
    missing_pairs = 0
    for instance, methods in sorted(indexed.items()):
        weighted = methods.get("weighted")
        lex = methods.get("lex-cos")
        if weighted is None or lex is None:
            missing_pairs += 1
            continue
        both_optimum = weighted["status"] == lex["status"] == "OPTIMUM"
        item: dict[str, Any] = {
            "instance_sha256": instance,
            "instance": weighted["instance"],
            "weighted_status": weighted["status"],
            "lex_cos_status": lex["status"],
            "both_optimum": both_optimum,
        }
        for metric in METRICS:
            left = _number(weighted.get(metric))
            right = _number(lex.get(metric))
            item[f"weighted_{metric}"] = left
            item[f"lex_cos_{metric}"] = right
            item[f"lex_minus_weighted_{metric}"] = (
                right - left
                if both_optimum and left is not None and right is not None
                else None
            )
        pairs.append(item)

    policy_rows = []
    for method in ("weighted", "lex-cos", "lex-overtime"):
        group = [row for row in rows if row["method"] == method]
        optimum = [row for row in group if row["status"] == "OPTIMUM"]
        policy_rows.append(
            {
                "method": method,
                "runs": len(group),
                "optimum_runs": len(optimum),
                "unsat_runs": sum(
                    row["status"] in {"UNSAT", "UNSATISFIABLE"}
                    for row in group
                ),
                "timeout_runs": sum(
                    row["status"].startswith("TIMEOUT") for row in group
                ),
                "par2_seconds": _par2(group),
                "median_peak_rss_mb": _median(group, "peak_rss_mb"),
                "median_similarity": _median(optimum, "similarity"),
                "median_continuity": _median(optimum, "continuity"),
                "median_overtime": _median(optimum, "overtime"),
            }
        )

    optimum_pairs = [row for row in pairs if row["both_optimum"]]
    paired_summary = [
        {
            "pairs": len(pairs),
            "both_optimum_pairs": len(optimum_pairs),
            "median_similarity_change": _median(
                optimum_pairs, "lex_minus_weighted_similarity"
            ),
            "median_continuity_change": _median(
                optimum_pairs, "lex_minus_weighted_continuity"
            ),
            "median_overtime_change": _median(
                optimum_pairs, "lex_minus_weighted_overtime"
            ),
            "continuity_improved": sum(
                row["lex_minus_weighted_continuity"] < 0 for row in optimum_pairs
            ),
            "continuity_equal": sum(
                row["lex_minus_weighted_continuity"] == 0 for row in optimum_pairs
            ),
            "continuity_worsened": sum(
                row["lex_minus_weighted_continuity"] > 0 for row in optimum_pairs
            ),
            "overtime_decreased": sum(
                row["lex_minus_weighted_overtime"] < 0 for row in optimum_pairs
            ),
            "overtime_equal": sum(
                row["lex_minus_weighted_overtime"] == 0 for row in optimum_pairs
            ),
            "overtime_increased": sum(
                row["lex_minus_weighted_overtime"] > 0 for row in optimum_pairs
            ),
        }
    ]

    result = {
        "runs": len(rows),
        "instances": len(indexed),
        "paired_instances": len(pairs),
        "missing_pairs": missing_pairs,
        "configuration_errors": configuration_errors,
        "load_profile_errors": load_profile_errors,
        "duplicate_logical_keys": duplicate_logical_keys,
        "hard_errors": sum(row["status"] not in ALLOWED for row in rows),
        "unverified_optima": sum(
            row["status"] == "OPTIMUM" and not _true(row.get("verified"))
            for row in rows
        ),
        "methods": {
            method: sum(row["method"] == method for row in rows)
            for method in ("weighted", "lex-cos", "lex-overtime")
        },
    }
    result["valid"] = (
        result["runs"] == 144
        and result["instances"] == 48
        and result["paired_instances"] == 48
        and result["missing_pairs"] == 0
        and result["configuration_errors"] == 0
        and result["load_profile_errors"] == 0
        and result["duplicate_logical_keys"] == 0
        and result["hard_errors"] == 0
        and result["unverified_optima"] == 0
        and result["methods"]
        == {"weighted": 48, "lex-cos": 48, "lex-overtime": 48}
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _write(output_dir / "corrected_policy_pairs.csv", pairs)
    _write(output_dir / "corrected_policy_summary.csv", policy_rows)
    _write(output_dir / "corrected_paired_summary.csv", paired_summary)
    (output_dir / "corrected_validation.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("experiments/results/gcp_corrected_primary"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/results/gcp_corrected_analysis"),
    )
    arguments = parser.parse_args()
    try:
        result = analyze(arguments.results, arguments.output_dir)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
