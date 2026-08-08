#!/usr/bin/env python3
"""Build confirmatory factorial and lexicographic manuscript tables."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


PROVED = {"OPTIMUM", "UNSAT", "UNSATISFIABLE"}
ALLOWED = PROVED | {"TIMEOUT", "TIMEOUT_FEASIBLE"}
BASELINE = ("sorting-network", "none", "none")
PROPOSED = ("totalizer", "both", "slot-service")
CONFIG_KEYS = ("cardinality", "implied", "symmetry")
METRICS = ("coverage", "similarity", "continuity", "overtime")


def _rows(result_dir: Path) -> list[dict[str, str]]:
    validation = json.loads(
        (result_dir / "validation.json").read_text(encoding="utf-8")
    )
    if not validation.get("complete"):
        raise ValueError(f"incomplete campaign: {result_dir}")
    with (result_dir / "runs.csv").open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _number(value: Any) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _configuration(row: dict[str, str]) -> tuple[str, str, str]:
    return tuple(row[key] for key in CONFIG_KEYS)  # type: ignore[return-value]


def _median(rows: Iterable[dict[str, Any]], key: str) -> float | None:
    values = [value for row in rows if (value := _number(row.get(key))) is not None]
    return statistics.median(values) if values else None


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _factorial(
    rows: list[dict[str, str]], output_dir: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    baseline = {
        row["instance_sha256"]: row
        for row in rows
        if _configuration(row) == BASELINE
    }
    paired = []
    for row in rows:
        left = baseline[row["instance_sha256"]]
        both_optimum = left["status"] == row["status"] == "OPTIMUM"
        both_proved = left["status"] in PROVED and row["status"] in PROVED
        left_elapsed = _number(left["elapsed_seconds"])
        right_elapsed = _number(row["elapsed_seconds"])
        objective_match = None
        if both_optimum:
            objective_match = (
                left["coverage"], left["weighted_reference_score"]
            ) == (row["coverage"], row["weighted_reference_score"])
        paired.append(
            {
                "instance": row["instance"],
                "instance_sha256": row["instance_sha256"],
                "users": row["users"],
                "agents": row["agents"],
                "visits": row["visits"],
                "cardinality": row["cardinality"],
                "implied": row["implied"],
                "symmetry": row["symmetry"],
                "baseline_status": left["status"],
                "configuration_status": row["status"],
                "both_proved": both_proved,
                "both_optimum": both_optimum,
                "objective_match": objective_match,
                "baseline_elapsed_seconds": left_elapsed,
                "configuration_elapsed_seconds": right_elapsed,
                "speedup_baseline_over_configuration": (
                    left_elapsed / right_elapsed
                    if both_proved and left_elapsed is not None
                    and right_elapsed not in (None, 0) else None
                ),
                "variables_difference": (
                    (_number(row["variables_max"]) or 0)
                    - (_number(left["variables_max"]) or 0)
                ),
                "hard_clauses_difference": (
                    (_number(row["hard_clauses_max"]) or 0)
                    - (_number(left["hard_clauses_max"]) or 0)
                ),
                "soft_clauses_difference": (
                    (_number(row["soft_clauses_max"]) or 0)
                    - (_number(left["soft_clauses_max"]) or 0)
                ),
            }
        )

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    source_grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for pair, source in zip(paired, rows):
        grouped[_configuration(source)].append(pair)
        source_grouped[_configuration(source)].append(source)
    summary = []
    for configuration in sorted(grouped):
        group = grouped[configuration]
        source = source_grouped[configuration]
        summary.append(
            {
                **dict(zip(CONFIG_KEYS, configuration)),
                "runs": len(group),
                "optimum_runs": sum(row["status"] == "OPTIMUM" for row in source),
                "proved_runs": sum(row["status"] in PROVED for row in source),
                "timeout_runs": sum(row["status"].startswith("TIMEOUT") for row in source),
                "paired_proved_runs": sum(bool(row["both_proved"]) for row in group),
                "paired_optimum_runs": sum(bool(row["both_optimum"]) for row in group),
                "objective_mismatches": sum(row["objective_match"] is False for row in group),
                "median_speedup": _median(group, "speedup_baseline_over_configuration"),
                "median_variables_difference": _median(group, "variables_difference"),
                "median_hard_clauses_difference": _median(group, "hard_clauses_difference"),
                "median_soft_clauses_difference": _median(group, "soft_clauses_difference"),
            }
        )
    _write(output_dir / "factorial_paired_runs.csv", paired)
    _write(output_dir / "factorial_summary.csv", summary)
    return paired, summary


def _lex_vs_weighted(
    weighted: list[dict[str, str]], lex: list[dict[str, str]], output_dir: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected_weighted = {
        (row["instance_sha256"], _configuration(row)): row
        for row in weighted
        if _configuration(row) in {BASELINE, PROPOSED}
    }
    pairs = []
    for lex_row in lex:
        key = (lex_row["instance_sha256"], _configuration(lex_row))
        if key not in selected_weighted:
            raise ValueError(f"LEX-COS row has no matching weighted row: {key}")
        weighted_row = selected_weighted[key]
        both_optimum = weighted_row["status"] == lex_row["status"] == "OPTIMUM"
        item: dict[str, Any] = {
            "instance": lex_row["instance"],
            "instance_sha256": lex_row["instance_sha256"],
            "users": lex_row["users"],
            "agents": lex_row["agents"],
            "visits": lex_row["visits"],
            **dict(zip(CONFIG_KEYS, _configuration(lex_row))),
            "weighted_status": weighted_row["status"],
            "lex_cos_status": lex_row["status"],
            "both_optimum": both_optimum,
            "weighted_elapsed_seconds": _number(weighted_row["elapsed_seconds"]),
            "lex_cos_elapsed_seconds": _number(lex_row["elapsed_seconds"]),
        }
        for metric in METRICS:
            left = _number(weighted_row[metric])
            right = _number(lex_row[metric])
            item[f"weighted_{metric}"] = left
            item[f"lex_cos_{metric}"] = right
            item[f"lex_minus_weighted_{metric}"] = (
                right - left if both_optimum and left is not None and right is not None else None
            )
        pairs.append(item)

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in pairs:
        grouped[tuple(row[key] for key in CONFIG_KEYS)].append(row)
    summary = []
    for configuration, group in sorted(grouped.items()):
        optimum = [row for row in group if row["both_optimum"]]
        summary.append(
            {
                **dict(zip(CONFIG_KEYS, configuration)),
                "pairs": len(group),
                "both_optimum_pairs": len(optimum),
                "continuity_improved": sum(row["lex_minus_weighted_continuity"] < 0 for row in optimum),
                "continuity_equal": sum(row["lex_minus_weighted_continuity"] == 0 for row in optimum),
                "overtime_increased": sum(row["lex_minus_weighted_overtime"] > 0 for row in optimum),
                "median_similarity_change": _median(optimum, "lex_minus_weighted_similarity"),
                "median_continuity_change": _median(optimum, "lex_minus_weighted_continuity"),
                "median_overtime_change": _median(optimum, "lex_minus_weighted_overtime"),
            }
        )
    _write(output_dir / "lex_confirmatory_pairs.csv", pairs)
    _write(output_dir / "lex_confirmatory_summary.csv", summary)
    return pairs, summary


def _policy_sensitivity(
    lex_cos: list[dict[str, str]], lex_ocs: list[dict[str, str]], output_dir: Path
) -> list[dict[str, Any]]:
    cos = {
        (row["instance_sha256"], _configuration(row)): row for row in lex_cos
    }
    pairs = []
    for ocs in lex_ocs:
        key = (ocs["instance_sha256"], _configuration(ocs))
        if key not in cos:
            raise ValueError(f"LEX-OCS row has no matching LEX-COS row: {key}")
        current = cos[key]
        both_optimum = current["status"] == ocs["status"] == "OPTIMUM"
        row: dict[str, Any] = {
            "instance": ocs["instance"],
            "instance_sha256": ocs["instance_sha256"],
            **dict(zip(CONFIG_KEYS, _configuration(ocs))),
            "lex_cos_status": current["status"],
            "lex_ocs_status": ocs["status"],
            "both_optimum": both_optimum,
        }
        for metric in METRICS:
            left = _number(current[metric])
            right = _number(ocs[metric])
            row[f"lex_cos_{metric}"] = left
            row[f"lex_ocs_{metric}"] = right
            row[f"ocs_minus_cos_{metric}"] = (
                right - left if both_optimum and left is not None and right is not None else None
            )
        row["same_objective_vector"] = both_optimum and all(
            row[f"ocs_minus_cos_{metric}"] == 0 for metric in METRICS
        )
        pairs.append(row)
    _write(output_dir / "lex_policy_sensitivity_pairs.csv", pairs)
    return pairs


def analyze(arguments: argparse.Namespace) -> dict[str, Any]:
    output_dir = arguments.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    ablation = _rows(arguments.ablation_results)
    weighted = _rows(arguments.weighted_results)
    lex = _rows(arguments.lex_results)
    sensitivity = _rows(arguments.sensitivity_results)
    scalability = _rows(arguments.scalability_results)
    scalability_lex = [row for row in scalability if row["method"] == "lex-cos"]
    factorial_pairs, factorial_summary = _factorial(ablation, output_dir)
    lex_pairs, lex_summary = _lex_vs_weighted(weighted, lex, output_dir)
    sensitivity_pairs = _policy_sensitivity(
        scalability_lex, sensitivity, output_dir
    )
    all_rows = [*ablation, *weighted, *lex, *sensitivity, *scalability]
    counts = {
        "ablation": len(ablation), "weighted": len(weighted),
        "lex_confirmatory": len(lex), "lex_sensitivity": len(sensitivity),
        "lex_scalability": len(scalability),
    }
    expected = {
        "ablation": 1280, "weighted": 1600, "lex_confirmatory": 560,
        "lex_sensitivity": 160, "lex_scalability": 320,
    }
    result = {
        "counts": counts,
        "expected_counts": expected,
        "count_match": counts == expected,
        "hard_errors": sum(row["status"] not in ALLOWED for row in all_rows),
        "unverified_optimum": sum(
            row["status"] == "OPTIMUM" and row["verified"] != "True"
            for row in all_rows
        ),
        "factorial_objective_mismatches": sum(
            row["objective_match"] is False for row in factorial_pairs
        ),
        "factorial_configurations": len(factorial_summary),
        "lex_confirmatory_pairs": len(lex_pairs),
        "lex_confirmatory_configurations": len(lex_summary),
        "policy_sensitivity_pairs": len(sensitivity_pairs),
        "policy_sensitivity_both_optimum": sum(
            bool(row["both_optimum"]) for row in sensitivity_pairs
        ),
        "policy_sensitivity_same_vector": sum(
            bool(row["same_objective_vector"]) for row in sensitivity_pairs
        ),
    }
    result["valid"] = (
        result["count_match"]
        and result["hard_errors"] == 0
        and result["unverified_optimum"] == 0
        and result["factorial_objective_mismatches"] == 0
        and result["factorial_configurations"] == 8
        and result["lex_confirmatory_configurations"] == 2
        and result["policy_sensitivity_pairs"] == 160
    )
    (output_dir / "analysis_validation.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ablation-results", type=Path, default=Path("experiments/results/gcp_original_ablation"))
    parser.add_argument("--weighted-results", type=Path, default=Path("experiments/results/gcp_original_weighted_primary"))
    parser.add_argument("--lex-results", type=Path, default=Path("experiments/results/gcp_original_lex_primary"))
    parser.add_argument("--sensitivity-results", type=Path, default=Path("experiments/results/gcp_original_lex_sensitivity"))
    parser.add_argument("--scalability-results", type=Path, default=Path("experiments/results/gcp_lex_scalability_screen"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/results/gcp_primary_analysis"))
    arguments = parser.parse_args()
    try:
        result = analyze(arguments)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
