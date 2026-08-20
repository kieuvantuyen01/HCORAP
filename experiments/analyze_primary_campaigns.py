#!/usr/bin/env python3
"""Build confirmatory factorial and lexicographic manuscript tables."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


PROVED = {"OPTIMUM", "UNSAT", "UNSATISFIABLE"}
ALLOWED = PROVED | {"TIMEOUT", "TIMEOUT_FEASIBLE"}
BASELINE = ("sorting-network", "none", "none")
REFERENCE_COMPOSITE = ("totalizer", "both", "slot-service")
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


def _par2_value(row: dict[str, Any]) -> float | None:
    elapsed = _number(row.get("elapsed_seconds"))
    timeout = _number(row.get("timeout_seconds"))
    if row.get("status") in PROVED:
        return elapsed
    return 2 * timeout if timeout is not None else None


def _par2(rows: Iterable[dict[str, Any]]) -> float | None:
    values = [value for row in rows if (value := _par2_value(row)) is not None]
    return statistics.fmean(values) if values else None


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _bootstrap_median_ci(
    values: list[float], label: str, repetitions: int = 2000
) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    seed = int.from_bytes(hashlib.sha256(label.encode("utf-8")).digest()[:8], "big")
    generator = random.Random(seed)
    medians = [
        statistics.median(generator.choices(values, k=len(values)))
        for _ in range(repetitions)
    ]
    return _percentile(medians, 0.025), _percentile(medians, 0.975)


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
                "unsat_runs": sum(
                    row["status"] in {"UNSAT", "UNSATISFIABLE"}
                    for row in source
                ),
                "proved_runs": sum(row["status"] in PROVED for row in source),
                "timeout_runs": sum(
                    row["status"].startswith("TIMEOUT") for row in source
                ),
                "par2_seconds": _par2(source),
                "median_proved_seconds": _median(
                    [row for row in source if row["status"] in PROVED],
                    "elapsed_seconds",
                ),
                "median_peak_rss_mb": _median(source, "peak_rss_mb"),
                "median_variables": _median(source, "variables_max"),
                "median_hard_clauses": _median(source, "hard_clauses_max"),
                "median_soft_clauses": _median(source, "soft_clauses_max"),
                "paired_proved_runs": sum(bool(row["both_proved"]) for row in group),
                "paired_optimum_runs": sum(bool(row["both_optimum"]) for row in group),
                "objective_mismatches": sum(
                    row["objective_match"] is False for row in group
                ),
                "median_speedup": _median(group, "speedup_baseline_over_configuration"),
                "median_variables_difference": _median(group, "variables_difference"),
                "median_hard_clauses_difference": _median(
                    group, "hard_clauses_difference"
                ),
                "median_soft_clauses_difference": _median(
                    group, "soft_clauses_difference"
                ),
            }
        )
    _write(output_dir / "factorial_paired_runs.csv", paired)
    _write(output_dir / "factorial_summary.csv", summary)
    return paired, summary


def _factorial_contrasts(
    rows: list[dict[str, str]], output_dir: Path
) -> list[dict[str, Any]]:
    indexed = {
        (row["instance_sha256"], _configuration(row)): row for row in rows
    }
    instances = sorted({row["instance_sha256"] for row in rows})
    specifications: list[
        tuple[str, tuple[str, str, str], tuple[str, str, str], str]
    ] = []
    for implied in ("none", "both"):
        for symmetry in ("none", "slot-service"):
            specifications.append(
                (
                    "encoding",
                    ("sorting-network", implied, symmetry),
                    ("totalizer", implied, symmetry),
                    f"IC={implied};SB={symmetry}",
                )
            )
    for cardinality in ("sorting-network", "totalizer"):
        for symmetry in ("none", "slot-service"):
            specifications.append(
                (
                    "implied",
                    (cardinality, "none", symmetry),
                    (cardinality, "both", symmetry),
                    f"Enc={cardinality};SB={symmetry}",
                )
            )
    for cardinality in ("sorting-network", "totalizer"):
        for implied in ("none", "both"):
            specifications.append(
                (
                    "symmetry",
                    (cardinality, implied, "none"),
                    (cardinality, implied, "slot-service"),
                    f"Enc={cardinality};IC={implied}",
                )
            )

    contrasts = []
    for factor, left_config, right_config, condition in specifications:
        pairs = []
        for instance in instances:
            left = indexed.get((instance, left_config))
            right = indexed.get((instance, right_config))
            if left is not None and right is not None:
                pairs.append((left, right))
        proved = [
            (left, right)
            for left, right in pairs
            if left["status"] in PROVED and right["status"] in PROVED
        ]
        optimum = [
            (left, right)
            for left, right in pairs
            if left["status"] == right["status"] == "OPTIMUM"
        ]
        speedups = []
        for left, right in proved:
            left_elapsed = _number(left.get("elapsed_seconds"))
            right_elapsed = _number(right.get("elapsed_seconds"))
            if left_elapsed is not None and right_elapsed not in (None, 0):
                speedups.append(left_elapsed / right_elapsed)
        lower, upper = _bootstrap_median_ci(
            speedups, f"{factor}:{left_config}:{right_config}"
        )

        def difference(key: str) -> list[float]:
            values = []
            for left, right in pairs:
                left_value = _number(left.get(key))
                right_value = _number(right.get(key))
                if left_value is not None and right_value is not None:
                    values.append(right_value - left_value)
            return values

        contrasts.append(
            {
                "factor": factor,
                "condition": condition,
                "left_cardinality": left_config[0],
                "left_implied": left_config[1],
                "left_symmetry": left_config[2],
                "right_cardinality": right_config[0],
                "right_implied": right_config[1],
                "right_symmetry": right_config[2],
                "pairs": len(pairs),
                "both_proved_pairs": len(proved),
                "both_optimum_pairs": len(optimum),
                "right_faster": sum(value > 1 + 1e-12 for value in speedups),
                "ties": sum(abs(value - 1) <= 1e-12 for value in speedups),
                "left_faster": sum(value < 1 - 1e-12 for value in speedups),
                "median_speedup_left_over_right": (
                    statistics.median(speedups) if speedups else None
                ),
                "bootstrap_95_ci_low": lower,
                "bootstrap_95_ci_high": upper,
                "median_peak_rss_difference_mb": (
                    statistics.median(values)
                    if (values := difference("peak_rss_mb")) else None
                ),
                "median_variables_difference": (
                    statistics.median(values)
                    if (values := difference("variables_max")) else None
                ),
                "median_hard_clauses_difference": (
                    statistics.median(values)
                    if (values := difference("hard_clauses_max")) else None
                ),
                "median_soft_clauses_difference": (
                    statistics.median(values)
                    if (values := difference("soft_clauses_max")) else None
                ),
                "objective_mismatches": sum(
                    (left["coverage"], left["weighted_reference_score"])
                    != (right["coverage"], right["weighted_reference_score"])
                    for left, right in optimum
                ),
            }
        )
    _write(output_dir / "factorial_contrasts.csv", contrasts)
    return contrasts


def _weighted_composite(
    rows: list[dict[str, str]], output_dir: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    indexed = {
        (row["instance_sha256"], _configuration(row)): row for row in rows
    }
    instances = sorted({row["instance_sha256"] for row in rows})
    pairs = []
    for instance in instances:
        left = indexed.get((instance, BASELINE))
        right = indexed.get((instance, REFERENCE_COMPOSITE))
        if left is None or right is None:
            continue
        both_proved = left["status"] in PROVED and right["status"] in PROVED
        both_optimum = left["status"] == right["status"] == "OPTIMUM"
        left_elapsed = _number(left.get("elapsed_seconds"))
        right_elapsed = _number(right.get("elapsed_seconds"))
        pairs.append(
            {
                "instance": left["instance"],
                "instance_sha256": instance,
                "baseline_status": left["status"],
                "reference_status": right["status"],
                "both_proved": both_proved,
                "both_optimum": both_optimum,
                "status_match": (
                    left["status"] == right["status"] if both_proved else None
                ),
                "objective_match": (
                    (
                        left["coverage"],
                        left["weighted_reference_score"],
                    )
                    == (
                        right["coverage"],
                        right["weighted_reference_score"],
                    )
                    if both_optimum
                    else None
                ),
                "baseline_elapsed_seconds": left_elapsed,
                "reference_elapsed_seconds": right_elapsed,
                "speedup_baseline_over_reference": (
                    left_elapsed / right_elapsed
                    if both_proved
                    and left_elapsed is not None
                    and right_elapsed not in (None, 0)
                    else None
                ),
                "peak_rss_difference_mb": (
                    right_rss - left_rss
                    if (left_rss := _number(left.get("peak_rss_mb"))) is not None
                    and (right_rss := _number(right.get("peak_rss_mb"))) is not None
                    else None
                ),
            }
        )

    cell_summary = []
    for configuration in (BASELINE, REFERENCE_COMPOSITE):
        group = [row for row in rows if _configuration(row) == configuration]
        cell_summary.append(
            {
                **dict(zip(CONFIG_KEYS, configuration)),
                "runs": len(group),
                "optimum_runs": sum(row["status"] == "OPTIMUM" for row in group),
                "unsat_runs": sum(
                    row["status"] in {"UNSAT", "UNSATISFIABLE"} for row in group
                ),
                "timeout_runs": sum(
                    row["status"].startswith("TIMEOUT") for row in group
                ),
                "par2_seconds": _par2(group),
                "median_peak_rss_mb": _median(group, "peak_rss_mb"),
            }
        )

    speedups = [
        value
        for row in pairs
        if (value := _number(row["speedup_baseline_over_reference"])) is not None
    ]
    lower, upper = _bootstrap_median_ci(speedups, "weighted-composite:B:R")
    paired_summary = [
        {
            "comparison": "B-to-R",
            "pairs": len(pairs),
            "both_proved_pairs": sum(bool(row["both_proved"]) for row in pairs),
            "both_optimum_pairs": sum(bool(row["both_optimum"]) for row in pairs),
            "status_mismatches": sum(row["status_match"] is False for row in pairs),
            "objective_mismatches": sum(
                row["objective_match"] is False for row in pairs
            ),
            "reference_faster": sum(value > 1 + 1e-12 for value in speedups),
            "ties": sum(abs(value - 1) <= 1e-12 for value in speedups),
            "baseline_faster": sum(value < 1 - 1e-12 for value in speedups),
            "median_speedup_baseline_over_reference": (
                statistics.median(speedups) if speedups else None
            ),
            "bootstrap_95_ci_low": lower,
            "bootstrap_95_ci_high": upper,
            "median_peak_rss_difference_mb": _median(
                pairs, "peak_rss_difference_mb"
            ),
        }
    ]
    _write(output_dir / "weighted_composite_pairs.csv", pairs)
    _write(output_dir / "weighted_composite_summary.csv", cell_summary)
    _write(output_dir / "weighted_composite_paired_summary.csv", paired_summary)
    return pairs, cell_summary, paired_summary


def _lex_vs_weighted(
    weighted: list[dict[str, str]], lex: list[dict[str, str]], output_dir: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected_weighted = {
        (row["instance_sha256"], _configuration(row)): row
        for row in weighted
        if _configuration(row) in {BASELINE, REFERENCE_COMPOSITE}
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
            "weighted_timeout_seconds": _number(weighted_row.get("timeout_seconds")),
            "lex_cos_timeout_seconds": _number(lex_row.get("timeout_seconds")),
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
                "weighted_proved_runs": sum(
                    row["weighted_status"] in PROVED for row in group
                ),
                "lex_cos_proved_runs": sum(
                    row["lex_cos_status"] in PROVED for row in group
                ),
                "both_optimum_pairs": len(optimum),
                "continuity_improved": sum(row["lex_minus_weighted_continuity"] < 0 for row in optimum),
                "continuity_equal": sum(row["lex_minus_weighted_continuity"] == 0 for row in optimum),
                "continuity_worsened": sum(row["lex_minus_weighted_continuity"] > 0 for row in optimum),
                "overtime_decreased": sum(row["lex_minus_weighted_overtime"] < 0 for row in optimum),
                "overtime_equal": sum(row["lex_minus_weighted_overtime"] == 0 for row in optimum),
                "overtime_increased": sum(row["lex_minus_weighted_overtime"] > 0 for row in optimum),
                "similarity_improved": sum(row["lex_minus_weighted_similarity"] > 0 for row in optimum),
                "similarity_equal": sum(row["lex_minus_weighted_similarity"] == 0 for row in optimum),
                "similarity_worsened": sum(row["lex_minus_weighted_similarity"] < 0 for row in optimum),
                "median_similarity_change": _median(optimum, "lex_minus_weighted_similarity"),
                "median_continuity_change": _median(optimum, "lex_minus_weighted_continuity"),
                "median_overtime_change": _median(optimum, "lex_minus_weighted_overtime"),
                "weighted_par2_seconds": _par2(
                    {
                        "status": row["weighted_status"],
                        "elapsed_seconds": row["weighted_elapsed_seconds"],
                        "timeout_seconds": row["weighted_timeout_seconds"],
                    }
                    for row in group
                ),
                "lex_cos_par2_seconds": _par2(
                    {
                        "status": row["lex_cos_status"],
                        "elapsed_seconds": row["lex_cos_elapsed_seconds"],
                        "timeout_seconds": row["lex_cos_timeout_seconds"],
                    }
                    for row in group
                ),
            }
        )
    _write(output_dir / "lex_confirmatory_pairs.csv", pairs)
    _write(output_dir / "lex_confirmatory_summary.csv", summary)
    return pairs, summary


def _policy_sensitivity(
    lex_cos: list[dict[str, str]], lex_ocs: list[dict[str, str]], output_dir: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
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
            "lex_cos_elapsed_seconds": _number(current.get("elapsed_seconds")),
            "lex_ocs_elapsed_seconds": _number(ocs.get("elapsed_seconds")),
            "lex_cos_timeout_seconds": _number(current.get("timeout_seconds")),
            "lex_ocs_timeout_seconds": _number(ocs.get("timeout_seconds")),
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
                "lex_cos_proved_runs": sum(
                    row["lex_cos_status"] in PROVED for row in group
                ),
                "lex_ocs_proved_runs": sum(
                    row["lex_ocs_status"] in PROVED for row in group
                ),
                "both_optimum_pairs": len(optimum),
                "same_objective_vector_pairs": sum(
                    bool(row["same_objective_vector"]) for row in optimum
                ),
                "similarity_improved": sum(
                    row["ocs_minus_cos_similarity"] > 0 for row in optimum
                ),
                "similarity_equal": sum(
                    row["ocs_minus_cos_similarity"] == 0 for row in optimum
                ),
                "similarity_worsened": sum(
                    row["ocs_minus_cos_similarity"] < 0 for row in optimum
                ),
                "continuity_improved": sum(
                    row["ocs_minus_cos_continuity"] < 0 for row in optimum
                ),
                "continuity_equal": sum(
                    row["ocs_minus_cos_continuity"] == 0 for row in optimum
                ),
                "continuity_worsened": sum(
                    row["ocs_minus_cos_continuity"] > 0 for row in optimum
                ),
                "overtime_decreased": sum(
                    row["ocs_minus_cos_overtime"] < 0 for row in optimum
                ),
                "overtime_equal": sum(
                    row["ocs_minus_cos_overtime"] == 0 for row in optimum
                ),
                "overtime_increased": sum(
                    row["ocs_minus_cos_overtime"] > 0 for row in optimum
                ),
                "median_similarity_change": _median(
                    optimum, "ocs_minus_cos_similarity"
                ),
                "median_continuity_change": _median(
                    optimum, "ocs_minus_cos_continuity"
                ),
                "median_overtime_change": _median(
                    optimum, "ocs_minus_cos_overtime"
                ),
                "lex_cos_par2_seconds": _par2(
                    {
                        "status": row["lex_cos_status"],
                        "elapsed_seconds": row["lex_cos_elapsed_seconds"],
                        "timeout_seconds": row["lex_cos_timeout_seconds"],
                    }
                    for row in group
                ),
                "lex_ocs_par2_seconds": _par2(
                    {
                        "status": row["lex_ocs_status"],
                        "elapsed_seconds": row["lex_ocs_elapsed_seconds"],
                        "timeout_seconds": row["lex_ocs_timeout_seconds"],
                    }
                    for row in group
                ),
            }
        )
    _write(output_dir / "lex_policy_sensitivity_pairs.csv", pairs)
    _write(output_dir / "lex_policy_sensitivity_summary.csv", summary)
    return pairs, summary


def analyze(arguments: argparse.Namespace) -> dict[str, Any]:
    output_dir = arguments.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    ablation = _rows(arguments.ablation_results)
    policy = _rows(arguments.lex_results)
    weighted_policy = [row for row in policy if row["method"] == "weighted"]
    lex = [row for row in policy if row["method"] == "lex-cos"]
    if len(weighted_policy) + len(lex) != len(policy):
        raise ValueError("original policy campaign contains an unexpected method")
    sensitivity = _rows(arguments.sensitivity_results)
    factorial_pairs, factorial_summary = _factorial(ablation, output_dir)
    factorial_contrasts = _factorial_contrasts(ablation, output_dir)
    weighted_pairs, weighted_summary, weighted_paired_summary = (
        _weighted_composite(ablation, output_dir)
    )
    lex_pairs, lex_summary = _lex_vs_weighted(weighted_policy, lex, output_dir)
    sensitivity_pairs, sensitivity_summary = _policy_sensitivity(
        lex, sensitivity, output_dir
    )
    all_rows = [*ablation, *policy, *sensitivity]
    counts = {
        "ablation": len(ablation),
        "weighted_policy": len(weighted_policy),
        "lex_confirmatory": len(lex),
        "lex_sensitivity": len(sensitivity),
    }
    expected = {
        "ablation": 640,
        "weighted_policy": 140,
        "lex_confirmatory": 140,
        "lex_sensitivity": 70,
    }
    result = {
        "scope": "compact",
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
        "factorial_contrasts": len(factorial_contrasts),
        "factorial_contrast_objective_mismatches": sum(
            int(row["objective_mismatches"]) for row in factorial_contrasts
        ),
        "weighted_composite_pairs": len(weighted_pairs),
        "weighted_composite_configurations": len(weighted_summary),
        "weighted_composite_status_mismatches": int(
            weighted_paired_summary[0]["status_mismatches"]
        ),
        "weighted_composite_objective_mismatches": int(
            weighted_paired_summary[0]["objective_mismatches"]
        ),
        "lex_confirmatory_pairs": len(lex_pairs),
        "lex_confirmatory_configurations": len(lex_summary),
        "policy_sensitivity_pairs": len(sensitivity_pairs),
        "policy_sensitivity_configurations": len(sensitivity_summary),
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
        and result["factorial_contrasts"] == 12
        and result["factorial_contrast_objective_mismatches"] == 0
        and result["weighted_composite_pairs"] == 80
        and result["weighted_composite_configurations"] == 2
        and result["weighted_composite_status_mismatches"] == 0
        and result["weighted_composite_objective_mismatches"] == 0
        and result["lex_confirmatory_pairs"] == 140
        and result["lex_confirmatory_configurations"] == 1
        and result["policy_sensitivity_pairs"] == 70
        and result["policy_sensitivity_configurations"] == 1
    )
    (output_dir / "analysis_validation.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ablation-results", type=Path, default=Path("experiments/results/gcp_original_ablation"))
    parser.add_argument("--lex-results", type=Path, default=Path("experiments/results/gcp_original_lex_primary"))
    parser.add_argument("--sensitivity-results", type=Path, default=Path("experiments/results/gcp_original_lex_sensitivity"))
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
