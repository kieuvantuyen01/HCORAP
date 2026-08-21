#!/usr/bin/env python3
"""Build manuscript-grade corrected-v2 policy evidence from exact MIP runs.

The existing EvalMaxSAT corrected-v2 campaign is retained as a scalability
result.  Policy effects are considered manuscript-eligible only when the full
Gurobi evaluation campaign and the predeclared CPLEX stratum audit pass the
evidence gates in ``corrected_exact_evidence_gates.json``.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable


METHODS = ("weighted", "lex-cos", "lex-overtime")
PAIRWISE = (
    ("weighted", "lex-cos", "weighted-to-continuity-first"),
    ("lex-cos", "lex-overtime", "continuity-first-to-overtime-first"),
)
METRICS = ("similarity", "continuity", "overtime")
NONTECHNICAL = {"OPTIMUM", "INFEASIBLE", "TIMEOUT", "TIMEOUT_FEASIBLE"}


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _rows(result_dir: Path) -> list[dict[str, str]]:
    validation = _json(result_dir / "validation.json")
    if validation.get("complete") is not True:
        raise ValueError(f"incomplete commercial campaign: {result_dir}")
    with (result_dir / "runs.csv").open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _number(value: Any) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _normalized(value: Any) -> Any:
    try:
        return Decimal(str(value)).normalize()
    except (InvalidOperation, ValueError):
        return value


def _true(value: Any) -> bool:
    return str(value).lower() == "true"


def _median(rows: Iterable[dict[str, Any]], key: str) -> float | None:
    values = [value for row in rows if (value := _number(row.get(key))) is not None]
    return statistics.median(values) if values else None


def _par2(rows: Iterable[dict[str, Any]]) -> float | None:
    values = []
    for row in rows:
        timeout = _number(row.get("timeout_seconds"))
        elapsed = _number(row.get("elapsed_seconds"))
        if row.get("status") in {"OPTIMUM", "INFEASIBLE"} and elapsed is not None:
            values.append(elapsed)
        elif timeout is not None:
            values.append(2 * timeout)
    return statistics.fmean(values) if values else None


def _index(
    rows: list[dict[str, str]], backend: str
) -> tuple[dict[str, dict[str, dict[str, str]]], int]:
    indexed: dict[str, dict[str, dict[str, str]]] = {}
    duplicates = 0
    for row in rows:
        if row.get("backend") != backend:
            continue
        methods = indexed.setdefault(row["instance_sha256"], {})
        duplicates += row["method"] in methods
        methods[row["method"]] = row
    return indexed, duplicates


def _stratum(row: dict[str, str]) -> tuple[str, str, str]:
    return row.get("users", ""), row.get("agents", ""), row.get("visits", "")


def _policy_summary(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    output = []
    for method in METHODS:
        group = [row for row in rows if row["method"] == method]
        optimum = [row for row in group if row["status"] == "OPTIMUM"]
        output.append(
            {
                "solver": "Gurobi",
                "method": method,
                "runs": len(group),
                "optimum_runs": len(optimum),
                "infeasible_runs": sum(row["status"] == "INFEASIBLE" for row in group),
                "timeout_runs": sum(row["status"].startswith("TIMEOUT") for row in group),
                "par2_seconds": _par2(group),
                "median_peak_rss_mb": _median(group, "peak_rss_mb"),
                "median_similarity": _median(optimum, "similarity"),
                "median_continuity": _median(optimum, "continuity"),
                "median_overtime": _median(optimum, "overtime"),
            }
        )
    return output


def _pairwise(
    indexed: dict[str, dict[str, dict[str, str]]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pairs = []
    summaries = []
    for left_method, right_method, comparison in PAIRWISE:
        current = []
        for instance, methods in sorted(indexed.items()):
            left = methods.get(left_method)
            right = methods.get(right_method)
            if left is None or right is None:
                continue
            both_optimum = left["status"] == right["status"] == "OPTIMUM"
            item: dict[str, Any] = {
                "comparison": comparison,
                "instance_sha256": instance,
                "instance": left["instance"],
                "users": left.get("users"),
                "agents": left.get("agents"),
                "visits": left.get("visits"),
                "seed": left.get("seed_instance"),
                "left_method": left_method,
                "right_method": right_method,
                "left_status": left["status"],
                "right_status": right["status"],
                "both_optimum": both_optimum,
            }
            for metric in METRICS:
                left_value = _number(left.get(metric))
                right_value = _number(right.get(metric))
                item[f"left_{metric}"] = left_value
                item[f"right_{metric}"] = right_value
                item[f"delta_{metric}"] = (
                    right_value - left_value
                    if both_optimum
                    and left_value is not None
                    and right_value is not None
                    else None
                )
            current.append(item)
            pairs.append(item)
        optimum = [row for row in current if row["both_optimum"]]
        summaries.append(
            {
                "solver": "Gurobi",
                "comparison": comparison,
                "left_method": left_method,
                "right_method": right_method,
                "pairs": len(current),
                "left_proved_runs": sum(
                    row["left_status"] in {"OPTIMUM", "INFEASIBLE"}
                    for row in current
                ),
                "right_proved_runs": sum(
                    row["right_status"] in {"OPTIMUM", "INFEASIBLE"}
                    for row in current
                ),
                "both_optimum_pairs": len(optimum),
                "left_par2_seconds": _par2(
                    methods[left_method]
                    for methods in indexed.values()
                    if left_method in methods
                ),
                "right_par2_seconds": _par2(
                    methods[right_method]
                    for methods in indexed.values()
                    if right_method in methods
                ),
                "same_objective_vector_pairs": sum(
                    all(row[f"delta_{metric}"] == 0 for metric in METRICS)
                    for row in optimum
                ),
                "median_similarity_change": _median(optimum, "delta_similarity"),
                "median_continuity_change": _median(optimum, "delta_continuity"),
                "median_overtime_change": _median(optimum, "delta_overtime"),
                "continuity_improved": sum(row["delta_continuity"] < 0 for row in optimum),
                "continuity_equal": sum(row["delta_continuity"] == 0 for row in optimum),
                "continuity_worsened": sum(row["delta_continuity"] > 0 for row in optimum),
                "overtime_decreased": sum(row["delta_overtime"] < 0 for row in optimum),
                "overtime_equal": sum(row["delta_overtime"] == 0 for row in optimum),
                "overtime_increased": sum(row["delta_overtime"] > 0 for row in optimum),
            }
        )
    return pairs, summaries


def _agreement(
    primary: dict[str, dict[str, dict[str, str]]],
    audit: dict[str, dict[str, dict[str, str]]],
) -> list[dict[str, Any]]:
    output = []
    for instance in sorted(set(primary) & set(audit)):
        for method in METHODS:
            gurobi = primary[instance].get(method)
            cplex = audit[instance].get(method)
            if gurobi is None or cplex is None:
                continue
            status_agreement = gurobi["status"] == cplex["status"]
            objective_agreement: bool | None = None
            comparison_fields = (
                ("coverage", "weighted_reference_score")
                if method == "weighted"
                else ("coverage", "similarity", "continuity", "overtime")
            )
            if gurobi["status"] == cplex["status"] == "OPTIMUM":
                objective_agreement = all(
                    _normalized(gurobi[field]) == _normalized(cplex[field])
                    for field in comparison_fields
                )
            output.append(
                {
                    "instance_sha256": instance,
                    "instance": gurobi["instance"],
                    "method": method,
                    "gurobi_status": gurobi["status"],
                    "cplex_status": cplex["status"],
                    "status_agreement": status_agreement,
                    "both_optimum": gurobi["status"] == cplex["status"] == "OPTIMUM",
                    "objective_agreement": objective_agreement,
                    "comparison_fields": "|".join(comparison_fields),
                    "gurobi_objective": str(tuple(gurobi[field] for field in comparison_fields)),
                    "cplex_objective": str(tuple(cplex[field] for field in comparison_fields)),
                }
            )
    return output


def analyze(
    primary_dir: Path,
    audit_dir: Path,
    gates_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    gates = _json(gates_path)
    primary_rows = _rows(primary_dir)
    audit_rows = _rows(audit_dir)
    primary, primary_duplicates = _index(primary_rows, "gurobi-mip")
    audit, audit_duplicates = _index(audit_rows, "cplex-mip")
    methods = set(gates["expected_methods"])
    primary_method_errors = sum(set(group) != methods for group in primary.values())
    audit_method_errors = sum(set(group) != methods for group in audit.values())
    primary_strata = {
        _stratum(next(iter(group.values()))) for group in primary.values() if group
    }
    audit_strata = {
        _stratum(next(iter(group.values()))) for group in audit.values() if group
    }
    audit_seeds = {
        next(iter(group.values())).get("seed_instance")
        for group in audit.values()
        if group
    }

    all_policy_optimum = {
        instance
        for instance, group in primary.items()
        if set(group) == methods
        and all(group[method]["status"] == "OPTIMUM" for method in methods)
    }
    optimum_by_stratum: Counter[tuple[str, str, str]] = Counter()
    for instance in all_policy_optimum:
        row = next(iter(primary[instance].values()))
        optimum_by_stratum[_stratum(row)] += 1
    strata_with_two = sum(count >= 2 for count in optimum_by_stratum.values())

    pair_rows, pair_summaries = _pairwise(primary)
    agreements = _agreement(primary, audit)
    status_disagreements = sum(not row["status_agreement"] for row in agreements)
    objective_disagreements = sum(
        row["objective_agreement"] is False for row in agreements
    )
    audit_optimum_groups = sum(row["both_optimum"] for row in agreements)
    source_rows = [*primary_rows, *audit_rows]
    technical_errors = sum(row["status"] not in NONTECHNICAL for row in source_rows)
    unverified_optima = sum(
        row["status"] == "OPTIMUM" and not _true(row.get("verified"))
        for row in source_rows
    )

    structural_checks = {
        "primary_runs": len(primary_rows) == int(gates["expected_primary_runs"]),
        "primary_instances": len(primary) == int(gates["expected_primary_instances"]),
        "primary_backend": all(row.get("backend") == "gurobi-mip" for row in primary_rows),
        "primary_methods": primary_method_errors == 0,
        "primary_strata": len(primary_strata) == int(gates["expected_strata"]),
        "primary_duplicate_keys": primary_duplicates == 0,
        "audit_runs": len(audit_rows) == int(gates["expected_audit_runs"]),
        "audit_instances": len(audit) == int(gates["expected_audit_instances"]),
        "audit_backend": all(row.get("backend") == "cplex-mip" for row in audit_rows),
        "audit_methods": audit_method_errors == 0,
        "audit_strata": len(audit_strata) == int(gates["expected_strata"]),
        "audit_seed": audit_seeds == {"1002"},
        "audit_duplicate_keys": audit_duplicates == 0,
    }
    evidence_checks = {
        "all_policy_optimum_instances": len(all_policy_optimum)
        >= int(gates["minimum_all_policy_optimum_instances"]),
        "strata_with_two_all_policy_optimum_seeds": strata_with_two
        >= int(gates["minimum_strata_with_two_all_policy_optimum_seeds"]),
        "audit_optimum_groups": audit_optimum_groups
        >= int(gates["required_audit_optimum_groups"]),
        "status_disagreements": status_disagreements
        <= int(gates["maximum_status_disagreements"]),
        "objective_disagreements": objective_disagreements
        <= int(gates["maximum_objective_disagreements"]),
        "technical_errors": technical_errors
        <= int(gates["maximum_technical_errors"]),
        "unverified_optima": unverified_optima
        <= int(gates["maximum_unverified_optima"]),
    }
    result = {
        "scope": "corrected-v2-exact-policy",
        "primary_runs": len(primary_rows),
        "primary_instances": len(primary),
        "primary_strata": len(primary_strata),
        "audit_runs": len(audit_rows),
        "audit_instances": len(audit),
        "audit_strata": len(audit_strata),
        "all_policy_optimum_instances": len(all_policy_optimum),
        "strata_with_two_all_policy_optimum_seeds": strata_with_two,
        "audit_agreement_groups": len(agreements),
        "audit_optimum_groups": audit_optimum_groups,
        "status_disagreements": status_disagreements,
        "objective_disagreements": objective_disagreements,
        "technical_errors": technical_errors,
        "unverified_optima": unverified_optima,
        "status_counts": dict(sorted(Counter(row["status"] for row in source_rows).items())),
        "structural_checks": structural_checks,
        "evidence_checks": evidence_checks,
        "structurally_valid": all(structural_checks.values()),
        "evidence_sufficient": all(evidence_checks.values()),
        "gates": gates,
    }
    result["manuscript_eligible"] = (
        result["structurally_valid"] and result["evidence_sufficient"]
    )
    result["valid"] = result["manuscript_eligible"]

    output_dir.mkdir(parents=True, exist_ok=True)
    _write(output_dir / "corrected_policy_summary.csv", _policy_summary(primary_rows))
    _write(output_dir / "corrected_pairwise_pairs.csv", pair_rows)
    _write(output_dir / "corrected_pairwise_summary.csv", pair_summaries)
    _write(output_dir / "corrected_solver_agreement.csv", agreements)
    (output_dir / "corrected_exact_validation.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--primary-results",
        type=Path,
        default=Path("experiments/results/gcp_commercial_corrected_primary"),
    )
    parser.add_argument(
        "--audit-results",
        type=Path,
        default=Path("experiments/results/gcp_commercial_corrected_audit"),
    )
    parser.add_argument(
        "--gates",
        type=Path,
        default=Path("experiments/configs/corrected_exact_evidence_gates.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/results/gcp_corrected_exact_analysis"),
    )
    arguments = parser.parse_args()
    try:
        result = analyze(
            arguments.primary_results,
            arguments.audit_results,
            arguments.gates,
            arguments.output_dir,
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["manuscript_eligible"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
