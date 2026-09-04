#!/usr/bin/env python3
"""Analyze the fixed Original-suite Policy x Encoding experiment.

The experiment has two scheduling policies (weighted and LEX-COS), two
cardinality encodings (sorting network and Totalizer), and no optional implied
or symmetry-breaking constraints.  Gurobi provides an exact objective-vector
reference; it is not treated as another runtime competitor.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import statistics
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable


METHODS = ("weighted", "lex-cos")
ENCODINGS = ("sorting-network", "totalizer")
MAXSAT_PROVED = {"OPTIMUM", "UNSAT", "UNSATISFIABLE"}
MAXSAT_ALLOWED = MAXSAT_PROVED | {"TIMEOUT", "TIMEOUT_FEASIBLE"}
EXACT_PROVED = {"OPTIMUM", "INFEASIBLE"}
EXACT_ALLOWED = EXACT_PROVED | {"TIMEOUT", "TIMEOUT_FEASIBLE"}
EXPECTED_TIMEOUT = Decimal("3600")
BOOTSTRAP_REPETITIONS = 10_000
PINNED_EVALMAXSAT_SHA256 = (
    "97614c996e1173ca0672ec46da153656046db1d84b9362a8561161ee750779f7"
)


def _read_rows(result_dir: Path) -> list[dict[str, str]]:
    validation_path = result_dir / "validation.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if validation.get("complete") is not True:
        raise ValueError(f"incomplete campaign: {result_dir}")
    with (result_dir / "runs.csv").open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _campaign_metadata(result_dir: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    return (
        _read_object(result_dir / "validation.json"),
        _read_object(result_dir / "environment.json"),
        _read_object(result_dir / "resolved_campaign.json"),
    )


def _decimal(value: Any) -> Decimal | None:
    if value in (None, ""):
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None


def _float(value: Any) -> float | None:
    number = _decimal(value)
    return float(number) if number is not None else None


def _boolean(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value in ("True", "true", "1", 1):
        return True
    if value in ("False", "false", "0", 0):
        return False
    return None


def _status_class(status: str) -> str:
    if status == "OPTIMUM":
        return "OPTIMUM"
    if status in {"UNSAT", "UNSATISFIABLE", "INFEASIBLE"}:
        return "INFEASIBLE"
    if status in {"TIMEOUT", "TIMEOUT_FEASIBLE"}:
        return "UNRESOLVED"
    return "INVALID"


def _objective(row: dict[str, str], method: str) -> tuple[Decimal | None, ...]:
    if method == "weighted":
        keys = ("coverage", "weighted_reference_score")
    elif method == "lex-cos":
        keys = ("coverage", "continuity", "overtime", "similarity")
    else:
        raise ValueError(f"unsupported method: {method}")
    return tuple(_decimal(row.get(key)) for key in keys)


def _objective_matches(
    left: dict[str, str], right: dict[str, str], method: str
) -> bool:
    left_vector = _objective(left, method)
    right_vector = _objective(right, method)
    return None not in left_vector and left_vector == right_vector


def _par2_value(row: dict[str, str]) -> float | None:
    timeout = _float(row.get("timeout_seconds"))
    elapsed = _float(row.get("elapsed_seconds"))
    if row.get("status") in MAXSAT_PROVED:
        return elapsed
    return 2 * timeout if timeout is not None else None


def _median(rows: Iterable[dict[str, Any]], key: str) -> float | None:
    values = [
        value for row in rows if (value := _float(row.get(key))) is not None
    ]
    return statistics.median(values) if values else None


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
    values: list[float], label: str
) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    seed = int.from_bytes(hashlib.sha256(label.encode("utf-8")).digest()[:8], "big")
    generator = random.Random(seed)
    estimates = [
        statistics.median(generator.choices(values, k=len(values)))
        for _ in range(BOOTSTRAP_REPETITIONS)
    ]
    return _percentile(estimates, 0.025), _percentile(estimates, 0.975)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _duplicate_count(rows: list[dict[str, str]], keys: tuple[str, ...]) -> int:
    identities = [tuple(row.get(key, "") for key in keys) for row in rows]
    return len(identities) - len(set(identities))


def _maxsat_row_valid(row: dict[str, str]) -> bool:
    method = row.get("method")
    expected_variant = "weighted" if method == "weighted" else "staged-aligned"
    solver_calls = _decimal(row.get("solver_calls"))
    solver_call_count_valid = (
        solver_calls == Decimal("1")
        if method == "weighted"
        else solver_calls in {Decimal("1"), Decimal("2"), Decimal("3")}
    )
    if method == "lex-cos" and row.get("status") == "OPTIMUM":
        solver_call_count_valid = solver_calls == Decimal("3")
    return all(
        (
            method in METHODS,
            row.get("variant") == expected_variant,
            _decimal(row.get("schema_version")) == Decimal("3"),
            row.get("cardinality") in ENCODINGS,
            row.get("implied") == "none",
            row.get("symmetry") == "none",
            _boolean(row.get("align_evalmaxsat_tct")) is True,
            _decimal(row.get("timeout_seconds")) == EXPECTED_TIMEOUT,
            solver_call_count_valid,
            row.get("status") in MAXSAT_ALLOWED,
            not row.get("validation_errors"),
            _boolean(row.get("hard_timeout")) is False,
            row.get("status") != "OPTIMUM"
            or _boolean(row.get("verified")) is True,
            row.get("status") != "TIMEOUT_FEASIBLE"
            or _boolean(row.get("verified")) is True,
        )
    )


def _exact_row_valid(row: dict[str, str]) -> bool:
    return all(
        (
            _decimal(row.get("schema_version")) == Decimal("1"),
            row.get("method") in METHODS,
            row.get("backend") == "gurobi-mip",
            row.get("formulation") == "mip-e",
            _decimal(row.get("timeout_seconds")) == EXPECTED_TIMEOUT,
            _decimal(row.get("threads")) == Decimal("1"),
            _decimal(row.get("solver_seed")) == Decimal("0"),
            _decimal(row.get("mip_gap")) == Decimal("0"),
            _decimal(row.get("absolute_mip_gap")) == Decimal("0"),
            row.get("status") in EXACT_ALLOWED,
            not row.get("validation_errors"),
            _boolean(row.get("hard_timeout")) is False,
            row.get("status") != "OPTIMUM"
            or _boolean(row.get("verified")) is True,
            row.get("status") != "TIMEOUT_FEASIBLE"
            or _boolean(row.get("verified")) is True,
        )
    )


def _validation_valid(validation: dict[str, Any], expected_runs: int) -> bool:
    return all(
        (
            validation.get("complete") is True,
            validation.get("expected_runs") == expected_runs,
            validation.get("complete_runs") == expected_runs,
            validation.get("manifest_runs") == expected_runs,
            validation.get("workers") == 1,
            not validation.get("invalid_run_ids"),
            not validation.get("missing_run_ids"),
            not validation.get("unexpected_run_ids"),
        )
    )


def _resolved_maxsat_config_valid(config: dict[str, Any], expected_instances: int) -> bool:
    configurations = {
        (item.get("cardinality"), item.get("implied"), item.get("symmetry"))
        for item in config.get("configurations", [])
    }
    runs = {
        (item.get("method"), item.get("variant"), item.get("align_evalmaxsat_tct"))
        for item in config.get("runs", [])
    }
    return all(
        (
            config.get("expected_instances") == expected_instances,
            config.get("expected_runs") == expected_instances * 4,
            _decimal(config.get("timeout_seconds")) == EXPECTED_TIMEOUT,
            config.get("workers") == 1,
            config.get("order_strategy") == "blocked-instance",
            config.get("order_seed") == 20270906,
            config.get("instances") == ["../../instances/paperInstances/**/*.txt"],
            (config.get("instance_filters") or {}).get("seeds") == [1, 2, 3],
            configurations
            == {
                ("sorting-network", "none", "none"),
                ("totalizer", "none", "none"),
            },
            runs
            == {
                ("weighted", "weighted", True),
                ("lex-cos", "staged-aligned", True),
            },
        )
    )


def _resolved_exact_config_valid(config: dict[str, Any], expected_instances: int) -> bool:
    backends = {
        (item.get("backend"), item.get("formulation"))
        for item in config.get("commercial_configurations", [])
    }
    methods = {item.get("method") for item in config.get("runs", [])}
    return all(
        (
            config.get("expected_instances") == expected_instances,
            config.get("expected_runs") == expected_instances * 2,
            _decimal(config.get("timeout_seconds")) == EXPECTED_TIMEOUT,
            config.get("workers") == 1,
            config.get("threads") == 1,
            config.get("seed") == 0,
            _decimal(config.get("mip_gap")) == Decimal("0"),
            _decimal(config.get("absolute_mip_gap")) == Decimal("0"),
            config.get("order_strategy") == "blocked-instance",
            config.get("order_seed") == 20270906,
            config.get("instances") == ["../../instances/paperInstances/**/*.txt"],
            (config.get("instance_filters") or {}).get("seeds") == [1, 2, 3],
            backends == {("gurobi-mip", "mip-e")},
            methods == set(METHODS),
        )
    )


def _environment_checks(
    maxsat: dict[str, Any], exact: dict[str, Any]
) -> dict[str, bool]:
    maxsat_git = maxsat.get("git") or {}
    exact_git = exact.get("git") or {}
    maxsat_affinity = maxsat.get("process_cpu_affinity") or []
    exact_affinity = exact.get("process_cpu_affinity") or []
    return {
        "maxsat_solver_hash": maxsat.get("solver_sha256")
        == PINNED_EVALMAXSAT_SHA256,
        "same_source_commit": bool(maxsat_git.get("commit"))
        and maxsat_git.get("commit") == exact_git.get("commit"),
        "clean_source": maxsat_git.get("dirty") is False
        and exact_git.get("dirty") is False,
        "linux_x86_64": maxsat.get("machine") == "x86_64"
        and exact.get("machine") == "x86_64"
        and str(maxsat.get("platform", "")).startswith("Linux")
        and str(exact.get("platform", "")).startswith("Linux"),
        "single_same_cpu_affinity": len(maxsat_affinity) == 1
        and maxsat_affinity == exact_affinity,
    }


def analyze(
    maxsat_results: Path,
    exact_results: Path,
    output_dir: Path,
    *,
    expected_instances: int = 48,
) -> dict[str, Any]:
    maxsat_rows = _read_rows(maxsat_results)
    exact_rows = _read_rows(exact_results)
    maxsat_validation, maxsat_environment, maxsat_resolved = _campaign_metadata(
        maxsat_results
    )
    exact_validation, exact_environment, exact_resolved = _campaign_metadata(
        exact_results
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    maxsat_instances = {row.get("instance_sha256", "") for row in maxsat_rows}
    exact_instances = {row.get("instance_sha256", "") for row in exact_rows}
    expected_maxsat_keys = {
        (identity, method, encoding)
        for identity in maxsat_instances
        for method in METHODS
        for encoding in ENCODINGS
    }
    observed_maxsat_keys = {
        (row.get("instance_sha256", ""), row.get("method", ""), row.get("cardinality", ""))
        for row in maxsat_rows
    }
    expected_exact_keys = {
        (identity, method) for identity in exact_instances for method in METHODS
    }
    observed_exact_keys = {
        (row.get("instance_sha256", ""), row.get("method", ""))
        for row in exact_rows
    }
    structural_checks = {
        "maxsat_collection_complete": _validation_valid(
            maxsat_validation, expected_instances * 4
        ),
        "exact_collection_complete": _validation_valid(
            exact_validation, expected_instances * 2
        ),
        "maxsat_resolved_config": _resolved_maxsat_config_valid(
            maxsat_resolved.get("config") or {}, expected_instances
        ),
        "exact_resolved_config": _resolved_exact_config_valid(
            exact_resolved.get("config") or {}, expected_instances
        ),
        "maxsat_run_count": len(maxsat_rows) == expected_instances * 4,
        "exact_run_count": len(exact_rows) == expected_instances * 2,
        "maxsat_instance_count": len(maxsat_instances) == expected_instances,
        "exact_instance_count": len(exact_instances) == expected_instances,
        "instance_sets_match": maxsat_instances == exact_instances,
        "maxsat_unique_keys": _duplicate_count(
            maxsat_rows, ("instance_sha256", "method", "cardinality")
        ) == 0,
        "exact_unique_keys": _duplicate_count(
            exact_rows, ("instance_sha256", "method")
        ) == 0,
        "maxsat_matrix_complete": observed_maxsat_keys == expected_maxsat_keys,
        "exact_matrix_complete": observed_exact_keys == expected_exact_keys,
        "maxsat_rows_valid": all(_maxsat_row_valid(row) for row in maxsat_rows),
        "exact_rows_valid": all(_exact_row_valid(row) for row in exact_rows),
    }
    structurally_valid = all(structural_checks.values())
    environment_checks = _environment_checks(maxsat_environment, exact_environment)

    exact_index = {
        (row["instance_sha256"], row["method"]): row for row in exact_rows
    }
    reference_agreement: list[dict[str, Any]] = []
    objective_mismatches = 0
    status_contradictions = 0
    for row in maxsat_rows:
        method = row.get("method", "")
        reference = exact_index.get((row.get("instance_sha256", ""), method))
        maxsat_class = _status_class(row.get("status", ""))
        exact_class = _status_class(reference.get("status", "")) if reference else "MISSING"
        status_match = None
        objective_match = None
        if reference is not None and maxsat_class != "UNRESOLVED":
            status_match = maxsat_class == exact_class
            if status_match is False:
                status_contradictions += 1
        if reference is not None and maxsat_class == exact_class == "OPTIMUM":
            objective_match = _objective_matches(row, reference, method)
            if not objective_match:
                objective_mismatches += 1
        reference_agreement.append(
            {
                "instance": row.get("instance"),
                "instance_sha256": row.get("instance_sha256"),
                "method": method,
                "cardinality": row.get("cardinality"),
                "maxsat_status": row.get("status"),
                "gurobi_status": reference.get("status") if reference else "MISSING",
                "status_match_when_decided": status_match,
                "objective_match_when_optimum": objective_match,
            }
        )

    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in maxsat_rows:
        grouped[(row.get("method", ""), row.get("cardinality", ""))].append(row)
    cell_summary: list[dict[str, Any]] = []
    for method in METHODS:
        for encoding in ENCODINGS:
            rows = grouped.get((method, encoding), [])
            par2_values = [
                value for row in rows if (value := _par2_value(row)) is not None
            ]
            cell_summary.append(
                {
                    "method": method,
                    "cardinality": encoding,
                    "implied": "none",
                    "symmetry": "none",
                    "runs": len(rows),
                    "optimum_runs": sum(row.get("status") == "OPTIMUM" for row in rows),
                    "unsat_runs": sum(
                        row.get("status") in {"UNSAT", "UNSATISFIABLE"}
                        for row in rows
                    ),
                    "proved_runs": sum(row.get("status") in MAXSAT_PROVED for row in rows),
                    "timeout_runs": sum(
                        row.get("status", "").startswith("TIMEOUT") for row in rows
                    ),
                    "par2_seconds": statistics.fmean(par2_values) if par2_values else None,
                    "median_proved_seconds": _median(
                        [row for row in rows if row.get("status") in MAXSAT_PROVED],
                        "elapsed_seconds",
                    ),
                    "median_peak_rss_mb": _median(rows, "peak_rss_mb"),
                    "median_variables": _median(rows, "variables_max"),
                    "median_hard_clauses": _median(rows, "hard_clauses_max"),
                    "median_soft_clauses": _median(rows, "soft_clauses_max"),
                }
            )

    maxsat_index = {
        (row["instance_sha256"], row["method"], row["cardinality"]): row
        for row in maxsat_rows
    }
    paired_rows: list[dict[str, Any]] = []
    contrast_rows: list[dict[str, Any]] = []
    claim_support: dict[str, bool] = {}
    for method in METHODS:
        speedups = []
        method_pairs = []
        for identity in sorted(maxsat_instances):
            sorting = maxsat_index.get((identity, method, "sorting-network"))
            totalizer = maxsat_index.get((identity, method, "totalizer"))
            if sorting is None or totalizer is None:
                continue
            sorting_class = _status_class(sorting.get("status", ""))
            totalizer_class = _status_class(totalizer.get("status", ""))
            both_proved = (
                sorting.get("status") in MAXSAT_PROVED
                and totalizer.get("status") in MAXSAT_PROVED
            )
            both_optimum = sorting_class == totalizer_class == "OPTIMUM"
            status_match = sorting_class == totalizer_class if both_proved else None
            objective_match = (
                _objective_matches(sorting, totalizer, method)
                if both_optimum else None
            )
            sorting_elapsed = _float(sorting.get("elapsed_seconds"))
            totalizer_elapsed = _float(totalizer.get("elapsed_seconds"))
            speedup = (
                sorting_elapsed / totalizer_elapsed
                if both_proved
                and sorting_elapsed is not None
                and totalizer_elapsed not in (None, 0)
                else None
            )
            if speedup is not None:
                speedups.append(speedup)
            pair = {
                "instance": sorting.get("instance"),
                "instance_sha256": identity,
                "method": method,
                "sorting_status": sorting.get("status"),
                "totalizer_status": totalizer.get("status"),
                "both_proved": both_proved,
                "both_optimum": both_optimum,
                "status_match_when_decided": status_match,
                "objective_match_when_optimum": objective_match,
                "sorting_elapsed_seconds": sorting_elapsed,
                "totalizer_elapsed_seconds": totalizer_elapsed,
                "speedup_sorting_over_totalizer": speedup,
            }
            method_pairs.append(pair)
            paired_rows.append(pair)
        lower, upper = _bootstrap_median_ci(speedups, f"{method}:sorting:totalizer")
        sorting_summary = next(
            row for row in cell_summary
            if row["method"] == method and row["cardinality"] == "sorting-network"
        )
        totalizer_summary = next(
            row for row in cell_summary
            if row["method"] == method and row["cardinality"] == "totalizer"
        )
        contrast_objective_mismatches = sum(
            row["objective_match_when_optimum"] is False for row in method_pairs
        )
        contrast_status_mismatches = sum(
            row["status_match_when_decided"] is False for row in method_pairs
        )
        supported = all(
            (
                len(method_pairs) == expected_instances,
                totalizer_summary["proved_runs"] >= sorting_summary["proved_runs"],
                contrast_objective_mismatches == 0,
                contrast_status_mismatches == 0,
                lower is not None and lower > 1.0,
                totalizer_summary["par2_seconds"] < sorting_summary["par2_seconds"],
            )
        )
        claim_support[method] = supported
        contrast_rows.append(
            {
                "method": method,
                "pairs": len(method_pairs),
                "both_proved_pairs": sum(bool(row["both_proved"]) for row in method_pairs),
                "both_optimum_pairs": sum(bool(row["both_optimum"]) for row in method_pairs),
                "totalizer_faster": sum(value > 1 + 1e-12 for value in speedups),
                "ties": sum(abs(value - 1) <= 1e-12 for value in speedups),
                "sorting_faster": sum(value < 1 - 1e-12 for value in speedups),
                "median_speedup_sorting_over_totalizer": (
                    statistics.median(speedups) if speedups else None
                ),
                "bootstrap_95_ci_low": lower,
                "bootstrap_95_ci_high": upper,
                "sorting_proved_runs": sorting_summary["proved_runs"],
                "totalizer_proved_runs": totalizer_summary["proved_runs"],
                "sorting_par2_seconds": sorting_summary["par2_seconds"],
                "totalizer_par2_seconds": totalizer_summary["par2_seconds"],
                "status_mismatches": contrast_status_mismatches,
                "objective_mismatches": contrast_objective_mismatches,
                "totalizer_faster_claim_supported": supported,
            }
        )

    reference_proved_runs = sum(
        row.get("status") in EXACT_PROVED for row in exact_rows
    )
    evidence_checks = {
        "reference_proves_all_runs": reference_proved_runs == expected_instances * 2,
        "no_reference_status_contradictions": status_contradictions == 0,
        "no_reference_objective_mismatches": objective_mismatches == 0,
    }
    evidence_valid = (
        structurally_valid
        and all(environment_checks.values())
        and all(evidence_checks.values())
    )
    report = {
        "scope": "original-policy-by-encoding-3600",
        "expected_instances": expected_instances,
        "maxsat_runs": len(maxsat_rows),
        "exact_reference_runs": len(exact_rows),
        "reference_proved_runs": reference_proved_runs,
        "status_contradictions": status_contradictions,
        "objective_mismatches": objective_mismatches,
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        "structural_checks": structural_checks,
        "structurally_valid": structurally_valid,
        "environment_checks": environment_checks,
        "evidence_checks": evidence_checks,
        "evidence_valid": evidence_valid,
        "totalizer_claim_supported": claim_support,
    }
    _write_csv(output_dir / "policy_encoding_summary.csv", cell_summary)
    _write_csv(output_dir / "policy_encoding_pairs.csv", paired_rows)
    _write_csv(output_dir / "policy_encoding_contrasts.csv", contrast_rows)
    _write_csv(output_dir / "policy_encoding_reference_agreement.csv", reference_agreement)
    (output_dir / "policy_encoding_validation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maxsat-results", type=Path, required=True)
    parser.add_argument("--exact-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-instances", type=int, default=48)
    arguments = parser.parse_args()
    try:
        report = analyze(
            arguments.maxsat_results.resolve(),
            arguments.exact_results.resolve(),
            arguments.output.resolve(),
            expected_instances=arguments.expected_instances,
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["evidence_valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
