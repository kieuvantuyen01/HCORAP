#!/usr/bin/env python3
"""Validate and summarize the full Gurobi/CPLEX baseline on Original HCORAP."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable

try:
    from .source_provenance import (
        GUROBI_SOURCE_PATHS,
        SHARED_PROBLEM_SOURCE_PATHS,
        git_paths_equivalent,
    )
except ImportError:
    from source_provenance import (
        GUROBI_SOURCE_PATHS,
        SHARED_PROBLEM_SOURCE_PATHS,
        git_paths_equivalent,
    )


METHODS = ("weighted", "lex-cos")
BACKENDS = ("gurobi-mip", "cplex-mip")
EXACT_PROVED = {"OPTIMUM", "INFEASIBLE"}
EXACT_ALLOWED = EXACT_PROVED | {"TIMEOUT", "TIMEOUT_FEASIBLE"}
MAXSAT_PROVED = {"OPTIMUM", "UNSAT", "UNSATISFIABLE"}
MAXSAT_ALLOWED = MAXSAT_PROVED | {"TIMEOUT", "TIMEOUT_FEASIBLE"}
EXPECTED_TIMEOUT = Decimal("3600")


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _read_campaign(
    result_dir: Path,
) -> tuple[list[dict[str, str]], dict[str, Any], dict[str, Any], dict[str, Any]]:
    validation = _read_object(result_dir / "validation.json")
    environment = _read_object(result_dir / "environment.json")
    resolved = _read_object(result_dir / "resolved_campaign.json")
    with (result_dir / "runs.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    return rows, validation, environment, resolved


def _decimal(value: Any) -> Decimal | None:
    if value in (None, ""):
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None


def _float(value: Any) -> float | None:
    parsed = _decimal(value)
    return float(parsed) if parsed is not None else None


def _truth(value: Any) -> bool:
    return str(value).lower() == "true"


def _status_class(status: str) -> str:
    if status == "OPTIMUM":
        return "OPTIMUM"
    if status in {"INFEASIBLE", "UNSAT", "UNSATISFIABLE"}:
        return "INFEASIBLE"
    if status in {"TIMEOUT", "TIMEOUT_FEASIBLE"}:
        return "UNRESOLVED"
    return "INVALID"


def _objective(row: dict[str, str], method: str) -> tuple[Decimal | None, ...]:
    fields = (
        ("coverage", "weighted_reference_score")
        if method == "weighted"
        else ("coverage", "continuity", "overtime", "similarity")
    )
    return tuple(_decimal(row.get(field)) for field in fields)


def _objective_matches(
    left: dict[str, str], right: dict[str, str], method: str
) -> bool:
    left_vector = _objective(left, method)
    right_vector = _objective(right, method)
    return None not in left_vector and left_vector == right_vector


def _duplicate_count(rows: list[dict[str, str]]) -> int:
    keys = [(row.get("instance_sha256"), row.get("method")) for row in rows]
    return len(keys) - len(set(keys))


def _campaign_complete(validation: dict[str, Any], expected_runs: int) -> bool:
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


def _resolved_config_valid(
    resolved: dict[str, Any], backend: str, expected_instances: int
) -> bool:
    config = resolved.get("config") or {}
    configurations = {
        (item.get("backend"), item.get("formulation"))
        for item in config.get("commercial_configurations", [])
    }
    runs = {
        (item.get("method"), int(item.get("wc", 1)), int(item.get("wo", 1)))
        for item in config.get("runs", [])
    }
    return all(
        (
            config.get("expected_instances") == expected_instances,
            config.get("expected_runs") == expected_instances * len(METHODS),
            _decimal(config.get("timeout_seconds")) == EXPECTED_TIMEOUT,
            config.get("threads") == 1,
            config.get("seed") == 0,
            _decimal(config.get("mip_gap")) == Decimal("0"),
            _decimal(config.get("absolute_mip_gap")) == Decimal("0"),
            config.get("workers") == 1,
            config.get("order_strategy") == "blocked-instance",
            config.get("order_seed") == 20270906,
            config.get("instances") == ["../../instances/paperInstances/**/*.txt"],
            (config.get("instance_filters") or {}).get("seeds") == [1, 2, 3],
            configurations == {(backend, "mip-e")},
            runs == {("weighted", 1, 1), ("lex-cos", 1, 1)},
        )
    )


def _resolved_maxsat_config_valid(
    resolved: dict[str, Any], expected_instances: int
) -> bool:
    config = resolved.get("config") or {}
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


def _row_valid(row: dict[str, str], backend: str) -> bool:
    status = row.get("status", "")
    return all(
        (
            _decimal(row.get("schema_version")) == Decimal("1"),
            row.get("backend") == backend,
            row.get("formulation") == "mip-e",
            row.get("method") in METHODS,
            _decimal(row.get("timeout_seconds")) == EXPECTED_TIMEOUT,
            _decimal(row.get("threads")) == Decimal("1"),
            _decimal(row.get("solver_seed")) == Decimal("0"),
            _decimal(row.get("mip_gap")) == Decimal("0"),
            _decimal(row.get("absolute_mip_gap")) == Decimal("0"),
            status in EXACT_ALLOWED,
            not row.get("validation_errors"),
            not _truth(row.get("hard_timeout")),
            status != "OPTIMUM" or _truth(row.get("verified")),
            status != "TIMEOUT_FEASIBLE" or _truth(row.get("verified")),
        )
    )


def _maxsat_row_valid(row: dict[str, str]) -> bool:
    method = row.get("method", "")
    expected_variant = "weighted" if method == "weighted" else "staged-aligned"
    solver_calls = _decimal(row.get("solver_calls"))
    solver_calls_valid = (
        solver_calls == Decimal("1")
        if method == "weighted"
        else solver_calls in {Decimal("1"), Decimal("2"), Decimal("3")}
    )
    if method == "lex-cos" and row.get("status") == "OPTIMUM":
        solver_calls_valid = solver_calls == Decimal("3")
    status = row.get("status", "")
    return all(
        (
            _decimal(row.get("schema_version")) == Decimal("3"),
            method in METHODS,
            row.get("variant") == expected_variant,
            row.get("cardinality") in {"sorting-network", "totalizer"},
            row.get("unit_objective_bound_encoding") == row.get("cardinality"),
            row.get("weighted_similarity_bound_encoding") == "pb-bdd",
            row.get("implied") == "none",
            row.get("symmetry") == "none",
            _truth(row.get("align_evalmaxsat_tct")),
            _decimal(row.get("timeout_seconds")) == EXPECTED_TIMEOUT,
            solver_calls_valid,
            status in MAXSAT_ALLOWED,
            not row.get("validation_errors"),
            not _truth(row.get("hard_timeout")),
            status != "OPTIMUM" or _truth(row.get("verified")),
            status != "TIMEOUT_FEASIBLE" or _truth(row.get("verified")),
        )
    )


def _environment_checks(
    maxsat: dict[str, Any], gurobi: dict[str, Any], cplex: dict[str, Any]
) -> dict[str, bool]:
    maxsat_git = maxsat.get("git") or {}
    gurobi_git = gurobi.get("git") or {}
    cplex_git = cplex.get("git") or {}
    maxsat_affinity = maxsat.get("process_cpu_affinity") or []
    gurobi_affinity = gurobi.get("process_cpu_affinity") or []
    cplex_affinity = cplex.get("process_cpu_affinity") or []
    return {
        "clean_source": maxsat_git.get("dirty") is False
        and gurobi_git.get("dirty") is False
        and cplex_git.get("dirty") is False,
        "shared_problem_source_equivalent": git_paths_equivalent(
            str(maxsat_git.get("commit") or ""),
            str(cplex_git.get("commit") or ""),
            SHARED_PROBLEM_SOURCE_PATHS,
        )
        and git_paths_equivalent(
            str(gurobi_git.get("commit") or ""),
            str(cplex_git.get("commit") or ""),
            SHARED_PROBLEM_SOURCE_PATHS,
        ),
        "gurobi_source_equivalent": git_paths_equivalent(
            str(gurobi_git.get("commit") or ""),
            str(cplex_git.get("commit") or ""),
            GUROBI_SOURCE_PATHS,
        ),
        "linux_x86_64": maxsat.get("machine") == "x86_64"
        and gurobi.get("machine") == "x86_64"
        and cplex.get("machine") == "x86_64"
        and str(maxsat.get("platform", "")).startswith("Linux")
        and str(gurobi.get("platform", "")).startswith("Linux")
        and str(cplex.get("platform", "")).startswith("Linux"),
        "single_same_cpu_affinity": len(maxsat_affinity) == 1
        and maxsat_affinity == gurobi_affinity == cplex_affinity,
        "same_logical_cpu_count": maxsat.get("logical_cpu_count") == 8
        and gurobi.get("logical_cpu_count") == 8
        and cplex.get("logical_cpu_count") == 8,
    }


def _par2(row: dict[str, str]) -> float | None:
    elapsed = _float(row.get("elapsed_seconds"))
    timeout = _float(row.get("timeout_seconds"))
    if _status_class(row.get("status", "")) in {"OPTIMUM", "INFEASIBLE"}:
        return elapsed
    return 2 * timeout if timeout is not None else None


def _median(rows: Iterable[dict[str, str]], field: str) -> float | None:
    values = [value for row in rows if (value := _float(row.get(field))) is not None]
    return statistics.median(values) if values else None


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def analyze(
    maxsat_results: Path,
    gurobi_results: Path,
    cplex_results: Path,
    output_dir: Path,
    *,
    expected_instances: int = 48,
) -> dict[str, Any]:
    (
        maxsat_rows,
        maxsat_validation,
        maxsat_environment,
        maxsat_resolved,
    ) = _read_campaign(maxsat_results)
    (
        gurobi_rows,
        gurobi_validation,
        gurobi_environment,
        gurobi_resolved,
    ) = _read_campaign(gurobi_results)
    (
        cplex_rows,
        cplex_validation,
        cplex_environment,
        cplex_resolved,
    ) = _read_campaign(cplex_results)
    expected_exact_runs = expected_instances * len(METHODS)
    expected_maxsat_runs = expected_instances * len(METHODS) * 2
    expected_keys = {
        (identity, method)
        for identity in {row.get("instance_sha256", "") for row in gurobi_rows}
        for method in METHODS
    }
    gurobi_keys = {
        (row.get("instance_sha256", ""), row.get("method", ""))
        for row in gurobi_rows
    }
    cplex_keys = {
        (row.get("instance_sha256", ""), row.get("method", ""))
        for row in cplex_rows
    }
    maxsat_keys = {
        (
            row.get("instance_sha256", ""),
            row.get("method", ""),
            row.get("cardinality", ""),
        )
        for row in maxsat_rows
    }
    expected_maxsat_keys = {
        (identity, method, cardinality)
        for identity, method in expected_keys
        for cardinality in ("sorting-network", "totalizer")
    }
    gurobi_instances = {key[0] for key in gurobi_keys}
    cplex_instances = {key[0] for key in cplex_keys}
    maxsat_instances = {key[0] for key in maxsat_keys}
    structural_checks = {
        "maxsat_collection_complete": _campaign_complete(
            maxsat_validation, expected_maxsat_runs
        ),
        "gurobi_collection_complete": _campaign_complete(
            gurobi_validation, expected_exact_runs
        ),
        "cplex_collection_complete": _campaign_complete(
            cplex_validation, expected_exact_runs
        ),
        "maxsat_resolved_config": _resolved_maxsat_config_valid(
            maxsat_resolved, expected_instances
        ),
        "gurobi_resolved_config": _resolved_config_valid(
            gurobi_resolved, "gurobi-mip", expected_instances
        ),
        "cplex_resolved_config": _resolved_config_valid(
            cplex_resolved, "cplex-mip", expected_instances
        ),
        "maxsat_run_count": len(maxsat_rows) == expected_maxsat_runs,
        "gurobi_run_count": len(gurobi_rows) == expected_exact_runs,
        "cplex_run_count": len(cplex_rows) == expected_exact_runs,
        "maxsat_instance_count": len(maxsat_instances) == expected_instances,
        "gurobi_instance_count": len(gurobi_instances) == expected_instances,
        "cplex_instance_count": len(cplex_instances) == expected_instances,
        "instance_sets_match": maxsat_instances
        == gurobi_instances
        == cplex_instances,
        "maxsat_unique_keys": len(maxsat_keys) == len(maxsat_rows),
        "gurobi_unique_keys": _duplicate_count(gurobi_rows) == 0,
        "cplex_unique_keys": _duplicate_count(cplex_rows) == 0,
        "maxsat_matrix_complete": maxsat_keys == expected_maxsat_keys,
        "gurobi_matrix_complete": gurobi_keys == expected_keys,
        "cplex_matrix_complete": cplex_keys == expected_keys,
        "maxsat_rows_valid": all(_maxsat_row_valid(row) for row in maxsat_rows),
        "gurobi_rows_valid": all(
            _row_valid(row, "gurobi-mip") for row in gurobi_rows
        ),
        "cplex_rows_valid": all(_row_valid(row, "cplex-mip") for row in cplex_rows),
    }
    environment_checks = _environment_checks(
        maxsat_environment, gurobi_environment, cplex_environment
    )

    maxsat_index = {
        (row["instance_sha256"], row["method"], row["cardinality"]): row
        for row in maxsat_rows
    }
    gurobi_index = {
        (row["instance_sha256"], row["method"]): row for row in gurobi_rows
    }
    cplex_index = {
        (row["instance_sha256"], row["method"]): row for row in cplex_rows
    }
    pair_rows: list[dict[str, Any]] = []
    for identity, method in sorted(expected_keys):
        maxsat = maxsat_index.get((identity, method, "totalizer"))
        gurobi = gurobi_index.get((identity, method))
        cplex = cplex_index.get((identity, method))
        if maxsat is None or gurobi is None or cplex is None:
            continue
        maxsat_class = _status_class(maxsat.get("status", ""))
        gurobi_class = _status_class(gurobi.get("status", ""))
        cplex_class = _status_class(cplex.get("status", ""))
        commercial_status_agreement = gurobi_class == cplex_class
        commercial_objective_agreement = (
            _objective_matches(gurobi, cplex, method)
            if gurobi_class == cplex_class == "OPTIMUM"
            else None
        )
        maxsat_status_agreement = (
            maxsat_class == gurobi_class if maxsat_class != "UNRESOLVED" else None
        )
        maxsat_objective_agreement = (
            _objective_matches(maxsat, gurobi, method)
            if maxsat_class == gurobi_class == "OPTIMUM"
            else None
        )
        maxsat_elapsed = _float(maxsat.get("elapsed_seconds"))
        gurobi_elapsed = _float(gurobi.get("elapsed_seconds"))
        cplex_elapsed = _float(cplex.get("elapsed_seconds"))
        pair_rows.append(
            {
                "instance": gurobi.get("instance"),
                "instance_sha256": identity,
                "method": method,
                "maxsat_configuration": "totalizer-none-none",
                "maxsat_status": maxsat.get("status"),
                "gurobi_status": gurobi.get("status"),
                "cplex_status": cplex.get("status"),
                "commercial_status_agreement": commercial_status_agreement,
                "commercial_objective_agreement_when_optimal": (
                    commercial_objective_agreement
                ),
                "maxsat_status_agreement_when_decided": maxsat_status_agreement,
                "maxsat_objective_agreement_when_optimal": (
                    maxsat_objective_agreement
                ),
                "maxsat_elapsed_seconds": maxsat_elapsed,
                "gurobi_elapsed_seconds": gurobi_elapsed,
                "cplex_elapsed_seconds": cplex_elapsed,
                "cplex_over_gurobi_runtime": (
                    cplex_elapsed / gurobi_elapsed
                    if gurobi_elapsed not in (None, 0) and cplex_elapsed is not None
                    else None
                ),
            }
        )

    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    totalizer_rows = [
        row for row in maxsat_rows if row.get("cardinality") == "totalizer"
    ]
    for row in totalizer_rows:
        grouped[("evalmaxsat-totalizer", row.get("method", ""))].append(row)
    for row in [*gurobi_rows, *cplex_rows]:
        grouped[(row.get("backend", ""), row.get("method", ""))].append(row)
    summary_rows: list[dict[str, Any]] = []
    for backend in ("evalmaxsat-totalizer", *BACKENDS):
        for method in METHODS:
            rows = grouped[(backend, method)]
            par2_values = [value for row in rows if (value := _par2(row)) is not None]
            summary_rows.append(
                {
                    "backend": backend,
                    "method": method,
                    "runs": len(rows),
                    "optimal_runs": sum(row.get("status") == "OPTIMUM" for row in rows),
                    "infeasible_runs": sum(
                        _status_class(row.get("status", "")) == "INFEASIBLE"
                        for row in rows
                    ),
                    "proved_runs": sum(
                        _status_class(row.get("status", ""))
                        in {"OPTIMUM", "INFEASIBLE"}
                        for row in rows
                    ),
                    "timeout_runs": sum(
                        row.get("status", "").startswith("TIMEOUT") for row in rows
                    ),
                    "median_elapsed_seconds": _median(rows, "elapsed_seconds"),
                    "par2_seconds": statistics.fmean(par2_values)
                    if par2_values
                    else None,
                    "median_peak_rss_mb": _median(rows, "peak_rss_mb"),
                }
            )

    commercial_checks = {
        "all_pairs_present": len(pair_rows) == expected_exact_runs,
        "both_solvers_prove_all": all(
            row.get("status") in EXACT_PROVED
            for row in [*gurobi_rows, *cplex_rows]
        ),
        "no_status_disagreement": all(
            row["commercial_status_agreement"] is True for row in pair_rows
        ),
        "no_objective_disagreement": all(
            row["commercial_objective_agreement_when_optimal"] is not False
            for row in pair_rows
        ),
    }
    maxsat_checks = {
        "all_totalizer_rows_present": len(totalizer_rows) == expected_exact_runs,
        "no_status_contradiction_when_decided": all(
            row["maxsat_status_agreement_when_decided"] is not False
            for row in pair_rows
        ),
        "no_objective_disagreement_when_optimal": all(
            row["maxsat_objective_agreement_when_optimal"] is not False
            for row in pair_rows
        ),
    }
    report = {
        "scope": "original-full-exact-method-baseline-3600",
        "expected_instances": expected_instances,
        "expected_maxsat_runs": expected_maxsat_runs,
        "expected_runs_per_commercial_solver": expected_exact_runs,
        "maxsat_runs": len(maxsat_rows),
        "gurobi_runs": len(gurobi_rows),
        "cplex_runs": len(cplex_rows),
        "structural_checks": structural_checks,
        "environment_checks": environment_checks,
        "commercial_agreement_checks": commercial_checks,
        "maxsat_reference_checks": maxsat_checks,
        "runtime_comparison_valid": all(structural_checks.values())
        and all(environment_checks.values()),
        "evidence_valid": all(structural_checks.values())
        and all(environment_checks.values())
        and all(commercial_checks.values())
        and all(maxsat_checks.values()),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "exact_method_summary.csv", summary_rows)
    _write_csv(output_dir / "cross_solver_pairs.csv", pair_rows)
    (output_dir / "full_exact_baseline_validation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maxsat-results", type=Path, required=True)
    parser.add_argument("--gurobi-results", type=Path, required=True)
    parser.add_argument("--cplex-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-instances", type=int, default=48)
    arguments = parser.parse_args()
    try:
        report = analyze(
            arguments.maxsat_results.resolve(),
            arguments.gurobi_results.resolve(),
            arguments.cplex_results.resolve(),
            arguments.output.resolve(),
            expected_instances=arguments.expected_instances,
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["evidence_valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
