from __future__ import annotations

import csv
import json
from pathlib import Path

from experiments.analyze_full_commercial_baseline import analyze


METHODS = ("weighted", "lex-cos")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _row(index: int, method: str, backend: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "instance": f"instance-{index}.txt",
        "instance_sha256": f"sha-{index}",
        "backend": backend,
        "formulation": "mip-e",
        "method": method,
        "timeout_seconds": 3600,
        "threads": 1,
        "solver_seed": 0,
        "mip_gap": 0,
        "absolute_mip_gap": 0,
        "status": "OPTIMUM",
        "validation_errors": "",
        "hard_timeout": "False",
        "verified": "True",
        "coverage": 100,
        "weighted_reference_score": 500 + index,
        "continuity": index % 3,
        "overtime": index % 2,
        "similarity": 490 + index,
        "elapsed_seconds": (index + 1) * (1 if backend == "gurobi-mip" else 2),
        "peak_rss_mb": 100 if backend == "gurobi-mip" else 120,
    }


def _maxsat_row(index: int, method: str, cardinality: str) -> dict[str, object]:
    return {
        "schema_version": 3,
        "instance": f"instance-{index}.txt",
        "instance_sha256": f"sha-{index}",
        "method": method,
        "variant": "weighted" if method == "weighted" else "staged-aligned",
        "cardinality": cardinality,
        "implied": "none",
        "symmetry": "none",
        "align_evalmaxsat_tct": "True",
        "timeout_seconds": 3600,
        "solver_calls": 1 if method == "weighted" else 3,
        "status": "OPTIMUM",
        "validation_errors": "",
        "hard_timeout": "False",
        "verified": "True",
        "coverage": 100,
        "weighted_reference_score": 500 + index,
        "continuity": index % 3,
        "overtime": index % 2,
        "similarity": 490 + index,
        "elapsed_seconds": (index + 1) * 10,
        "peak_rss_mb": 200,
    }


def _resolved_config(backend: str, instances: int) -> dict[str, object]:
    return {
        "config": {
            "expected_instances": instances,
            "expected_runs": instances * len(METHODS),
            "timeout_seconds": 3600,
            "threads": 1,
            "seed": 0,
            "mip_gap": 0,
            "absolute_mip_gap": 0,
            "workers": 1,
            "order_strategy": "blocked-instance",
            "order_seed": 20270906,
            "instances": ["../../instances/paperInstances/**/*.txt"],
            "instance_filters": {"seeds": [1, 2, 3]},
            "commercial_configurations": [
                {"backend": backend, "formulation": "mip-e"}
            ],
            "runs": [
                {"method": "weighted", "wc": 1, "wo": 1},
                {"method": "lex-cos"},
            ],
        }
    }


def _maxsat_resolved_config(instances: int) -> dict[str, object]:
    return {
        "config": {
            "expected_instances": instances,
            "expected_runs": instances * len(METHODS) * 2,
            "timeout_seconds": 3600,
            "workers": 1,
            "order_strategy": "blocked-instance",
            "order_seed": 20270906,
            "instances": ["../../instances/paperInstances/**/*.txt"],
            "instance_filters": {"seeds": [1, 2, 3]},
            "configurations": [
                {
                    "cardinality": "sorting-network",
                    "implied": "none",
                    "symmetry": "none",
                },
                {
                    "cardinality": "totalizer",
                    "implied": "none",
                    "symmetry": "none",
                },
            ],
            "runs": [
                {
                    "method": "weighted",
                    "variant": "weighted",
                    "align_evalmaxsat_tct": True,
                },
                {
                    "method": "lex-cos",
                    "variant": "staged-aligned",
                    "align_evalmaxsat_tct": True,
                },
            ],
        }
    }


def _write_campaign_metadata(
    path: Path, expected_runs: int, resolved: dict[str, object]
) -> None:
    (path / "validation.json").write_text(
        json.dumps(
            {
                "complete": True,
                "expected_runs": expected_runs,
                "complete_runs": expected_runs,
                "manifest_runs": expected_runs,
                "workers": 1,
                "invalid_run_ids": [],
                "missing_run_ids": [],
                "unexpected_run_ids": [],
            }
        ),
        encoding="utf-8",
    )
    (path / "environment.json").write_text(
        json.dumps(
            {
                "machine": "x86_64",
                "platform": "Linux-test",
                "logical_cpu_count": 8,
                "process_cpu_affinity": [0],
                "git": {"commit": "same-test-commit", "dirty": False},
            }
        ),
        encoding="utf-8",
    )
    (path / "resolved_campaign.json").write_text(
        json.dumps(resolved), encoding="utf-8"
    )


def _campaign(path: Path, backend: str, instances: int) -> list[dict[str, object]]:
    path.mkdir()
    rows = [
        _row(index, method, backend)
        for index in range(instances)
        for method in METHODS
    ]
    _write_csv(path / "runs.csv", rows)
    expected_runs = instances * len(METHODS)
    _write_campaign_metadata(
        path, expected_runs, _resolved_config(backend, instances)
    )
    return rows


def _maxsat_campaign(path: Path, instances: int) -> list[dict[str, object]]:
    path.mkdir()
    rows = [
        _maxsat_row(index, method, cardinality)
        for index in range(instances)
        for method in METHODS
        for cardinality in ("sorting-network", "totalizer")
    ]
    _write_csv(path / "runs.csv", rows)
    _write_campaign_metadata(
        path, instances * len(METHODS) * 2, _maxsat_resolved_config(instances)
    )
    return rows


def test_full_commercial_analyzer_accepts_complete_agreeing_matrix(
    tmp_path: Path,
) -> None:
    maxsat = tmp_path / "maxsat"
    gurobi = tmp_path / "gurobi"
    cplex = tmp_path / "cplex"
    output = tmp_path / "analysis"
    _maxsat_campaign(maxsat, 8)
    _campaign(gurobi, "gurobi-mip", 8)
    _campaign(cplex, "cplex-mip", 8)

    report = analyze(maxsat, gurobi, cplex, output, expected_instances=8)

    assert report["evidence_valid"] is True
    assert report["runtime_comparison_valid"] is True
    assert report["commercial_agreement_checks"]["all_pairs_present"] is True
    assert (output / "exact_method_summary.csv").is_file()
    assert (output / "cross_solver_pairs.csv").is_file()
    assert (output / "full_exact_baseline_validation.json").is_file()


def test_full_commercial_analyzer_rejects_objective_disagreement(
    tmp_path: Path,
) -> None:
    maxsat = tmp_path / "maxsat"
    gurobi = tmp_path / "gurobi"
    cplex = tmp_path / "cplex"
    output = tmp_path / "analysis"
    _maxsat_campaign(maxsat, 8)
    _campaign(gurobi, "gurobi-mip", 8)
    cplex_rows = _campaign(cplex, "cplex-mip", 8)
    cplex_rows[0]["weighted_reference_score"] = 999
    _write_csv(cplex / "runs.csv", cplex_rows)

    report = analyze(maxsat, gurobi, cplex, output, expected_instances=8)

    assert report["evidence_valid"] is False
    assert (
        report["commercial_agreement_checks"]["no_objective_disagreement"]
        is False
    )
