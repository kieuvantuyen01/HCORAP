from __future__ import annotations

import csv
import json
from pathlib import Path

from experiments.analyze_policy_encoding_matrix import analyze
from experiments.analyze_policy_encoding_matrix import PINNED_EVALMAXSAT_SHA256


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _exact_row(index: int, method: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "instance": f"instance-{index}.txt",
        "instance_sha256": f"sha-{index}",
        "method": method,
        "backend": "gurobi-mip",
        "formulation": "mip-e",
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
    }


def _maxsat_row(index: int, method: str, encoding: str) -> dict[str, object]:
    totalizer = encoding == "totalizer"
    return {
        "schema_version": 3,
        "instance": f"instance-{index}.txt",
        "instance_sha256": f"sha-{index}",
        "variant": "weighted" if method == "weighted" else "staged-aligned",
        "method": method,
        "cardinality": encoding,
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
        "elapsed_seconds": (10 + index) if totalizer else (20 + 2 * index),
        "peak_rss_mb": 80 if totalizer else 90,
        "variables_max": 1000 if totalizer else 2000,
        "hard_clauses_max": 4000 if totalizer else 3500,
        "soft_clauses_max": 100,
    }


def _campaigns(tmp_path: Path, instances: int = 8) -> tuple[Path, Path, Path]:
    maxsat = tmp_path / "maxsat"
    exact = tmp_path / "exact"
    output = tmp_path / "analysis"
    maxsat.mkdir()
    exact.mkdir()
    maxsat_rows = [
        _maxsat_row(index, method, encoding)
        for index in range(instances)
        for method in ("weighted", "lex-cos")
        for encoding in ("sorting-network", "totalizer")
    ]
    exact_rows = [
        _exact_row(index, method)
        for index in range(instances)
        for method in ("weighted", "lex-cos")
    ]
    _write_csv(maxsat / "runs.csv", maxsat_rows)
    _write_csv(exact / "runs.csv", exact_rows)
    for root, expected_runs in ((maxsat, instances * 4), (exact, instances * 2)):
        (root / "validation.json").write_text(
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
    common_environment = {
        "machine": "x86_64",
        "platform": "Linux-test",
        "process_cpu_affinity": [0],
        "git": {"commit": "test-commit", "dirty": False},
    }
    (maxsat / "environment.json").write_text(
        json.dumps(
            {**common_environment, "solver_sha256": PINNED_EVALMAXSAT_SHA256}
        ),
        encoding="utf-8",
    )
    (exact / "environment.json").write_text(
        json.dumps(common_environment), encoding="utf-8"
    )
    (maxsat / "resolved_campaign.json").write_text(
        json.dumps(
            {
                "config": {
                    "expected_instances": instances,
                    "expected_runs": instances * 4,
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
        ),
        encoding="utf-8",
    )
    (exact / "resolved_campaign.json").write_text(
        json.dumps(
            {
                "config": {
                    "expected_instances": instances,
                    "expected_runs": instances * 2,
                    "timeout_seconds": 3600,
                    "workers": 1,
                    "threads": 1,
                    "seed": 0,
                    "mip_gap": 0,
                    "absolute_mip_gap": 0,
                    "order_strategy": "blocked-instance",
                    "order_seed": 20270906,
                    "instances": ["../../instances/paperInstances/**/*.txt"],
                    "instance_filters": {"seeds": [1, 2, 3]},
                    "commercial_configurations": [
                        {"backend": "gurobi-mip", "formulation": "mip-e"}
                    ],
                    "runs": [{"method": method} for method in ("weighted", "lex-cos")],
                }
            }
        ),
        encoding="utf-8",
    )
    return maxsat, exact, output


def test_analyzer_accepts_complete_fixed_matrix(tmp_path: Path) -> None:
    maxsat, exact, output = _campaigns(tmp_path)

    report = analyze(maxsat, exact, output, expected_instances=8)

    assert report["structurally_valid"] is True
    assert report["evidence_valid"] is True
    assert report["objective_mismatches"] == 0
    assert report["totalizer_claim_supported"] == {
        "weighted": True,
        "lex-cos": True,
    }
    assert (output / "policy_encoding_summary.csv").is_file()
    assert (output / "policy_encoding_contrasts.csv").is_file()
    assert (output / "policy_encoding_reference_agreement.csv").is_file()


def test_analyzer_rejects_an_objective_mismatch_against_reference(
    tmp_path: Path,
) -> None:
    maxsat, exact, output = _campaigns(tmp_path)
    rows = list(csv.DictReader((maxsat / "runs.csv").open(newline="", encoding="utf-8")))
    rows[0]["weighted_reference_score"] = "999"
    _write_csv(maxsat / "runs.csv", rows)

    report = analyze(maxsat, exact, output, expected_instances=8)

    assert report["structurally_valid"] is True
    assert report["evidence_valid"] is False
    assert report["objective_mismatches"] == 1


def test_analyzer_keeps_timeout_as_unresolved_not_infeasible(tmp_path: Path) -> None:
    maxsat, exact, output = _campaigns(tmp_path)
    rows = list(csv.DictReader((maxsat / "runs.csv").open(newline="", encoding="utf-8")))
    rows[0].update(
        status="TIMEOUT_FEASIBLE",
        verified="True",
        elapsed_seconds="3600",
    )
    _write_csv(maxsat / "runs.csv", rows)

    report = analyze(maxsat, exact, output, expected_instances=8)

    assert report["structurally_valid"] is True
    assert report["evidence_valid"] is True
    assert report["status_contradictions"] == 0


def test_analyzer_rejects_unpinned_solver_provenance(tmp_path: Path) -> None:
    maxsat, exact, output = _campaigns(tmp_path)
    environment = json.loads(
        (maxsat / "environment.json").read_text(encoding="utf-8")
    )
    environment["solver_sha256"] = "unreviewed-solver"
    (maxsat / "environment.json").write_text(
        json.dumps(environment), encoding="utf-8"
    )

    report = analyze(maxsat, exact, output, expected_instances=8)

    assert report["environment_checks"]["maxsat_solver_hash"] is False
    assert report["evidence_valid"] is False


def test_fixed_configs_contain_only_the_declared_matrix() -> None:
    root = Path(__file__).resolve().parents[1]
    maxsat = json.loads(
        (root / "experiments/configs/gcp_original_policy_encoding_3600.json")
        .read_text(encoding="utf-8")
    )
    exact = json.loads(
        (root / "experiments/configs/gcp_original_policy_reference_3600.json")
        .read_text(encoding="utf-8")
    )

    assert maxsat["expected_instances"] == exact["expected_instances"] == 48
    assert maxsat["timeout_seconds"] == exact["timeout_seconds"] == 3600
    assert maxsat["expected_runs"] == 192
    assert exact["expected_runs"] == 96
    assert {item["method"] for item in maxsat["runs"]} == {"weighted", "lex-cos"}
    assert {item["cardinality"] for item in maxsat["configurations"]} == {
        "sorting-network",
        "totalizer",
    }
    assert {
        (item["implied"], item["symmetry"])
        for item in maxsat["configurations"]
    } == {("none", "none")}
