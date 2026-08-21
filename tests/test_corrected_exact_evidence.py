from __future__ import annotations

import csv
import json
from pathlib import Path

from experiments.analyze_corrected_exact_evidence import analyze


METHODS = ("weighted", "lex-cos", "lex-overtime")


def _write(path: Path, rows: list[dict[str, str]]) -> None:
    path.mkdir()
    (path / "validation.json").write_text(
        json.dumps({"complete": True}), encoding="utf-8"
    )
    with (path / "runs.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _row(stratum: int, seed: int, method: str, backend: str) -> dict[str, str]:
    instance = f"s{stratum}-seed{seed}"
    return {
        "instance_sha256": instance,
        "instance": f"{instance}.txt",
        "users": str(20 + stratum),
        "agents": str(10 + stratum),
        "visits": str(4 + stratum % 2),
        "seed_instance": str(seed),
        "backend": backend,
        "method": method,
        "status": "OPTIMUM",
        "verified": "True",
        "elapsed_seconds": "2",
        "timeout_seconds": "300",
        "peak_rss_mb": "100",
        "coverage": "20",
        "similarity": "90" if method != "weighted" else "100",
        "continuity": "1" if method != "weighted" else "3",
        "overtime": "0" if method == "lex-overtime" else "1",
        "weighted_reference_score": "88",
    }


def test_exact_corrected_evidence_requires_full_matrix_and_solver_agreement(
    tmp_path: Path,
) -> None:
    primary_rows = [
        _row(stratum, seed, method, "gurobi-mip")
        for stratum in range(16)
        for seed in (1001, 1002, 1003)
        for method in METHODS
    ]
    audit_rows = [
        _row(stratum, 1002, method, "cplex-mip")
        for stratum in range(16)
        for method in METHODS
    ]
    primary = tmp_path / "primary"
    audit = tmp_path / "audit"
    output = tmp_path / "analysis"
    _write(primary, primary_rows)
    _write(audit, audit_rows)

    result = analyze(
        primary,
        audit,
        Path("experiments/configs/corrected_exact_evidence_gates.json"),
        output,
    )
    assert result["manuscript_eligible"] is True
    assert result["all_policy_optimum_instances"] == 48
    assert result["audit_optimum_groups"] == 48
    assert result["strata_with_two_all_policy_optimum_seeds"] == 16
    assert (output / "corrected_pairwise_summary.csv").is_file()

    audit_rows[0]["weighted_reference_score"] = "999"
    _write_csv = audit / "runs.csv"
    with _write_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(audit_rows[0]))
        writer.writeheader()
        writer.writerows(audit_rows)
    mismatched = analyze(
        primary,
        audit,
        Path("experiments/configs/corrected_exact_evidence_gates.json"),
        output,
    )
    assert mismatched["manuscript_eligible"] is False
    assert mismatched["objective_disagreements"] == 1
