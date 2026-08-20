from __future__ import annotations

import json
from pathlib import Path

from experiments.audit_results_addition import (
    agreement_audit,
    audit,
    coverage_balance,
    objective_signature,
)


def _record(config: str, instance: str, score: int) -> dict[str, object]:
    return {
        "analysis_unit": "main_8cfg_evalmaxsat",
        "configuration": config,
        "instance_name": instance,
        "status": "OPTIMUM",
        "method": "weighted",
        "delta": "0.05",
        "objective_signature": json.dumps([120, score], separators=(",", ":")),
    }


def test_weighted_signature_allows_different_tie_vectors() -> None:
    left = {
        "objective_mode": "weighted",
        "metrics": {
            "coverage": 120,
            "similarity": 428,
            "continuity": 10,
            "overtime": 0,
            "weighted_reference_score": 418,
        },
    }
    right = {
        "objective_mode": "weighted",
        "metrics": {
            "coverage": 120,
            "similarity": 426,
            "continuity": 8,
            "overtime": 0,
            "weighted_reference_score": 418,
        },
    }
    assert objective_signature(left) == objective_signature(right)


def test_balance_and_agreement_detect_different_failures() -> None:
    rows = [
        _record("cfg1", "i1", 10),
        _record("cfg2", "i1", 11),
        _record("cfg1", "i2", 12),
    ]
    balance = coverage_balance(rows)
    assert balance[0]["balanced_instance_sets"] is False
    assert balance[0]["intersection_instances"] == 1
    agreement = agreement_audit(rows)
    assert agreement[0]["objective_disagreements"] == 1


def test_audit_is_idempotent_and_excludes_organized_output(tmp_path: Path) -> None:
    root = tmp_path / "results_addition"
    result_dir = root / "main_8cfg_evalmaxsat" / "cfg1_ORIGINAL"
    result_dir.mkdir(parents=True)
    (root / "main_8cfg_evalmaxsat" / "environment.txt").write_text(
        "git_commit=abc\nhcorap_sha256=def\n",
        encoding="utf-8",
    )
    (result_dir / "instance_1.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "instance": "instances/instance_1.txt",
                "status": "OPTIMUM",
                "method": "weighted",
                "objective_mode": "weighted",
                "continuity_weight": 1,
                "overtime_weight": 1,
                "elapsed_seconds": 1.0,
                "timeout_seconds": 10,
                "metrics": {
                    "coverage": 120,
                    "similarity": 12,
                    "continuity": 2,
                    "overtime": 0,
                    "weighted_reference_score": 10,
                    "verified": True,
                },
            }
        ),
        encoding="utf-8",
    )
    first = audit(root, root / "organized", "2026-08-19T00:00:00+00:00")
    second = audit(root, root / "organized", "2026-08-19T00:00:00+00:00")
    assert first["source_files"] == second["source_files"] == 2
    campaigns = json.loads(
        (root / "organized" / "catalog" / "campaign_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert all(row["campaign"] != "organized" for row in campaigns)
