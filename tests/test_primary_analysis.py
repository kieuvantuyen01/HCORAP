from __future__ import annotations

from pathlib import Path

from experiments.analyze_primary_campaigns import (
    _factorial,
    _lex_vs_weighted,
    _policy_sensitivity,
)


def _row(**updates: str) -> dict[str, str]:
    row = {
        "instance": "instance_30_10_4_1.txt",
        "instance_sha256": "instance-1",
        "users": "30",
        "agents": "10",
        "visits": "4",
        "cardinality": "sorting-network",
        "implied": "none",
        "symmetry": "none",
        "status": "OPTIMUM",
        "elapsed_seconds": "2",
        "coverage": "120",
        "similarity": "100",
        "continuity": "3",
        "overtime": "2",
        "weighted_reference_score": "95",
        "variables_max": "1000",
        "hard_clauses_max": "2000",
        "soft_clauses_max": "300",
        "verified": "True",
    }
    row.update(updates)
    return row


def test_primary_analysis_pairs_factorial_and_policies(tmp_path: Path) -> None:
    baseline = _row()
    proposed = _row(
        cardinality="totalizer",
        implied="both",
        symmetry="slot-service",
        elapsed_seconds="1",
        variables_max="800",
        hard_clauses_max="1500",
    )
    pairs, summary = _factorial([baseline, proposed], tmp_path)
    assert len(summary) == 2
    assert all(row["objective_match"] is True for row in pairs)
    assert pairs[1]["speedup_baseline_over_configuration"] == 2

    lex = _row(similarity="90", continuity="1", overtime="1")
    lex_pairs, lex_summary = _lex_vs_weighted([baseline], [lex], tmp_path)
    assert lex_pairs[0]["lex_minus_weighted_continuity"] == -2
    assert lex_summary[0]["continuity_improved"] == 1

    ocs = _row(method="lex-overtime", similarity="88", continuity="2", overtime="0")
    sensitivity = _policy_sensitivity([lex], [ocs], tmp_path)
    assert sensitivity[0]["ocs_minus_cos_overtime"] == -1
    assert sensitivity[0]["same_objective_vector"] is False
