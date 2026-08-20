from __future__ import annotations

from pathlib import Path

from experiments.analyze_primary_campaigns import (
    _factorial,
    _factorial_contrasts,
    _lex_vs_weighted,
    _policy_sensitivity,
    _weighted_composite,
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
        "timeout_seconds": "120",
        "peak_rss_mb": "100",
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

    direct_totalizer = _row(
        cardinality="totalizer",
        elapsed_seconds="1",
        peak_rss_mb="80",
        variables_max="800",
        hard_clauses_max="2500",
    )
    contrasts = _factorial_contrasts([baseline, direct_totalizer], tmp_path)
    encoding = next(
        row
        for row in contrasts
        if row["factor"] == "encoding"
        and row["condition"] == "IC=none;SB=none"
    )
    assert encoding["both_proved_pairs"] == 1
    assert encoding["right_faster"] == 1
    assert encoding["median_speedup_left_over_right"] == 2
    assert encoding["median_variables_difference"] == -200
    assert encoding["median_hard_clauses_difference"] == 500

    weighted_pairs, weighted_summary, weighted_paired = _weighted_composite(
        [baseline, proposed], tmp_path
    )
    assert len(weighted_pairs) == 1
    assert len(weighted_summary) == 2
    assert weighted_paired[0]["objective_mismatches"] == 0
    assert weighted_paired[0]["median_speedup_baseline_over_reference"] == 2

    weighted_policy = _row(
        cardinality="totalizer",
        implied="both",
        symmetry="slot-service",
        elapsed_seconds="3",
        timeout_seconds="300",
    )
    lex = _row(
        cardinality="totalizer",
        implied="both",
        symmetry="slot-service",
        similarity="90",
        continuity="1",
        overtime="1",
        timeout_seconds="300",
    )
    lex_pairs, lex_summary = _lex_vs_weighted([weighted_policy], [lex], tmp_path)
    assert lex_pairs[0]["lex_minus_weighted_continuity"] == -2
    assert lex_summary[0]["continuity_improved"] == 1
    assert lex_summary[0]["weighted_proved_runs"] == 1
    assert lex_summary[0]["lex_cos_proved_runs"] == 1
    assert lex_summary[0]["weighted_par2_seconds"] == 3
    assert lex_summary[0]["lex_cos_par2_seconds"] == 2

    ocs = _row(
        method="lex-overtime",
        cardinality="totalizer",
        implied="both",
        symmetry="slot-service",
        similarity="88",
        continuity="2",
        overtime="0",
        elapsed_seconds="4",
        timeout_seconds="300",
    )
    sensitivity, sensitivity_summary = _policy_sensitivity(
        [lex], [ocs], tmp_path
    )
    assert sensitivity[0]["ocs_minus_cos_overtime"] == -1
    assert sensitivity[0]["same_objective_vector"] is False
    assert sensitivity_summary[0]["both_optimum_pairs"] == 1
    assert sensitivity_summary[0]["lex_ocs_proved_runs"] == 1
    assert sensitivity_summary[0]["lex_cos_par2_seconds"] == 2
    assert sensitivity_summary[0]["lex_ocs_par2_seconds"] == 4
