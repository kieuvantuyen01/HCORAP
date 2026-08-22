from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from experiments.freeze_manuscript_bundle import validate_and_render_marker
from experiments.generate_manuscript_results import FACTORIAL_ORDER, generate


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _configuration(configuration: tuple[str, str, str]) -> dict[str, str]:
    return dict(zip(("cardinality", "implied", "symmetry"), configuration))


def test_generate_complete_branch_aware_manuscript_bundle(tmp_path: Path) -> None:
    primary = tmp_path / "primary"
    corrected = tmp_path / "corrected"
    corrected_maxsat = tmp_path / "corrected-maxsat"
    corrected_maxsat_results = tmp_path / "corrected-maxsat-results"
    cross = tmp_path / "cross"
    output = tmp_path / "generated"
    primary.mkdir()
    corrected.mkdir()
    corrected_maxsat.mkdir()
    corrected_maxsat_results.mkdir()
    cross.mkdir()
    screening = tmp_path / "screening.json"
    screening.write_text(
        json.dumps(
            {
                "decision": "GO",
                "expected_measured_runs": 924,
                "branches": {
                    "original_lexicographic": {"enabled": True},
                    "corrected_v2_lexicographic": {"enabled": True},
                },
            }
        ),
        encoding="utf-8",
    )
    (primary / "analysis_validation.json").write_text(
        json.dumps({"valid": True, "scope": "compact"}), encoding="utf-8"
    )
    (corrected / "corrected_exact_validation.json").write_text(
        json.dumps(
            {
                "valid": True,
                "manuscript_eligible": True,
                "audit_instances": 16,
                "audit_runs": 48,
            }
        ),
        encoding="utf-8",
    )
    (corrected_maxsat / "corrected_validation.json").write_text(
        json.dumps({"valid": True, "structurally_valid": True}),
        encoding="utf-8",
    )
    _write_csv(
        corrected_maxsat / "corrected_policy_summary.csv",
        [
            {
                "method": method,
                "runs": 48,
                "optimum_runs": 3,
                "timeout_runs": 45,
                "par2_seconds": 570,
            }
            for method in ("weighted", "lex-cos", "lex-overtime")
        ],
    )
    corrected_maxsat_run_rows = []
    for method in ("weighted", "lex-cos", "lex-overtime"):
        for index in range(48):
            optimum = index < (1 if method == "weighted" else 3)
            corrected_maxsat_run_rows.append(
                {
                    "method": method,
                    "status": "OPTIMUM" if optimum else "TIMEOUT",
                    "stage_count": (
                        0 if method == "weighted" else (3 if optimum else (1 if index == 47 else 2))
                    ),
                }
            )
    _write_csv(corrected_maxsat_results / "runs.csv", corrected_maxsat_run_rows)
    (cross / "cross_paradigm_validation.json").write_text(
        json.dumps({"valid": True, "scope": "full"}), encoding="utf-8"
    )

    _write_csv(
        primary / "factorial_summary.csv",
        [
            {
                **_configuration(configuration),
                "runs": 48,
                "optimum_runs": 36,
                "unsat_runs": 6,
                "timeout_runs": 6,
                "par2_seconds": 35,
                "median_peak_rss_mb": 140,
                "median_variables": 12000,
            }
            for configuration in FACTORIAL_ORDER
        ],
    )
    factorial_pair_rows = []
    for instance_index in range(3):
        for configuration in FACTORIAL_ORDER:
            totalizer_only = configuration == ("totalizer", "none", "none")
            full = configuration == ("totalizer", "both", "slot-service")
            elapsed = 8 if totalizer_only else (10 if full else 12)
            factorial_pair_rows.append(
                {
                    "instance_sha256": f"factorial-{instance_index}",
                    **_configuration(configuration),
                    "both_proved": "True",
                    "configuration_elapsed_seconds": elapsed,
                }
            )
    _write_csv(primary / "factorial_paired_runs.csv", factorial_pair_rows)
    contrasts = []
    for implied in ("none", "both"):
        for symmetry in ("none", "slot-service"):
            contrasts.append(
                {
                    "factor": "encoding",
                    "condition": f"IC={implied};SB={symmetry}",
                    "both_proved_pairs": 40,
                    "right_faster": 22,
                    "ties": 0,
                    "left_faster": 18,
                    "median_speedup_left_over_right": 1.2,
                    "bootstrap_95_ci_low": 1.05,
                    "bootstrap_95_ci_high": 1.35,
                    "median_variables_difference": -200,
                    "median_hard_clauses_difference": 300,
                }
            )
    for cardinality in ("sorting-network", "totalizer"):
        for symmetry in ("none", "slot-service"):
            contrasts.append(
                {
                    "factor": "implied",
                    "condition": f"Enc={cardinality};SB={symmetry}",
                    "both_proved_pairs": 38,
                    "right_faster": 20,
                    "ties": 1,
                    "left_faster": 17,
                    "median_speedup_left_over_right": 1.1,
                    "bootstrap_95_ci_low": 0.95,
                    "bootstrap_95_ci_high": 1.25,
                    "median_variables_difference": 25,
                    "median_hard_clauses_difference": 50,
                }
            )
    for cardinality in ("sorting-network", "totalizer"):
        for implied in ("none", "both"):
            contrasts.append(
                {
                    "factor": "symmetry",
                    "condition": f"Enc={cardinality};IC={implied}",
                    "both_proved_pairs": 36,
                    "right_faster": 17,
                    "ties": 2,
                    "left_faster": 17,
                    "median_speedup_left_over_right": 0.98,
                    "bootstrap_95_ci_low": 0.85,
                    "bootstrap_95_ci_high": 1.12,
                    "median_variables_difference": 0,
                    "median_hard_clauses_difference": 40,
                }
            )
    _write_csv(primary / "factorial_contrasts.csv", contrasts)
    _write_csv(
        primary / "weighted_composite_paired_summary.csv",
        [
            {
                "both_proved_pairs": 40,
                "reference_faster": 22,
                "ties": 0,
                "baseline_faster": 18,
                "median_speedup_baseline_over_reference": 1.18,
                "bootstrap_95_ci_low": 1.08,
                "bootstrap_95_ci_high": 1.28,
            }
        ],
    )
    policy_rows = []
    for configuration in (FACTORIAL_ORDER[-1],):
        policy_rows.append(
            {
                **_configuration(configuration),
                "pairs": 42,
                "weighted_proved_runs": 38,
                "lex_cos_proved_runs": 36,
                "both_optimum_pairs": 34,
                "median_similarity_change": -5,
                "median_continuity_change": -2,
                "median_overtime_change": 0,
                "weighted_par2_seconds": 80,
                "lex_cos_par2_seconds": 110,
            }
        )
    _write_csv(primary / "lex_confirmatory_summary.csv", policy_rows)
    _write_csv(
        primary / "lex_confirmatory_pairs.csv",
        [
            {
                "instance_sha256": f"original-{index}",
                "both_optimum": "True",
                "weighted_similarity": 100,
                "weighted_overtime": 1 if index == 0 else 0,
                "lex_minus_weighted_similarity": -5,
                "lex_minus_weighted_continuity": -1,
                "lex_minus_weighted_overtime": -1 if index == 0 else 0,
            }
            for index in range(4)
        ],
    )
    _write_csv(
        corrected / "corrected_policy_summary.csv",
        [
            {
                "method": method,
                "runs": 48,
                "optimum_runs": 42,
                "timeout_runs": 6,
                "median_similarity": 90 if method == "weighted" else 85,
                "median_continuity": 4 if method == "weighted" else 2,
                "median_overtime": 2 if method == "weighted" else 1,
                "par2_seconds": 75 if method == "weighted" else 105,
            }
            for method in ("weighted", "lex-cos", "lex-overtime")
        ],
    )
    _write_csv(
        corrected / "corrected_pairwise_summary.csv",
        [
            {
                "solver": "Gurobi",
                "comparison": comparison,
                "pairs": 48,
                "left_proved_runs": 45,
                "right_proved_runs": 44,
                "both_optimum_pairs": 38,
                "same_objective_vector_pairs": 30,
                "median_similarity_change": -5,
                "median_continuity_change": -2,
                "median_overtime_change": -1,
                "left_par2_seconds": 95,
                "right_par2_seconds": 100,
                "continuity_improved": 35,
                "continuity_equal": 3,
                "continuity_worsened": 0,
                "overtime_decreased": 36,
                "overtime_equal": 2,
                "overtime_increased": 0,
            }
            for comparison in (
                "weighted-to-continuity-first",
                "continuity-first-to-overtime-first",
            )
        ],
    )
    corrected_pair_rows = []
    for index in range(48):
        corrected_pair_rows.append(
            {
                "comparison": "weighted-to-continuity-first",
                "instance_sha256": f"corrected-{index}",
                "both_optimum": "True",
                "left_similarity": 100,
                "left_overtime": 1 if index < 47 else 0,
                "delta_similarity": -6,
                "delta_continuity": -1 if index < 42 else 0,
                "delta_overtime": -1 if index < 47 else 0,
            }
        )
    priority_similarity = (-3, 0, 8)
    for index in range(48):
        differs = index >= 45
        corrected_pair_rows.append(
            {
                "comparison": "continuity-first-to-overtime-first",
                "instance_sha256": f"corrected-{index}",
                "both_optimum": "True",
                "left_similarity": 94,
                "left_overtime": 1,
                "delta_similarity": priority_similarity[index - 45] if differs else 0,
                "delta_continuity": 1 if differs else 0,
                "delta_overtime": -1 if differs else 0,
            }
        )
    _write_csv(corrected / "corrected_pairwise_pairs.csv", corrected_pair_rows)
    _write_csv(
        cross / "cross_paradigm_agreement.csv",
        [
            {
                "method": method,
                "all_exact_optimum": "True",
                "all_exact_infeasible": "False",
                "status_agreement": "True",
                "objective_agreement": "True",
            }
            for method in ("weighted", "lex-cos")
            for _ in range(20)
        ],
    )

    provenance = generate(
        argparse.Namespace(
            screening_decision=screening,
            primary_dir=primary,
            corrected_dir=corrected,
            corrected_maxsat_dir=corrected_maxsat,
            corrected_maxsat_results_dir=corrected_maxsat_results,
            cross_dir=cross,
            output_dir=output,
        )
    )
    assert provenance["valid"]
    assert provenance["expected_measured_runs"] == 924
    results = (output / "results.tex").read_text(encoding="utf-8")
    assert results.count(r"\begin{table}") == 2
    assert r"\begin{table*}" not in results
    assert results.count(r"\begin{figure*}") == 1
    assert r"\label{tab:factorial-footprint}" in results
    assert r"\label{tab:benchmark-signal}" in results
    assert "PAR-2 includes" in results
    assert r"12{,}000" in results
    assert "Totalizer helps; extra constraints do not" in results
    assert "final compatibility criterion" in results
    assert r"\resultplaceholder" not in results
    assert "EvalMaxSAT, Gurobi, and CPLEX" in results
    assert (output / "manuscript-provenance.json").is_file()
    marker = validate_and_render_marker(
        generated_dir=output,
        primary_validation=primary / "analysis_validation.json",
        cross_validation=cross / "cross_paradigm_validation.json",
        corrected_validation=corrected / "corrected_exact_validation.json",
        generation_provenance=output / "manuscript-provenance.json",
        screening_decision=screening,
        source_commit="publication-commit",
        expected_commit="publication-commit",
        source_clean=True,
    )
    assert r"\def\HCORAPFrozenValidationStatus{VALID}" in marker
