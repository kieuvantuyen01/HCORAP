from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from experiments.generate_compact_manuscript_results import generate, _runtime_reduction_summary


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    policy = tmp_path / "policy"
    encoding = tmp_path / "encoding"
    commercial = tmp_path / "commercial"
    output = tmp_path / "generated"
    policy.mkdir()
    encoding.mkdir()
    commercial.mkdir()

    (policy / "corrected_exact_validation.json").write_text(
        json.dumps({"manuscript_eligible": True}), encoding="utf-8"
    )
    _write_csv(
        policy / "corrected_pairwise_summary.csv",
        [
            {
                "solver": "Gurobi",
                "comparison": "weighted-to-continuity-first",
                "left_method": "weighted",
                "right_method": "lex-cos",
                "pairs": 48,
                "both_optimum_pairs": 48,
                "median_similarity_change": -36,
                "median_continuity_change": -5.5,
                "median_overtime_change": -12,
                "continuity_improved": 43,
                "overtime_decreased": 47,
            }
        ],
    )
    policy_details = []
    for index in range(48):
        policy_details.append(
            {
                "comparison": "weighted-to-continuity-first",
                "instance_sha256": f"sha-{index}",
                "left_method": "weighted",
                "right_method": "lex-cos",
                "both_optimum": True,
                "delta_continuity": -5.5 if index < 43 else 0,
                "delta_overtime": -12 if index != 42 else 0,
                "delta_similarity": -36,
                "left_similarity": 600,
            }
        )
    _write_csv(policy / "corrected_pairwise_pairs.csv", policy_details)

    (encoding / "policy_encoding_validation.json").write_text(
        json.dumps({"evidence_valid": True}), encoding="utf-8"
    )
    summaries = []
    for method in ("weighted", "lex-cos"):
        for cardinality in ("sorting-network", "totalizer"):
            totalizer = cardinality == "totalizer"
            summaries.append(
                {
                    "method": method,
                    "cardinality": cardinality,
                    "implied": "none",
                    "symmetry": "none",
                    "runs": 48,
                    "optimum_runs": 40,
                    "unsat_runs": 6,
                    "proved_runs": 46,
                    "timeout_runs": 2,
                    "par2_seconds": 100 if totalizer else 120,
                    "median_proved_seconds": 10 if totalizer else 15,
                    "median_peak_rss_mb": 80 if totalizer else 90,
                    "median_variables": 1000 if totalizer else 2000,
                    "median_hard_clauses": 4000 if totalizer else 3500,
                    "median_soft_clauses": 100,
                }
            )
    _write_csv(encoding / "policy_encoding_summary.csv", summaries)
    pair_rows = []
    for method in ("weighted", "lex-cos"):
        for index in range(48):
            users = 30 if index < 24 else 40
            agents = (10, 15, 20, 25)[index % 4]
            visits = 4 if index % 2 == 0 else 5
            pair_rows.append(
                {
                    "instance": f"instance_{users}_{agents}_{visits}_{index % 3 + 1}.txt",
                    "method": method,
                    "both_proved": index < 46,
                    "speedup_sorting_over_totalizer": 1.5,
                    "sorting_elapsed_seconds": 15,
                    "totalizer_elapsed_seconds": 10,
                }
            )
    _write_csv(encoding / "policy_encoding_pairs.csv", pair_rows)
    _write_csv(
        encoding / "policy_encoding_contrasts.csv",
        [
            {
                "method": method,
                "pairs": 48,
                "both_proved_pairs": 46,
                "totalizer_faster": 40,
                "median_speedup_sorting_over_totalizer": 1.5,
                "bootstrap_95_ci_low": 1.2,
                "bootstrap_95_ci_high": 1.8,
                "totalizer_faster_claim_supported": True,
            }
            for method in ("weighted", "lex-cos")
        ],
    )
    (commercial / "full_exact_baseline_validation.json").write_text(
        json.dumps({
            "evidence_valid": True,
            "runtime_comparison_valid": True,
            "maxsat_runs": 192,
            "gurobi_runs": 96,
            "cplex_runs": 96,
        }), encoding="utf-8"
    )
    exact_summary = []
    for backend in ("evalmaxsat-totalizer", "gurobi-mip", "cplex-mip"):
        for method in ("weighted", "lex-cos"):
            maxsat = backend == "evalmaxsat-totalizer"
            exact_summary.append({
                "backend": backend, "method": method, "runs": 48,
                "optimal_runs": 40 if maxsat else 46,
                "infeasible_runs": 6 if maxsat else 2,
                "proved_runs": 46 if maxsat else 48,
                "timeout_runs": 2 if maxsat else 0, "median_elapsed_seconds": 10,
                "par2_seconds": 100, "median_peak_rss_mb": 80,
            })
    _write_csv(commercial / "exact_method_summary.csv", exact_summary)
    cross_rows = []
    for method in ("weighted", "lex-cos"):
        for index in range(48):
            proved = index < 46
            cross_rows.append({
                "method": method,
                "maxsat_status": "OPTIMUM" if proved else "TIMEOUT",
                "gurobi_status": "OPTIMUM" if proved else "INFEASIBLE",
                "cplex_status": "OPTIMUM" if proved else "INFEASIBLE",
                "commercial_status_agreement": True,
                "commercial_objective_agreement_when_optimal": True if proved else "",
                "maxsat_status_agreement_when_decided": True if proved else "",
                "maxsat_objective_agreement_when_optimal": True if proved else "",
                "maxsat_elapsed_seconds": 20,
                "gurobi_elapsed_seconds": 1,
                "cplex_elapsed_seconds": 2,
            })
    _write_csv(commercial / "cross_solver_pairs.csv", cross_rows)
    return policy, encoding, commercial, output


def test_generator_emits_only_after_all_gates_pass(tmp_path: Path) -> None:
    policy, encoding, commercial, output = _inputs(tmp_path)

    report = generate(policy, encoding, commercial, output)

    assert report["policy_gate"] is True
    assert report["encoding_gate"] is True
    assert report["commercial_gate"] is True
    assert report["totalizer_claim_supported"] == {
        "weighted": True,
        "lex-cos": True,
    }
    macros = (output / "compact_result_macros.tex").read_text(encoding="utf-8")
    table = (output / "compact_encoding_table.tex").read_text(encoding="utf-8")
    policy_table = (output / "compact_policy_table.tex").read_text(encoding="utf-8")
    assert r"\BothPriorityMeasuresImprovedCount}{42}" in macros
    assert r"\PolicyEffectCoordinates" in macros
    assert r"\LexCosVariableReductionPercent}{50}" in macros
    assert r"\LexCosHardClauseChangePercent}{14}" in macros
    assert r"\CommercialComparisonPairCount}{96}" in macros
    assert r"\EncodingSizeStrataConclusion" in macros
    assert "Weighted SIM score" in policy_table
    assert "Better & Unchanged & Worse" in policy_table
    assert r"Compatibility (\%) & Higher & 0 & 0 & 48 & 6.0 & [6.0, 6.0]" in policy_table
    assert "Continuity violations & Lower & 43 & 5 & 0" in policy_table
    assert "Weighted & SN & 40 / 6 / 2 & 120.0 & 15.0" in table
    assert r"LEX-COS & TOT & 40 / 6 / 2 & \textbf{100.0} & \textbf{10.0}" in table
    assert "Median runtime includes proved runs; PAR-2 includes all 48 runs" in table
    assert (output / "compact_result_provenance.json").is_file()
    stats = json.loads((output / "compact_figure_statistics.json").read_text())
    assert stats["runtime_reductions"]["weighted"]["median"] == pytest.approx(100 / 3)
    assert stats["runtime_reductions"]["weighted"]["excluded_pairs"] == 2
    assert "compact_policy_figure.tex" in report["outputs"]
    assert "compact_encoding_figure.tex" in report["outputs"]
    assert "compact_commercial_table.tex" in report["outputs"]
    commercial_table = (output / "compact_commercial_table.tex").read_text(
        encoding="utf-8"
    )
    assert "EvalMaxSAT" in commercial_table
    assert "Gurobi" in commercial_table
    assert "CPLEX" in commercial_table
    assert "Opt / Inf / TO" in commercial_table
    assert r"Gurobi & Weighted & \textbf{46 / 2 / 0}" in commercial_table


def test_runtime_reduction_uses_paired_percentages_and_excludes_timeouts() -> None:
    rows = [
        {"both_proved": "True", "sorting_elapsed_seconds": "100", "totalizer_elapsed_seconds": "50"},
        {"both_proved": "True", "sorting_elapsed_seconds": "100", "totalizer_elapsed_seconds": "200"},
        {"both_proved": "False", "sorting_elapsed_seconds": "3600", "totalizer_elapsed_seconds": "1"},
    ]
    stats = _runtime_reduction_summary(rows)
    assert stats["median"] == -25  # Median of 50% and -100%, not transformed median ratio.
    assert stats["reductions_percent"] == [-100, 50]
    assert stats["ci_low"] == -100
    assert stats["ci_high"] == 50
    assert stats["excluded_pairs"] == 1
    assert stats == _runtime_reduction_summary(rows[::-1])


@pytest.mark.parametrize("runtime", ["0", "-1", "nan", "inf"])
def test_runtime_reduction_rejects_invalid_times(runtime: str) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        _runtime_reduction_summary([{
            "both_proved": "True", "sorting_elapsed_seconds": runtime,
            "totalizer_elapsed_seconds": "1",
        }])


def test_generator_rejects_failed_encoding_gate(tmp_path: Path) -> None:
    policy, encoding, commercial, output = _inputs(tmp_path)
    (encoding / "policy_encoding_validation.json").write_text(
        json.dumps({"evidence_valid": False}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="evidence gate"):
        generate(policy, encoding, commercial, output)

    assert not output.exists()
