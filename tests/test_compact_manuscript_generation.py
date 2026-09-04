from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from experiments.generate_compact_manuscript_results import generate


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    policy = tmp_path / "policy"
    encoding = tmp_path / "encoding"
    output = tmp_path / "generated"
    policy.mkdir()
    encoding.mkdir()

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
                    "proved_runs": 46 if totalizer else 45,
                    "par2_seconds": 100 if totalizer else 120,
                    "median_proved_seconds": 10 if totalizer else 15,
                    "median_peak_rss_mb": 80 if totalizer else 90,
                    "median_variables": 1000 if totalizer else 2000,
                    "median_hard_clauses": 4000 if totalizer else 3500,
                    "median_soft_clauses": 100,
                }
            )
    _write_csv(encoding / "policy_encoding_summary.csv", summaries)
    _write_csv(
        encoding / "policy_encoding_contrasts.csv",
        [
            {
                "method": method,
                "pairs": 48,
                "both_proved_pairs": 45,
                "totalizer_faster": 40,
                "median_speedup_sorting_over_totalizer": 1.5,
                "bootstrap_95_ci_low": 1.2,
                "bootstrap_95_ci_high": 1.8,
                "totalizer_faster_claim_supported": True,
            }
            for method in ("weighted", "lex-cos")
        ],
    )
    return policy, encoding, output


def test_generator_emits_only_after_both_gates_pass(tmp_path: Path) -> None:
    policy, encoding, output = _inputs(tmp_path)

    report = generate(policy, encoding, output)

    assert report["policy_gate"] is True
    assert report["encoding_gate"] is True
    assert report["totalizer_claim_supported"] == {
        "weighted": True,
        "lex-cos": True,
    }
    macros = (output / "compact_result_macros.tex").read_text(encoding="utf-8")
    table = (output / "compact_encoding_table.tex").read_text(encoding="utf-8")
    assert r"\BothPriorityMeasuresImprovedCount}{42}" in macros
    assert "Weighted & SN & 45" in table
    assert "LEX-COS & TOT & 46" in table
    assert (output / "compact_result_provenance.json").is_file()


def test_generator_rejects_failed_encoding_gate(tmp_path: Path) -> None:
    policy, encoding, output = _inputs(tmp_path)
    (encoding / "policy_encoding_validation.json").write_text(
        json.dumps({"evidence_valid": False}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="evidence gate"):
        generate(policy, encoding, output)

    assert not output.exists()
