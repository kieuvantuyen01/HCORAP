from __future__ import annotations

import csv
import json
from pathlib import Path

from experiments.evaluate_screening_gates import evaluate


FIELDS = (
    "instance_sha256",
    "method",
    "delta",
    "wc",
    "wo",
    "cardinality",
    "implied",
    "symmetry",
    "status",
    "coverage",
    "similarity",
    "continuity",
    "overtime",
    "weighted_reference_score",
    "peak_rss_mb",
)


def _write_campaign(path: Path, rows: list[dict[str, str]]) -> None:
    path.mkdir()
    (path / "validation.json").write_text(
        json.dumps({"complete": True}), encoding="utf-8"
    )
    with (path / "runs.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _row(index: int, **updates: str) -> dict[str, str]:
    row = {
        "instance_sha256": f"instance-{index}",
        "method": "weighted",
        "delta": "-",
        "wc": "1",
        "wo": "1",
        "cardinality": "totalizer",
        "implied": "none",
        "symmetry": "slot-service",
        "status": "OPTIMUM",
        "coverage": "16",
        "similarity": str(40 + index),
        "continuity": str(index % 3),
        "overtime": str(index % 2),
        "weighted_reference_score": str(100 + index),
        "peak_rss_mb": "100",
    }
    row.update(updates)
    return row


def test_screening_gate_reports_go_and_detects_objective_mismatch(
    tmp_path: Path,
) -> None:
    encoding_rows = []
    for index in range(5):
        encoding_rows.extend(
            [
                _row(
                    index,
                    cardinality="sorting-network",
                    symmetry="none",
                ),
                _row(index, implied="both"),
            ]
        )
    multi_rows = [
        _row(index, method="lex-cos", status="OPTIMUM" if index == 0 else "TIMEOUT")
        for index in range(10)
    ] + [
        _row(
            20 + index,
            method="epsilon",
            delta="0.05",
            status="OPTIMUM" if index == 0 else "TIMEOUT",
        )
        for index in range(10)
    ]
    weight_rows = [
        _row(
            40,
            wc=str(index + 1),
            status="OPTIMUM" if index < 2 else "TIMEOUT",
            similarity=str(50 + index),
        )
        for index in range(10)
    ]

    encoding = tmp_path / "encoding"
    multiobjective = tmp_path / "multiobjective"
    weights = tmp_path / "weights"
    lex_scalability = tmp_path / "lex_scalability"
    _write_campaign(encoding, encoding_rows)
    _write_campaign(multiobjective, multi_rows)
    _write_campaign(weights, weight_rows)
    scalability_rows = []
    for configuration in (
        {"cardinality": "sorting-network", "implied": "none", "symmetry": "none"},
        {"cardinality": "totalizer", "implied": "both", "symmetry": "slot-service"},
    ):
        for index in range(5):
            scalability_rows.extend(
                [
                    _row(100 + index, **configuration),
                    _row(
                        100 + index,
                        method="lex-cos",
                        status="OPTIMUM" if index < 3 else "TIMEOUT",
                        **configuration,
                    ),
                ]
            )
    _write_campaign(lex_scalability, scalability_rows)
    config = {
        "encoding_result_dir": str(encoding),
        "multiobjective_result_dir": str(multiobjective),
        "weight_result_dir": str(weights),
        "lex_scalability_result_dir": str(lex_scalability),
        "output": str(tmp_path / "decision.json"),
        "encoding": {
            "baseline": {
                "cardinality": "sorting-network",
                "implied": "none",
                "symmetry": "none",
            },
            "proposed": {
                "cardinality": "totalizer",
                "implied": "both",
                "symmetry": "slot-service",
            },
            "maximum_objective_mismatches": 0,
            "minimum_paired_optimum_runs": 5,
            "minimum_proposed_to_baseline_optimum_ratio": 0.95,
        },
        "multiobjective": {
            "minimum_lex_cos_optimum_rate": 0.10,
            "minimum_epsilon_optimum_rate": 0.10,
        },
        "weights": {
            "minimum_optimum_rate": 0.10,
            "minimum_instances_with_multiple_vectors": 1,
        },
        "lex_scalability": {
            "minimum_b0_optimum_runs": 5,
            "minimum_best_configuration_completion_rate": 0.60,
            "maximum_peak_rss_mb": 12288,
        },
        "maximum_hard_errors_per_campaign": 0,
    }
    config_path = tmp_path / "gates.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    assert evaluate(config_path)["decision"] == "GO"
    encoding_rows[1]["weighted_reference_score"] = "999"
    with (encoding / "runs.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(encoding_rows)
    decision = evaluate(config_path)
    assert decision["decision"] == "NO-GO"
    assert len(decision["encoding"]["objective_mismatches"]) == 1
