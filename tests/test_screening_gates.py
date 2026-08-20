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
    encoding = tmp_path / "encoding"
    _write_campaign(encoding, encoding_rows)
    config = {
        "encoding_result_dir": str(encoding),
        "output": str(tmp_path / "decision.json"),
        "expected_measured_runs": 732,
        "encoding": {
            "baseline": {
                "cardinality": "sorting-network",
                "implied": "none",
                "symmetry": "none",
            },
            "reference_composite": {
                "cardinality": "totalizer",
                "implied": "both",
                "symmetry": "slot-service",
            },
            "maximum_objective_mismatches": 0,
            "minimum_paired_optimum_runs": 5,
            "minimum_reference_to_baseline_optimum_ratio": 0.95,
        },
        "maximum_peak_rss_mb": 12288,
        "maximum_hard_errors_per_campaign": 0,
    }
    config_path = tmp_path / "gates.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    initial = evaluate(config_path)
    assert initial["decision"] == "GO"
    assert initial["publication_scope"] == "COMPACT"
    assert initial["expected_measured_runs"] == 732
    assert initial["branches"]["original_lexicographic"]["enabled"]
    assert initial["branches"]["corrected_v2_lexicographic"]["enabled"]

    encoding_rows[1]["weighted_reference_score"] = "999"
    with (encoding / "runs.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(encoding_rows)
    decision = evaluate(config_path)
    assert decision["decision"] == "NO-GO"
    assert not decision["hard_stop_pass"]
    assert len(decision["encoding"]["objective_mismatches"]) == 1
