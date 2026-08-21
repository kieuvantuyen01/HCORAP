from __future__ import annotations

import csv
import json
from pathlib import Path

from experiments.analyze_corrected_validation import analyze


def test_corrected_validation_builds_policy_and_paired_tables(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "validation.json").write_text(
        json.dumps({"complete": True}), encoding="utf-8"
    )
    rows = []
    for index in range(48):
        for method in ("weighted", "lex-cos", "lex-overtime"):
            rows.append(
                {
                    "instance_sha256": f"sha-{index}",
                    "instance": f"instance-{index}.txt",
                    "method": method,
                    "cardinality": "totalizer",
                    "implied": "both",
                    "symmetry": "slot-service",
                    "load_profile": "critical",
                    "status": "OPTIMUM",
                    "verified": "True",
                    "elapsed_seconds": "2",
                    "timeout_seconds": "300",
                    "peak_rss_mb": "100",
                    "similarity": "90" if method != "weighted" else "100",
                    "continuity": "1" if method != "weighted" else "3",
                    "overtime": "0" if method != "weighted" else "1",
                }
            )
    with (source / "runs.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    result = analyze(source, tmp_path / "analysis")
    assert result["valid"]
    assert result["manuscript_eligible"]
    assert result["scope"] == "corrected-v2-evalmaxsat-scalability"
    with (tmp_path / "analysis" / "corrected_paired_summary.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        summary = next(csv.DictReader(stream))
    assert summary["both_optimum_pairs"] == "48"
    assert summary["median_continuity_change"] == "-2.0"
    assert summary["median_overtime_change"] == "-1.0"


def test_corrected_maxsat_structure_can_pass_without_policy_evidence(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "validation.json").write_text(
        json.dumps({"complete": True}), encoding="utf-8"
    )
    rows = []
    for index in range(48):
        for method in ("weighted", "lex-cos", "lex-overtime"):
            rows.append(
                {
                    "instance_sha256": f"sha-{index}",
                    "instance": f"instance-{index}.txt",
                    "method": method,
                    "cardinality": "totalizer",
                    "implied": "both",
                    "symmetry": "slot-service",
                    "load_profile": "critical",
                    "status": "TIMEOUT",
                    "verified": "False",
                    "elapsed_seconds": "300",
                    "timeout_seconds": "300",
                    "peak_rss_mb": "100",
                    "similarity": "",
                    "continuity": "",
                    "overtime": "",
                }
            )
    with (source / "runs.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = analyze(source, tmp_path / "analysis")
    assert result["valid"] is True
    assert result["evidence_sufficient"] is False
    assert result["manuscript_eligible"] is False
