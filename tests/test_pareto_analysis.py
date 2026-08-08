from __future__ import annotations

import csv
import json
from pathlib import Path

from experiments.analyze_pareto_results import analyze


def test_pareto_analysis_deduplicates_delta_outcomes(tmp_path: Path) -> None:
    result_dir = tmp_path / "results"
    result_dir.mkdir()
    (result_dir / "validation.json").write_text(
        json.dumps({"complete": True}), encoding="utf-8"
    )
    fields = (
        "instance", "instance_sha256", "method", "cardinality", "implied",
        "symmetry", "delta", "status", "coverage", "similarity",
        "continuity", "overtime", "similarity_reference_optimum",
        "similarity_lower_bound", "similarity_realized_loss_absolute", "verified",
    )
    base = {
        "instance": "instance.txt",
        "instance_sha256": "instance-1",
        "method": "epsilon",
        "cardinality": "totalizer",
        "implied": "both",
        "symmetry": "slot-service",
        "status": "OPTIMUM",
        "coverage": "10",
        "similarity_reference_optimum": "50",
        "similarity_lower_bound": "45",
        "verified": "True",
    }
    rows = [
        {**base, "delta": "0", "similarity": "50", "continuity": "2", "overtime": "1", "similarity_realized_loss_absolute": "0"},
        {**base, "delta": "0.01", "similarity": "50", "continuity": "2", "overtime": "1", "similarity_realized_loss_absolute": "0"},
        {**base, "delta": "0.10", "similarity": "45", "continuity": "1", "overtime": "0", "similarity_realized_loss_absolute": "5"},
    ]
    with (result_dir / "runs.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    result = analyze(result_dir, tmp_path / "analysis")
    assert result["valid"] is True
    assert result["unique_points"] == 2
    assert result["nondominated_points"] == 2
