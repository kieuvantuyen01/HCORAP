from __future__ import annotations

import csv
import json
from pathlib import Path

from experiments.analyze_weight_sensitivity import analyze


ROOT = Path(__file__).resolve().parents[1]


def test_weight_analysis_tracks_scale_equivalent_runs(tmp_path: Path) -> None:
    result_dir = tmp_path / "results"
    result_dir.mkdir()
    (result_dir / "validation.json").write_text(
        json.dumps({"complete": True}), encoding="utf-8"
    )
    fields = (
        "instance", "instance_sha256", "users", "agents", "visits",
        "load_profile", "method", "wc", "wo", "status", "coverage",
        "similarity", "continuity", "overtime", "assignment_sha256",
        "verified",
    )
    base = {
        "instance": str(ROOT / "tests" / "instances" / "tradeoff.txt"),
        "instance_sha256": "instance-1",
        "users": "1",
        "agents": "2",
        "visits": "2",
        "load_profile": "",
        "method": "weighted",
        "status": "OPTIMUM",
        "coverage": "2",
        "similarity": "8",
        "continuity": "0",
        "overtime": "1",
        "assignment_sha256": "assignment-1",
        "verified": "True",
    }
    rows = [{**base, "wc": "1", "wo": "1"}, {**base, "wc": "2", "wo": "2"}]
    with (result_dir / "runs.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    result = analyze(result_dir, tmp_path / "analysis")
    assert result["valid"] is True
    assert result["repeated_scale_groups"] == 1
    assert result["scale_groups_with_one_vector"] == 1
