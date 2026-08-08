from __future__ import annotations

import json
from pathlib import Path

from experiments.collect_commercial_campaign import collect
from experiments.run_commercial_campaign import run_campaign


ROOT = Path(__file__).resolve().parents[1]


def test_commercial_campaign_preflights_runs_collects_and_resumes(tmp_path: Path) -> None:
    result_dir = tmp_path / "results"
    config = {
        "binary": str(ROOT / "bin" / "release" / "hcorap_commercial"),
        "result_dir": str(result_dir),
        "instances": [str(ROOT / "tests" / "instances" / "sparse_users.txt")],
        "expected_instances": 1,
        "expected_runs": 2,
        "timeout_seconds": 10,
        "preflight_timeout_seconds": 10,
        "threads": 1,
        "seed": 0,
        "mip_gap": 0,
        "absolute_mip_gap": 0,
        "workers": 1,
        "commercial_configurations": [
            {
                "backend": "reference-enumerator",
                "formulation": "direct-schedule-enumeration",
            }
        ],
        "runs": [
            {"method": "weighted", "native_log": False},
            {"method": "lex-cos", "native_log": False},
        ],
    }
    config_path = tmp_path / "commercial.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    dry = run_campaign(config_path, dry_run=True)
    assert dry["valid"] is True
    assert dry["tasks"] == 2
    first = run_campaign(config_path)
    assert first["complete"] is True
    assert first["complete_runs"] == 2
    second = run_campaign(config_path, resume=True)
    assert second["complete"] is True
    summary = collect(result_dir)
    assert summary["runs"] == 2
    rows = [
        json.loads(line)
        for line in (result_dir / "manifest.jsonl").read_text().splitlines()
    ]
    assert {row["result_status"] for row in rows} == {"OPTIMUM"}
    assert all(not row["validation_errors"] for row in rows)
