from __future__ import annotations

import json
from pathlib import Path

from experiments.validate_campaign_manifest import validate


def test_reduced_campaign_manifest_matches_locked_budget() -> None:
    report = validate(Path("experiments/configs/reduced_campaign_manifest.json"))
    assert report["valid"] is True
    assert report["measured_runs"] == 4896
    assert report["worst_case_seconds"] == 1_176_960


def test_campaign_manifest_rejects_config_drift(tmp_path: Path) -> None:
    (tmp_path / "campaign.json").write_text(
        json.dumps(
            {
                "expected_runs": 9,
                "timeout_seconds": 60,
                "result_dir": "../results/example",
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "measured_campaigns": [
                    {
                        "name": "example",
                        "config": "campaign.json",
                        "expected_runs": 10,
                        "timeout_seconds": 60,
                    }
                ],
                "non_measured_campaigns": [],
                "expected_measured_runs": 10,
                "expected_worst_case_seconds": 600,
                "expected_worst_case_core_hours": 1 / 6,
            }
        ),
        encoding="utf-8",
    )

    report = validate(tmp_path / "manifest.json")
    assert report["valid"] is False
    assert any("expected_runs" in error for error in report["errors"])
