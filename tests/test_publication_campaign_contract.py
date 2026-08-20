from __future__ import annotations

import json
import shutil
from pathlib import Path

from experiments.validate_publication_campaign import validate


MANIFEST = Path("experiments/configs/reduced_campaign_manifest.json")


def test_publication_campaign_contract_is_locked() -> None:
    report = validate(MANIFEST)
    assert report["valid"] is True
    assert report["measured_runs"] == 1270
    assert report["measured_timeout_seconds"] == 300
    assert report["maxsat_solver"]["name"] == "EvalMaxSAT"
    assert report["worst_case_seconds"] == 381_000


def test_publication_campaign_contract_rejects_timeout_drift(tmp_path: Path) -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for campaign in manifest["measured_campaigns"] + manifest["non_measured_campaigns"]:
        shutil.copy(MANIFEST.parent / campaign["config"], tmp_path / campaign["config"])
    (tmp_path / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    factorial_path = tmp_path / "gcp_original_ablation.json"
    factorial = json.loads(factorial_path.read_text(encoding="utf-8"))
    factorial["timeout_seconds"] = 120
    factorial_path.write_text(json.dumps(factorial), encoding="utf-8")

    report = validate(tmp_path / "manifest.json")
    assert report["valid"] is False
    assert any("gcp_original_ablation.json timeout" in error for error in report["errors"])
