from __future__ import annotations

from pathlib import Path

from experiments.audit_existing_results import audit


def test_historical_audit_counts_only_run_level_raw_provenance(
    tmp_path: Path,
) -> None:
    results = tmp_path / "results"
    results.mkdir()
    (results / "audit_20260808.json").write_text("{}\n", encoding="utf-8")
    ignored_pilot = tmp_path / "experiments" / "results" / "pilot" / "raw"
    ignored_pilot.mkdir(parents=True)
    (ignored_pilot / "run.json").write_text("{}\n", encoding="utf-8")

    without_raw = audit(tmp_path)
    assert without_raw["provenance"]["raw_json_files_under_historical_results"] == 0
    assert without_raw["provenance"]["historical_weighted_csv_has_full_raw_json"] is False

    historical_raw = results / "campaign" / "raw"
    historical_raw.mkdir(parents=True)
    (historical_raw / "run.json").write_text("{}\n", encoding="utf-8")
    with_raw = audit(tmp_path)
    assert with_raw["provenance"]["raw_json_files_under_historical_results"] == 1
    assert with_raw["provenance"]["historical_weighted_csv_has_full_raw_json"] is True
