from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments" / "run_reproducible_campaign.py"


def _module():
    specification = importlib.util.spec_from_file_location("campaign_runner", SCRIPT)
    assert specification and specification.loader
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_reproducible_campaign_runs_validates_and_resumes(tmp_path: Path) -> None:
    module = _module()
    result_dir = tmp_path / "results"
    config = {
        "binary": str(ROOT / "bin" / "release" / "hcorap_multi"),
        "solver": str(ROOT / "tests" / "rc2_open_wbo.py"),
        "result_dir": str(result_dir),
        "instances": [str(ROOT / "tests" / "instances" / "lex_cos_tie.txt")],
        "timeout_seconds": 30,
        "workers": 1,
        "configurations": [
            {"cardinality": "totalizer", "implied": "none", "symmetry": "none"}
        ],
        "runs": [
            {"method": "weighted", "wc": 1, "wo": 1},
            {"method": "lex-cos"},
        ],
    }
    config_path = tmp_path / "campaign.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    dry = module.run_campaign(config_path, dry_run=True)
    assert dry["valid"] is True
    assert dry["tasks"] == 2
    first = module.run_campaign(config_path)
    assert first["complete"] is True
    assert first["expected_runs"] == 2
    second = module.run_campaign(config_path, resume=True)
    assert second["complete"] is True
    records = [
        json.loads(line)
        for line in (result_dir / "manifest.jsonl").read_text().splitlines()
    ]
    assert len(records) == 2
    assert all(not record["validation_errors"] for record in records)
    assert {record["result_status"] for record in records} == {"OPTIMUM"}
    manifest_path = result_dir / "manifest.jsonl"
    relocated = []
    for record in records:
        record["result"] = f"/retired-gcp/results/raw/{record['run_id']}.json"
        record["instance"] = "/retired-gcp/tests/instances/lex_cos_tie.txt"
        relocated.append(json.dumps(record, sort_keys=True))
    manifest_path.write_text("\n".join(relocated) + "\n", encoding="utf-8")

    collector_spec = importlib.util.spec_from_file_location(
        "campaign_collector", ROOT / "experiments" / "collect_reproducible_campaign.py"
    )
    assert collector_spec and collector_spec.loader
    collector = importlib.util.module_from_spec(collector_spec)
    collector_spec.loader.exec_module(collector)
    summary = collector.collect(result_dir)
    assert summary["runs"] == 2
    assert summary["optimum_runs"] == 2
    assert (result_dir / "runs.csv").is_file()
    assert (result_dir / "summary_by_class.csv").is_file()


def test_campaign_preserves_a_verified_evalmaxsat_timeout_incumbent(
    tmp_path: Path,
) -> None:
    module = _module()
    result_dir = tmp_path / "timeout-results"
    config = {
        "binary": str(ROOT / "bin" / "release" / "hcorap_multi"),
        "solver": str(ROOT / "tests" / "rc2_timeout_incumbent.py"),
        "result_dir": str(result_dir),
        "instances": [str(ROOT / "tests" / "instances" / "lex_cos_tie.txt")],
        "timeout_seconds": 0.5,
        "solver_shutdown_grace_seconds": 2,
        "hard_grace_seconds": 10,
        "workers": 1,
        "configurations": [
            {"cardinality": "totalizer", "implied": "none", "symmetry": "none"}
        ],
        "runs": [
            {
                "variant": "staged-incumbent-bound",
                "method": "lex-cos",
                "align_evalmaxsat_tct": True,
                "stage3_incumbent_bound": True,
                "print_assignments": True,
            }
        ],
    }
    config_path = tmp_path / "timeout-campaign.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    result = module.run_campaign(config_path)
    assert result["complete"] is True
    record = json.loads(
        (result_dir / "manifest.jsonl").read_text(encoding="utf-8").strip()
    )
    assert record["result_status"] == "TIMEOUT_FEASIBLE"
    assert record["validation_errors"] == []
    payload = json.loads(Path(record["result"]).read_text(encoding="utf-8"))
    assert payload["schema_version"] == 3
    assert payload["metrics"]["verified"] is True
    assert payload["align_evalmaxsat_tct"] is True
    assert payload["stage3_incumbent_bound"] is True
    assert payload["certified_lexicographic_prefix"] == 2

    collector_spec = importlib.util.spec_from_file_location(
        "timeout_campaign_collector",
        ROOT / "experiments" / "collect_reproducible_campaign.py",
    )
    assert collector_spec and collector_spec.loader
    collector = importlib.util.module_from_spec(collector_spec)
    collector_spec.loader.exec_module(collector)
    summary = collector.collect(result_dir)
    assert summary["timeout_runs"] == 1
