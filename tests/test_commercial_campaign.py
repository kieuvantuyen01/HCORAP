from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.collect_commercial_campaign import collect
from experiments.run_commercial_campaign import _build_tasks, run_campaign


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("name", "expected_instances", "expected_tasks"),
    [
        ("gcp_commercial_original.json", 20, 80),
        ("gcp_commercial_correctness_smoke.json", 3, 18),
    ],
)
def test_publication_commercial_configs_expand_to_locked_task_counts(
    name: str, expected_instances: int, expected_tasks: int
) -> None:
    config_path = ROOT / "experiments" / "configs" / name
    config = json.loads(config_path.read_text(encoding="utf-8"))
    commercial_configs = [
        {
            **item,
            "resolved_parameter_file": item.get("parameter_file"),
        }
        for item in config["commercial_configurations"]
    ]
    tasks = _build_tasks(
        config,
        base=config_path.parent,
        binary_hash="publication-test-binary",
        commercial_configs=commercial_configs,
    )
    assert len({task["instance_sha256"] for task in tasks}) == expected_instances
    assert len(tasks) == expected_tasks


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
    manifest_path = result_dir / "manifest.jsonl"
    relocated = []
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        record["result"] = f"/retired-gcp/results/raw/{record['run_id']}.json"
        record["instance"] = "/retired-gcp/tests/instances/sparse_users.txt"
        relocated.append(json.dumps(record, sort_keys=True))
    manifest_path.write_text("\n".join(relocated) + "\n", encoding="utf-8")
    summary = collect(result_dir)
    assert summary["runs"] == 2
    rows = [
        json.loads(line)
        for line in manifest_path.read_text().splitlines()
    ]
    assert {row["result_status"] for row in rows} == {"OPTIMUM"}
    assert all(not row["validation_errors"] for row in rows)
