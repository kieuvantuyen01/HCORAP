from __future__ import annotations

import json
from pathlib import Path

from hcorap.experiment import run_experiment_config
from hcorap.io import write_instance


def test_experiment_jsonl_is_reproducible_and_resumable(
    tradeoff_instance, tmp_path: Path
) -> None:
    instance_path = tmp_path / "tiny.txt"
    write_instance(tradeoff_instance, instance_path)
    config_path = tmp_path / "pilot.json"
    config_path.write_text(
        json.dumps(
            {
                "instances": ["tiny.txt"],
                "output": "result.jsonl",
                "methods": [
                    {
                        "method": "weighted",
                        "continuity_weight": 1,
                        "overtime_weight": 1,
                        "timeout_seconds": 5,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    summary = run_experiment_config(config_path)
    assert summary["executed"] == 1
    record = json.loads((tmp_path / "result.jsonl").read_text(encoding="utf-8"))
    assert record["result"]["status"] == "OPTIMUM"
    assert record["instance"]["sha256"]
    assert record["environment"]["python"]

    resumed = run_experiment_config(config_path, resume=True)
    assert resumed["executed"] == 0
    assert resumed["skipped"] == 1
