from __future__ import annotations

import csv
import json
from pathlib import Path

from experiments.analyze_lex_encoding_transfer import FULL, TOTALIZER_ONLY, analyze


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_pilot_gate_accepts_stage_progress(tmp_path: Path) -> None:
    results = tmp_path / "results"
    output = tmp_path / "analysis"
    results.mkdir()
    (results / "validation.json").write_text(
        json.dumps({"complete": True}), encoding="utf-8"
    )
    rows = []
    for index in range(16):
        for configuration in (TOTALIZER_ONLY, FULL):
            totalizer_only = configuration == TOTALIZER_ONLY
            stage_count = 2 if totalizer_only and index < 4 else 1
            rows.append(
                {
                    "instance_sha256": f"sha-{index}",
                    "instance": f"instance-{index}.txt",
                    "users": 30,
                    "agents": 10,
                    "visits": 4,
                    "seed": 1002,
                    "method": "lex-cos",
                    "cardinality": configuration[0],
                    "implied": configuration[1],
                    "symmetry": configuration[2],
                    "status": "TIMEOUT",
                    "verified": "",
                    "stage_count": stage_count,
                    "elapsed_seconds": 300,
                    "timeout_seconds": 300,
                }
            )
    _write_csv(results / "runs.csv", rows)

    report = analyze(results, output, expected_instances=16)

    assert report["structurally_valid"]
    assert report["stage_wins"] == 4
    assert report["decision"] == "GO"
    assert (output / "lex_encoding_transfer_pairs.csv").is_file()
