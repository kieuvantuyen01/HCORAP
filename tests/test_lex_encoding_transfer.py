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


def _row(
    *, index: int, configuration: tuple[str, str, str], stage_count: int
) -> dict[str, object]:
    objectives = " | ".join(("continuity", "overtime")[:stage_count])
    optima = " | ".join(("1", "0")[:stage_count])
    full = configuration == FULL
    return {
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
        "stage_objectives": objectives,
        "stage_optima": optima,
        "solver_calls": stage_count + 1,
        "elapsed_seconds": 300,
        "timeout_seconds": 300,
        "solve_seconds_sum": 12 if full else 10,
        "encode_seconds_sum": 1.2 if full else 1,
        "variables_max": 1200 if full else 1000,
        "hard_clauses_max": 2400 if full else 2000,
        "peak_rss_mb": 115 if full else 100,
    }


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
                _row(
                    index=index,
                    configuration=configuration,
                    stage_count=stage_count,
                )
            )
    _write_csv(results / "runs.csv", rows)

    report = analyze(results, output, expected_instances=16)

    assert report["structurally_valid"]
    assert report["stage_wins"] == 4
    assert report["decision"] == "GO"
    assert (output / "lex_encoding_transfer_pairs.csv").is_file()


def test_pilot_stop_reports_paired_footprint(tmp_path: Path) -> None:
    results = tmp_path / "results"
    output = tmp_path / "analysis"
    results.mkdir()
    (results / "validation.json").write_text(
        json.dumps({"complete": True}), encoding="utf-8"
    )
    rows = [
        _row(index=index, configuration=configuration, stage_count=2)
        for index in range(16)
        for configuration in (TOTALIZER_ONLY, FULL)
    ]
    _write_csv(results / "runs.csv", rows)

    report = analyze(results, output, expected_instances=16)

    assert report["structurally_valid"]
    assert report["decision"] == "STOP"
    assert report["common_stage_value_matches"] == 16
    assert report["both_reached_final_stage"] == 16
    assert report["full_slower_on_completed_stages"] == 16
    assert report["median_full_over_totalizer_completed_solve_ratio"] == 1.2
    assert report["median_variables_difference"] == 200
    assert report["median_hard_clauses_difference"] == 400
    assert report["median_peak_rss_difference_mb"] == 15


def test_pilot_rejects_disagreement_on_completed_stage_optimum(tmp_path: Path) -> None:
    results = tmp_path / "results"
    output = tmp_path / "analysis"
    results.mkdir()
    (results / "validation.json").write_text(
        json.dumps({"complete": True}), encoding="utf-8"
    )
    rows = [
        _row(index=index, configuration=configuration, stage_count=2)
        for index in range(16)
        for configuration in (TOTALIZER_ONLY, FULL)
    ]
    rows[1]["stage_optima"] = "2 | 0"
    _write_csv(results / "runs.csv", rows)

    report = analyze(results, output, expected_instances=16)

    assert not report["structurally_valid"]
    assert report["decision"] == "INVALID"
    assert not report["structural_checks"]["common_stage_values"]
