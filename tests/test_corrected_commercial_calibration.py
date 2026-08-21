from __future__ import annotations

import csv
import json
from pathlib import Path

from experiments.evaluate_corrected_commercial_calibration import evaluate


def test_corrected_commercial_calibration_gate(tmp_path: Path) -> None:
    rows = []
    for index in range(8):
        for backend in ("gurobi-mip", "cplex-mip"):
            for method in ("weighted", "lex-cos", "lex-overtime"):
                rows.append(
                    {
                        "instance_sha256": f"sha-{index}",
                        "backend": backend,
                        "method": method,
                        "status": "OPTIMUM",
                        "verified": "True",
                        "coverage": "20",
                        "similarity": "90",
                        "continuity": "1",
                        "overtime": "0",
                        "weighted_reference_score": "88",
                    }
                )
    result_dir = tmp_path / "calibration"
    result_dir.mkdir()
    (result_dir / "validation.json").write_text(
        json.dumps({"complete": True}), encoding="utf-8"
    )
    with (result_dir / "runs.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    gates = Path(
        "experiments/configs/corrected_commercial_calibration_gates.json"
    )
    assert evaluate(result_dir, gates)["pass"] is True
    rows[0]["weighted_reference_score"] = "999"
    with (result_dir / "runs.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = evaluate(result_dir, gates)
    assert result["pass"] is False
    assert result["objective_disagreements"] == 1
