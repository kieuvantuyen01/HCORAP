from __future__ import annotations

import csv
import json
from pathlib import Path

from experiments.evaluate_commercial_correctness_smoke import (
    evaluate as evaluate_smoke,
)
from experiments.evaluate_evalmaxsat_calibration import (
    evaluate as evaluate_evalmaxsat,
)


def _campaign(path: Path, rows: list[dict[str, str]]) -> None:
    path.mkdir()
    (path / "validation.json").write_text(
        json.dumps({"complete": True}), encoding="utf-8"
    )
    with (path / "runs.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_smoke_gate_accepts_agreed_optimum_and_infeasible_groups(
    tmp_path: Path,
) -> None:
    rows = []
    backends = ("gurobi-mip", "cplex-mip", "reference-enumerator")
    for instance in range(3):
        for method in ("weighted", "lex-cos"):
            for backend in backends:
                infeasible = instance == 2
                rows.append(
                    {
                        "instance_sha256": f"sha-{instance}",
                        "method": method,
                        "backend": backend,
                        "status": "INFEASIBLE" if infeasible else "OPTIMUM",
                        "verified": "False" if infeasible else "True",
                        "coverage": "" if infeasible else "2",
                        "similarity": "" if infeasible else "8",
                        "continuity": "" if infeasible else "0",
                        "overtime": "" if infeasible else "1",
                        "weighted_reference_score": "" if infeasible else "7",
                    }
                )
    result_dir = tmp_path / "smoke"
    _campaign(result_dir, rows)
    assert evaluate_smoke(result_dir)["pass"] is True
    rows[0]["weighted_reference_score"] = "999"
    with (result_dir / "runs.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    assert evaluate_smoke(result_dir)["pass"] is False


def test_evalmaxsat_calibration_gate_requires_two_verified_optima(
    tmp_path: Path,
) -> None:
    rows = [
        {
            "status": status,
            "verified": "True" if status == "OPTIMUM" else "False",
        }
        for status in ("OPTIMUM", "OPTIMUM", "TIMEOUT", "TIMEOUT")
    ]
    result_dir = tmp_path / "calibration"
    _campaign(result_dir, rows)
    assert evaluate_evalmaxsat(result_dir)["pass"] is True
    rows[1] = {"status": "TIMEOUT", "verified": "False"}
    with (result_dir / "runs.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    assert evaluate_evalmaxsat(result_dir)["pass"] is False
