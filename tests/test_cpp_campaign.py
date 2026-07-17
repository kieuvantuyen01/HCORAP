from __future__ import annotations

import csv
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "experiments" / "run_cpp_experiments.sh"
SOLVER = ROOT / "tests" / "rc2_open_wbo.py"
INSTANCE = ROOT / "tests" / "instances" / "symmetry.txt"
BINARY = ROOT / "bin" / "release" / "hcorap_multi"


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        delimiter = "\t" if path.suffix == ".tsv" else ","
        return list(csv.DictReader(stream, delimiter=delimiter))


def test_cpp_campaign_writes_matrix_and_excel_ready_logs(tmp_path: Path) -> None:
    result_dir = tmp_path / "campaign"
    environment = os.environ.copy()
    environment.update(
        {
            "HCORAP_MULTI_BIN": str(BINARY),
            "CARDINALITY_ENCODINGS": "sorting-network totalizer",
            "IMPLIED_CONFIGS": "none both-plus",
            "SYMMETRY_CONFIGS": "none all",
            "DELTAS": "0",
            "TIMEOUT": "30",
            "SOLVER_ID": "rc2-test",
        }
    )
    subprocess.run(
        [str(RUNNER), str(SOLVER), str(result_dir), str(INSTANCE)],
        cwd=ROOT,
        env=environment,
        check=True,
        text=True,
        capture_output=True,
        timeout=30,
    )

    matrix = _csv_rows(result_dir / "configuration_matrix.tsv")
    raw = _csv_rows(result_dir / "runs.csv")
    summary = _csv_rows(result_dir / "configuration_summary.csv")

    assert len(matrix) == 8
    assert len(raw) == 32
    assert len(summary) == 32
    assert {row["status"] for row in raw} == {"OPTIMUM"}
    assert {row["verified"] for row in raw} == {"True"}
    assert len(
        {
            (
                row["cardinality_encoding"],
                row["implied_constraints"],
                row["symmetry_breaking"],
            )
            for row in raw
        }
    ) == 8
    assert {row["method"] for row in raw} == {
        "weighted",
        "lex-continuity",
        "lex-overtime",
        "epsilon",
    }
