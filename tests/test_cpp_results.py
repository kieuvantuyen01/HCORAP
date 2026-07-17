from __future__ import annotations

import pytest

from experiments.collect_cpp_results import summarize


def test_summary_recognizes_cpp_unsatisfiable_status() -> None:
    common = {
        "cardinality_encoding": "sorting-network",
        "implied_constraints": "none",
        "symmetry_breaking": "none",
        "method": "weighted",
        "delta": "-",
        "timeout_seconds": 10,
        "verified": False,
    }
    rows = [
        {**common, "status": "OPTIMUM", "elapsed_seconds": 1},
        {**common, "status": "UNSATISFIABLE", "elapsed_seconds": 2},
        {**common, "status": "TIMEOUT", "elapsed_seconds": 10},
    ]

    summary = summarize(rows)[0]
    assert summary["optimum_runs"] == 1
    assert summary["unsat_runs"] == 1
    assert summary["timeout_runs"] == 1
    assert summary["error_runs"] == 0
    assert summary["par2_seconds"] == pytest.approx((1 + 2 + 20) / 3)
