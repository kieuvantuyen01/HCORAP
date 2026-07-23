from __future__ import annotations

import pytest

from experiments.collect_epsilon_results import deduplicate_points
from experiments.collect_main_results import compute_par2


def test_epsilon_points_are_deduplicated_per_instance_and_configuration() -> None:
    common = {
        "instance_name": "tradeoff",
        "cfg_id": "1",
        "label": "ORIGINAL",
        "cardinality": "sorting-network",
        "ic": "none",
        "sb": "none",
        "status": "OPTIMUM",
        "coverage": "2",
        "similarity_reference_optimum": "9",
    }
    rows = [
        {
            **common,
            "delta": "0",
            "similarity": "9",
            "continuity": "1",
            "overtime": "0",
            "elapsed_s": "1.0",
        },
        {
            **common,
            "delta": "0.01",
            "similarity": "9",
            "continuity": "1",
            "overtime": "0",
            "elapsed_s": "2.0",
        },
        {
            **common,
            "delta": "0.2",
            "similarity": "8",
            "continuity": "0",
            "overtime": "1",
            "elapsed_s": "3.0",
        },
        {
            **common,
            "status": "TIMEOUT",
            "delta": "0.3",
            "similarity": "",
            "continuity": "",
            "overtime": "",
            "elapsed_s": "30",
        },
    ]

    points = deduplicate_points(rows)
    assert len(points) == 2
    assert points[0]["similarity"] == "9"
    assert points[0]["delta_count"] == 2
    assert points[0]["deltas"] == "0 | 0.01"
    assert points[0]["mean_elapsed_s"] == 1.5
    assert points[1]["similarity"] == "8"
    assert points[1]["similarity_realized_loss_absolute"] == 1


def test_main_collector_par2_uses_per_run_timeout_and_accepts_unsat() -> None:
    rows = [
        {"status": "OPTIMUM", "elapsed_s": 1, "timeout_seconds": 10},
        {"status": "TIMEOUT", "elapsed_s": 10, "timeout_seconds": 10},
        {"status": "UNSATISFIABLE", "elapsed_s": 2, "timeout_seconds": 10},
    ]
    assert compute_par2(rows) == pytest.approx((1 + 20 + 2) / 3)
