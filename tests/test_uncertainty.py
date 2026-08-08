from __future__ import annotations

import csv
from pathlib import Path

from experiments.generate_uncertainty_scenarios import generate_scenarios
from experiments.verify_uncertainty_scenarios import verify
from hcorap.generator import generate_benchmark_batch
from hcorap.io import read_instance


def test_agent_day_scenarios_are_paired_nested_and_capacity_safe(tmp_path: Path) -> None:
    benchmark = tmp_path / "benchmark"
    generate_benchmark_batch(
        users=(3,),
        agent_counts=(2,),
        services_per_user_counts=(2,),
        calibration_seeds=(11,),
        evaluation_seeds=(),
        load_profiles=("critical",),
        normal_fraction=0.85,
        output_dir=benchmark,
        days=2,
        slots_per_day=6,
    )
    base = next(benchmark.glob("calibration/critical/*.txt"))
    scenario_root = tmp_path / "scenarios"
    result = generate_scenarios(
        (base,),
        probabilities=("0", "0.5", "1"),
        scenario_seeds=(101,),
        output_dir=scenario_root,
    )
    assert result["scenarios"] == 3
    checked = verify(scenario_root)
    assert checked["valid"] is True

    with (scenario_root / "scenarios.csv").open(newline="") as stream:
        rows = sorted(csv.DictReader(stream), key=lambda row: float(row["absence_probability"]))
    original = read_instance(base)
    no_disruption = read_instance(Path(rows[0]["scenario_instance"]))
    full_disruption = read_instance(Path(rows[-1]["scenario_instance"]))
    assert no_disruption.agent_availability == original.agent_availability
    assert all(value == 0 for row in full_disruption.agent_availability for value in row)
    assert sum(full_disruption.normal_hours) + sum(full_disruption.extra_hours) == 0
    removed = [int(row["removed_available_slots"]) for row in rows]
    assert removed == sorted(removed)
