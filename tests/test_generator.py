from __future__ import annotations

import csv
import json

from hcorap.generator import (
    LANGUAGES,
    benchmark_diagnostics,
    calibrate_capacity,
    generation_witness,
    generate_benchmark_batch,
    generate_nested_family,
)
from hcorap.metrics import verify_assignments


def test_corrected_generator_is_seeded_exact_and_nested() -> None:
    first = generate_nested_family(
        users=3,
        agent_counts=(2, 4),
        services_per_user_counts=(2, 3),
        seed=19,
        days=2,
        slots_per_day=6,
    )
    second = generate_nested_family(
        users=3,
        agent_counts=(2, 4),
        services_per_user_counts=(2, 3),
        seed=19,
        days=2,
        slots_per_day=6,
    )
    assert first == second
    for (agents, visits), instance in first.items():
        assert instance.agents == agents
        assert instance.services == 3 * visits
        assert all(len(group) == visits for group in instance.services_by_user)
        assert instance.to_summary()["services_without_candidates"] == 0
        assert all(
            user["language"] in LANGUAGES
            for user in instance.metadata["users_raw"]
        )
        witness = generation_witness(instance)
        assert len(witness) == instance.services
        assert verify_assignments(instance, witness).valid is True

    small = first[(2, 2)]
    large = first[(4, 3)]
    parent_ids = small.metadata["nested_parent"]["selected_parent_service_ids"]
    assert small.agent_availability == large.agent_availability[:2]
    assert small.service_availability == tuple(
        large.service_availability[parent] for parent in parent_ids
    )


def test_capacity_calibration_hits_declared_rho_and_splits_overtime() -> None:
    instance = generate_nested_family(
        users=3,
        agent_counts=(2,),
        services_per_user_counts=(2,),
        seed=7,
        days=2,
        slots_per_day=6,
    )[(2, 2)]
    calibrated = calibrate_capacity(
        instance,
        target_rho=0.85,
        normal_fraction=0.75,
        load_profile="critical",
    )
    total = sum(calibrated.normal_hours) + sum(calibrated.extra_hours)
    assert total == 8  # ceil(6 / 0.85)
    assert sum(calibrated.normal_hours) == 6
    assert all(
        normal + extra <= sum(availability)
        for normal, extra, availability in zip(
            calibrated.normal_hours,
            calibrated.extra_hours,
            calibrated.agent_availability,
        )
    )
    diagnostics = benchmark_diagnostics(calibrated)
    assert diagnostics["load_profile"] == "critical"
    assert diagnostics["rho"] == 0.75
    assert verify_assignments(calibrated, generation_witness(calibrated)).valid is True


def test_benchmark_batch_freezes_disjoint_splits_and_hashes(tmp_path) -> None:
    result = generate_benchmark_batch(
        users=(3,),
        agent_counts=(2,),
        services_per_user_counts=(2,),
        calibration_seeds=(1,),
        evaluation_seeds=(2,),
        load_profiles=("critical",),
        normal_fraction=0.85,
        output_dir=tmp_path,
        days=2,
        slots_per_day=6,
    )
    assert result["instances"] == 2
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["calibration_seeds"] == [1]
    assert manifest["evaluation_seeds"] == [2]
    with (tmp_path / "diagnostics.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert {row["split"] for row in rows} == {"calibration", "evaluation"}
    assert all(len(row["sha256"]) == 64 for row in rows)
