from __future__ import annotations

from fractions import Fraction
from itertools import product

import pytest

from hcorap.cpsat import (
    solve_cpsat_epsilon_constraint,
    solve_cpsat_lexicographic,
    solve_cpsat_weighted,
)
from hcorap.metrics import verify_assignments
from hcorap.model import Assignment
from hcorap.solvers import (
    similarity_budget,
    solve_epsilon_constraint,
    solve_lexicographic,
    solve_weighted,
)


def _feasible_metrics(instance, *, full_coverage=True):
    choices = []
    for service in range(instance.services):
        candidates = [
            Assignment(agent, service, slot)
            for agent, slot in instance.candidate_triplets(service)
        ]
        if not full_coverage:
            candidates.append(None)
        choices.append(candidates)
    metrics = []
    for selected in product(*choices):
        schedule = tuple(item for item in selected if item is not None)
        checked = verify_assignments(
            instance, schedule, require_full_coverage=full_coverage
        )
        if checked.valid:
            metrics.append(checked.metrics)
    return metrics


def _triple(result):
    return (
        result.metrics.similarity,
        result.metrics.continuity_penalty,
        result.metrics.overtime,
    )


def test_weighted_matches_bruteforce_and_cpsat(tradeoff_instance) -> None:
    oracle = max(
        _feasible_metrics(tradeoff_instance),
        key=lambda metric: metric.similarity - metric.continuity_penalty - metric.overtime,
    )
    maxsat = solve_weighted(tradeoff_instance)
    cpsat = solve_cpsat_weighted(tradeoff_instance)
    expected_score = oracle.similarity - oracle.continuity_penalty - oracle.overtime
    for result in (maxsat, cpsat):
        assert result.status == "OPTIMUM"
        assert (
            result.metrics.similarity
            - result.metrics.continuity_penalty
            - result.metrics.overtime
        ) == expected_score
    assert _triple(maxsat) == _triple(cpsat)


@pytest.mark.parametrize(
    "policy,key",
    [
        ("continuity-priority", lambda m: (m.continuity_penalty, -m.similarity, m.overtime)),
        ("overtime-priority", lambda m: (m.overtime, m.continuity_penalty, -m.similarity)),
    ],
)
def test_lexicographic_matches_bruteforce_and_cpsat(
    tradeoff_instance, policy, key
) -> None:
    oracle = min(_feasible_metrics(tradeoff_instance), key=key)
    maxsat = solve_lexicographic(tradeoff_instance, policy=policy)
    cpsat = solve_cpsat_lexicographic(tradeoff_instance, policy=policy)
    assert _triple(maxsat) == (
        oracle.similarity,
        oracle.continuity_penalty,
        oracle.overtime,
    )
    assert _triple(cpsat) == _triple(maxsat)


@pytest.mark.parametrize("delta", ["0", "0.2"])
def test_epsilon_constraint_matches_bruteforce_and_cpsat(
    tradeoff_instance, delta
) -> None:
    feasible = _feasible_metrics(tradeoff_instance)
    sim_star = max(metric.similarity for metric in feasible)
    threshold = similarity_budget(sim_star, delta)
    eligible = [metric for metric in feasible if metric.similarity >= threshold]
    oracle = min(
        eligible,
        key=lambda metric: (
            metric.continuity_penalty,
            metric.overtime,
            -metric.similarity,
        ),
    )
    maxsat = solve_epsilon_constraint(tradeoff_instance, delta=delta)
    cpsat = solve_cpsat_epsilon_constraint(tradeoff_instance, delta=delta)
    expected = (oracle.similarity, oracle.continuity_penalty, oracle.overtime)
    assert _triple(maxsat) == expected
    assert _triple(cpsat) == expected


def test_similarity_budget_uses_exact_ceiling() -> None:
    assert similarity_budget(101, Fraction(1, 100)) == 100
    assert similarity_budget(39, "0.025") == 39


def test_soft_coverage_is_optimized_first(partially_infeasible_instance) -> None:
    for result in (
        solve_weighted(partially_infeasible_instance, require_full_coverage=False),
        solve_lexicographic(partially_infeasible_instance, require_full_coverage=False),
        solve_cpsat_weighted(partially_infeasible_instance, require_full_coverage=False),
    ):
        assert result.status == "OPTIMUM"
        assert result.metrics.coverage == 1
        assert result.metrics.original_stability is None


def test_zero_timeout_is_reported(tradeoff_instance) -> None:
    result = solve_weighted(tradeoff_instance, timeout_seconds=0)
    assert result.status == "TIMEOUT"


def test_zero_overtime_weight_still_reports_exact_metric(tradeoff_instance) -> None:
    result = solve_weighted(tradeoff_instance, overtime_weight=0)
    assert result.status == "OPTIMUM"
    checked = verify_assignments(tradeoff_instance, result.assignments)
    assert result.metrics.overtime == checked.metrics.overtime
