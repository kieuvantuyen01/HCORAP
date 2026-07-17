from __future__ import annotations

from hcorap.metrics import verify_assignments
from hcorap.model import Assignment


def test_metrics_match_documented_definitions(tradeoff_instance) -> None:
    schedule = (Assignment(1, 0, 0), Assignment(1, 1, 1))
    result = verify_assignments(tradeoff_instance, schedule)
    assert result.valid
    assert result.metrics.coverage == 2
    assert result.metrics.similarity == 8
    assert result.metrics.continuity_penalty == 0
    assert result.metrics.continuity_normalized == 1.0
    assert result.metrics.overtime == 1
    assert result.metrics.overtime_normalized == 0.5
    assert result.metrics.original_stability == 1


def test_verifier_reports_resource_conflicts(tradeoff_instance) -> None:
    schedule = (Assignment(0, 0, 0), Assignment(0, 1, 0))
    result = verify_assignments(tradeoff_instance, schedule)
    assert not result.valid
    assert any("agent 0 has 2 services" in item for item in result.violations)
    assert any("user 0 receives 2 services" in item for item in result.violations)


def test_partial_coverage_has_served_conditioned_continuity(tradeoff_instance) -> None:
    result = verify_assignments(
        tradeoff_instance,
        (Assignment(0, 0, 0),),
        require_full_coverage=False,
    )
    assert result.valid
    assert result.metrics.coverage == 1
    assert result.metrics.continuity_penalty == 0
    assert result.metrics.continuity_normalized == 1.0
    assert result.metrics.partially_served_sequences == 1
    assert result.metrics.original_stability is None
