"""Independent solution verification and metric computation.

The verifier deliberately works from assignment triples rather than solver
variables.  It is therefore suitable for checking MaxSAT, CP-SAT, brute-force
and (after parsing their output) the authors' C++ solutions with the same code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .model import Assignment, HCORAPInstance, Metrics


@dataclass(frozen=True)
class VerificationResult:
    """Result of validating a concrete schedule against an instance."""

    valid: bool
    violations: Tuple[str, ...]
    metrics: Metrics


def _attainable_similarity_upper_bound(instance: HCORAPInstance) -> int:
    """Return a fast optimistic similarity bound.

    This ignores conflicts between services, agents and time slots, so it is an
    upper bound rather than a claim that the value is jointly attainable.
    """

    upper_bound = 0
    for service in range(instance.services):
        feasible_agents = {
            agent for agent, _slot in instance.candidate_triplets(service)
        }
        upper_bound += max(
            (instance.rewards[agent][service] for agent in feasible_agents),
            default=0,
        )
    return upper_bound


def compute_metrics(
    instance: HCORAPInstance, assignments: Iterable[Assignment]
) -> Metrics:
    """Compute all reported criteria from a concrete assignment set.

    Invalid assignments are still counted.  Call :func:`verify_assignments`
    when feasibility is required; keeping metric computation total makes error
    reports more useful during encoder development.
    """

    schedule = tuple(assignments)
    served_services = {
        item.service for item in schedule if 0 <= item.service < instance.services
    }
    coverage = len(served_services)

    similarity = sum(
        instance.rewards[item.agent][item.service]
        for item in schedule
        if 0 <= item.agent < instance.agents
        and 0 <= item.service < instance.services
    )
    theoretical_similarity = instance.max_reward * instance.services
    attainable_upper_bound = _attainable_similarity_upper_bound(instance)

    agents_by_service: Dict[int, set[int]] = {}
    workload = [0] * instance.agents
    for item in schedule:
        if 0 <= item.service < instance.services and 0 <= item.agent < instance.agents:
            agents_by_service.setdefault(item.service, set()).add(item.agent)
            workload[item.agent] += 1

    continuity_penalty = 0
    continuity_denominator = 0
    unserved_sequences = 0
    partially_served_sequences = 0
    distinct_agents_total = 0
    for sequence in instance.sequences:
        sequence_agents = set()
        served_in_sequence = 0
        for service in sequence:
            assigned = agents_by_service.get(service, set())
            if assigned:
                served_in_sequence += 1
                sequence_agents.update(assigned)
        if served_in_sequence == 0:
            unserved_sequences += 1
        elif served_in_sequence < len(sequence):
            partially_served_sequences += 1
        distinct_agents = len(sequence_agents)
        distinct_agents_total += distinct_agents
        continuity_penalty += max(0, distinct_agents - 1)
        continuity_denominator += max(0, served_in_sequence - 1)

    overtime_by_agent = [
        max(0, workload[agent] - instance.normal_hours[agent])
        for agent in range(instance.agents)
    ]
    overtime = sum(overtime_by_agent)
    full_coverage = coverage == instance.services

    return Metrics(
        coverage=coverage,
        coverage_normalized=coverage / instance.services,
        similarity=similarity,
        similarity_normalized=(
            similarity / theoretical_similarity
            if theoretical_similarity > 0
            else 0.0
        ),
        similarity_attainable=(
            similarity / attainable_upper_bound
            if attainable_upper_bound > 0
            else None
        ),
        continuity_penalty=continuity_penalty,
        continuity_normalized=(
            1.0 - continuity_penalty / continuity_denominator
            if continuity_denominator > 0
            else 1.0
        ),
        overtime=overtime,
        overtime_normalized=overtime / instance.services,
        overtime_cost=overtime * instance.penalty,
        agents_with_overtime=sum(value > 0 for value in overtime_by_agent),
        max_agent_overtime=max(overtime_by_agent, default=0),
        max_agent_workload=max(workload, default=0),
        unserved_sequences=unserved_sequences,
        partially_served_sequences=partially_served_sequences,
        original_stability=(
            sum(len(sequence) for sequence in instance.sequences)
            - distinct_agents_total
            if full_coverage
            else None
        ),
    )


def verify_assignments(
    instance: HCORAPInstance,
    assignments: Iterable[Assignment],
    *,
    require_full_coverage: bool = True,
) -> VerificationResult:
    """Check all HCORAP hard constraints and return independent metrics."""

    schedule = tuple(assignments)
    violations: List[str] = []
    by_service: Dict[int, List[Assignment]] = {}
    by_agent_slot: Dict[Tuple[int, int], List[Assignment]] = {}
    by_user_slot: Dict[Tuple[int, int], List[Assignment]] = {}
    workload = [0] * instance.agents
    service_to_user = instance.service_to_user

    seen = set()
    for position, item in enumerate(schedule):
        if item in seen:
            violations.append(f"duplicate assignment triple: {item}")
        seen.add(item)

        if not 0 <= item.agent < instance.agents:
            violations.append(
                f"assignment {position} has invalid agent {item.agent}"
            )
            continue
        if not 0 <= item.service < instance.services:
            violations.append(
                f"assignment {position} has invalid service {item.service}"
            )
            continue
        if not 0 <= item.time_slot < instance.time_slots:
            violations.append(
                f"assignment {position} has invalid time slot {item.time_slot}"
            )
            continue

        agent, service, slot = item.agent, item.service, item.time_slot
        if instance.rewards[agent][service] <= 0:
            violations.append(
                f"agent {agent} is not qualified for service {service}"
            )
        if not instance.agent_availability[agent][slot]:
            violations.append(f"agent {agent} is unavailable at slot {slot}")
        if not instance.service_availability[service][slot]:
            violations.append(f"service {service} is unavailable at slot {slot}")

        by_service.setdefault(service, []).append(item)
        by_agent_slot.setdefault((agent, slot), []).append(item)
        user = service_to_user[service]
        by_user_slot.setdefault((user, slot), []).append(item)
        workload[agent] += 1

    for service, items in sorted(by_service.items()):
        if len(items) > 1:
            violations.append(
                f"service {service} is assigned {len(items)} times (maximum is 1)"
            )
    for (agent, slot), items in sorted(by_agent_slot.items()):
        if len(items) > 1:
            violations.append(
                f"agent {agent} has {len(items)} services at slot {slot}"
            )
    for (user, slot), items in sorted(by_user_slot.items()):
        if len(items) > 1:
            violations.append(
                f"user {user} receives {len(items)} services at slot {slot}"
            )
    for agent, hours in enumerate(workload):
        capacity = instance.normal_hours[agent] + instance.extra_hours[agent]
        if hours > capacity:
            violations.append(
                f"agent {agent} works {hours} hours, exceeding capacity {capacity}"
            )

    missing = sorted(set(range(instance.services)) - set(by_service))
    if require_full_coverage and missing:
        preview = ", ".join(str(service) for service in missing[:10])
        suffix = "..." if len(missing) > 10 else ""
        violations.append(
            f"{len(missing)} services are unserved: {preview}{suffix}"
        )

    return VerificationResult(
        valid=not violations,
        violations=tuple(violations),
        metrics=compute_metrics(instance, schedule),
    )
