"""Core immutable data structures used by every HCORAP backend."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


IntTuple = Tuple[int, ...]
IntMatrix = Tuple[IntTuple, ...]


@dataclass(frozen=True)
class HCORAPInstance:
    """A parsed HCORAP instance using zero-based identifiers."""

    users: int
    services: int
    agents: int
    time_slots: int
    services_by_user: Tuple[IntTuple, ...]
    sequences: Tuple[IntTuple, ...]
    agent_availability: IntMatrix
    service_availability: IntMatrix
    rewards: IntMatrix
    overtime_penalty: int
    normal_hours: IntTuple
    extra_hours: IntTuple
    source: Optional[Path] = None
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)

    @property
    def penalty(self) -> int:
        """Return the non-negative cost of one overtime service/time slot."""

        return abs(self.overtime_penalty)

    @property
    def max_reward(self) -> int:
        return max((max(row, default=0) for row in self.rewards), default=0)

    @property
    def service_to_user(self) -> Tuple[int, ...]:
        mapping = [-1] * self.services
        for user, group in enumerate(self.services_by_user):
            for service in group:
                mapping[service] = user
        return tuple(mapping)

    def candidate_triplets(self, service: int) -> Tuple[Tuple[int, int], ...]:
        """Return feasible ``(agent, time_slot)`` pairs for one service."""

        pairs = []
        for agent in range(self.agents):
            if self.rewards[agent][service] <= 0:
                continue
            for slot in range(self.time_slots):
                if (
                    self.agent_availability[agent][slot]
                    and self.service_availability[service][slot]
                ):
                    pairs.append((agent, slot))
        return tuple(pairs)

    def to_summary(self) -> Dict[str, Any]:
        candidates = [len(self.candidate_triplets(s)) for s in range(self.services)]
        total_capacity = sum(
            normal + extra
            for normal, extra in zip(self.normal_hours, self.extra_hours)
        )
        return {
            "source": str(self.source) if self.source else None,
            "users": self.users,
            "services": self.services,
            "agents": self.agents,
            "time_slots": self.time_slots,
            "sequences": len(self.sequences),
            "penalty": self.penalty,
            "capacity": total_capacity,
            "rho": self.services / total_capacity if total_capacity else None,
            "candidate_pairs_min": min(candidates, default=0),
            "candidate_pairs_mean": (
                sum(candidates) / len(candidates) if candidates else 0.0
            ),
            "services_without_candidates": sum(value == 0 for value in candidates),
            "singleton_candidate_services": sum(value == 1 for value in candidates),
        }

@dataclass(frozen=True, order=True)
class Assignment:
    agent: int
    service: int
    time_slot: int


@dataclass(frozen=True)
class Metrics:
    coverage: int
    coverage_normalized: float
    similarity: int
    similarity_normalized: float
    similarity_attainable: Optional[float]
    continuity_penalty: int
    continuity_normalized: float
    overtime: int
    overtime_normalized: float
    overtime_cost: int
    agents_with_overtime: int
    max_agent_overtime: int
    max_agent_workload: int
    unserved_sequences: int
    partially_served_sequences: int
    original_stability: Optional[int]

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StageResult:
    objective: str
    sense: str
    optimum: int
    elapsed_seconds: float

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SolveResult:
    status: str
    method: str
    assignments: Tuple[Assignment, ...] = ()
    metrics: Optional[Metrics] = None
    stages: Tuple[StageResult, ...] = ()
    elapsed_seconds: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "method": self.method,
            "assignments": [asdict(item) for item in self.assignments],
            "metrics": self.metrics.as_dict() if self.metrics else None,
            "stages": [stage.as_dict() for stage in self.stages],
            "elapsed_seconds": self.elapsed_seconds,
            "metadata": dict(self.metadata),
        }
