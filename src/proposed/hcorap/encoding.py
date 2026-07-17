"""Shared CNF encoding for the proposed HCORAP optimization methods."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from pysat.card import CardEnc, EncType, ITotalizer
from pysat.formula import CNF, IDPool
from pysat.pb import PBEnc

from .model import Assignment, HCORAPInstance


WeightedLiteral = Tuple[int, int]


@dataclass(frozen=True)
class LinearObjective:
    """A non-negative pseudo-Boolean sum plus a constant offset."""

    name: str
    sense: str
    terms: Tuple[WeightedLiteral, ...]
    offset: int = 0

    def __post_init__(self) -> None:
        if self.sense not in {"min", "max"}:
            raise ValueError(f"invalid objective sense: {self.sense}")
        if any(weight <= 0 for _literal, weight in self.terms):
            raise ValueError("objective weights must be positive")

    @property
    def term_sum(self) -> int:
        return sum(weight for _literal, weight in self.terms)

    @property
    def minimum(self) -> int:
        return self.offset

    @property
    def maximum(self) -> int:
        return self.offset + self.term_sum

    def evaluate(self, model: Iterable[int]) -> int:
        positive = {literal for literal in model if literal > 0}
        value = self.offset
        for literal, weight in self.terms:
            satisfied = literal in positive if literal > 0 else -literal not in positive
            if satisfied:
                value += weight
        return value


class HCORAPEncoding:
    """Encode feasibility once and expose consistent objective expressions.

    Only feasible ``x[a,s,t]`` variables are created.  The remaining semantic
    variables are exact OR reifications, which prevents objective variables
    from taking values unrelated to the decoded schedule.
    """

    def __init__(
        self, instance: HCORAPInstance, *, require_full_coverage: bool = True
    ) -> None:
        self.instance = instance
        self.require_full_coverage = require_full_coverage
        self.vpool = IDPool()
        self.cnf = CNF()

        self.x: Dict[Tuple[int, int, int], int] = {}
        self.y: Dict[Tuple[int, int], int] = {}
        self.z: Dict[int, int] = {}
        self.sequence_agent: Dict[Tuple[int, int], int] = {}
        self.sequence_active: Dict[int, int] = {}
        self.overtime_literals: Dict[Tuple[int, int], int] = {}
        self._objectives: Dict[str, LinearObjective] = {}

        self._allocate_semantic_variables()
        self._encode_reifications()
        self._encode_resource_constraints()
        self._encode_workload_and_overtime()
        self._build_objectives()

    def _allocate_semantic_variables(self) -> None:
        instance = self.instance
        for service in range(instance.services):
            for agent, slot in instance.candidate_triplets(service):
                self.x[(agent, service, slot)] = self.vpool.id(
                    ("x", agent, service, slot)
                )

        for agent, service, _slot in self.x:
            key = (agent, service)
            if key not in self.y:
                self.y[key] = self.vpool.id(("y", agent, service))

        for service in range(instance.services):
            self.z[service] = self.vpool.id(("z", service))

        for sequence_index, sequence in enumerate(instance.sequences):
            self.sequence_active[sequence_index] = self.vpool.id(
                ("sequence_active", sequence_index)
            )
            for agent in range(instance.agents):
                if any((agent, service) in self.y for service in sequence):
                    self.sequence_agent[(agent, sequence_index)] = self.vpool.id(
                        ("sequence_agent", agent, sequence_index)
                    )

    def _encode_or_reification(self, output: int, inputs: Sequence[int]) -> None:
        """Add ``output <-> OR(inputs)`` (including the empty OR)."""

        for literal in inputs:
            self.cnf.append([-literal, output])
        self.cnf.append([-output, *inputs])

    def _encode_reifications(self) -> None:
        instance = self.instance
        for (agent, service), output in self.y.items():
            inputs = [
                literal
                for (a, s, _slot), literal in self.x.items()
                if a == agent and s == service
            ]
            self._encode_or_reification(output, inputs)

        for service, output in self.z.items():
            inputs = [
                literal
                for (_agent, s, _slot), literal in self.x.items()
                if s == service
            ]
            self._encode_or_reification(output, inputs)
            if self.require_full_coverage:
                self.cnf.append([output])

        for (agent, sequence_index), output in self.sequence_agent.items():
            inputs = [
                self.y[(agent, service)]
                for service in instance.sequences[sequence_index]
                if (agent, service) in self.y
            ]
            self._encode_or_reification(output, inputs)

        for sequence_index, output in self.sequence_active.items():
            inputs = [self.z[service] for service in instance.sequences[sequence_index]]
            self._encode_or_reification(output, inputs)

    def _append_at_most_one(self, literals: Sequence[int]) -> None:
        if len(literals) <= 1:
            return
        encoded = CardEnc.atmost(
            lits=list(literals),
            bound=1,
            vpool=self.vpool,
            encoding=EncType.seqcounter,
        )
        self.cnf.extend(encoded.clauses)

    def _encode_resource_constraints(self) -> None:
        instance = self.instance

        for service in range(instance.services):
            self._append_at_most_one(
                [
                    literal
                    for (_agent, s, _slot), literal in self.x.items()
                    if s == service
                ]
            )

        for agent in range(instance.agents):
            for slot in range(instance.time_slots):
                self._append_at_most_one(
                    [
                        literal
                        for (a, _service, t), literal in self.x.items()
                        if a == agent and t == slot
                    ]
                )

        for services in instance.services_by_user:
            service_set = set(services)
            for slot in range(instance.time_slots):
                self._append_at_most_one(
                    [
                        literal
                        for (_agent, service, t), literal in self.x.items()
                        if service in service_set and t == slot
                    ]
                )

    def _encode_workload_and_overtime(self) -> None:
        instance = self.instance
        for agent in range(instance.agents):
            workload = [
                literal
                for (a, _service), literal in self.y.items()
                if a == agent
            ]
            normal = instance.normal_hours[agent]
            maximum = normal + instance.extra_hours[agent]
            if not workload:
                continue
            if maximum == 0:
                self.cnf.extend([[-literal] for literal in workload])
                continue

            capacity_index: Optional[int] = (
                maximum if len(workload) > maximum else None
            )
            last_overtime_index: Optional[int] = None
            if instance.extra_hours[agent] > 0 and len(workload) > normal:
                last_overtime_index = min(maximum - 1, len(workload) - 1)

            required_indexes = [
                index
                for index in (capacity_index, last_overtime_index)
                if index is not None
            ]
            if not required_indexes:
                continue

            upper_index = max(required_indexes)
            totalizer = ITotalizer(
                lits=workload, ubound=upper_index, top_id=self.vpool.top
            )
            self.cnf.extend(totalizer.cnf.clauses)
            self.vpool.top = max(self.vpool.top, totalizer.top_id)

            if capacity_index is not None:
                self.cnf.append([-totalizer.rhs[capacity_index]])
            if last_overtime_index is not None:
                for threshold_index in range(normal, last_overtime_index + 1):
                    overtime_unit = threshold_index - normal + 1
                    threshold_literal = totalizer.rhs[threshold_index]
                    self.overtime_literals[(agent, overtime_unit)] = threshold_literal
                    # ITotalizer guarantees count >= b -> rhs[b-1], but an
                    # unconstrained rhs may otherwise be set to true.  Add the
                    # reverse implication so reported overtime remains exact
                    # even when its scalarization weight is zero.
                    reverse = CardEnc.atleast(
                        lits=workload,
                        bound=threshold_index + 1,
                        vpool=self.vpool,
                        encoding=EncType.seqcounter,
                    )
                    self.cnf.extend(
                        [[-threshold_literal, *clause] for clause in reverse.clauses]
                    )
            totalizer.delete()

    def _build_objectives(self) -> None:
        instance = self.instance
        self._objectives["coverage"] = LinearObjective(
            name="coverage",
            sense="max",
            terms=tuple((self.z[service], 1) for service in range(instance.services)),
        )
        self._objectives["similarity"] = LinearObjective(
            name="similarity",
            sense="max",
            terms=tuple(
                (literal, instance.rewards[agent][service])
                for (agent, service), literal in sorted(self.y.items())
            ),
        )

        continuity_terms: List[WeightedLiteral] = [
            (literal, 1) for _key, literal in sorted(self.sequence_agent.items())
        ]
        continuity_terms.extend(
            (-literal, 1)
            for _sequence, literal in sorted(self.sequence_active.items())
        )
        self._objectives["continuity"] = LinearObjective(
            name="continuity",
            sense="min",
            terms=tuple(continuity_terms),
            offset=-len(instance.sequences),
        )
        self._objectives["overtime"] = LinearObjective(
            name="overtime",
            sense="min",
            terms=tuple(
                (literal, 1)
                for _key, literal in sorted(self.overtime_literals.items())
            ),
        )

    @property
    def objectives(self) -> Mapping[str, LinearObjective]:
        return dict(self._objectives)

    def objective(self, name: str) -> LinearObjective:
        try:
            return self._objectives[name]
        except KeyError as exc:
            raise ValueError(f"unknown objective {name!r}") from exc

    def assignments_from_model(self, model: Iterable[int]) -> Tuple[Assignment, ...]:
        positive = {literal for literal in model if literal > 0}
        return tuple(
            sorted(
                Assignment(agent=agent, service=service, time_slot=slot)
                for (agent, service, slot), literal in self.x.items()
                if literal in positive
            )
        )

    def stats(self) -> Mapping[str, int]:
        return {
            "variables": self.vpool.top,
            "clauses": len(self.cnf.clauses),
            "x_variables": len(self.x),
            "y_variables": len(self.y),
            "sequence_agent_variables": len(self.sequence_agent),
            "overtime_literals": len(self.overtime_literals),
        }


def objective_bound_clauses(
    objective: LinearObjective,
    relation: str,
    value: int,
    *,
    vpool: IDPool,
) -> List[List[int]]:
    """Encode an exact integer bound on a :class:`LinearObjective`."""

    if relation not in {"atleast", "atmost", "equals"}:
        raise ValueError(f"invalid bound relation: {relation}")
    target = value - objective.offset
    total = objective.term_sum

    def one_side(side: str) -> List[List[int]]:
        if side == "atleast":
            if target <= 0:
                return []
            if target > total:
                return [[]]
            encoded = PBEnc.atleast(
                lits=[literal for literal, _weight in objective.terms],
                weights=[weight for _literal, weight in objective.terms],
                bound=target,
                vpool=vpool,
            )
            return encoded.clauses
        if target < 0:
            return [[]]
        if target >= total:
            return []
        encoded = PBEnc.atmost(
            lits=[literal for literal, _weight in objective.terms],
            weights=[weight for _literal, weight in objective.terms],
            bound=target,
            vpool=vpool,
        )
        return encoded.clauses

    if relation == "equals":
        return one_side("atleast") + one_side("atmost")
    return one_side(relation)
