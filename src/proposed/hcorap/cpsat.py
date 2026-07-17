"""Independent OR-Tools CP-SAT baseline for the HCORAP formulations.

The module is optional at installation time.  It intentionally rebuilds the
model from the instance instead of translating the MaxSAT CNF, which makes it
useful for detecting shared encoding mistakes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .metrics import verify_assignments
from .model import Assignment, HCORAPInstance, SolveResult, StageResult
from .solvers import Delta, LEXICOGRAPHIC_POLICIES, similarity_budget


def _load_cp_model() -> Any:
    try:
        from ortools.sat.python import cp_model
    except ImportError as exc:  # pragma: no cover - exercised without the extra
        raise RuntimeError(
            "CP-SAT support is optional; install it with "
            "`python -m pip install -e '.[cpsat]'`"
        ) from exc
    return cp_model


@dataclass
class _Bundle:
    model: Any
    x: Dict[Tuple[int, int, int], Any]
    y: Dict[Tuple[int, int], Any]
    sequence_agent: Dict[Tuple[int, int], Any]
    overtime: Dict[int, Any]
    objectives: Mapping[str, Any]
    require_full_coverage: bool


def _build_model(
    instance: HCORAPInstance, *, require_full_coverage: bool
) -> _Bundle:
    cp_model = _load_cp_model()
    model = cp_model.CpModel()
    x: Dict[Tuple[int, int, int], Any] = {}
    y: Dict[Tuple[int, int], Any] = {}
    z: Dict[int, Any] = {}
    sequence_agent: Dict[Tuple[int, int], Any] = {}
    sequence_active: Dict[int, Any] = {}
    overtime: Dict[int, Any] = {}

    for service in range(instance.services):
        for agent, slot in instance.candidate_triplets(service):
            x[(agent, service, slot)] = model.new_bool_var(
                f"x_{agent}_{service}_{slot}"
            )

    for agent, service, _slot in x:
        if (agent, service) not in y:
            y[(agent, service)] = model.new_bool_var(f"y_{agent}_{service}")
    for (agent, service), variable in y.items():
        inputs = [
            value
            for (a, s, _slot), value in x.items()
            if a == agent and s == service
        ]
        model.add(variable == sum(inputs))

    for service in range(instance.services):
        z[service] = model.new_bool_var(f"z_{service}")
        inputs = [
            value
            for (_agent, s, _slot), value in x.items()
            if s == service
        ]
        if inputs:
            model.add(z[service] == sum(inputs))
            model.add_at_most_one(inputs)
        else:
            model.add(z[service] == 0)
        if require_full_coverage:
            model.add(z[service] == 1)

    for agent in range(instance.agents):
        for slot in range(instance.time_slots):
            model.add_at_most_one(
                value
                for (a, _service, t), value in x.items()
                if a == agent and t == slot
            )

    for services in instance.services_by_user:
        service_set = set(services)
        for slot in range(instance.time_slots):
            model.add_at_most_one(
                value
                for (_agent, service, t), value in x.items()
                if service in service_set and t == slot
            )

    for agent in range(instance.agents):
        workload_variables = [
            value for (a, _service), value in y.items() if a == agent
        ]
        workload = sum(workload_variables)
        maximum = instance.normal_hours[agent] + instance.extra_hours[agent]
        model.add(workload <= maximum)
        extra = model.new_int_var(
            0, instance.extra_hours[agent], f"overtime_{agent}"
        )
        model.add_max_equality(
            extra, [workload - instance.normal_hours[agent], 0]
        )
        overtime[agent] = extra

    for sequence_index, sequence in enumerate(instance.sequences):
        sequence_active[sequence_index] = model.new_bool_var(
            f"sequence_active_{sequence_index}"
        )
        model.add_max_equality(
            sequence_active[sequence_index], [z[service] for service in sequence]
        )
        for agent in range(instance.agents):
            inputs = [
                y[(agent, service)]
                for service in sequence
                if (agent, service) in y
            ]
            if inputs:
                variable = model.new_bool_var(
                    f"sequence_agent_{agent}_{sequence_index}"
                )
                model.add_max_equality(variable, inputs)
                sequence_agent[(agent, sequence_index)] = variable

    objectives = {
        "coverage": sum(z.values()),
        "similarity": sum(
            instance.rewards[agent][service] * variable
            for (agent, service), variable in y.items()
        ),
        "continuity": sum(sequence_agent.values()) - sum(sequence_active.values()),
        "overtime": sum(overtime.values()),
    }
    return _Bundle(
        model=model,
        x=x,
        y=y,
        sequence_agent=sequence_agent,
        overtime=overtime,
        objectives=objectives,
        require_full_coverage=require_full_coverage,
    )


def _remaining(started: float, timeout_seconds: Optional[float]) -> Optional[float]:
    if timeout_seconds is None:
        return None
    return max(0.0, timeout_seconds - (time.perf_counter() - started))


def _solve_stage(
    bundle: _Bundle,
    objective_name: str,
    sense: str,
    *,
    timeout_seconds: Optional[float],
    workers: int,
    random_seed: int,
) -> Tuple[Any, Any, float]:
    cp_model = _load_cp_model()
    expression = bundle.objectives[objective_name]
    if sense == "max":
        bundle.model.maximize(expression)
    elif sense == "min":
        bundle.model.minimize(expression)
    else:
        raise ValueError(f"invalid objective sense: {sense}")

    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = workers
    solver.parameters.random_seed = random_seed
    if timeout_seconds is not None:
        solver.parameters.max_time_in_seconds = max(0.001, timeout_seconds)
    started = time.perf_counter()
    status = solver.solve(bundle.model)
    return status, solver, time.perf_counter() - started


def _status_name(status: Any) -> str:
    cp_model = _load_cp_model()
    return {
        cp_model.OPTIMAL: "OPTIMUM",
        cp_model.FEASIBLE: "TIMEOUT_FEASIBLE",
        cp_model.INFEASIBLE: "UNSATISFIABLE",
        cp_model.MODEL_INVALID: "MODEL_INVALID",
        cp_model.UNKNOWN: "TIMEOUT",
    }.get(status, str(status))


def _result(
    instance: HCORAPInstance,
    bundle: _Bundle,
    solver: Optional[Any],
    *,
    status: str,
    method: str,
    stages: Sequence[StageResult],
    elapsed_seconds: float,
    metadata: Mapping[str, Any],
) -> SolveResult:
    assignments: Tuple[Assignment, ...] = ()
    metrics = None
    if solver is not None and status in {"OPTIMUM", "TIMEOUT_FEASIBLE"}:
        assignments = tuple(
            sorted(
                Assignment(agent, service, slot)
                for (agent, service, slot), variable in bundle.x.items()
                if solver.boolean_value(variable)
            )
        )
        verification = verify_assignments(
            instance,
            assignments,
            require_full_coverage=bundle.require_full_coverage,
        )
        if not verification.valid:
            raise RuntimeError(
                "CP-SAT baseline returned an invalid schedule: "
                + "; ".join(verification.violations[:5])
            )
        metrics = verification.metrics
        expected = {
            "coverage": metrics.coverage,
            "similarity": metrics.similarity,
            "continuity": metrics.continuity_penalty,
            "overtime": metrics.overtime,
        }
        actual = {
            name: solver.value(expression)
            for name, expression in bundle.objectives.items()
        }
        if actual != expected:
            raise RuntimeError(
                f"CP-SAT variables disagree with decoded schedule: {actual} != {expected}"
            )

    details = dict(metadata)
    details["backend"] = "ortools-cp-sat"
    details["model"] = {
        "x_variables": len(bundle.x),
        "y_variables": len(bundle.y),
        "sequence_agent_variables": len(bundle.sequence_agent),
        "overtime_variables": len(bundle.overtime),
    }
    return SolveResult(
        status=status,
        method=method,
        assignments=assignments,
        metrics=metrics,
        stages=tuple(stages),
        elapsed_seconds=elapsed_seconds,
        metadata=details,
    )


def _sequential(
    instance: HCORAPInstance,
    *,
    method: str,
    objective_order: Sequence[str],
    initial_bounds: Sequence[Tuple[str, str, int]] = (),
    require_full_coverage: bool,
    timeout_seconds: Optional[float],
    workers: int,
    random_seed: int,
    metadata: Mapping[str, Any],
) -> SolveResult:
    cp_model = _load_cp_model()
    started = time.perf_counter()
    bundle = _build_model(
        instance, require_full_coverage=require_full_coverage
    )
    for objective_name, relation, value in initial_bounds:
        expression = bundle.objectives[objective_name]
        if relation == "atleast":
            bundle.model.add(expression >= value)
        elif relation == "atmost":
            bundle.model.add(expression <= value)
        else:
            raise ValueError(f"invalid bound relation: {relation}")

    stages: List[StageResult] = []
    last_solver = None
    for objective_name in objective_order:
        sense = "max" if objective_name in {"coverage", "similarity"} else "min"
        status, solver, elapsed = _solve_stage(
            bundle,
            objective_name,
            sense,
            timeout_seconds=_remaining(started, timeout_seconds),
            workers=workers,
            random_seed=random_seed,
        )
        status_name = _status_name(status)
        last_solver = solver if status in {cp_model.OPTIMAL, cp_model.FEASIBLE} else None
        if status != cp_model.OPTIMAL:
            incomplete = dict(metadata)
            incomplete["incomplete_stage"] = objective_name
            return _result(
                instance,
                bundle,
                last_solver,
                status=status_name,
                method=method,
                stages=stages,
                elapsed_seconds=time.perf_counter() - started,
                metadata=incomplete,
            )
        optimum = int(solver.value(bundle.objectives[objective_name]))
        stages.append(StageResult(objective_name, sense, optimum, elapsed))
        if sense == "max":
            bundle.model.add(bundle.objectives[objective_name] >= optimum)
        else:
            bundle.model.add(bundle.objectives[objective_name] <= optimum)

    assert last_solver is not None
    return _result(
        instance,
        bundle,
        last_solver,
        status="OPTIMUM",
        method=method,
        stages=stages,
        elapsed_seconds=time.perf_counter() - started,
        metadata=metadata,
    )


def solve_cpsat_weighted(
    instance: HCORAPInstance,
    *,
    continuity_weight: int = 1,
    overtime_weight: int = 1,
    require_full_coverage: bool = True,
    timeout_seconds: Optional[float] = None,
    workers: int = 1,
    random_seed: int = 0,
) -> SolveResult:
    """Solve the calibrated weighted objective with the CP-SAT baseline."""

    if continuity_weight < 0 or overtime_weight < 0:
        raise ValueError("continuity and overtime weights must be non-negative")
    cp_model = _load_cp_model()
    started = time.perf_counter()
    bundle = _build_model(
        instance, require_full_coverage=require_full_coverage
    )
    stages: List[StageResult] = []

    if not require_full_coverage:
        status, solver, elapsed = _solve_stage(
            bundle,
            "coverage",
            "max",
            timeout_seconds=_remaining(started, timeout_seconds),
            workers=workers,
            random_seed=random_seed,
        )
        if status != cp_model.OPTIMAL:
            return _result(
                instance,
                bundle,
                solver if status == cp_model.FEASIBLE else None,
                status=_status_name(status),
                method="cpsat:weighted",
                stages=stages,
                elapsed_seconds=time.perf_counter() - started,
                metadata={"incomplete_stage": "coverage"},
            )
        optimum = int(solver.value(bundle.objectives["coverage"]))
        stages.append(StageResult("coverage", "max", optimum, elapsed))
        bundle.model.add(bundle.objectives["coverage"] >= optimum)

    weighted = (
        bundle.objectives["similarity"]
        - continuity_weight * bundle.objectives["continuity"]
        - overtime_weight * instance.penalty * bundle.objectives["overtime"]
    )
    bundle.objectives = dict(bundle.objectives)
    bundle.objectives["weighted_score"] = weighted
    status, solver, elapsed = _solve_stage(
        bundle,
        "weighted_score",
        "max",
        timeout_seconds=_remaining(started, timeout_seconds),
        workers=workers,
        random_seed=random_seed,
    )
    status_name = _status_name(status)
    if status == cp_model.OPTIMAL:
        optimum = int(solver.value(weighted))
        stages.append(StageResult("weighted_score", "max", optimum, elapsed))
    metadata = {
        "continuity_weight": continuity_weight,
        "overtime_weight": overtime_weight,
        "overtime_penalty": instance.penalty,
        "workers": workers,
        "random_seed": random_seed,
        "timeout_seconds": timeout_seconds,
    }
    # weighted_score is not an independently reported metric.
    bundle.objectives = {
        name: value
        for name, value in bundle.objectives.items()
        if name != "weighted_score"
    }
    return _result(
        instance,
        bundle,
        solver if status in {cp_model.OPTIMAL, cp_model.FEASIBLE} else None,
        status=status_name,
        method="cpsat:weighted",
        stages=stages,
        elapsed_seconds=time.perf_counter() - started,
        metadata=metadata,
    )


def solve_cpsat_lexicographic(
    instance: HCORAPInstance,
    *,
    policy: str = "continuity-priority",
    require_full_coverage: bool = True,
    timeout_seconds: Optional[float] = None,
    workers: int = 1,
    random_seed: int = 0,
) -> SolveResult:
    """Run the same lexicographic policies through CP-SAT."""

    if policy not in LEXICOGRAPHIC_POLICIES:
        raise ValueError(f"unknown lexicographic policy: {policy}")
    order = list(LEXICOGRAPHIC_POLICIES[policy])
    if not require_full_coverage:
        order.insert(0, "coverage")
    return _sequential(
        instance,
        method=f"cpsat:lexicographic:{policy}",
        objective_order=order,
        require_full_coverage=require_full_coverage,
        timeout_seconds=timeout_seconds,
        workers=workers,
        random_seed=random_seed,
        metadata={
            "policy": policy,
            "order": order,
            "workers": workers,
            "random_seed": random_seed,
            "timeout_seconds": timeout_seconds,
        },
    )


def solve_cpsat_epsilon_constraint(
    instance: HCORAPInstance,
    *,
    delta: Delta,
    require_full_coverage: bool = True,
    timeout_seconds: Optional[float] = None,
    workers: int = 1,
    random_seed: int = 0,
) -> SolveResult:
    """Run the similarity-budget method with CP-SAT and the same semantics."""

    # Similarity reference needs its own first solve; the completion is then
    # rebuilt with only the declared budget, not with SIM fixed at SIM*.
    prefix = ["similarity"]
    if not require_full_coverage:
        prefix.insert(0, "coverage")
    reference = _sequential(
        instance,
        method="cpsat:epsilon-reference",
        objective_order=prefix,
        require_full_coverage=require_full_coverage,
        timeout_seconds=timeout_seconds,
        workers=workers,
        random_seed=random_seed,
        metadata={"phase": "reference"},
    )
    if reference.status != "OPTIMUM":
        return reference
    similarity_optimum = next(
        stage.optimum
        for stage in reference.stages
        if stage.objective == "similarity"
    )
    threshold = similarity_budget(similarity_optimum, delta)
    elapsed_reference = reference.elapsed_seconds
    remaining_timeout = (
        None
        if timeout_seconds is None
        else max(0.0, timeout_seconds - elapsed_reference)
    )
    bounds: List[Tuple[str, str, int]] = [("similarity", "atleast", threshold)]
    order = ["continuity", "overtime", "similarity"]
    if not require_full_coverage:
        coverage_optimum = next(
            stage.optimum
            for stage in reference.stages
            if stage.objective == "coverage"
        )
        bounds.append(("coverage", "atleast", coverage_optimum))
    completion = _sequential(
        instance,
        method="cpsat:epsilon-constraint",
        objective_order=order,
        initial_bounds=bounds,
        require_full_coverage=require_full_coverage,
        timeout_seconds=remaining_timeout,
        workers=workers,
        random_seed=random_seed,
        metadata={
            "delta": str(delta),
            "similarity_optimum": similarity_optimum,
            "similarity_threshold": threshold,
            "workers": workers,
            "random_seed": random_seed,
            "timeout_seconds": timeout_seconds,
        },
    )
    reference_stages = tuple(
        StageResult(
            "similarity_reference" if stage.objective == "similarity" else stage.objective,
            stage.sense,
            stage.optimum,
            stage.elapsed_seconds,
        )
        for stage in reference.stages
    )
    completion_stages = tuple(
        StageResult(
            "similarity_tiebreak" if stage.objective == "similarity" else stage.objective,
            stage.sense,
            stage.optimum,
            stage.elapsed_seconds,
        )
        for stage in completion.stages
    )
    return SolveResult(
        status=completion.status,
        method=completion.method,
        assignments=completion.assignments,
        metrics=completion.metrics,
        stages=reference_stages + completion_stages,
        elapsed_seconds=elapsed_reference + completion.elapsed_seconds,
        metadata=completion.metadata,
    )
