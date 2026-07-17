"""Weighted, lexicographic and epsilon-constraint MaxSAT solvers."""

from __future__ import annotations

import time
from fractions import Fraction
from threading import Event, Timer
from typing import Iterable, List, Optional, Sequence, Tuple, Union

from pysat.examples.rc2 import RC2, RC2Stratified
from pysat.formula import WCNF
from pysat.solvers import Solver

from .encoding import HCORAPEncoding, LinearObjective, objective_bound_clauses
from .metrics import verify_assignments
from .model import HCORAPInstance, SolveResult, StageResult


Delta = Union[str, float, int, Fraction]


class SolverConsistencyError(RuntimeError):
    """Raised when a solver model disagrees with the independent verifier."""


def _build_wcnf(
    hard_clauses: Iterable[Sequence[int]],
    soft_clauses: Iterable[Tuple[Sequence[int], int]],
) -> WCNF:
    formula = WCNF()
    for clause in hard_clauses:
        formula.append(list(clause))
    for clause, weight in soft_clauses:
        if weight < 0:
            raise ValueError("MaxSAT soft weights cannot be negative")
        if weight > 0:
            formula.append(list(clause), weight=weight)
    return formula


def _objective_soft_clauses(
    objective: LinearObjective,
) -> List[Tuple[List[int], int]]:
    if objective.sense == "max":
        return [([literal], weight) for literal, weight in objective.terms]
    return [([-literal], weight) for literal, weight in objective.terms]


def _run_maxsat(
    formula: WCNF,
    *,
    sat_solver: str,
    maxsat_algorithm: str,
    timeout_seconds: Optional[float],
) -> Tuple[Optional[List[int]], float, bool]:
    """Run RC2 with an interruptible wall-clock limit."""

    algorithms = {"rc2": RC2, "rc2-stratified": RC2Stratified}
    try:
        optimizer_class = algorithms[maxsat_algorithm]
    except KeyError as exc:
        raise ValueError(
            f"unknown MaxSAT algorithm {maxsat_algorithm!r}; "
            f"choose from {sorted(algorithms)}"
        ) from exc
    if timeout_seconds is not None and timeout_seconds <= 0:
        return None, 0.0, True

    started = time.perf_counter()
    interrupted = Event()
    with optimizer_class(formula, solver=sat_solver) as optimizer:
        timer = None
        if timeout_seconds is not None:
            def request_interrupt() -> None:
                interrupted.set()
                optimizer.interrupt()

            timer = Timer(timeout_seconds, request_interrupt)
            timer.daemon = True
            timer.start()
        try:
            model = optimizer.compute(expect_interrupt=timer is not None)
        finally:
            if timer is not None:
                timer.cancel()
    elapsed = time.perf_counter() - started
    timed_out = interrupted.is_set() and model is None
    return model, elapsed, timed_out


def _remaining_timeout(started: float, timeout_seconds: Optional[float]) -> Optional[float]:
    if timeout_seconds is None:
        return None
    return max(0.0, timeout_seconds - (time.perf_counter() - started))


def _optimize(
    hard_clauses: Sequence[Sequence[int]],
    objective: LinearObjective,
    *,
    sat_solver: str,
    maxsat_algorithm: str,
    timeout_seconds: Optional[float],
) -> Tuple[Optional[List[int]], Optional[int], float, bool]:
    if not objective.terms:
        if timeout_seconds is not None and timeout_seconds <= 0:
            return None, None, 0.0, True
        interrupted = Event()
        started = time.perf_counter()
        with Solver(name=sat_solver, bootstrap_with=hard_clauses) as solver:
            timer = None
            if timeout_seconds is not None:
                def request_interrupt() -> None:
                    interrupted.set()
                    solver.interrupt()

                timer = Timer(timeout_seconds, request_interrupt)
                timer.daemon = True
                timer.start()
            try:
                status = (
                    solver.solve_limited(expect_interrupt=True)
                    if timer is not None
                    else solver.solve()
                )
                model = solver.get_model() if status else None
            finally:
                if timer is not None:
                    timer.cancel()
        elapsed = time.perf_counter() - started
        if model is None:
            return None, None, elapsed, interrupted.is_set() or status is None
        return model, objective.offset, elapsed, False

    formula = _build_wcnf(hard_clauses, _objective_soft_clauses(objective))
    model, elapsed, timed_out = _run_maxsat(
        formula,
        sat_solver=sat_solver,
        maxsat_algorithm=maxsat_algorithm,
        timeout_seconds=timeout_seconds,
    )
    if model is None:
        return None, None, elapsed, timed_out
    return model, objective.evaluate(model), elapsed, False


def _fix_optimum(
    hard_clauses: List[List[int]],
    encoding: HCORAPEncoding,
    objective: LinearObjective,
    optimum: int,
) -> None:
    relation = "atleast" if objective.sense == "max" else "atmost"
    hard_clauses.extend(
        objective_bound_clauses(
            objective, relation, optimum, vpool=encoding.vpool
        )
    )


def _checked_result(
    instance: HCORAPInstance,
    encoding: HCORAPEncoding,
    model: Sequence[int],
    *,
    method: str,
    stages: Sequence[StageResult],
    elapsed_seconds: float,
    metadata: Optional[dict] = None,
) -> SolveResult:
    assignments = encoding.assignments_from_model(model)
    verification = verify_assignments(
        instance,
        assignments,
        require_full_coverage=encoding.require_full_coverage,
    )
    if not verification.valid:
        raise SolverConsistencyError(
            "decoded MaxSAT model violates HCORAP constraints: "
            + "; ".join(verification.violations[:5])
        )

    metrics = verification.metrics
    objective_values = {
        "coverage": metrics.coverage,
        "similarity": metrics.similarity,
        "continuity": metrics.continuity_penalty,
        "overtime": metrics.overtime,
    }
    disagreements = {
        name: (objective.evaluate(model), objective_values[name])
        for name, objective in encoding.objectives.items()
        if objective.evaluate(model) != objective_values[name]
    }
    if disagreements:
        raise SolverConsistencyError(
            f"objective variables disagree with decoded schedule: {disagreements}"
        )

    result_metadata = dict(metadata or {})
    result_metadata["encoding"] = dict(encoding.stats())
    return SolveResult(
        status="OPTIMUM",
        method=method,
        assignments=assignments,
        metrics=metrics,
        stages=tuple(stages),
        elapsed_seconds=elapsed_seconds,
        metadata=result_metadata,
    )


def _empty_result(
    status: str,
    method: str,
    stages: Sequence[StageResult],
    elapsed_seconds: float,
    *,
    metadata: Optional[dict] = None,
) -> SolveResult:
    return SolveResult(
        status=status,
        method=method,
        stages=tuple(stages),
        elapsed_seconds=elapsed_seconds,
        metadata=dict(metadata or {}),
    )


def solve_weighted(
    instance: HCORAPInstance,
    *,
    continuity_weight: int = 1,
    overtime_weight: int = 1,
    require_full_coverage: bool = True,
    sat_solver: str = "g4",
    maxsat_algorithm: str = "rc2-stratified",
    timeout_seconds: Optional[float] = None,
) -> SolveResult:
    """Solve ``max SIM - wc*CONT - wo*penalty*OT``.

    When coverage is soft, it is optimized and fixed in a separate first stage
    rather than hidden in an instance-dependent big weight.
    """

    if continuity_weight < 0 or overtime_weight < 0:
        raise ValueError("continuity and overtime weights must be non-negative")

    started = time.perf_counter()
    encoding = HCORAPEncoding(
        instance, require_full_coverage=require_full_coverage
    )
    hard = [list(clause) for clause in encoding.cnf.clauses]
    stages: List[StageResult] = []

    if not require_full_coverage:
        coverage = encoding.objective("coverage")
        model, optimum, elapsed, timed_out = _optimize(
            hard,
            coverage,
            sat_solver=sat_solver,
            maxsat_algorithm=maxsat_algorithm,
            timeout_seconds=_remaining_timeout(started, timeout_seconds),
        )
        if model is None or optimum is None:
            return _empty_result(
                "TIMEOUT" if timed_out else "UNSATISFIABLE",
                "weighted",
                stages,
                time.perf_counter() - started,
            )
        stages.append(StageResult("coverage", "max", optimum, elapsed))
        _fix_optimum(hard, encoding, coverage, optimum)

    soft: List[Tuple[List[int], int]] = []
    soft.extend(_objective_soft_clauses(encoding.objective("similarity")))
    soft.extend(
        (clause, weight * continuity_weight)
        for clause, weight in _objective_soft_clauses(
            encoding.objective("continuity")
        )
    )
    overtime_multiplier = overtime_weight * instance.penalty
    soft.extend(
        (clause, weight * overtime_multiplier)
        for clause, weight in _objective_soft_clauses(
            encoding.objective("overtime")
        )
    )

    formula = _build_wcnf(hard, soft)
    model, stage_elapsed, timed_out = _run_maxsat(
        formula,
        sat_solver=sat_solver,
        maxsat_algorithm=maxsat_algorithm,
        timeout_seconds=_remaining_timeout(started, timeout_seconds),
    )
    total_elapsed = time.perf_counter() - started
    if model is None:
        return _empty_result(
            "TIMEOUT" if timed_out else "UNSATISFIABLE",
            "weighted",
            stages,
            total_elapsed,
        )

    similarity = encoding.objective("similarity").evaluate(model)
    continuity = encoding.objective("continuity").evaluate(model)
    overtime = encoding.objective("overtime").evaluate(model)
    weighted_score = (
        similarity
        - continuity_weight * continuity
        - overtime_multiplier * overtime
    )
    stages.append(
        StageResult("weighted_score", "max", weighted_score, stage_elapsed)
    )
    return _checked_result(
        instance,
        encoding,
        model,
        method="weighted",
        stages=stages,
        elapsed_seconds=total_elapsed,
        metadata={
            "continuity_weight": continuity_weight,
            "overtime_weight": overtime_weight,
            "overtime_penalty": instance.penalty,
            "weighted_score": weighted_score,
            "sat_solver": sat_solver,
            "maxsat_algorithm": maxsat_algorithm,
            "timeout_seconds": timeout_seconds,
            "full_coverage": require_full_coverage,
        },
    )


LEXICOGRAPHIC_POLICIES = {
    "continuity-priority": ("continuity", "similarity", "overtime"),
    "overtime-priority": ("overtime", "continuity", "similarity"),
}


def solve_lexicographic(
    instance: HCORAPInstance,
    *,
    policy: str = "continuity-priority",
    require_full_coverage: bool = True,
    sat_solver: str = "g4",
    maxsat_algorithm: str = "rc2-stratified",
    timeout_seconds: Optional[float] = None,
) -> SolveResult:
    """Solve one of the documented sequential lexicographic policies."""

    try:
        order = LEXICOGRAPHIC_POLICIES[policy]
    except KeyError as exc:
        raise ValueError(
            f"unknown policy {policy!r}; choose from {sorted(LEXICOGRAPHIC_POLICIES)}"
        ) from exc

    started = time.perf_counter()
    encoding = HCORAPEncoding(
        instance, require_full_coverage=require_full_coverage
    )
    hard = [list(clause) for clause in encoding.cnf.clauses]
    stages: List[StageResult] = []
    model: Optional[List[int]] = None

    objective_order = list(order)
    if not require_full_coverage:
        objective_order.insert(0, "coverage")

    for objective_name in objective_order:
        objective = encoding.objective(objective_name)
        model, optimum, elapsed, timed_out = _optimize(
            hard,
            objective,
            sat_solver=sat_solver,
            maxsat_algorithm=maxsat_algorithm,
            timeout_seconds=_remaining_timeout(started, timeout_seconds),
        )
        if model is None or optimum is None:
            return _empty_result(
                "TIMEOUT" if timed_out else "UNSATISFIABLE",
                f"lexicographic:{policy}",
                stages,
                time.perf_counter() - started,
            )
        stages.append(
            StageResult(objective.name, objective.sense, optimum, elapsed)
        )
        _fix_optimum(hard, encoding, objective, optimum)

    assert model is not None
    return _checked_result(
        instance,
        encoding,
        model,
        method=f"lexicographic:{policy}",
        stages=stages,
        elapsed_seconds=time.perf_counter() - started,
        metadata={
            "policy": policy,
            "order": objective_order,
            "sat_solver": sat_solver,
            "maxsat_algorithm": maxsat_algorithm,
            "timeout_seconds": timeout_seconds,
            "full_coverage": require_full_coverage,
        },
    )


def _delta_fraction(delta: Delta) -> Fraction:
    if isinstance(delta, Fraction):
        value = delta
    elif isinstance(delta, float):
        value = Fraction(str(delta))
    else:
        value = Fraction(delta)
    if value < 0 or value > 1:
        raise ValueError("delta must lie in the closed interval [0, 1]")
    return value


def similarity_budget(similarity_optimum: int, delta: Delta) -> int:
    """Compute ``ceil((1-delta)*SIM*)`` without floating-point rounding."""

    fraction = _delta_fraction(delta)
    retained = (1 - fraction) * similarity_optimum
    return (retained.numerator + retained.denominator - 1) // retained.denominator


def solve_epsilon_constraint(
    instance: HCORAPInstance,
    *,
    delta: Delta,
    require_full_coverage: bool = True,
    sat_solver: str = "g4",
    maxsat_algorithm: str = "rc2-stratified",
    timeout_seconds: Optional[float] = None,
) -> SolveResult:
    """Solve one similarity-budget point with lexicographic completion."""

    delta_value = _delta_fraction(delta)
    started = time.perf_counter()
    encoding = HCORAPEncoding(
        instance, require_full_coverage=require_full_coverage
    )
    hard = [list(clause) for clause in encoding.cnf.clauses]
    stages: List[StageResult] = []

    if not require_full_coverage:
        coverage = encoding.objective("coverage")
        model, optimum, elapsed, timed_out = _optimize(
            hard,
            coverage,
            sat_solver=sat_solver,
            maxsat_algorithm=maxsat_algorithm,
            timeout_seconds=_remaining_timeout(started, timeout_seconds),
        )
        if model is None or optimum is None:
            return _empty_result(
                "TIMEOUT" if timed_out else "UNSATISFIABLE",
                "epsilon-constraint",
                stages,
                time.perf_counter() - started,
            )
        stages.append(StageResult("coverage", "max", optimum, elapsed))
        _fix_optimum(hard, encoding, coverage, optimum)

    similarity = encoding.objective("similarity")
    model, similarity_optimum, elapsed, timed_out = _optimize(
        hard,
        similarity,
        sat_solver=sat_solver,
        maxsat_algorithm=maxsat_algorithm,
        timeout_seconds=_remaining_timeout(started, timeout_seconds),
    )
    if model is None or similarity_optimum is None:
        return _empty_result(
            "TIMEOUT" if timed_out else "UNSATISFIABLE",
            "epsilon-constraint",
            stages,
            time.perf_counter() - started,
        )
    stages.append(
        StageResult("similarity_reference", "max", similarity_optimum, elapsed)
    )

    threshold = similarity_budget(similarity_optimum, delta_value)
    hard.extend(
        objective_bound_clauses(
            similarity, "atleast", threshold, vpool=encoding.vpool
        )
    )

    completion = ("continuity", "overtime", "similarity")
    for index, objective_name in enumerate(completion):
        objective = encoding.objective(objective_name)
        model, optimum, elapsed, timed_out = _optimize(
            hard,
            objective,
            sat_solver=sat_solver,
            maxsat_algorithm=maxsat_algorithm,
            timeout_seconds=_remaining_timeout(started, timeout_seconds),
        )
        if model is None or optimum is None:
            return _empty_result(
                "TIMEOUT" if timed_out else "UNSATISFIABLE",
                "epsilon-constraint",
                stages,
                time.perf_counter() - started,
            )
        stage_name = (
            "similarity_tiebreak"
            if objective_name == "similarity"
            else objective_name
        )
        stages.append(StageResult(stage_name, objective.sense, optimum, elapsed))
        if index < len(completion) - 1:
            _fix_optimum(hard, encoding, objective, optimum)

    assert model is not None
    return _checked_result(
        instance,
        encoding,
        model,
        method="epsilon-constraint",
        stages=stages,
        elapsed_seconds=time.perf_counter() - started,
        metadata={
            "delta": str(delta_value),
            "similarity_optimum": similarity_optimum,
            "similarity_threshold": threshold,
            "completion_order": list(completion),
            "sat_solver": sat_solver,
            "maxsat_algorithm": maxsat_algorithm,
            "timeout_seconds": timeout_seconds,
            "full_coverage": require_full_coverage,
        },
    )


def solve_epsilon_grid(
    instance: HCORAPInstance,
    deltas: Iterable[Delta] = ("0", "0.01", "0.025", "0.05", "0.10"),
    *,
    require_full_coverage: bool = True,
    sat_solver: str = "g4",
    maxsat_algorithm: str = "rc2-stratified",
    timeout_seconds: Optional[float] = None,
) -> Tuple[SolveResult, ...]:
    """Solve the documented discrete budget grid in deterministic order."""

    return tuple(
        solve_epsilon_constraint(
            instance,
            delta=delta,
            require_full_coverage=require_full_coverage,
            sat_solver=sat_solver,
            maxsat_algorithm=maxsat_algorithm,
            timeout_seconds=timeout_seconds,
        )
        for delta in deltas
    )
