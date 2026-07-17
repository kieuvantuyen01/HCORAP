from __future__ import annotations

from pysat.solvers import Solver

from hcorap.encoding import HCORAPEncoding
from hcorap.metrics import verify_assignments


def test_encoding_decodes_to_verified_schedule(tradeoff_instance) -> None:
    encoding = HCORAPEncoding(tradeoff_instance)
    with Solver(name="g4", bootstrap_with=encoding.cnf.clauses) as solver:
        assert solver.solve()
        model = solver.get_model()
    assignments = encoding.assignments_from_model(model)
    verification = verify_assignments(tradeoff_instance, assignments)
    assert verification.valid
    assert encoding.objective("coverage").evaluate(model) == verification.metrics.coverage
    assert encoding.objective("similarity").evaluate(model) == verification.metrics.similarity
    assert encoding.objective("continuity").evaluate(model) == verification.metrics.continuity_penalty
    assert encoding.objective("overtime").evaluate(model) == verification.metrics.overtime


def test_hard_and_soft_coverage_have_distinct_status(partially_infeasible_instance) -> None:
    hard = HCORAPEncoding(partially_infeasible_instance, require_full_coverage=True)
    with Solver(name="g4", bootstrap_with=hard.cnf.clauses) as solver:
        assert not solver.solve()

    soft = HCORAPEncoding(partially_infeasible_instance, require_full_coverage=False)
    with Solver(name="g4", bootstrap_with=soft.cnf.clauses) as solver:
        assert solver.solve()
