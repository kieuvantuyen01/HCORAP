"""Research implementation of HCORAP optimization methods.

The C++ files elsewhere under :mod:`src` are the authors' baseline.  This
package is the proposed, independently tested Python implementation.
"""

from .io import InstanceFormatError, read_instance, write_instance
from .metrics import VerificationResult, compute_metrics, verify_assignments
from .model import Assignment, HCORAPInstance, Metrics, SolveResult, StageResult
from .solvers import (
    solve_epsilon_constraint,
    solve_epsilon_grid,
    solve_lexicographic,
    solve_weighted,
)

__all__ = [
    "Assignment",
    "HCORAPInstance",
    "InstanceFormatError",
    "Metrics",
    "SolveResult",
    "StageResult",
    "VerificationResult",
    "compute_metrics",
    "read_instance",
    "solve_epsilon_constraint",
    "solve_epsilon_grid",
    "solve_lexicographic",
    "solve_weighted",
    "verify_assignments",
    "write_instance",
]

__version__ = "0.1.0"
