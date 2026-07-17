"""Cross-check the proposed weighted model against the authors' C++ encoder."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

from pysat.formula import WCNF

from .io import read_instance
from .solvers import _run_maxsat, solve_weighted


COMMENT_TOTALS = re.compile(r"^c\s+(-?\d+)\s+(-?\d+)\s*$")


def parse_post_2022_wcnf(text: str) -> Tuple[WCNF, int, int]:
    """Parse the headerless ``h`` hard-clause format emitted by the C++ code."""

    formula = WCNF()
    total_soft = None
    constant_revenue = None
    for line_number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        if line.startswith("c"):
            match = COMMENT_TOTALS.match(line)
            if match and total_soft is None:
                total_soft = int(match.group(1))
                constant_revenue = int(match.group(2))
            continue
        tokens = line.split()
        if tokens[-1] != "0":
            raise ValueError(f"unterminated WCNF clause at line {line_number}")
        if tokens[0] == "h":
            formula.append([int(token) for token in tokens[1:-1]])
        else:
            weight = int(tokens[0])
            formula.append([int(token) for token in tokens[1:-1]], weight=weight)
    if total_soft is None or constant_revenue is None:
        raise ValueError("C++ output does not contain the expected objective comment")
    return formula, total_soft, constant_revenue


def _clause_satisfied(clause: Iterable[int], positive: set[int]) -> bool:
    return any(
        literal in positive if literal > 0 else -literal not in positive
        for literal in clause
    )


def _unsatisfied_soft_cost(formula: WCNF, model: Iterable[int]) -> int:
    positive = {literal for literal in model if literal > 0}
    return sum(
        int(weight)
        for clause, weight in zip(formula.soft, formula.wght)
        if not _clause_satisfied(clause, positive)
    )


def crosscheck_cpp_instance(
    instance_path: Path,
    *,
    binary: Path = Path("bin/release/hcorap2sat"),
    timeout_seconds: Optional[float] = 60.0,
    sat_solver: str = "g4",
) -> Dict[str, Any]:
    """Compare certified objective values after removing the proven constant."""

    instance_path = Path(instance_path).resolve()
    binary = Path(binary).resolve()
    if not binary.is_file():
        raise FileNotFoundError(
            f"C++ encoder not found at {binary}; run `make YICES=0` first"
        )
    instance = read_instance(instance_path)
    if instance.overtime_penalty > 0:
        raise ValueError("the authors' C++ encoding expects a non-positive P")

    completed = subprocess.run(
        [str(binary), "-e=1", "-f=dimacs", "-S=0", str(instance_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    formula, total_soft, constant_revenue = parse_post_2022_wcnf(
        completed.stdout
    )
    cpp_model, cpp_elapsed, cpp_timeout = _run_maxsat(
        formula,
        sat_solver=sat_solver,
        maxsat_algorithm="rc2-stratified",
        timeout_seconds=timeout_seconds,
    )
    if cpp_model is None:
        return {
            "status": "TIMEOUT" if cpp_timeout else "UNSATISFIABLE",
            "cpp_elapsed_seconds": cpp_elapsed,
            "match": None,
        }

    cpp_cost = _unsatisfied_soft_cost(formula, cpp_model)
    cpp_original_reward = total_soft - cpp_cost + constant_revenue
    continuity_constant = sum(len(sequence) - 1 for sequence in instance.sequences)
    cpp_equivalent_score = cpp_original_reward - continuity_constant

    proposed = solve_weighted(
        instance,
        continuity_weight=1,
        overtime_weight=1,
        sat_solver=sat_solver,
        maxsat_algorithm="rc2-stratified",
        timeout_seconds=timeout_seconds,
    )
    proposed_score = None
    if proposed.metrics is not None:
        proposed_score = (
            proposed.metrics.similarity
            - proposed.metrics.continuity_penalty
            - instance.penalty * proposed.metrics.overtime
        )
    return {
        "status": proposed.status,
        "match": proposed.status == "OPTIMUM" and cpp_equivalent_score == proposed_score,
        "instance": str(instance_path),
        "cpp": {
            "variables": formula.nv,
            "hard_clauses": len(formula.hard),
            "soft_clauses": len(formula.soft),
            "total_soft": total_soft,
            "constant_revenue": constant_revenue,
            "unsatisfied_cost": cpp_cost,
            "original_reward": cpp_original_reward,
            "continuity_constant": continuity_constant,
            "equivalent_score": cpp_equivalent_score,
            "elapsed_seconds": cpp_elapsed,
        },
        "proposed": {
            "score": proposed_score,
            "result": proposed.as_dict(),
        },
    }
