#!/usr/bin/env python3
"""EvalMaxSAT-like stub that returns an incumbent when interrupted.

The stub solves the WCNF immediately. Files for the first two ordinary stages
are reported as optimal, while the later stage and the dominance-weighted
single-call formula wait for SIGTERM and then emit the already computed model
as SATISFIABLE. It intentionally requires the
``--TCT`` option so tests also verify target-time propagation.
"""

from __future__ import annotations

import signal
import sys
import time
from pathlib import Path

from pysat.examples.rc2 import RC2
from pysat.formula import WCNF


bit_string = ""


def _emit(status: str) -> None:
    print(f"s {status}", flush=True)
    print("v " + bit_string, flush=True)


def _handle_term(_signum: int, _frame: object) -> None:
    _emit("SATISFIABLE")
    raise SystemExit(10)


def main() -> int:
    if len(sys.argv) != 4 or sys.argv[1] != "--TCT":
        print("c usage: rc2_timeout_incumbent.py --TCT SECONDS FORMULA.wcnf")
        return 2
    int(sys.argv[2])
    formula_path = Path(sys.argv[3])
    formula = WCNF(from_file=str(formula_path))
    with RC2(formula) as solver:
        model = solver.compute()
    if model is None:
        print("s UNSATISFIABLE", flush=True)
        return 0

    positive = {literal for literal in model if literal > 0}
    global bit_string
    bit_string = "".join(
        "1" if variable in positive else "0"
        for variable in range(1, formula.nv + 1)
    )

    dominance_weighted = max(formula.wght, default=0) > 10
    if not dominance_weighted and (
        formula_path.name.endswith("_0.wcnf")
        or formula_path.name.endswith("_1.wcnf")
    ):
        _emit("OPTIMUM FOUND")
        return 0

    signal.signal(signal.SIGTERM, _handle_term)
    while True:
        time.sleep(0.05)


if __name__ == "__main__":
    raise SystemExit(main())
