#!/usr/bin/env python3
"""Tiny EvalMaxSAT-output-compatible RC2 wrapper for parser regression tests."""

from __future__ import annotations

import sys

from pysat.examples.rc2 import RC2
from pysat.formula import WCNF


def main() -> int:
    if len(sys.argv) != 2:
        print("c usage: rc2_evalmaxsat.py FORMULA.wcnf")
        return 2

    formula = WCNF(from_file=sys.argv[1])
    with RC2(formula) as solver:
        model = solver.compute()
    if model is None:
        print("s UNSATISFIABLE")
        return 0

    positive = {literal for literal in model if literal > 0}
    bit_string = "".join(
        "1" if variable in positive else "0"
        for variable in range(1, formula.nv + 1)
    )
    print("s OPTIMUM FOUND")
    print("v " + bit_string)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
