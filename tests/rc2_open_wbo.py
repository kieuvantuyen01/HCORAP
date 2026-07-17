#!/usr/bin/env python3
"""Tiny Open-WBO-compatible RC2 wrapper for C++ correctness tests only."""

from __future__ import annotations

import sys

from pysat.examples.rc2 import RC2
from pysat.formula import WCNF


def main() -> int:
    if len(sys.argv) != 2:
        print("c usage: rc2_open_wbo.py FORMULA.wcnf")
        return 2

    formula = WCNF(from_file=sys.argv[1])
    with RC2(formula) as solver:
        model = solver.compute()
    if model is None:
        print("s UNSATISFIABLE")
        return 0

    print("s OPTIMUM FOUND")
    print("v " + " ".join(str(value) for value in model) + " 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
