#!/usr/bin/env python3
"""
MaxSAT solver for the HCORAP problem.

Uses PySAT's RC2 (core-guided MaxSAT solver) to solve the
partial weighted MaxSAT encoding in a single invocation.

Corresponds to Section 6.3 in doc/main.tex.

Usage:
    python3 maxsat_solver.py <instance>
"""

import time
import argparse
from pysat.formula import WCNF
from pysat.examples.rc2 import RC2

from hcorap_encoding import HCORAPInstance, HCORAPEncoding, verify_solution


def solve_maxsat(enc: HCORAPEncoding):
    """
    Solve using PySAT's RC2 (core-guided MaxSAT solver).

    Creates a WCNF formula with:
      - Hard clauses: all constraints (weight = ∞)
      - Soft clauses: objective components with finite weights
    RC2 maximises total weight of satisfied soft clauses.
    """
    wcnf = WCNF()

    for cl in enc.hard_clauses:
        wcnf.append(cl)

    for lits, w in enc.soft_clauses:
        if isinstance(lits, int):
            wcnf.append([lits], weight=w)
        elif isinstance(lits, list):
            wcnf.append(lits, weight=w)

    total_W = sum(w for _, w in enc.soft_clauses)
    print(f"[INFO] WCNF: {wcnf.nv} vars, "
          f"{len(wcnf.hard)} hard, {len(wcnf.soft)} soft, W={total_W}")

    t0 = time.time()
    with RC2(wcnf) as rc2:
        model = rc2.compute()
        cost = rc2.cost
    t_solve = time.time() - t0

    if model is None:
        print(f"[INFO] UNSATISFIABLE ({t_solve:.2f}s)")
        return None

    obj = total_W - cost  # RC2 minimises cost = unsatisfied weight
    print(f"[INFO] RC2: cost={cost}, obj={obj}, time={t_solve:.2f}s")
    return (obj, model)


def main():
    parser = argparse.ArgumentParser(
        description='MaxSAT (RC2) solver for HCORAP')
    parser.add_argument('instance', help='Path to instance file')
    args = parser.parse_args()

    print("=" * 60)
    print("  HCORAP MaxSAT Solver (RC2)")
    print("=" * 60)

    # Parse
    print(f"\n[INFO] Parsing: {args.instance}")
    t0 = time.time()
    inst = HCORAPInstance(args.instance)
    print(f"[INFO] {inst}  ({time.time()-t0:.3f}s)")

    # Encode
    print(f"\n[INFO] Encoding...")
    t0 = time.time()
    enc = HCORAPEncoding(inst)
    t_enc = time.time() - t0
    print(f"[INFO] Encoded in {t_enc:.2f}s: "
          f"{enc.vm.num_vars} vars, {len(enc.hard_clauses)} hard, "
          f"{len(enc.soft_clauses)} soft")

    # Solve
    print(f"\n[INFO] Solving (MaxSAT RC2)...")
    t0 = time.time()
    result = solve_maxsat(enc)
    t_solve = time.time() - t0

    if result is not None:
        obj, model = result
        print(f"\n[INFO] Total solve time: {t_solve:.2f}s")
        verify_solution(inst, enc, model)
    else:
        print(f"\n[INFO] UNSATISFIABLE ({t_solve:.2f}s)")

    print()


if __name__ == '__main__':
    main()
