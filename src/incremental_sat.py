#!/usr/bin/env python3
"""
SAT-based optimisation solver for the HCORAP problem.

Implements two approaches:
  1. Incremental SAT: single solver instance with clause-addition
     top-down search (learned clauses preserved across iterations)
  2. MaxSAT (RC2): single call to PySAT's RC2 core-guided solver

Usage:
    python3 incremental_sat.py <instance> [--mode incr|maxsat]
"""

import sys
import time
import argparse
from collections import Counter
from pysat.solvers import Solver
from pysat.card import CardEnc, EncType
from pysat.formula import WCNF
from pysat.examples.rc2 import RC2


# =============================================================================
# Instance Parser
# =============================================================================

class HCORAPInstance:
    """Parsed HCORAP instance data."""

    def __init__(self, filepath):
        self.filepath = filepath
        self._parse(filepath)

    def _parse(self, filepath):
        with open(filepath, 'r') as f:
            lines = f.read().splitlines()

        idx = 0

        def skip_to(tag):
            nonlocal idx
            while idx < len(lines) and lines[idx].strip() != tag:
                idx += 1
            idx += 1

        def read_int():
            nonlocal idx
            val = int(lines[idx].strip())
            idx += 1
            return val

        skip_to('#U');  self.U = read_int()
        skip_to('#S');  self.S = read_int()
        skip_to('#A');  self.A = read_int()
        skip_to('#TS'); self.TS = read_int()

        skip_to('#SU')
        self.SU = []
        while idx < len(lines) and lines[idx].strip() != '#SEQ':
            line = lines[idx].strip()
            if line:
                self.SU.append([int(x) for x in line.split()])
            idx += 1

        idx += 1  # skip '#SEQ'
        self.SEQ = []
        while idx < len(lines) and lines[idx].strip() != '#TSA(i)':
            line = lines[idx].strip()
            if line:
                self.SEQ.append([int(x) for x in line.split()])
            idx += 1

        idx += 1
        self.TSA = []
        for a in range(self.A):
            row = []
            while len(row) < self.TS:
                row.extend([int(x) for x in lines[idx].split()])
                idx += 1
            self.TSA.append(row[:self.TS])

        skip_to('#TSS(i)')
        self.TSS = []
        for s in range(self.S):
            row = []
            while len(row) < self.TS:
                row.extend([int(x) for x in lines[idx].split()])
                idx += 1
            self.TSS.append(row[:self.TS])

        skip_to('#r(i,j)')
        self.r = []
        for a in range(self.A):
            self.r.append([int(x) for x in lines[idx].split()])
            idx += 1

        skip_to('#P');  self.P = read_int()

        skip_to('#HN(i)')
        self.HN = [read_int() for _ in range(self.A)]

        skip_to('#HE(i)')
        self.HE = [read_int() for _ in range(self.A)]

    def __repr__(self):
        return (f"HCORAP(U={self.U}, S={self.S}, A={self.A}, TS={self.TS}, "
                f"|SEQ|={len(self.SEQ)}, P={self.P})")


# =============================================================================
# Variable Manager
# =============================================================================

class VarManager:
    def __init__(self):
        self._next_var = 1

    def new_var(self):
        v = self._next_var
        self._next_var += 1
        return v

    @property
    def num_vars(self):
        return self._next_var - 1


# =============================================================================
# HCORAP SAT Encoding
# =============================================================================

class HCORAPEncoding:
    """
    Encodes the HCORAP problem as SAT (hard clauses) + soft objective
    (list of (literal, weight) pairs).

    The encoding matches the C++ HCORAPEncoding in the HCORAP codebase,
    using sorting network counters for continuity and extra-hour objectives.
    """

    def __init__(self, instance: HCORAPInstance):
        self.inst = instance
        self.vm = VarManager()

        self.x = {}      # x[(a,s,t)] → var
        self.y = {}      # y[(a,s)]   → var
        self.su_var = {}  # su[(s,t)]  → var
        self.w = {}      # w[(a,q)]   → var

        self.hard_clauses = []
        self.soft_clauses = []   # list of (clause_lits, weight)

        self._create_variables()
        self._encode_hard()
        self._encode_soft()

    def _add_hard(self, clause):
        self.hard_clauses.append(clause)

    def _add_soft(self, lits, weight):
        """Add a soft clause: satisfied when at least one lit is true."""
        if isinstance(lits, int):
            lits = [lits]
        self.soft_clauses.append((lits, weight))

    # ---- Variables ----

    def _create_variables(self):
        inst = self.inst
        for a in range(inst.A):
            for s in range(inst.S):
                if inst.r[a][s] == 0:
                    continue
                self.y[(a, s)] = self.vm.new_var()
                for t in range(inst.TS):
                    if inst.TSA[a][t] and inst.TSS[s][t]:
                        self.x[(a, s, t)] = self.vm.new_var()

        for s in range(inst.S):
            for t in range(inst.TS):
                if inst.TSS[s][t]:
                    self.su_var[(s, t)] = self.vm.new_var()

        for a in range(inst.A):
            for q in range(len(inst.SEQ)):
                if len(inst.SEQ[q]) == 1:
                    s0 = inst.SEQ[q][0]
                    if (a, s0) in self.y:
                        self.w[(a, q)] = self.y[(a, s0)]
                else:
                    self.w[(a, q)] = self.vm.new_var()

    # ---- AMO ----

    def _add_amo(self, lits):
        if len(lits) <= 1:
            return
        if len(lits) <= 6:
            for i in range(len(lits)):
                for j in range(i + 1, len(lits)):
                    self._add_hard([-lits[i], -lits[j]])
        else:
            cnf = CardEnc.atmost(lits, bound=1,
                                 top_id=self.vm.num_vars,
                                 encoding=EncType.seqcounter)
            if cnf.nv > self.vm.num_vars:
                while self.vm.num_vars < cnf.nv:
                    self.vm.new_var()
            self.hard_clauses.extend(cnf.clauses)

    # ---- Hard constraints ----

    def _encode_hard(self):
        inst = self.inst

        # Reification: y <=> OR_t x
        for a in range(inst.A):
            for s in range(inst.S):
                if (a, s) not in self.y:
                    continue
                yv = self.y[(a, s)]
                xvs = [self.x[(a, s, t)] for t in range(inst.TS)
                       if (a, s, t) in self.x]
                if not xvs:
                    self._add_hard([-yv])
                    continue
                for xv in xvs:
                    self._add_hard([-xv, yv])
                self._add_hard([-yv] + xvs)

        # Reification: su <=> OR_a x
        for s in range(inst.S):
            for t in range(inst.TS):
                if (s, t) not in self.su_var:
                    continue
                sv = self.su_var[(s, t)]
                xvs = [self.x[(a, s, t)] for a in range(inst.A)
                       if (a, s, t) in self.x]
                if not xvs:
                    self._add_hard([-sv])
                    continue
                for xv in xvs:
                    self._add_hard([-xv, sv])
                self._add_hard([-sv] + xvs)

        # Reification: w <=> OR_{s in SEQ} y
        for a in range(inst.A):
            for q in range(len(inst.SEQ)):
                if len(inst.SEQ[q]) == 1:
                    continue
                if (a, q) not in self.w:
                    continue
                wv = self.w[(a, q)]
                yvs = [self.y[(a, s)] for s in inst.SEQ[q]
                       if (a, s) in self.y]
                if not yvs:
                    self._add_hard([-wv])
                    continue
                for yv in yvs:
                    self._add_hard([wv, -yv])
                self._add_hard([-wv] + yvs)

        # C1: AMO per service
        for s in range(inst.S):
            xvs = [self.x[(a, s, t)] for a in range(inst.A)
                   for t in range(inst.TS) if (a, s, t) in self.x]
            self._add_amo(xvs)

        # C2: AMO per agent per timeslot
        for a in range(inst.A):
            for t in range(inst.TS):
                xvs = [self.x[(a, s, t)] for s in range(inst.S)
                       if (a, s, t) in self.x]
                self._add_amo(xvs)

        # C4: AMO per user per timeslot
        for su_group in inst.SU:
            for t in range(inst.TS):
                svs = [self.su_var[(s, t)] for s in su_group
                       if (s, t) in self.su_var]
                self._add_amo(svs)

        # C5: Service coverage (hard)
        for s in range(inst.S):
            svs = [self.su_var[(s, t)] for t in range(inst.TS)
                   if (s, t) in self.su_var]
            if svs:
                self._add_hard(svs)

        # C6: Max working hours
        for a in range(inst.A):
            max_h = inst.HN[a] + inst.HE[a]
            yvs = [self.y[(a, s)] for s in range(inst.S)
                   if (a, s) in self.y]
            if len(yvs) > max_h:
                cnf = CardEnc.atmost(yvs, bound=max_h,
                                     top_id=self.vm.num_vars,
                                     encoding=EncType.seqcounter)
                if cnf.nv > self.vm.num_vars:
                    while self.vm.num_vars < cnf.nv:
                        self.vm.new_var()
                self.hard_clauses.extend(cnf.clauses)

    # ---- Sequential counter for small groups ----

    def _build_counter(self, lits):
        """Sequential counter for list of lits.
           Returns outputs[k] = "at least k+1 true"."""
        n = len(lits)
        if n == 0:
            return []
        if n == 1:
            return [lits[0]]

        prev = [0] * n
        prev[0] = lits[0]
        for j in range(1, n):
            v = self.vm.new_var()
            self._add_hard([-v])
            prev[j] = v

        for i in range(1, n):
            curr = [0] * n
            for j in range(min(i + 1, n)):
                v = self.vm.new_var()
                curr[j] = v
                if j == 0:
                    self._add_hard([-prev[0], v])
                    self._add_hard([-lits[i], v])
                    self._add_hard([prev[0], lits[i], -v])
                else:
                    self._add_hard([-prev[j], v])
                    self._add_hard([-prev[j - 1], -lits[i], v])
                    self._add_hard([-v, prev[j], prev[j - 1]])
                    self._add_hard([-v, prev[j], lits[i]])
            for j in range(i + 1, n):
                v = self.vm.new_var()
                self._add_hard([-v])
                curr[j] = v
            prev = curr
        return prev

    # ---- Soft objectives ----

    def _encode_soft(self):
        inst = self.inst

        # O1: Expertise reward
        for a in range(inst.A):
            for s in range(inst.S):
                if inst.r[a][s] > 0 and (a, s) in self.y:
                    self._add_soft(self.y[(a, s)], inst.r[a][s])

        # O2: Continuity penalty (using sorting networks)
        for q in range(len(inst.SEQ)):
            if len(inst.SEQ[q]) == 1:
                continue
            wvs = [self.w[(a, q)] for a in range(inst.A)
                   if (a, q) in self.w]
            if not wvs:
                continue
            p = min(inst.A, len(inst.SEQ[q]))
            outs = self._build_counter(wvs)
            for i in range(min(p, len(outs))):
                self._add_soft(-outs[i], 1)

        # O3: Extra-hour penalty
        for a in range(inst.A):
            yvs = [self.y[(a, s)] for s in range(inst.S)
                   if (a, s) in self.y]
            if len(yvs) <= inst.HN[a]:
                continue
            outs = self._build_counter(yvs)
            max_h = inst.HN[a] + inst.HE[a]
            p = min(max_h, len(outs))
            pw = -inst.P
            for k in range(inst.HN[a], p):
                self._add_soft(-outs[k], pw)


# =============================================================================
# Incremental SAT Solver (top-down with clause addition)
# =============================================================================

def solve_incremental(enc: HCORAPEncoding, solver_name='glucose4'):
    """
    Incremental top-down search: find initial solution, then add
    permanent bound-tightening constraints. Learned clauses are reused.
    """
    solver = Solver(name=solver_name)
    for cl in enc.hard_clauses:
        solver.add_clause(cl)

    print(f"[INFO] Solver: {solver_name}, {enc.vm.num_vars} vars, "
          f"{len(enc.hard_clauses)} hard, {len(enc.soft_clauses)} soft")

    # Compute weighted objective from model
    def count_obj(model):
        ms = set(model)
        return sum(w for lits, w in enc.soft_clauses
                   if any(l in ms for l in (lits if isinstance(lits, list) else [lits])))

    # --- Build permanent objective representation ---
    # Collect (literal, weight) pairs from soft clauses (unit soft clauses only)
    obj_pairs = []
    for lits, w in enc.soft_clauses:
        if isinstance(lits, int):
            obj_pairs.append((lits, w))
        elif len(lits) == 1:
            obj_pairs.append((lits[0], w))
        else:
            # Clause soft — need auxiliary variable
            # aux <=> OR(lits)
            aux = enc.vm.new_var()
            for l in lits:
                solver.add_clause([-l, aux])  # l => aux
            solver.add_clause([-aux] + lits)  # aux => OR(lits)
            obj_pairs.append((aux, w))

    total_W = sum(w for _, w in obj_pairs)
    print(f"[INFO] Weighted obj: {len(obj_pairs)} lits, W={total_W}")

    t_start = time.time()
    n_calls = 0

    # Find initial feasible solution
    n_calls += 1
    if not solver.solve():
        print("[INFO] UNSATISFIABLE")
        solver.delete()
        return None

    model = solver.get_model()
    ms = set(model)
    best_obj = sum(w for l, w in obj_pairs if l in ms)
    best_model = model
    print(f"  [{n_calls:2d}] Initial: obj={best_obj}")

    # Iteratively tighten: require obj >= best_obj + 1
    while best_obj < total_W:
        target = best_obj + 1

        # Encode: sum of w_i * lit_i >= target
        # Equivalent: sum of w_i * (¬lit_i) <= total_W - target
        neg_lits = [-l for l, w in obj_pairs]
        weights_list = [w for l, w in obj_pairs]
        bound = total_W - target

        if bound < 0:
            break

        # Weighted atmost via expansion (each lit repeated w times)
        expanded = []
        for nl, wt in zip(neg_lits, weights_list):
            expanded.extend([nl] * wt)

        cnf = CardEnc.atmost(expanded, bound=bound,
                             top_id=enc.vm.num_vars,
                             encoding=EncType.totalizer)
        if cnf.nv > enc.vm.num_vars:
            while enc.vm.num_vars < cnf.nv:
                enc.vm.new_var()
        for cl in cnf.clauses:
            solver.add_clause(cl)

        n_calls += 1
        t0 = time.time()
        sat = solver.solve()
        t_call = time.time() - t0

        if sat:
            model = solver.get_model()
            ms = set(model)
            obj = sum(w for l, w in obj_pairs if l in ms)
            print(f"  [{n_calls:2d}] target>={target:4d}  SAT   "
                  f"obj={obj:4d}  ({t_call:.3f}s)")
            best_obj = obj
            best_model = model
        else:
            print(f"  [{n_calls:2d}] target>={target:4d}  UNSAT "
                  f"             ({t_call:.3f}s)")
            break

    t_total = time.time() - t_start
    print(f"[INFO] Done: {n_calls} calls, {t_total:.2f}s, optimal={best_obj}")

    solver.delete()
    return (best_obj, best_model)


# =============================================================================
# MaxSAT Solver (RC2)
# =============================================================================

def solve_maxsat(enc: HCORAPEncoding):
    """
    Solve using PySAT's RC2 (core-guided MaxSAT solver).
    """
    wcnf = WCNF()

    # Hard clauses (infinite weight = top weight)
    for cl in enc.hard_clauses:
        wcnf.append(cl)

    # Soft clauses
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


# =============================================================================
# Solution Verification
# =============================================================================

def verify_solution(inst, enc, model):
    model_set = set(model)
    ok = True

    assignments = [(a, s, t) for (a, s, t), v in enc.x.items()
                   if v in model_set]

    # C1
    svc_cnt = Counter(s for a, s, t in assignments)
    for s, c in svc_cnt.items():
        if c > 1:
            print(f"  [FAIL] C1: svc {s} = {c}x"); ok = False

    # C2
    at_cnt = Counter((a, t) for a, s, t in assignments)
    for k, c in at_cnt.items():
        if c > 1:
            print(f"  [FAIL] C2: {k} = {c}"); ok = False

    # C3
    for a, s, t in assignments:
        if inst.r[a][s] == 0 or not inst.TSA[a][t] or not inst.TSS[s][t]:
            print(f"  [FAIL] C3: ({a},{s},{t})"); ok = False

    # C4
    for su in inst.SU:
        su_set = set(su)
        ts_cnt = Counter(t for a, s, t in assignments if s in su_set)
        for t, c in ts_cnt.items():
            if c > 1:
                print(f"  [FAIL] C4: t={t} c={c}"); ok = False

    covered = set(s for a, s, t in assignments)
    uncovered = set(range(inst.S)) - covered

    agent_h = Counter(a for a, s, t in assignments)
    for a in range(inst.A):
        if agent_h.get(a, 0) > inst.HN[a] + inst.HE[a]:
            print(f"  [FAIL] C6: agent {a}"); ok = False

    sim = sum(inst.r[a][s] for a, s, t in assignments)
    cont = 0
    for seq in inst.SEQ:
        agents = set(a for a, s, t in assignments if s in seq)
        if len(agents) > 1:
            cont += len(agents) - 1
    extra = 0
    for a in range(inst.A):
        h = agent_h.get(a, 0)
        if h > inst.HN[a]:
            extra += (h - inst.HN[a]) * (-inst.P)
    total = sim - cont - extra

    print(f"\n  === Solution ===")
    print(f"  Constraints:   {'OK' if ok else 'FAILED'}")
    print(f"  Assigned:      {len(assignments)}/{inst.S}")
    print(f"  Uncovered:     {len(uncovered)}")
    print(f"  Similarity:    {sim}")
    print(f"  Continuity:    -{cont}")
    print(f"  Extra hours:   -{extra}")
    print(f"  Objective:     {total}")
    return ok


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='SAT-based solver for HCORAP')
    parser.add_argument('instance', help='Path to instance file')
    parser.add_argument('--mode', default='incr',
                        choices=['incr', 'maxsat'],
                        help='incr=incremental SAT, maxsat=RC2 (default: incr)')
    parser.add_argument('--solver', default='glucose4',
                        choices=['glucose4', 'cadical153', 'minisat22'])
    args = parser.parse_args()

    print("=" * 60)
    print(f"  HCORAP SAT-Based Solver  (mode={args.mode})")
    print("=" * 60)

    print(f"\n[INFO] Parsing: {args.instance}")
    t0 = time.time()
    inst = HCORAPInstance(args.instance)
    print(f"[INFO] {inst}  ({time.time()-t0:.3f}s)")

    print(f"\n[INFO] Encoding...")
    t0 = time.time()
    enc = HCORAPEncoding(inst)
    t_enc = time.time() - t0
    print(f"[INFO] Encoded in {t_enc:.2f}s: "
          f"{enc.vm.num_vars} vars, {len(enc.hard_clauses)} hard, "
          f"{len(enc.soft_clauses)} soft")

    print(f"\n[INFO] Solving ({args.mode})...")
    t0 = time.time()
    if args.mode == 'incr':
        result = solve_incremental(enc, solver_name=args.solver)
    else:
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
