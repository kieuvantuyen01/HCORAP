#!/usr/bin/env python3
"""
SAT-based optimisation solver for the HCORAP problem.

Encoding follows the MaxSAT model described in doc/main.tex (Section 5),
which is based on the original paper by Unceta et al. (2025).

Variables (§5.1):
    x[(a,s,t)]  — main variable: agent a performs service s at time slot t
    y[(a,s)]    — assignment variable: agent a performs service s (eq:reif-y)
    z[(s,t)]    — service time-slot variable (su in paper) (eq:reif-z)
    w[(a,q)]    — sequence assignment variable (ss_{a,q} in paper) (eq:reif-w)
    w_hat       — service count outputs via sorting network (eq:w-count)
    c           — distinct agent count outputs via sorting network (eq:c-count)

Hard Clauses (§5.2):
    Reification (eq:reif-y, eq:reif-z, eq:reif-w)
    AMO per service (C1, eq:amo1)
    AMO per agent per timeslot (C2, eq:amo2)
    AMO per user per timeslot (C4, eq:amo4)
    Implicit C3: invalid (a,s,t) variables not created
    Coverage (C5, eq:cov-hard)
    Max hours (C6, eq:max-hours) via sorting network

Soft Clauses — Objective (§5.3):
    objective = similarity + stability − cost  (eq:sat-obj)
    O1: ⟨y_{a,s}, r(a,s)⟩              (eq:soft-reward)
    O2: ⟨¬c_{q,i}, 1⟩                  (eq:soft-cont)
    O3: ⟨¬ŵ_{a,i}, |P|⟩               (eq:soft-extra)

Implements two solving approaches:
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
    """Parsed HCORAP instance data (Section 1.1 of main.tex)."""

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
    """Allocates fresh Boolean variable IDs for PySAT."""

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
# HCORAP SAT Encoding (Section 5 of main.tex)
# =============================================================================

class HCORAPEncoding:
    """
    Full SAT encoding of the HCORAP problem.

    Produces:
      - hard_clauses: list of clauses (all must be satisfied)
      - soft_clauses: list of (literal_or_clause, weight) pairs

    The encoding follows Section 5 of main.tex:
      §5.1 Boolean Variables + Definition Constraints
      §5.2 Hard Clauses (C1–C6)
      §5.3 Soft Clauses (O1–O3)
    """

    def __init__(self, instance: HCORAPInstance):
        self.inst = instance
        self.vm = VarManager()

        # --- §5.1 Boolean Variables ---
        self.x = {}       # x[(a,s,t)] : main variable (eq 1)
        self.y = {}       # y[(a,s)]   : assignment variable (eq:reif-y)
        self.z = {}       # z[(s,t)]   : service time-slot var (eq:reif-z)
        self.w = {}       # w[(a,q)]   : sequence assignment var (eq:reif-w)
        # w_hat and c are created by _build_sorting_network() in _encode_soft()

        self.hard_clauses = []
        self.soft_clauses = []   # list of ([lits], weight)

        self._create_variables()
        self._encode_hard()
        self._encode_soft()

    def _add_hard(self, clause):
        """Add a hard clause (must be satisfied)."""
        self.hard_clauses.append(clause)

    def _add_soft(self, lits, weight):
        """Add a weighted soft clause ⟨c, w⟩ (eq:sat-obj)."""
        if isinstance(lits, int):
            lits = [lits]
        self.soft_clauses.append((lits, weight))

    # =========================================================================
    # §5.1 Variable Creation
    # =========================================================================

    def _create_variables(self):
        """
        Create Boolean variables for all valid combinations.

        - x[(a,s,t)] created only when r(a,s)>0, TSA(a,t)=1, TSS(s,t)=1
          (implicit C3: eq 11,12,13 in paper)
        - y[(a,s)] created only when r(a,s)>0
        - z[(s,t)] created only when TSS(s,t)=1
        - w[(a,q)] created for each agent-sequence pair
        """
        inst = self.inst

        # Main variables x and assignment variables y
        for a in range(inst.A):
            for s in range(inst.S):
                if inst.r[a][s] == 0:
                    continue  # C3/qualification: r(a,s)=0 → forbidden
                self.y[(a, s)] = self.vm.new_var()
                for t in range(inst.TS):
                    if inst.TSA[a][t] and inst.TSS[s][t]:
                        self.x[(a, s, t)] = self.vm.new_var()

        # Service time-slot variables z (= su in paper)
        for s in range(inst.S):
            for t in range(inst.TS):
                if inst.TSS[s][t]:
                    self.z[(s, t)] = self.vm.new_var()

        # Sequence assignment variables w (= ss_{a,q} in paper)
        # For singleton sequences, w[(a,q)] = y[(a,s)] directly
        for a in range(inst.A):
            for q in range(len(inst.SEQ)):
                if len(inst.SEQ[q]) == 1:
                    s0 = inst.SEQ[q][0]
                    if (a, s0) in self.y:
                        self.w[(a, q)] = self.y[(a, s0)]
                else:
                    self.w[(a, q)] = self.vm.new_var()

    # =========================================================================
    # AMO Encoding
    # =========================================================================

    def _add_amo(self, lits):
        """
        At-Most-One constraint.
        Uses pairwise encoding for ≤6 variables, sequential counter otherwise.
        (Paper §4.3 notes quadratic encoding for small AMO.)
        """
        if len(lits) <= 1:
            return
        if len(lits) <= 6:
            # Pairwise: O(n²) clauses
            for i in range(len(lits)):
                for j in range(i + 1, len(lits)):
                    self._add_hard([-lits[i], -lits[j]])
        else:
            # Sequential counter: O(n) aux vars, O(n) clauses
            cnf = CardEnc.atmost(lits, bound=1,
                                 top_id=self.vm.num_vars,
                                 encoding=EncType.seqcounter)
            if cnf.nv > self.vm.num_vars:
                while self.vm.num_vars < cnf.nv:
                    self.vm.new_var()
            self.hard_clauses.extend(cnf.clauses)

    # =========================================================================
    # §5.2 Hard Clauses
    # =========================================================================

    def _encode_hard(self):
        """Encode all hard constraints (§5.2 of main.tex)."""
        inst = self.inst

        # --- Reification: y_{a,s} ⟺ ∨_t x_{a,s,t}  (eq:reif-y) ---
        for a in range(inst.A):
            for s in range(inst.S):
                if (a, s) not in self.y:
                    continue
                y_var = self.y[(a, s)]
                x_vars = [self.x[(a, s, t)] for t in range(inst.TS)
                          if (a, s, t) in self.x]
                if not x_vars:
                    self._add_hard([-y_var])
                    continue
                # Forward: x_{a,s,t} → y_{a,s}
                for xv in x_vars:
                    self._add_hard([-xv, y_var])
                # Backward: y_{a,s} → ∨_t x_{a,s,t}
                self._add_hard([-y_var] + x_vars)

        # --- Reification: z_{s,t} ⟺ ∨_a x_{a,s,t}  (eq:reif-z) ---
        for s in range(inst.S):
            for t in range(inst.TS):
                if (s, t) not in self.z:
                    continue
                z_var = self.z[(s, t)]
                x_vars = [self.x[(a, s, t)] for a in range(inst.A)
                          if (a, s, t) in self.x]
                if not x_vars:
                    self._add_hard([-z_var])
                    continue
                # Forward: x_{a,s,t} → z_{s,t}
                for xv in x_vars:
                    self._add_hard([-xv, z_var])
                # Backward: z_{s,t} → ∨_a x_{a,s,t}
                self._add_hard([-z_var] + x_vars)

        # --- Reification: w_{a,q} ⟺ ∨_{s∈SEQ_q} y_{a,s}  (eq:reif-w) ---
        # (only for |SEQ_q| > 1; singleton handled in _create_variables)
        for a in range(inst.A):
            for q in range(len(inst.SEQ)):
                if len(inst.SEQ[q]) == 1:
                    continue  # w[(a,q)] = y[(a,s)] already
                if (a, q) not in self.w:
                    continue
                w_var = self.w[(a, q)]
                y_vars = [self.y[(a, s)] for s in inst.SEQ[q]
                          if (a, s) in self.y]
                if not y_vars:
                    self._add_hard([-w_var])
                    continue
                # Forward: y_{a,s} → w_{a,q}
                for yv in y_vars:
                    self._add_hard([-yv, w_var])
                # Backward: w_{a,q} → ∨_{s∈SEQ_q} y_{a,s}
                self._add_hard([-w_var] + y_vars)

        # --- C1: AMO per service (eq:amo1) ---
        # atMostOne({x_{a,s,t} : a ∈ A, t ∈ T})  ∀s
        for s in range(inst.S):
            x_vars = [self.x[(a, s, t)] for a in range(inst.A)
                      for t in range(inst.TS) if (a, s, t) in self.x]
            self._add_amo(x_vars)

        # --- C2: AMO per agent per timeslot (eq:amo2) ---
        # atMostOne({x_{a,s,t} : s ∈ S})  ∀a, ∀t
        for a in range(inst.A):
            for t in range(inst.TS):
                x_vars = [self.x[(a, s, t)] for s in range(inst.S)
                          if (a, s, t) in self.x]
                self._add_amo(x_vars)

        # --- C4: AMO per user per timeslot (eq:amo4) ---
        # atMostOne({z_{s,t} : s ∈ SU_u})  ∀u, ∀t
        for su_group in inst.SU:
            for t in range(inst.TS):
                z_vars = [self.z[(s, t)] for s in su_group
                          if (s, t) in self.z]
                self._add_amo(z_vars)

        # --- C5: Service coverage (eq:cov-hard) ---
        # ∨_{t∈T} z_{s,t}  ∀s
        for s in range(inst.S):
            z_vars = [self.z[(s, t)] for t in range(inst.TS)
                      if (s, t) in self.z]
            if z_vars:
                self._add_hard(z_vars)

        # --- C6: Maximum working hours (eq:max-hours) ---
        # Uses sorting network on {y_{a,s}}, forces out[HN_a + HE_a] = false
        for a in range(inst.A):
            max_hours = inst.HN[a] + inst.HE[a]
            y_vars = [self.y[(a, s)] for s in range(inst.S)
                      if (a, s) in self.y]
            if len(y_vars) > max_hours:
                cnf = CardEnc.atmost(y_vars, bound=max_hours,
                                     top_id=self.vm.num_vars,
                                     encoding=EncType.seqcounter)
                if cnf.nv > self.vm.num_vars:
                    while self.vm.num_vars < cnf.nv:
                        self.vm.new_var()
                self.hard_clauses.extend(cnf.clauses)

    # =========================================================================
    # Sorting Network (Sequential Counter)
    # =========================================================================

    def _build_sorting_network(self, lits):
        """
        Build a sequential counter (sorting network) over a list of literals.

        Given n input literals {l_1, ..., l_n}, produces n output variables
        {o_0, ..., o_{n-1}} where:
            o_k is true  ⟺  at least (k+1) of the inputs are true

        This implements the sorting network encoding from
        Asín et al. (2011), used for:
          - eq:w-count (ŵ_{a,i}): sorting over y_{a,s} variables
          - eq:c-count (c_{q,i}): sorting over w_{a,q} variables

        Returns:
            List of output variable IDs [o_0, ..., o_{n-1}]
        """
        n = len(lits)
        if n == 0:
            return []
        if n == 1:
            return [lits[0]]

        # Initialise: prev[0] = lit[0], prev[j>0] = false
        prev = [0] * n
        prev[0] = lits[0]
        for j in range(1, n):
            v = self.vm.new_var()
            self._add_hard([-v])  # force false
            prev[j] = v

        # Iteratively merge each new literal
        for i in range(1, n):
            curr = [0] * n
            for j in range(min(i + 1, n)):
                v = self.vm.new_var()
                curr[j] = v
                if j == 0:
                    # o_0 ⟺ (prev[0] ∨ lit[i])
                    self._add_hard([-prev[0], v])
                    self._add_hard([-lits[i], v])
                    self._add_hard([prev[0], lits[i], -v])
                else:
                    # o_j ⟺ (prev[j] ∨ (prev[j-1] ∧ lit[i]))
                    self._add_hard([-prev[j], v])
                    self._add_hard([-prev[j - 1], -lits[i], v])
                    self._add_hard([-v, prev[j], prev[j - 1]])
                    self._add_hard([-v, prev[j], lits[i]])
            for j in range(i + 1, n):
                v = self.vm.new_var()
                self._add_hard([-v])  # force false
                curr[j] = v
            prev = curr
        return prev

    # =========================================================================
    # §5.3 Soft Clauses (Objective Function)
    # =========================================================================

    def _encode_soft(self):
        """
        Encode the three objective components as weighted soft clauses.

        objective = similarity(O1) + stability(O2) − cost(O3)  (eq:sat-obj)

        where:
          similarity = Σ r(a,s) · y_{a,s}              (eq:similarity)
          stability  = Σ_q Σ_i (1 − c_{q,i})           (eq:stability)
          cost       = Σ_a Σ_i |P| · ŵ_{a,i}           (eq:cost)
        """
        inst = self.inst

        # --- O1: Similarity — ⟨y_{a,s}, r(a,s)⟩  (eq:soft-reward) ---
        for a in range(inst.A):
            for s in range(inst.S):
                if inst.r[a][s] > 0 and (a, s) in self.y:
                    self._add_soft(self.y[(a, s)], inst.r[a][s])

        # --- O2: Stability — ⟨¬c_{q,i}, 1⟩  (eq:soft-cont) ---
        # c_{q,i} = sorting network output over {w_{a,q}}_a  (eq:c-count)
        # c_{q,i} true ⟺ ≥i distinct agents work in SEQ_q
        # Soft clause ⟨¬c_{q,i}, 1⟩ rewards having fewer agents
        for q in range(len(inst.SEQ)):
            w_vars = [self.w[(a, q)] for a in range(inst.A)
                      if (a, q) in self.w]
            if not w_vars:
                continue
            # Build sorting network → c_{q,i} outputs  (eq:c-count)
            c_outputs = self._build_sorting_network(w_vars)
            # Add soft clauses for i = 1, ..., |SEQ_q|  (eq:soft-cont)
            p = min(len(inst.SEQ[q]), len(c_outputs))
            for i in range(p):
                self._add_soft(-c_outputs[i], 1)

        # --- O3: Extra-hour cost — ⟨¬ŵ_{a,i}, |P|⟩  (eq:soft-extra) ---
        # ŵ_{a,i} = sorting network output over {y_{a,s}}_s  (eq:w-count)
        # ŵ_{a,i} true ⟺ agent a assigned ≥i services
        # Soft clause ⟨¬ŵ_{a,i}, |P|⟩ rewards NOT working extra hours
        # (cost transformation: eq:cost-transform)
        abs_P = -inst.P  # |P|, since P < 0 in instance
        for a in range(inst.A):
            y_vars = [self.y[(a, s)] for s in range(inst.S)
                      if (a, s) in self.y]
            if len(y_vars) <= inst.HN[a]:
                continue  # no extra hours possible
            # Build sorting network → ŵ_{a,i} outputs  (eq:w-count)
            w_hat_outputs = self._build_sorting_network(y_vars)
            # Add soft clauses for i = HN_a+1, ..., HN_a+HE_a  (eq:soft-extra)
            # In 0-indexed outputs: w_hat_outputs[k] = "≥k+1 services"
            # So i=HN_a+1 corresponds to index k=HN_a
            upper = min(inst.HN[a] + inst.HE[a], len(w_hat_outputs))
            for k in range(inst.HN[a], upper):
                self._add_soft(-w_hat_outputs[k], abs_P)


# =============================================================================
# Incremental SAT Solver (top-down with clause addition)
# =============================================================================

def solve_incremental(enc: HCORAPEncoding, solver_name='glucose4'):
    """
    Incremental top-down linear search (Algorithm 5 in main.tex).

    Uses a single solver instance. After finding a solution with obj=K,
    adds permanent clauses enforcing obj ≥ K+1 and re-solves.
    Learned clauses from previous iterations are preserved.
    """
    solver = Solver(name=solver_name)
    for cl in enc.hard_clauses:
        solver.add_clause(cl)

    print(f"[INFO] Solver: {solver_name}, {enc.vm.num_vars} vars, "
          f"{len(enc.hard_clauses)} hard, {len(enc.soft_clauses)} soft")

    # --- Build objective representation ---
    # Convert all soft clauses to (literal, weight) pairs
    obj_pairs = []
    for lits, w in enc.soft_clauses:
        if isinstance(lits, int):
            obj_pairs.append((lits, w))
        elif len(lits) == 1:
            obj_pairs.append((lits[0], w))
        else:
            # Multi-literal soft clause: introduce aux ⟺ ∨(lits)
            aux = enc.vm.new_var()
            for l in lits:
                solver.add_clause([-l, aux])
            solver.add_clause([-aux] + lits)
            obj_pairs.append((aux, w))

    total_W = sum(w for _, w in obj_pairs)
    print(f"[INFO] Weighted obj: {len(obj_pairs)} lits, W={total_W}")

    t_start = time.time()
    n_calls = 0

    # Step 1: Find initial feasible solution
    n_calls += 1
    if not solver.solve():
        print("[INFO] UNSATISFIABLE")
        solver.delete()
        return None

    model = solver.get_model()
    model_set = set(model)
    best_obj = sum(w for l, w in obj_pairs if l in model_set)
    best_model = model
    print(f"  [{n_calls:2d}] Initial: obj={best_obj}")

    # Step 2: Iteratively tighten bound (top-down)
    while best_obj < total_W:
        target = best_obj + 1

        # Encode: Σ w_i · lit_i ≥ target
        # ⟺  Σ w_i · (¬lit_i) ≤ total_W − target
        neg_lits = [-l for l, w in obj_pairs]
        weights = [w for l, w in obj_pairs]
        bound = total_W - target

        if bound < 0:
            break

        # Weighted at-most via expansion (each ¬lit repeated w times)
        expanded = []
        for nl, wt in zip(neg_lits, weights):
            expanded.extend([nl] * wt)

        cnf = CardEnc.atmost(expanded, bound=bound,
                             top_id=enc.vm.num_vars,
                             encoding=EncType.totalizer)
        if cnf.nv > enc.vm.num_vars:
            while enc.vm.num_vars < cnf.nv:
                enc.vm.new_var()

        # Add permanent clauses (clause-addition approach)
        for cl in cnf.clauses:
            solver.add_clause(cl)

        n_calls += 1
        t0 = time.time()
        sat = solver.solve()
        t_call = time.time() - t0

        if sat:
            model = solver.get_model()
            model_set = set(model)
            obj = sum(w for l, w in obj_pairs if l in model_set)
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


# =============================================================================
# Solution Verification
# =============================================================================

def verify_solution(inst: HCORAPInstance, enc: HCORAPEncoding, model):
    """
    Verify hard constraints and compute objective breakdown.

    Checks: C1 (AMO/service), C2 (AMO/agent-timeslot), C3 (availability),
            C4 (AMO/user-timeslot), C5 (coverage), C6 (max hours).

    Objective = similarity − continuity_penalty − extra_hour_cost  (eq:sat-obj)
    """
    model_set = set(model)
    ok = True

    # Extract assignments: {(a, s, t) : x[(a,s,t)] is true}
    assignments = [(a, s, t) for (a, s, t), v in enc.x.items()
                   if v in model_set]

    # --- C1: At most one assignment per service ---
    svc_cnt = Counter(s for a, s, t in assignments)
    for s, c in svc_cnt.items():
        if c > 1:
            print(f"  [FAIL] C1: service {s} assigned {c} times"); ok = False

    # --- C2: At most one service per agent per timeslot ---
    at_cnt = Counter((a, t) for a, s, t in assignments)
    for (a, t), c in at_cnt.items():
        if c > 1:
            print(f"  [FAIL] C2: agent {a}, slot {t} = {c}"); ok = False

    # --- C3: Agent availability, service suitability, qualification ---
    for a, s, t in assignments:
        if inst.r[a][s] == 0:
            print(f"  [FAIL] C3: r({a},{s})=0"); ok = False
        if not inst.TSA[a][t]:
            print(f"  [FAIL] C3: TSA({a},{t})=0"); ok = False
        if not inst.TSS[s][t]:
            print(f"  [FAIL] C3: TSS({s},{t})=0"); ok = False

    # --- C4: At most one service per user per timeslot ---
    for su_group in inst.SU:
        su_set = set(su_group)
        ts_cnt = Counter(t for a, s, t in assignments if s in su_set)
        for t, c in ts_cnt.items():
            if c > 1:
                print(f"  [FAIL] C4: user-group, slot {t} = {c}"); ok = False

    # --- C5: Coverage ---
    covered = set(s for a, s, t in assignments)
    uncovered = set(range(inst.S)) - covered

    # --- C6: Maximum working hours ---
    agent_hours = Counter(a for a, s, t in assignments)
    for a in range(inst.A):
        h = agent_hours.get(a, 0)
        if h > inst.HN[a] + inst.HE[a]:
            print(f"  [FAIL] C6: agent {a} works {h} > "
                  f"{inst.HN[a]}+{inst.HE[a]}"); ok = False

    # --- Objective breakdown (eq:sat-obj) ---
    # O1: Similarity = Σ r(a,s) · y_{a,s}  (eq:similarity)
    similarity = sum(inst.r[a][s] for a, s, t in assignments)

    # O2: Continuity penalty = Σ_q max(#agents_in_q − 1, 0)
    # (complement of stability: eq:stability)
    continuity_penalty = 0
    for seq in inst.SEQ:
        agents_in_seq = set(a for a, s, t in assignments if s in seq)
        if len(agents_in_seq) > 1:
            continuity_penalty += len(agents_in_seq) - 1

    # O3: Extra-hour cost = Σ_a |P| · max(hours_a − HN_a, 0)  (eq:cost)
    abs_P = -inst.P
    extra_hour_cost = 0
    for a in range(inst.A):
        h = agent_hours.get(a, 0)
        if h > inst.HN[a]:
            extra_hour_cost += (h - inst.HN[a]) * abs_P

    # objective = similarity + stability − cost
    #           = similarity − continuity_penalty − extra_hour_cost
    objective = similarity - continuity_penalty - extra_hour_cost

    print(f"\n  === Solution Verification ===")
    print(f"  Hard constraints:  {'OK' if ok else 'FAILED'}")
    print(f"  Assigned:          {len(assignments)}/{inst.S}")
    print(f"  Uncovered:         {len(uncovered)}")
    print(f"  ---")
    print(f"  O1 Similarity:     +{similarity}")
    print(f"  O2 Continuity:     −{continuity_penalty}")
    print(f"  O3 Extra hours:    −{extra_hour_cost}")
    print(f"  ---")
    print(f"  Objective:         {objective}")
    return ok


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='SAT-based solver for HCORAP '
                    '(Section 5 of main.tex)')
    parser.add_argument('instance', help='Path to instance file')
    parser.add_argument('--mode', default='incr',
                        choices=['incr', 'maxsat'],
                        help='incr=incremental SAT, maxsat=RC2 (default: incr)')
    parser.add_argument('--solver', default='glucose4',
                        choices=['glucose4', 'cadical153', 'minisat22'],
                        help='SAT solver backend (default: glucose4)')
    args = parser.parse_args()

    print("=" * 60)
    print(f"  HCORAP SAT-Based Solver  (mode={args.mode})")
    print("=" * 60)

    # Parse instance
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
