#!/usr/bin/env python3
"""
HCORAP SAT Encoding — Shared module.

Provides the instance parser, variable manager, and SAT encoding
used by both the Incremental SAT solver and the MaxSAT solver.

Encoding follows the MaxSAT model described in doc/main.tex (Section 5),
based on the original paper by Unceta et al. (2025).

Variables (§5.1):
    x[(a,s,t)]  — main variable: agent a performs service s at time slot t
    y[(a,s)]    — assignment variable (eq:reif-y)
    z[(s,t)]    — service time-slot variable (eq:reif-z)
    w[(a,q)]    — sequence assignment variable (eq:reif-w)
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
"""

from collections import Counter


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
        """
        if len(lits) <= 1:
            return
        if len(lits) <= 6:
            for i in range(len(lits)):
                for j in range(i + 1, len(lits)):
                    self._add_hard([-lits[i], -lits[j]])
        else:
            from pysat.card import CardEnc, EncType
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
                for xv in x_vars:
                    self._add_hard([-xv, y_var])
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
                for xv in x_vars:
                    self._add_hard([-xv, z_var])
                self._add_hard([-z_var] + x_vars)

        # --- Reification: w_{a,q} ⟺ ∨_{s∈SEQ_q} y_{a,s}  (eq:reif-w) ---
        for a in range(inst.A):
            for q in range(len(inst.SEQ)):
                if len(inst.SEQ[q]) == 1:
                    continue
                if (a, q) not in self.w:
                    continue
                w_var = self.w[(a, q)]
                y_vars = [self.y[(a, s)] for s in inst.SEQ[q]
                          if (a, s) in self.y]
                if not y_vars:
                    self._add_hard([-w_var])
                    continue
                for yv in y_vars:
                    self._add_hard([-yv, w_var])
                self._add_hard([-w_var] + y_vars)

        # --- C1: AMO per service (eq:amo1) ---
        for s in range(inst.S):
            x_vars = [self.x[(a, s, t)] for a in range(inst.A)
                      for t in range(inst.TS) if (a, s, t) in self.x]
            self._add_amo(x_vars)

        # --- C2: AMO per agent per timeslot (eq:amo2) ---
        for a in range(inst.A):
            for t in range(inst.TS):
                x_vars = [self.x[(a, s, t)] for s in range(inst.S)
                          if (a, s, t) in self.x]
                self._add_amo(x_vars)

        # --- C4: AMO per user per timeslot (eq:amo4) ---
        for su_group in inst.SU:
            for t in range(inst.TS):
                z_vars = [self.z[(s, t)] for s in su_group
                          if (s, t) in self.z]
                self._add_amo(z_vars)

        # --- C5: Service coverage (eq:cov-hard) ---
        for s in range(inst.S):
            z_vars = [self.z[(s, t)] for t in range(inst.TS)
                      if (s, t) in self.z]
            if z_vars:
                self._add_hard(z_vars)

        # --- C6: Maximum working hours (eq:max-hours) ---
        for a in range(inst.A):
            max_hours = inst.HN[a] + inst.HE[a]
            y_vars = [self.y[(a, s)] for s in range(inst.S)
                      if (a, s) in self.y]
            if len(y_vars) > max_hours:
                from pysat.card import CardEnc, EncType
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

        Given n input literals, produces n output variables {o_0,...,o_{n-1}}:
            o_k is true  ⟺  at least (k+1) of the inputs are true

        Used for:
          - eq:w-count (ŵ_{a,i}): sorting over y_{a,s} variables
          - eq:c-count (c_{q,i}): sorting over w_{a,q} variables

        Returns: list of output variable IDs [o_0, ..., o_{n-1}]
        """
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

    # =========================================================================
    # §5.3 Soft Clauses (Objective Function)
    # =========================================================================

    def _encode_soft(self):
        """
        Encode the three objective components as weighted soft clauses.

        objective = similarity(O1) + stability(O2) − cost(O3)  (eq:sat-obj)
        """
        inst = self.inst

        # --- O1: Similarity — ⟨y_{a,s}, r(a,s)⟩  (eq:soft-reward) ---
        for a in range(inst.A):
            for s in range(inst.S):
                if inst.r[a][s] > 0 and (a, s) in self.y:
                    self._add_soft(self.y[(a, s)], inst.r[a][s])

        # --- O2: Stability — ⟨¬c_{q,i}, 1⟩  (eq:soft-cont) ---
        for q in range(len(inst.SEQ)):
            w_vars = [self.w[(a, q)] for a in range(inst.A)
                      if (a, q) in self.w]
            if not w_vars:
                continue
            c_outputs = self._build_sorting_network(w_vars)
            p = min(len(inst.SEQ[q]), len(c_outputs))
            for i in range(p):
                self._add_soft(-c_outputs[i], 1)

        # --- O3: Extra-hour cost — ⟨¬ŵ_{a,i}, |P|⟩  (eq:soft-extra) ---
        abs_P = -inst.P
        for a in range(inst.A):
            y_vars = [self.y[(a, s)] for s in range(inst.S)
                      if (a, s) in self.y]
            if len(y_vars) <= inst.HN[a]:
                continue
            w_hat_outputs = self._build_sorting_network(y_vars)
            upper = min(inst.HN[a] + inst.HE[a], len(w_hat_outputs))
            for k in range(inst.HN[a], upper):
                self._add_soft(-w_hat_outputs[k], abs_P)


# =============================================================================
# Solution Verification
# =============================================================================

def verify_solution(inst: HCORAPInstance, enc: HCORAPEncoding, model):
    """
    Verify hard constraints and compute objective breakdown.

    Checks: C1–C6.
    Objective = similarity − continuity_penalty − extra_hour_cost  (eq:sat-obj)
    """
    model_set = set(model)
    ok = True

    assignments = [(a, s, t) for (a, s, t), v in enc.x.items()
                   if v in model_set]

    # C1: At most one assignment per service
    svc_cnt = Counter(s for a, s, t in assignments)
    for s, c in svc_cnt.items():
        if c > 1:
            print(f"  [FAIL] C1: service {s} assigned {c} times"); ok = False

    # C2: At most one service per agent per timeslot
    at_cnt = Counter((a, t) for a, s, t in assignments)
    for (a, t), c in at_cnt.items():
        if c > 1:
            print(f"  [FAIL] C2: agent {a}, slot {t} = {c}"); ok = False

    # C3: Availability / suitability / qualification
    for a, s, t in assignments:
        if inst.r[a][s] == 0:
            print(f"  [FAIL] C3: r({a},{s})=0"); ok = False
        if not inst.TSA[a][t]:
            print(f"  [FAIL] C3: TSA({a},{t})=0"); ok = False
        if not inst.TSS[s][t]:
            print(f"  [FAIL] C3: TSS({s},{t})=0"); ok = False

    # C4: At most one service per user per timeslot
    for su_group in inst.SU:
        su_set = set(su_group)
        ts_cnt = Counter(t for a, s, t in assignments if s in su_set)
        for t, c in ts_cnt.items():
            if c > 1:
                print(f"  [FAIL] C4: user-group, slot {t} = {c}"); ok = False

    # C5: Coverage
    covered = set(s for a, s, t in assignments)
    uncovered = set(range(inst.S)) - covered

    # C6: Maximum working hours
    agent_hours = Counter(a for a, s, t in assignments)
    for a in range(inst.A):
        h = agent_hours.get(a, 0)
        if h > inst.HN[a] + inst.HE[a]:
            print(f"  [FAIL] C6: agent {a} works {h} > "
                  f"{inst.HN[a]}+{inst.HE[a]}"); ok = False

    # Objective breakdown (eq:sat-obj)
    similarity = sum(inst.r[a][s] for a, s, t in assignments)

    continuity_penalty = 0
    for seq in inst.SEQ:
        agents_in_seq = set(a for a, s, t in assignments if s in seq)
        if len(agents_in_seq) > 1:
            continuity_penalty += len(agents_in_seq) - 1

    abs_P = -inst.P
    extra_hour_cost = 0
    for a in range(inst.A):
        h = agent_hours.get(a, 0)
        if h > inst.HN[a]:
            extra_hour_cost += (h - inst.HN[a]) * abs_P

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
