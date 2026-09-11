# Experimental supplement: exact multi-criteria HCORAP

This supplement accompanies *Exact Multi-Criteria Optimization for Home-Care
Resource Allocation with MaxSAT*. It contains objective derivations,
implementation details, supplementary results, and reproduction instructions.
The manuscript presents the allocation model, the CONT → OT → SIM policy,
the experimental design, and the main findings.

The [analysis snapshot](artifacts/20260911/) contains machine-readable tables
for the reported results. Its [manifest](artifacts/20260911/manifest.json)
records source paths, source and exported SHA-256 hashes, row counts, and
transformations. Absolute VM paths have been made repository-relative;
numerical cells are preserved. These are derived analysis tables. Raw solver
logs, allocation JSON files, and instance archives remain separate from this
snapshot and are required for a fresh solver run or independent schedule check.

## 1. Objective equivalence and exact priorities

### Stability reward and continuity penalty

For each nonempty continuity group $q$, let $n_q$ be the number of caregivers
assigned to its repeated services. Full coverage implies $n_q\geq1$, so

$$
\mathrm{CONT}=\sum_q(n_q-1),\qquad
\mathrm{STAB}=\sum_q(|\mathcal S_q|-n_q).
$$

Consequently,

$$
\mathrm{STAB}+\mathrm{CONT}
=\sum_q(|\mathcal S_q|-1).
$$

The right-hand side is an instance constant. Maximizing the original stability
reward is therefore equivalent to minimizing CONT on the full-coverage
feasible set. The published weighted objective can be written, up to a constant,
as $\mathrm{SIM}-w_c\mathrm{CONT}-w_op\mathrm{OT}$, where $p$ is the magnitude
of the instance's overtime penalty. The baseline uses $(w_c,w_o)=(1,1)$.
Workload and overtime are measured in service units.

### Sequential optimization

Let $F_0$ be the feasible allocations and
$(f_1,f_2,f_3)=(\mathrm{CONT},\mathrm{OT},-\mathrm{SIM})$. For each stage,

$$
z_i=\min_{x\in F_{i-1}}f_i(x),\qquad
F_i=\{x\in F_{i-1}:f_i(x)=z_i\}.
$$

Inductively, $F_i$ contains exactly the allocations attaining the first $i$
optimal values. Every allocation in $F_3$ is therefore lexicographically
optimal. A stage proceeds only after its predecessor proves optimality; the
three stages share one cumulative time limit. A feasible incumbent from an
unfinished stage does not establish the complete priority optimum. LEX-OT
reverses the first two criteria and uses the same procedure.

### Weighted optimal face and continuity budgets

The `weighted-face` mode first proves the optimal Weighted score $W^*$, fixes
that score, and minimizes or maximizes one requested criterion. Four probes
per instance find the CONT and OT intervals. These are separate extrema:
the minimum CONT and minimum OT on that face need not occur in one allocation.
If its minimum CONT exceeds the LEX-COS value, every Weighted optimum has
worse continuity. The corresponding OT comparison has the same interpretation.

The `continuity-budget` mode first proves the minimum continuity penalty
$C^*$, adds $\mathrm{CONT}\leq C^*+k$, then minimizes OT and maximizes SIM.
The budget counts aggregate penalty units, not patients. At $k=0$, the optimum
vector equals LEX-COS. Optimal OT cannot increase as $k$ increases. SIM can
decrease if the additional freedom permits lower OT, because OT still precedes
SIM; when optimal OT is unchanged, optimal SIM cannot decrease.

Implementation options and driver commands are documented in the
[research-depth implementation guide](RESEARCH_DEPTH_IMPLEMENTATION_20260909.md).

## 2. Boolean representation and auxiliary experiments

Sorting networks pass Boolean indicators through comparator circuits; output
$k$ indicates whether at least $k$ inputs are true. A Totalizer recursively
merges unary counts and exposes the same threshold information at the root.
The selected encoding represents workload counts and unit-coefficient bounds
between optimization stages. Bounds on weighted compatibility use the
pseudo-Boolean representation described in the
[encoding runbook](COMPACT_RESULTS_RUNBOOK.md).

The main comparison disables implied constraints and symmetry breaking in both
encodings. Implied constraints add consequences of the existing allocation
model. Symmetry breaking removes equivalent choices of interchangeable slots
or services. The earlier 300-second configuration study crossed encoding with
these two options. Implied constraints increased PAR-2 in all four conditional
comparisons and added approximately 4,041–9,657 median variables. Symmetry
breaking gave no consistent runtime improvement.

The [eight-configuration summary](artifacts/20260911/preliminary/factorial_summary.csv)
and [conditional comparisons](artifacts/20260911/preliminary/factorial_contrasts.csv)
document this preliminary study. Its 300-second results support configuration
selection; the manuscript's runtime estimates use the later 3,600-second study.

### Formula size and memory on the Original suite

| Policy | Encoding | Variables | Hard clauses | Peak memory (MB) |
|---|---|---:|---:|---:|
| Weighted | Sorting network | 32,164 | 129,734 | 88.0 |
| Weighted | Totalizer | 12,042 | 143,287 | 86.3 |
| LEX-COS | Sorting network | 168,408 | 531,536 | 201.1 |
| LEX-COS | Totalizer | 31,288 | 3,344,787 | 738.7 |

Entries are medians across all 48 runs per configuration. Formula size uses
the maximum count across stages within each run. Hard clauses represent
mandatory constraints; soft-clause counts are unchanged between encodings
within each policy. Peak RSS measures physical memory use. The
[source table](artifacts/20260911/encoding/policy_encoding_summary.csv) retains
all counts and runtime summaries.

### Runtime variation by instance size

The [size breakdown](artifacts/20260911/encoding/runtime_by_size.csv) reports
the median per-instance percentage runtime reduction and the number of jointly
proved pairs for each patient, caregiver, and service-count stratum. Under
Weighted, median reductions range from 9.8% to 16.2% across these strata.
Under LEX-COS, they range from −2.4% to +2.7%. These are descriptive summaries;
strata for different size factors overlap.

## 3. Additional policy results

### Returned optima and the complete Weighted optimal face

The main 48-pair comparison improves CONT in 43 instances and OT in 47, with
both improving in 42. LEX-COS reaches zero OT in 45 instances and zero CONT in
31. The median compatibility reduction is 36 points, or 6.3% when each pair
is normalized by its Weighted SIM score before taking the median.

| Patient count | Instances | Both CONT and OT improve | Median CONT reduction | Median OT reduction |
|---|---:|---:|---:|---:|
| 30 | 24 | 18 | 4 | 10 |
| 40 | 24 | 24 | 8 | 13 |

See the [paired policy results](artifacts/20260911/policy/corrected_pairwise_pairs.csv)
and [summary](artifacts/20260911/policy/corrected_pairwise_summary.csv).
The returned Weighted optima use overtime in 47 of 48 HCORAP-LC instances.
On the Original suite, 38 of the 42 feasible returned Weighted solutions use
zero overtime.

The full-face probes establish worse CONT for every Weighted optimum in 41
instances, worse OT in 43, and both in 36. All 24 instances with 40 patients
have both unavoidable losses. With 30 patients, the counts are 17 for CONT,
19 for OT, and 12 for both. Weighted optima span multiple CONT values in 46
instances and multiple OT values in all 48; the median interval widths are
3 CONT units and 6.5 OT units. These statements concern the published weight
setting. The [per-instance intervals](artifacts/20260911/diagnostics/weighted_face_intervals.csv)
include all four probe statuses and the LEX-COS reference values.

### Continuity budgets

Moving from $k=0$ to $k=1$ reduces OT in three instances by one service unit
each. SIM rises in 46 instances, is unchanged in one, and falls in one where
OT improves. The median SIM change is +8 points. Moving from $k=1$ to $k=2$
leaves optimal OT unchanged everywhere and raises SIM in all 48 instances,
with a median gain of 4 points. The
[budget curves](artifacts/20260911/diagnostics/continuity_budget_curves.csv)
and [adjacent changes](artifacts/20260911/diagnostics/continuity_budget_changes.csv)
give the complete results.

### Weight and capacity sensitivity

The nine $(w_c,w_o)\in\{1,4,8\}^2$ settings yield between four and nine distinct
returned quality vectors per instance. The SIM coefficient remains one.
The settings $(1,1)$, $(4,4)$, and $(8,8)$ return the LEX-COS vector in 0, 8,
and 25 of 48 instances. These are matches of returned optima; the full Weighted
optimal face was examined at the published setting. Detailed
[weight results](artifacts/20260911/weights/weight_optimum_runs.csv) and
[instance summaries](artifacts/20260911/weights/weight_instance_stability.csv)
support these comparisons.

| Target load | Regular-capacity fraction | Weighted differs from COS (/48) | LEX-OT differs from COS (/48) |
|---:|---:|---:|---:|
| 0.55 | 0.70 | 46 | 0 |
| 0.55 | 0.85 | 40 | 0 |
| 0.55 | 1.00 | 40 | 0 |
| 0.85 | 0.70 | 48 | 0 |
| 0.85 | 0.85 | 48 | 3 |
| 0.85 | 1.00 | 44 | 0 |
| 0.98 | 0.70 | 46 | 0 |
| 0.98 | 0.85 | 48 | 0 |
| 0.98 | 1.00 | 45 | 0 |

The capacity sweep changes regular and overtime allowances while preserving
the remaining instance data and a feasible witness. Its 432 variants share
six parent patient-seed families, and comparisons are descriptive. The
[cell summaries](artifacts/20260911/load/load_cell_summary.csv) and
[matched pairs](artifacts/20260911/load/load_policy_pairs.csv) also record
realized load, parent hashes, and changes in allocation structure. The nine
tested settings locate observed order conflicts; they do not determine a
continuous transition boundary or an infeasibility threshold.

## 4. Execution, statistical summaries, and validation

Measured campaigns use a Google Cloud `c4-highcpu-8` VM, 8 vCPUs, 16 GB RAM,
and one solver thread. The paired encoding runs use the same binary and
source revision, with one worker pinned to the allocated processor core.
The four configurations are randomized within each instance using order
seed `20270906`. Original instances use seeds 1–3; HCORAP-LC uses 1001–1003.
Gurobi is version 11.0.3 and CPLEX is version 22.1.2 in the reported audits.

Study A uses a 300-second limit. Study B uses 3,600 seconds for each complete
policy run. Its 192 EvalMaxSAT runs cover 48 instances × two policies × two
encodings. Each commercial solver covers 96 instance-policy combinations.
Optimal and infeasible statuses both count as proved. Runtime medians include
proved runs. PAR-2 includes all runs and assigns $2T$ to each unfinished run,
so a Study B timeout contributes 7,200 seconds.

Paired runtime reductions use $100(t_{SN}-t_{TOT})/t_{SN}$ for each jointly
proved pair. The manuscript uses the median of these percentages, with a
percentile 95% interval from 10,000 bootstrap resamples of complete pairs
(seed `20260908`). There are 46 Weighted pairs and 48 LEX-COS pairs. This
statistic differs from a percentage computed from separately aggregated
runtimes. The [figure statistics](artifacts/20260911/encoding/compact_figure_statistics.json)
retain every percentage and the unrounded confidence limits. The
[paired runtimes](artifacts/20260911/encoding/policy_encoding_pairs.csv) retain
the two unfinished Weighted pairs as well.

Gurobi and CPLEX agree on all 96 Original instance-policy statuses and all 84
optimal quality vectors. EvalMaxSAT agrees with both on its 94 decided cases,
including 82 optima; its two remaining Weighted runs time out. Median paired
EvalMaxSAT-to-Gurobi runtime ratios are 331 for Weighted and 426 for LEX-COS;
the corresponding CPLEX ratios are 204 and 217. These ratios use jointly
completed runs. See the [cross-solver pairs](artifacts/20260911/commercial/cross_solver_pairs.csv).

On HCORAP-LC, the CPLEX audit agrees with Gurobi on all 32 Weighted/LEX-COS
runs and all 16 LEX-OT runs. An independent allocation checker verifies
coverage, assignment feasibility, and the three recomputed quality measures.
Run records retain configuration, solver version, source revision, instance
hashes, and checksums. The full robustness campaign contains 480 policy
diagnostic runs, 432 weight runs, and 1,296 capacity-policy runs; all 2,208
prove optimality and pass the recorded validation checks.

The full campaign's source commit is
[`8b35c809`](https://github.com/kieuvantuyen01/HCORAP/tree/8b35c809daf37073c458307deab96ac2b55dc433).
The measured driver SHA-256 is
`7234074473777532ffd91bb1046434a1feda8f36d1eea1b8ee306a0adbdb6fc7`.
Overlapping pilot and full cells agree; the full campaign supplies the
48-instance robustness evidence.

## 5. Reproduction

The [GCP run instructions](RESEARCH_DEPTH_IMPLEMENTATION_20260909.md) cover
dependencies, licenses, input archives, configuration, and resuming a campaign.
The [compact-results runbook](COMPACT_RESULTS_RUNBOOK.md) covers the original
policy, encoding, and commercial comparisons. The
[full-results review](RESEARCH_DEPTH_FULL_REVIEW_20260911.md) records the audit
of the delivered full campaign.

With the raw full campaign and its instances restored at their repository
paths, the following commands regenerate its analyses without invoking solvers:

```bash
python3 experiments/analyze_policy_diagnostics.py \
  results_research_depth_full/research_depth_diagnostics_full \
  results_research_depth_full/research_depth_diagnostics_full/analysis

python3 experiments/analyze_weight_sensitivity.py \
  --results results_research_depth_full/research_depth_weights_full \
  --output-dir results_research_depth_full/research_depth_weights_full/analysis

python3 experiments/analyze_load_sweep.py \
  results_research_depth_full/research_depth_load_full \
  instances/research_depth_load_sweep/load_sweep_manifest.json \
  results_research_depth_full/research_depth_load_full/analysis

python3 experiments/generate_research_depth_manuscript_results.py \
  --results-root results_research_depth_full \
  --output-dir LaTeX-Templates/paper/generated_research_depth
```

The snapshot supports inspection of the reported numbers without running
commercial solvers. Its manifest distinguishes copied analysis data from the
size breakdown computed from the supplied paired runtimes.
