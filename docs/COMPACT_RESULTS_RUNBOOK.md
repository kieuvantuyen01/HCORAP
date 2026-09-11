# Original-suite encoding and exact-solver campaign

This runbook reproduces the manuscript comparison on the Original benchmark.
It evaluates the weighted and LEX-COS policies with two cardinality encodings,
then compares the decided MaxSAT results with Gurobi and CPLEX MIP-E.

## Fixed matrices

| Component | Matrix | Runs |
|---|---|---:|
| EvalMaxSAT | 48 instances × 2 policies × 2 encodings | 192 |
| Gurobi reference | 48 instances × 2 policies | 96 |
| CPLEX baseline | 48 instances × 2 policies | 96 |

All measurements use a 3,600-second cumulative limit, one worker, and one
solver thread. Implied constraints and symmetry breaking are disabled in the
main comparison. The configurations are:

- `experiments/configs/gcp_original_policy_encoding_3600.json`
- `experiments/configs/gcp_original_policy_reference_3600.json`
- `experiments/configs/gcp_original_cplex_reference_3600.json`

## Required software

Use Linux x86-64 with Python 3, a C++ compiler, EvalMaxSAT, and CPLEX
Optimization Studio. Gurobi is needed only if a complete validated 96-run
reference campaign is unavailable. Commercial solvers require valid licenses.

Set the executable and SDK paths, then freeze the measured revision:

```bash
export EVALMAXSAT_BIN=/opt/evalmaxsat/EvalMaxSAT_bin
export CPLEX_STUDIO_DIR=/opt/ibm/ILOG/CPLEX_Studio2211
export GUROBI_HOME=/opt/gurobi/linux64
export HCORAP_EXPECTED_COMMIT=$(git rev-parse HEAD)
export HCORAP_BACKUP_DIR=/mnt/disks/hcorap-backup
```

When reusing the complete Gurobi reference, point to it explicitly:

```bash
export HCORAP_GUROBI_RESULTS=/absolute/path/to/gurobi-reference-results
```

The reuse validator requires the complete 96-row protocol, exact instance and
configuration hashes, validated objective vectors, and compatible MIP-E source
files.

## Preflight

```bash
./experiments/run_cardinality_aligned_full_campaign.sh preflight
```

Preflight builds the relevant binaries, checks the solver installations and
licenses, validates the 48-instance selection, resolves both fixed matrices,
and runs the associated tests. It does not start the measured campaign.

## Run or resume

```bash
export CONFIRM_COMPACT_POLICY_ENCODING=YES
export CONFIRM_FULL_CPLEX_BASELINE=YES

nohup ./experiments/run_cardinality_aligned_full_campaign.sh all \
  > cardinality-aligned-full.log 2>&1 &
```

The runner keeps valid completed run identifiers and executes only missing or
invalid tasks. Individual phases are also available:

```bash
./experiments/run_cardinality_aligned_full_campaign.sh policy-encoding
./experiments/run_cardinality_aligned_full_campaign.sh cplex
```

The policy/encoding analyzer uses Gurobi objective vectors as exact references
and retains unfinished MaxSAT cases in coverage and PAR-2 calculations. The
commercial analyzer compares statuses and quality vectors across EvalMaxSAT,
Gurobi, and CPLEX using jointly completed cases for runtime ratios.

## Generate manuscript outputs

After the three validated analysis directories are on one machine:

```bash
export HCORAP_POLICY_ANALYSIS=/path/to/corrected-policy-analysis
export HCORAP_ENCODING_ANALYSIS=/path/to/policy-encoding-analysis
export HCORAP_COMMERCIAL_ANALYSIS=/path/to/commercial-analysis
export HCORAP_MANUSCRIPT_RESULTS=LaTeX-Templates/paper/generated_compact

./experiments/run_compact_policy_encoding.sh manuscript
```

This creates the LaTeX tables, figures, macros, and their provenance record.
The checksummed, review-facing tables are already available under
[`../artifact/results/`](../artifact/results/).

## Reported pairing rule

Runtime reduction is calculated for each jointly proved instance as
`100 × (t_SN - t_TOT) / t_SN`, followed by the median across pairs. The 95%
interval uses 10,000 percentile-bootstrap resamples with seed `20260908`.
Solver ratios likewise use matching completed instance-policy pairs. These
definitions are implemented in the analyzers and preserved in the published
tables rather than recomputed in the manuscript.
