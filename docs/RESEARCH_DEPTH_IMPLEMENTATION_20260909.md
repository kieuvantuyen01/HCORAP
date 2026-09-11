# Reproducing the HCORAP-LC robustness study

This runbook reproduces the experiments that examine the difference between
the weighted objective and the lexicographic order CONT → OT → SIM. It covers
the Weighted optimal face, continuity budgets, objective-weight sensitivity,
and a paired capacity sweep. The published outcomes and per-instance tables
are in the [`experimental supplement`](../artifact/SUPPLEMENT.md).

## Fixed design

The base evaluation set contains 48 HCORAP-LC instances: two patient counts,
four caregiver counts, two service-count settings, and three seeds. All
commercial runs use MIP-E, one Gurobi thread, a zero optimality gap, and a
300-second cumulative time limit.

| Campaign | Design | Runs |
|---|---|---:|
| Diagnostics | 48 instances × 10 policy/probe settings | 480 |
| Weights | 48 instances × 9 `(w_CONT, w_OT)` settings | 432 |
| Capacity | 432 paired variants × 3 policies | 1,296 |

The capacity grid crosses target load `{0.55, 0.85, 0.98}` with the fraction
of capacity treated as regular hours `{0.70, 0.85, 1.00}`. Each variant changes
only regular and overtime capacity and retains a verified feasible witness.

The configuration files under `experiments/configs/research_depth_*_full.json`
are the source of truth for the exact task matrices.

## GCP environment

The reported runs used a Linux x86-64 `c4-highcpu-8` VM with 8 vCPUs and
16 GB RAM. Install Python 3, a C++ compiler, Gurobi, CPLEX Optimization Studio,
and the Linux EvalMaxSAT binary. Gurobi and CPLEX require valid licenses.

Copy the environment template outside the repository and edit its local paths:

```bash
cp experiments/research_depth_gcp.env.example \
  /path/outside/repo/research_depth_gcp.env

set -a
source /path/outside/repo/research_depth_gcp.env
set +a
```

Set `HCORAP_EXPECTED_COMMIT` to the commit used for the measured campaign and
push that commit before starting. Set `HCORAP_BACKUP_DIR` to a persistent disk
outside the worktree if checkpoints are required.

## Validate the public inputs

From the repository root:

```bash
python3 experiments/verify_public_artifact.py
python3 experiments/generate_load_sweep.py
python3 experiments/prepare_research_depth_campaigns.py
```

The load generator reads the 48 versioned instance paths and hashes from the
public policy table. It creates 432 variants and a manifest under
`instances/research_depth_load_sweep/`. It refuses to overwrite an existing
sweep.

## Preflight and execute

```bash
experiments/run_research_depth_gcp.sh preflight
```

Preflight checks the machine, instances, configuration hashes, binary and
licenses; performs a clean build; runs the relevant tests; resolves every task;
and solves a short commercial and EvalMaxSAT check.

The pilot is optional when reproducing the complete reported design. To run it:

```bash
export CONFIRM_RESEARCH_DEPTH_PILOT=YES
nohup experiments/run_research_depth_gcp.sh pilot \
  > research-depth-pilot.log 2>&1 &
```

Run or resume the complete Gurobi study with:

```bash
export CONFIRM_RESEARCH_DEPTH_FULL=YES
nohup experiments/run_research_depth_gcp.sh full \
  > research-depth-full.log 2>&1 &
```

The same command resumes after an interrupted SSH session or VM restart.
Inspect progress with:

```bash
experiments/run_research_depth_gcp.sh status
tail -f research-depth-full.log
```

The convenience launcher loads the environment, selects the current commit,
and invokes the resumable full phase:

```bash
nohup experiments/launch_research_depth_full_gcp.sh \
  /path/outside/repo/research_depth_gcp.env \
  > research-depth-full.log 2>&1 &
```

## Rebuild analyses without solver calls

With the raw campaign directories present at their configured locations:

```bash
experiments/run_research_depth_gcp.sh analyze-full

python3 experiments/generate_research_depth_manuscript_results.py \
  --results-root results_research_depth_full \
  --output-dir LaTeX-Templates/paper/generated_research_depth
```

The analyzers match observations by instance hash and policy setting, recompute
the reported quality vectors from recorded assignments, and reject incomplete
or inconsistent rows. The artifact snapshot allows inspection of all reported
comparisons without installing or running a commercial solver.

## Provenance and comparison rules

Every raw record stores the source commit, configuration and binary hashes,
instance hash, solver version, termination status, elapsed time, and independent
solution-check result. Full and pilot directories remain separate. Paired
comparisons use only matching instance hashes; timeouts remain in coverage and
PAR-2 summaries. Capacity cells are descriptive because variants from a common
parent share the same underlying demand and preference data.
