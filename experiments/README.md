# Experiment pipeline

The repository exposes two current campaign entry points. Both resolve a fixed
task matrix, validate inputs and solver availability, run one worker with one
solver thread, resume completed tasks, and analyze only validated records.

## HCORAP-LC policy and sensitivity study

```bash
./experiments/run_research_depth_gcp.sh preflight
```

The full campaign contains:

| Component | Inputs and settings | Runs |
|---|---|---:|
| Policy diagnostics | 48 instances × 10 policy/probe settings | 480 |
| Objective weights | 48 instances × 9 weight pairs | 432 |
| Capacity sweep | 432 generated variants × 3 policies | 1,296 |

The capacity variants are regenerated from the versioned 48-instance suite
when absent. The complete GCP setup and phase commands are in
[`../docs/RESEARCH_DEPTH_IMPLEMENTATION_20260909.md`](../docs/RESEARCH_DEPTH_IMPLEMENTATION_20260909.md).

## Original-suite encoding and solver study

```bash
./experiments/run_cardinality_aligned_full_campaign.sh preflight
```

This combines 192 EvalMaxSAT policy/encoding runs with 96 Gurobi and 96 CPLEX
policy runs on the fixed 48-instance Original subset. The Gurobi campaign can
be reused after its provenance and complete 96-row protocol pass validation.
See [`../docs/COMPACT_RESULTS_RUNBOOK.md`](../docs/COMPACT_RESULTS_RUNBOOK.md).

## Pipeline components

| Stage | Main programs |
|---|---|
| Configure | `configs/*.json`, `publication_contract.py` |
| Execute | `run_reproducible_campaign.py`, `run_commercial_campaign.py` |
| Collect | `collect_reproducible_campaign.py`, `collect_commercial_campaign.py` |
| Validate | `validate_*.py`, `evaluate_*.py`, `audit_publication_evidence.py` |
| Analyze | `analyze_policy_diagnostics.py`, `analyze_weight_sensitivity.py`, `analyze_load_sweep.py`, `analyze_policy_encoding_matrix.py` |
| Publish | `generate_*manuscript_results.py`, `package_experiment_artifacts.sh` |

Result directories are ignored by Git. The compact, checksummed tables cited
by the paper are versioned under [`../artifact/results/`](../artifact/results/).
Validate that package with:

```bash
python3 experiments/verify_public_artifact.py
```

Scripts for earlier screening and calibration studies remain available for
provenance, while the two entry points above define the manuscript campaigns.
