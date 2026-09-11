# HCORAP exact multi-criteria optimization

This repository accompanies **Exact Multi-Criteria Optimization for Home-Care
Resource Allocation with MaxSAT**. It contains the exact MaxSAT and commercial
solver implementations, the benchmark instances used in the paper, the fixed
experiment configurations, and a compact snapshot of the reported results.

The main policy, **LEX-COS**, optimizes continuity of care first, overtime
second, and caregiver-service compatibility third. The experiments compare it
with the published weighted objective and evaluate sorting-network and
Totalizer cardinality encodings. Gurobi and CPLEX provide independent exact
comparisons through the same model and result checker.

## Repository map

| Path | Contents |
|---|---|
| [`src/`](src/) | C++ solver code and the Python `hcorap` package |
| [`instances/`](instances/) | Original benchmark and the 48-instance HCORAP-LC evaluation suite |
| [`experiments/`](experiments/) | Fixed configurations, runners, analyzers, and validation tools |
| [`artifact/`](artifact/) | Paper supplement, derived result tables, and checksums |
| [`docs/`](docs/) | Model, solver, protocol, and GCP reproduction notes |
| [`tests/`](tests/) | Unit and integration tests |

Raw solver logs and superseded experiment campaigns are intentionally excluded
from the current tree. The derived tables needed to inspect every claim in the
paper are versioned under `artifact/results/`.

## Inspect the published artifact

No solver license is needed to validate the result snapshot and its benchmark
inputs:

```bash
python3 experiments/verify_public_artifact.py
```

The command checks every published table against
[`artifact/results/manifest.json`](artifact/results/manifest.json), verifies
the declared row counts, and matches all 48 HCORAP-LC instance hashes to the
paired policy table. The full interpretation of these tables is in the
[`experimental supplement`](artifact/SUPPLEMENT.md).

## Install the Python tools

Python 3.9 or newer is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e '.[test]'
```

The Python package provides instance parsing, independent solution checks,
reference methods for small cases, and experiment utilities. The measured
MaxSAT and commercial-solver runs use the C++ executables.

## Build the C++ solvers

The open build does not require a commercial solver SDK:

```bash
make -j4 YICES=0
```

This creates the MaxSAT encoders and drivers under `bin/release/`. The measured
campaign used EvalMaxSAT as the external MaxSAT solver. See
[`docs/FAIR_EXPERIMENT_PROTOCOL.md`](docs/FAIR_EXPERIMENT_PROTOCOL.md) for the
locked execution settings.

Run the complete test suite after building:

```bash
python3 -m pytest -q
```

To include the Gurobi and CPLEX backends:

```bash
GUROBI_HOME=/path/to/gurobi \
CPLEX_STUDIO_DIR=/path/to/CPLEX_Studio \
make -j4 YICES=0 GUROBI=1 CPLEX=1 hcorap_commercial

bin/release/hcorap_commercial --list-backends
```

Both products require their own installations and valid licenses. Build and
link details are in
[`docs/COMMERCIAL_SOLVERS.md`](docs/COMMERCIAL_SOLVERS.md).

## Reproduce the experiments on GCP

Two entry points cover the experiments reported in the manuscript:

```bash
# HCORAP-LC policy diagnostics, weights, and capacity sensitivity
cp experiments/research_depth_gcp.env.example /path/outside/repo/research_depth_gcp.env
set -a
source /path/outside/repo/research_depth_gcp.env
set +a
experiments/run_research_depth_gcp.sh preflight

# Original-suite policy/encoding study and complete CPLEX comparison
experiments/run_cardinality_aligned_full_campaign.sh preflight
```

The runners print and validate the complete task matrices before execution,
resume completed tasks, use one solver thread, and record code, configuration,
binary, and instance provenance. Exact commands, environment variables, and
expected task counts are documented in
[`docs/RESEARCH_DEPTH_IMPLEMENTATION_20260909.md`](docs/RESEARCH_DEPTH_IMPLEMENTATION_20260909.md)
and [`docs/COMPACT_RESULTS_RUNBOOK.md`](docs/COMPACT_RESULTS_RUNBOOK.md).

## License and citation

The repository is distributed under the [`MIT License`](LICENSE). Citation
metadata is provided in [`CITATION.cff`](CITATION.cff).
