# HCORAP experiment pipeline

The publication pipeline separates execution, collection, analysis, evidence
gates, and artifact generation. Do not add manuscript numbers directly to TeX.

## Sources of truth

- `publication_contract.py`: independently reviewable scientific contract
  (campaign identities, totals, reference configuration, locked gates).
- `configs/reduced_campaign_manifest.json`: machine-readable execution matrix.
- `validate_publication_campaign.py`: checks that every JSON config and gate
  still matches the contract.
- `results/<campaign>/resolved_campaign.json`: immutable resolved task matrix
  captured at execution time.

The Python contract and JSON manifest intentionally duplicate the reviewed
matrix: their agreement is a drift detector. Other scripts must derive counts
and result-directory names from the manifest rather than add new hard-coded
totals.

## Pipeline layers

| Layer | Entry points | Responsibility |
|---|---|---|
| Run | `run_reproducible_campaign.py`, `run_commercial_campaign.py` | Resolve deterministic tasks, execute/resume, capture raw evidence |
| Collect | `collect_reproducible_campaign.py`, `collect_commercial_campaign.py` | Flatten relocatable raw artifacts and validate completeness |
| Gate | `evaluate_*.py`, `audit_publication_evidence.py` | Enforce predeclared correctness, coverage, agreement, and provenance rules |
| Analyze | `analyze_*campaign*.py`, `analyze_*evidence.py` | Produce paired summaries only from validator-approved rows |
| Publish | `generate_manuscript_results.py`, `freeze_manuscript_bundle.py`, `package_experiment_artifacts.sh` | Generate, freeze, and checksum manuscript evidence |

## Supported GCP entry points

- Full clean-room campaign: `run_all_remaining_publication.sh`.
- Only the corrected-v2 exact-policy supplement after C1--C5 already exist:
  `run_remaining_corrected_evidence.sh`.
- Conditional Totalizer-only transfer test for corrected-v2 LEX-COS:
  `run_corrected_lex_encoding_transfer.sh`.
- Individual resumable phases: `gcp_prepare_and_run.sh PHASE`.

Always run the contract check before starting a solver:

```bash
python3 experiments/validate_campaign_manifest.py
python3 experiments/validate_publication_campaign.py
```

## Adding a campaign

1. Add one JSON config with a new result directory and deterministic order seed.
2. Add the campaign to the manifest and to `publication_contract.py`.
3. Extend `validate_publication_campaign.py` with scientific invariants, not
   only a run-count assertion.
4. Add a calibration/evidence gate before exposing results to the manuscript.
5. Add task-matrix, failure, and relocation tests.
6. Update the GCP runbook and artifact include list.

Never reuse a result directory after changing the binary, config, timeout, or
instance sample. Never treat timeout as infeasible or compute objective deltas
from pairs that are not jointly verified optimum.
