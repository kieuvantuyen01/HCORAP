# Legacy experiment results

The CSV and JSON files in this directory are historical inputs to the audit in
`docs/EXPERIMENT_GAP_AUDIT_20260808.md`. They are retained so that the reported
missing-row, parser-error, and objective-agreement counts can be reproduced.

These files are **not** the ICIIT 2027 publication dataset:

- the eight weighted CSV files do not have complete raw JSON, solver logs,
  environment snapshots, or manifests;
- 54 weighted run records are missing across the eight configurations;
- the historical commercial data contains parser errors fixed after collection;
- `results_per_instance.csv` preserves the original `PARSE_ERROR` rows, whose
  trailing empty placeholders exceed the header width; the audit uses only the
  instance and status fields for those rows;
- `summary_by_config.csv` has a known cumulative-count issue.

Do not use these runtime values in the manuscript's primary tables. The new GCP
campaign writes validated data under `experiments/results/`, which is ignored
during execution and included only in the checksummed reproducibility artifact.
