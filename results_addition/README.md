# Additional experimental results

This directory is an immutable import of historical and development experiment
outputs collected in July--August 2026.  Its contents are **not** automatically
part of the ICIIT 2027 publication dataset.

Do not rename, move, deduplicate, or overwrite the campaign directories.  The
non-destructive audit and normalized catalog are generated under `organized/`:

```bash
python3 experiments/audit_results_addition.py --root results_addition
```

Read `organized/reports/RESULTS_ADDITION_AUDIT_20260819.md` before reusing any
result.  In particular, the EvalMaxSAT campaigns, incomplete commercial runs,
smoke tests, and aggregate epsilon/Pareto files must not be merged into the
publication campaign without satisfying the frozen protocol in
`submission_plan.md`.
