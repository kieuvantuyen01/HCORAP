# Paper artifact

This directory is the review-facing result package for the HCORAP manuscript.

- [`SUPPLEMENT.md`](SUPPLEMENT.md) explains the experimental design and the
  claims supported by each analysis.
- [`results/`](results/) contains the machine-readable tables cited by the
  supplement.
- [`results/manifest.json`](results/manifest.json) records the SHA-256 digest
  and expected row count of every published file.

Run the integrity check from the repository root:

```bash
python3 experiments/verify_public_artifact.py
```

The snapshot contains derived tables rather than thousands of per-run JSON
records. Each table preserves the identifiers and measurements needed for the
reported paired comparisons. The experiment runners and analyzers remain in
[`../experiments/`](../experiments/).
