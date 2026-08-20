# Frozen manuscript outputs

This directory is the interface between the validated analysis pipeline and the
paper. A submission build requires all three fragments below **and** a validated
`freeze-manifest.tex` marker:

- `abstract-findings.tex`: two or three quantitative sentences, with every
  value traceable to a generated table;
- `results.tex`: the complete `Results` section, including generated tables or
  figures and self-contained captions;
- `conclusion.tex`: the complete `Conclusion` section, containing only claims
  supported in `results.tex`.

The fragments currently present are exploratory layout drafts and are
intentionally rejected because no marker exists. File existence is not a data
freeze.

After all selected analysis validators pass, generate all three fragments and
their hash provenance from the repository root:

```sh
python3 experiments/generate_manuscript_results.py
```

Review the generated diff and PDF, then commit the three fragments plus
`manuscript-provenance.json`. From that clean publication commit, issue the
marker:

```sh
HCORAP_EXPECTED_COMMIT=<full-publication-commit> \
  python3 experiments/freeze_manuscript_bundle.py
```

The generator refuses invalid primary, enabled corrected-v2, or cross-paradigm
analyses and removes no branch silently. The freeze command additionally
refuses changed source/fragment hashes, a scope inconsistent with the screening
branches, a non-`GO` screening decision, dirty source, a commit mismatch,
missing fragments, or draft tokens. On failure it removes any stale marker.
`main.tex` ignores an unmarked bundle, while `submission.tex` raises a fatal
error.

Do not copy values from legacy CSV files or edit numbers in LaTeX manually.
Generated visuals must state the sample, timeout, aggregation rule, eligibility
rule, and better/worse direction in their captions.
