# ICIIT 2027 manuscript source

Place the HCORAP conference-paper LaTeX sources, bibliography, figures, and
generated tables in this directory. The downloaded ACM template bundle in the
parent directory is intentionally ignored; it remains available locally and
can be restored from the conference template package.

`main.tex` is a compileable, visually clean pre-results manuscript. The
Introduction, Related Work, model, methods, experimental design, and threats to
validity are written; quantitative abstract findings, Results, and Conclusion
still depend on frozen evidence. The clean appearance does **not** make a
pre-results PDF submission-ready.

The source deliberately has three build modes:

- `main.tex` is the clean manuscript preview. Editorial notes and pending-result
  floats are absent rather than disguised as evidence.
- `review.tex` is the internal collaborator build. It exposes the intended
  Results structure and clearly marked reservations for generated visuals.
- `submission.tex` is the release build. It fails immediately unless the full
  frozen-results bundle has a validator-issued marker; this is the only PDF
  eligible for submission.

Do not edit numerical findings into either source by hand. After validation and
data freeze, `experiments/generate_manuscript_results.py` must create the
complete bundle documented in [`generated/README.md`](generated/README.md).
`main.tex` loads the bundle only
when all required files and a valid `freeze-manifest.tex` exist, while
`submission.tex` turns an unvalidated, missing, or partial bundle into a fatal
build error.

Build from this directory while resolving the local ACM class and bibliography
style from the parent template directory:

```sh
# Clean pre-results preview
TEXINPUTS=..: latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

# Internal evidence-layout review
TEXINPUTS=..: latexmk -pdf -interaction=nonstopmode -halt-on-error review.tex

# Final release build; expected to fail before validated data freeze
TEXINPUTS=..: latexmk -pdf -interaction=nonstopmode -halt-on-error submission.tex
```

Do not create `generated/freeze-manifest.tex` manually. It is issued only by
`experiments/freeze_manuscript_bundle.py` from a clean publication commit after
the primary, every enabled corrected-v2 branch, and cross-paradigm validation
reports pass with scopes matching the recorded screening decision and the
generator provenance still matches every source and fragment hash.

The target is five double-column Letter pages including references. Before
frozen results, the clean build should remain comfortably below five pages;
the unused space is a page budget for the generated Results and Conclusion, not
an invitation to expand background or protocol text.

Do not commit auxiliary build files or the submission PDF here. Only
`submission.pdf`, produced after the frozen-results guard passes, and the
reproducibility archive should be attached to a release.
