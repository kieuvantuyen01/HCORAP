# Benchmark instances

Two instance collections support the manuscript experiments:

- `paperInstances/` is the Original benchmark used for the MaxSAT encoding and
  cross-solver study. The fixed experiment configurations select the reported
  48-instance subset.
- `corrected_v2_reduced_suite/evaluation_critical/evaluation/critical/` contains
  the 48 HCORAP-LC evaluation instances used for the policy study. Each `.txt`
  file has a `.txt.json` sidecar with generation metadata and a feasible
  witness used by the capacity-sweep generator.

The filenames encode the patient, caregiver, service-count, and seed factors.
SHA-256 identifiers in
[`../artifact/results/policy/corrected_pairwise_pairs.csv`](../artifact/results/policy/corrected_pairwise_pairs.csv)
bind each HCORAP-LC result to its exact input file.

The 432 capacity variants are deterministic generated data. Run
`python3 experiments/generate_load_sweep.py` to create them under
`instances/research_depth_load_sweep/`; the generated directory is ignored by
Git.
