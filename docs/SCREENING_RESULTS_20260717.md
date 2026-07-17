# HCORAP screening results — 2026-07-17

## Scope

All timed runs used the pinned Open-WBO 2.1 commit
`80f3073e41028b219b0b0ad7c61fba28351f88e6` and the C++
`hcorap_multi` binary. The initial official screening used seed 1 from all 16
`U x A x V` groups, weighted `(wc, wo)=(1,1)`, baseline encoding
`sorting-network/none/none`, and a 10-second cumulative timeout.

The 16-run screening produced 4 `OPTIMUM`, 1 `UNSATISFIABLE`, and 11
`TIMEOUT` results. Five instances were then selected using baseline status and
runtime only:

- easy: `instance_30_10_4_1`;
- medium: `instance_30_15_5_1`;
- larger solved: `instance_40_10_4_1`;
- fast UNSAT: `instance_40_10_5_1`;
- hard timeout: `instance_30_15_4_1`.

This is a functional screening sample, not a statistically sufficient sample.
No final performance claim should be based on one seed.

## Paired cardinality ablation

The other axes were fixed at `none/none`.

| Cardinality | OPTIMUM | UNSAT | TIMEOUT | Median (s) | PAR-2 (s) |
|---|---:|---:|---:|---:|---:|
| sorting-network | 3 | 1 | 1 | 1.937 | 4.952 |
| totalizer | 3 | 1 | 1 | 1.041 | 5.801 |

Totalizer was faster on two solved instances but substantially slower on
`instance_40_10_4_1`. Sorting network remains the provisional fixed
cardinality encoding because it had the lower paired PAR-2 while preserving
the baseline configuration.

## Paired implied-constraint ablation

Cardinality was fixed to `sorting-network` and symmetry to `none`.

| Implied constraints | OPTIMUM | UNSAT | TIMEOUT | Median (s) | PAR-2 (s) |
|---|---:|---:|---:|---:|---:|
| none | 3 | 1 | 1 | 1.814 | 4.929 |
| user-slots | 2 | 1 | 2 | 3.945 | 8.917 |
| slot-capacity | 2 | 1 | 2 | 4.827 | 9.084 |
| both | 2 | 1 | 2 | 6.817 | 9.503 |
| both-plus | 3 | 1 | 1 | 0.513 | 4.441 |

`both-plus` changed `instance_30_15_4_1` from timeout to verified optimum in
0.513 seconds, but changed `instance_40_10_4_1` from optimum to timeout. It is
therefore a promising candidate, not a universal dominance result.

## Paired symmetry ablation

Cardinality was fixed to `sorting-network` and implied constraints to
`both-plus`. Every setting produced 3 `OPTIMUM`, 1 `UNSATISFIABLE`, and 1
`TIMEOUT`.

| Symmetry | Median (s) | PAR-2 (s) |
|---|---:|---:|
| none | 0.526 | 4.430 |
| slots | 0.503 | 4.458 |
| services | 0.609 | 4.458 |
| slot-service | 0.492 | 4.433 |
| all | 0.523 | 4.435 |

The differences are too small for a performance claim. Symmetry `none`
remains frozen provisionally because it had the lowest PAR-2 and preserves the
baseline model.

## Official policy recheck

The provisional candidate `sorting-network/both-plus/none` was rechecked on
`instance_30_15_4_47` with a 60-second cumulative timeout.

| Policy | Status | Elapsed (s) | SIM | CONT | OT |
|---|---|---:|---:|---:|---:|
| weighted | OPTIMUM | 0.356 | 433 | 5 | 0 |
| lex-continuity | TIMEOUT | 60.157 | — | — | — |
| lex-overtime | OPTIMUM | 11.611 | 418 | 1 | 0 |
| epsilon, delta 0.05 | TIMEOUT | 60.153 | — | — | — |

The same lex-overtime run timed out with baseline implied constraints. This is
evidence that `both-plus` can help the multi-stage policy on this instance,
but lex-continuity and epsilon still require algorithmic or incremental-solver
work before a full official campaign.

## Provisional freeze and next gate

- Always retain audit baseline: `sorting-network/none/none`.
- Candidate for the next paired pilot: `sorting-network/both-plus/none`.
- Expand the paired pilot to additional preselected seeds before freezing a
  main-table encoding.
- Do not launch the 800-instance, 50-configuration matrix yet. Run the full
  interaction matrix only on a compute-budgeted screening subset.

Raw and summary CSV files are under `experiments/results/` in the campaign
directories ending in `_20260717`.

## Expanded paired pilot: seeds 2--5

After the initial screening, seeds 2--5 were fixed in advance for the same
five instance groups. This produced 20 new instances and 40 paired weighted
runs with a 10-second timeout.

| Implied constraints | OPTIMUM | UNSAT | TIMEOUT | PAR-2 (s) |
|---|---:|---:|---:|---:|
| none | 4 | 7 | 9 | 9.218 |
| both-plus | 8 | 7 | 5 | 5.456 |

`both-plus` converted four baseline timeouts to verified optima and did not
lose any instance solved or certified UNSAT by the baseline. This expanded
sample supports retaining `both-plus` for the next policy pilot, while still
not constituting a final full-benchmark result.

## Expanded lex-overtime policy pilot

Six instances were selected from the expanded weighted strata: three
`both-plus` rescues, one solved by both configurations, and two timeouts under
both configurations. Lex-overtime was run paired with a 30-second cumulative
timeout.

| Implied constraints | OPTIMUM | TIMEOUT | Verified | PAR-2 (s) |
|---|---:|---:|---:|---:|
| none | 0 | 6 | 0 | 60.000 |
| both-plus | 5 | 1 | 5 | 14.822 |

All five `both-plus` optima had zero overtime and zero continuity penalty. The
remaining larger instance, `instance_40_10_4_2`, timed out under both
configurations.

The next-pilot encoding is now frozen as
`sorting-network/both-plus/none`, with `sorting-network/none/none` retained as
the mandatory audit baseline. This freeze applies to the next paired pilot;
it is not yet a final main-table choice.

## Lex-continuity and epsilon gate

The same six-instance subset was run for lex-continuity and epsilon at
`delta=0.05`, paired between `none` and `both-plus`, with a 30-second cumulative
timeout.

| Policy | Implied constraints | OPTIMUM | TIMEOUT | PAR-2 (s) |
|---|---|---:|---:|---:|
| lex-continuity | none | 0 | 6 | 60.000 |
| lex-continuity | both-plus | 0 | 6 | 60.000 |
| epsilon 0.05 | none | 0 | 6 | 60.000 |
| epsilon 0.05 | both-plus | 0 | 6 | 60.000 |

Although no run completed, `both-plus` made more stage progress. For
lex-continuity it completed the continuity and similarity stages on five of
six instances, compared with one of six for `none`. For epsilon it completed
the similarity-reference stage on four of six instances, compared with one of
six for `none`.

The go/no-go decision is therefore:

- continue paired pilots for weighted and lex-overtime with
  `sorting-network/both-plus/none`;
- do not expand lex-continuity or epsilon campaigns using the current
  restart-per-stage implementation and 30-second budget;
- first implement or validate incremental/native multi-objective solving, or
  run a separately budgeted timeout-scaling experiment on a very small frozen
  subset.
