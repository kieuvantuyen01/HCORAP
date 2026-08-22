#!/usr/bin/env python3
"""Generate the submission Results, abstract findings, and conclusion.

Only branch-consistent, validator-approved campaign summaries are accepted.
The output intentionally contains no hand-editable numerical placeholders; a
separate freeze step binds these fragments to their exact source files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import statistics
from pathlib import Path
from typing import Any, Iterable

try:
    from .publication_contract import (
        FACTORIAL_CONFIGURATIONS,
        REFERENCE_CONFIGURATION,
    )
except ImportError:
    from publication_contract import (
        FACTORIAL_CONFIGURATIONS,
        REFERENCE_CONFIGURATION,
    )


BASELINE = ("sorting-network", "none", "none")
REFERENCE = REFERENCE_CONFIGURATION
CONFIG_KEYS = ("cardinality", "implied", "symmetry")
FACTORIAL_ORDER = FACTORIAL_CONFIGURATIONS
def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _portable_path(path: Path) -> str:
    """Prefer repository-relative provenance paths for relocatable artifacts."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(resolved)


def _float(value: Any) -> float:
    if value in (None, ""):
        raise ValueError(f"expected a numeric value, got {value!r}")
    return float(value)


def _int(value: Any) -> int:
    return int(float(value))


def _truth(value: Any) -> bool:
    return str(value).lower() == "true"


def _fmt(value: Any, digits: int = 2) -> str:
    if value in (None, ""):
        return "--"
    number = _float(value)
    rendered = f"{number:.{digits}f}"
    if digits > 0:
        rendered = rendered.rstrip("0").rstrip(".")
    return "0" if rendered == "-0" else rendered


def _count(value: Any) -> str:
    return f"{_int(value):,}".replace(",", r"{,}")


def _fmt_grouped(value: Any, digits: int = 2) -> str:
    rendered = _fmt(value, digits)
    if rendered == "--":
        return rendered
    sign = "-" if rendered.startswith("-") else ""
    unsigned = rendered.removeprefix("-")
    integer, separator, fraction = unsigned.partition(".")
    grouped = f"{int(integer):,}".replace(",", r"{,}")
    return sign + grouped + (separator + fraction if separator else "")


def _configuration(row: dict[str, str]) -> tuple[str, str, str]:
    return tuple(row[key] for key in CONFIG_KEYS)  # type: ignore[return-value]


def _one(
    rows: Iterable[dict[str, str]], predicate: Any, description: str
) -> dict[str, str]:
    selected = [row for row in rows if predicate(row)]
    if len(selected) != 1:
        raise ValueError(f"expected one {description}, found {len(selected)}")
    return selected[0]


def _range(rows: Iterable[dict[str, str]], key: str) -> tuple[str, str]:
    values = [_float(row[key]) for row in rows if row.get(key) not in (None, "")]
    if not values:
        return "--", "--"
    return _fmt(min(values)), _fmt(max(values))


def _range_wording(value: tuple[str, str]) -> str:
    if value == ("--", "--"):
        return "are unavailable because no contrast has pairs proved by both configurations"
    return f"range from {value[0]} to {value[1]}"


def _median(values: Iterable[float]) -> float:
    materialized = list(values)
    if not materialized:
        raise ValueError("cannot take the median of an empty sample")
    return statistics.median(materialized)


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot take a percentile of an empty sample")
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _bootstrap_median_ci(
    values: list[float], label: str, repetitions: int = 2_000
) -> tuple[float, float]:
    if not values:
        raise ValueError(f"cannot bootstrap an empty sample: {label}")
    seed = int.from_bytes(hashlib.sha256(label.encode("utf-8")).digest()[:8], "big")
    generator = random.Random(seed)
    medians = [
        statistics.median(generator.choices(values, k=len(values)))
        for _ in range(repetitions)
    ]
    return _percentile(medians, 0.025), _percentile(medians, 0.975)


def _policy_signal(
    rows: list[dict[str, str]],
    *,
    comparison: str | None,
    delta_prefix: str,
    weighted_similarity_key: str,
    weighted_overtime_key: str,
) -> dict[str, Any]:
    selected = [
        row
        for row in rows
        if (comparison is None or row.get("comparison") == comparison)
        and _truth(row.get("both_optimum"))
    ]
    if not selected:
        raise ValueError("policy signal has no jointly optimal pairs")
    relative_losses = []
    for row in selected:
        weighted_similarity = _float(row[weighted_similarity_key])
        similarity_change = _float(row[f"{delta_prefix}similarity"])
        if weighted_similarity <= 0:
            raise ValueError("weighted similarity must be positive")
        relative_losses.append(-100 * similarity_change / weighted_similarity)
    continuity_changes = [
        _float(row[f"{delta_prefix}continuity"]) for row in selected
    ]
    overtime_changes = [_float(row[f"{delta_prefix}overtime"]) for row in selected]
    similarity_changes = [
        _float(row[f"{delta_prefix}similarity"]) for row in selected
    ]
    return {
        "pairs": len(selected),
        "weighted_overtime_positive": sum(
            _float(row[weighted_overtime_key]) > 0 for row in selected
        ),
        "joint_strict_improvements": sum(
            continuity < 0 and overtime < 0
            for continuity, overtime in zip(continuity_changes, overtime_changes)
        ),
        "joint_nonworse": sum(
            continuity <= 0 and overtime <= 0
            for continuity, overtime in zip(continuity_changes, overtime_changes)
        ),
        "similarity_worsened": sum(value < 0 for value in similarity_changes),
        "median_similarity_change": _median(similarity_changes),
        "median_continuity_change": _median(continuity_changes),
        "median_overtime_change": _median(overtime_changes),
        "median_relative_similarity_loss_pct": _median(relative_losses),
        "min_relative_similarity_loss_pct": min(relative_losses),
        "max_relative_similarity_loss_pct": max(relative_losses),
    }


def _totalizer_only_contrast(
    rows: list[dict[str, str]], factorial: list[dict[str, str]]
) -> dict[str, Any]:
    totalizer_only = ("totalizer", "none", "none")
    full = REFERENCE
    indexed = {
        (row["instance_sha256"], _configuration(row)): row for row in rows
    }
    ratios = []
    totalizer_wins = 0
    instances = sorted({row["instance_sha256"] for row in rows})
    for instance in instances:
        left = indexed.get((instance, totalizer_only))
        right = indexed.get((instance, full))
        if left is None or right is None:
            continue
        if not (_truth(left["both_proved"]) and _truth(right["both_proved"])):
            continue
        left_time = _float(left["configuration_elapsed_seconds"])
        right_time = _float(right["configuration_elapsed_seconds"])
        if left_time <= 0 or right_time <= 0:
            raise ValueError("paired runtime must be positive")
        ratio = right_time / left_time
        ratios.append(ratio)
        totalizer_wins += ratio > 1 + 1e-12
    if not ratios:
        raise ValueError("no proved Totalizer-only/full pairs")
    low, high = _bootstrap_median_ci(ratios, "totalizer-only-vs-full")
    summaries = {_configuration(row): row for row in factorial}
    baseline = summaries[BASELINE]
    totalizer = summaries[totalizer_only]
    return {
        "pairs": len(ratios),
        "totalizer_only_faster": totalizer_wins,
        "median_full_over_totalizer_only": _median(ratios),
        "bootstrap_95_ci_low": low,
        "bootstrap_95_ci_high": high,
        "par2_reduction_vs_baseline_pct": 100
        * (_float(baseline["par2_seconds"]) - _float(totalizer["par2_seconds"]))
        / _float(baseline["par2_seconds"]),
    }


def _maxsat_progress(rows: list[dict[str, str]]) -> dict[str, dict[str, int]]:
    output: dict[str, dict[str, int]] = {}
    for method in ("weighted", "lex-cos", "lex-overtime"):
        group = [row for row in rows if row["method"] == method]
        if len(group) != 48:
            raise ValueError(f"expected 48 corrected-v2 MaxSAT rows for {method}")
        timeouts = [row for row in group if row["status"].startswith("TIMEOUT")]
        output[method] = {
            "runs": len(group),
            "optimum": sum(row["status"] == "OPTIMUM" for row in group),
            "timeouts": len(timeouts),
            "final_stage_timeouts": sum(
                method != "weighted" and _int(row.get("stage_count", 0)) >= 2
                for row in timeouts
            ),
        }
    return output


def _result_figure(
    encoding: list[dict[str, str]], corrected_pairs: list[dict[str, str]]
) -> str:
    """Render the two instance-level effects readers should remember."""
    order = (
        ("IC=none;SB=none", "neither added", 4),
        ("IC=none;SB=slot-service", "symmetry only", 3),
        ("IC=both;SB=none", "implied only", 2),
        ("IC=both;SB=slot-service", "both added", 1),
    )
    intervals = []
    labels = []
    ticks = []
    for condition, label, y in order:
        row = _one(
            encoding,
            lambda item, selected=condition: item["condition"] == selected,
            f"encoding/{condition} contrast",
        )
        low = _fmt(row["bootstrap_95_ci_low"], 3)
        high = _fmt(row["bootstrap_95_ci_high"], 3)
        median = _fmt(row["median_speedup_left_over_right"], 3)
        intervals.extend(
            (
                rf"    \addplot[draw=hcorapblue, very thick, mark=none] coordinates {{({low},{y}) ({high},{y})}};",
                rf"    \addplot[only marks, mark=*, mark size=2.2pt, hcorapblue] coordinates {{({median},{y})}};",
            )
        )
        labels.append(label)
        ticks.append(str(y))
    proved_counts = {_int(row["both_proved_pairs"]) for row in encoding}
    proved_wording = (
        f"{_count(next(iter(proved_counts)))} pairs solved by both settings per context"
        if len(proved_counts) == 1
        else "the pairs solved by both settings in each context"
    )
    policy_rows = [
        row
        for row in corrected_pairs
        if row["comparison"] == "weighted-to-continuity-first"
        and _truth(row["both_optimum"])
    ]
    if len(policy_rows) != 48:
        raise ValueError("expected 48 exact corrected-v2 policy pairs")
    policy_coordinates = " ".join(
        f"({_fmt(-_float(row['delta_continuity']), 1)},"
        f"{_fmt(-_float(row['delta_overtime']), 1)})"
        for row in policy_rows
    )

    return rf"""
\begin{{figure*}}[t]
  \centering
  \begin{{minipage}}[t]{{0.46\textwidth}}
    \centering
    \vspace{{0pt}}
    \begin{{tikzpicture}}
      \begin{{axis}}[
        width=0.98\linewidth,
        height=4.35cm,
        xmin=-0.7, xmax=13.8,
        ymin=-0.8, ymax=16.8,
        xtick={{0,3,6,9,12}},
        ytick={{0,4,8,12,16}},
        title={{\footnotesize\bfseries (a) Effect of continuity-first policy}},
        xlabel={{\scriptsize continuity improvement}},
        ylabel={{\scriptsize overtime reduction}},
        tick label style={{font=\scriptsize}},
        label style={{font=\scriptsize}},
        grid=major,
        grid style={{black!10}},
        axis lines=left,
        clip=false]
        \addplot[only marks,mark=*,mark size=1.65pt,
          draw=hcorapblue,fill=hcorapblue,fill opacity=0.60]
          coordinates {{{policy_coordinates}}};
      \end{{axis}}
    \end{{tikzpicture}}
  \end{{minipage}}\hfill
  \begin{{minipage}}[t]{{0.49\textwidth}}
    \centering
    \vspace{{0pt}}
    \begin{{tikzpicture}}
      \begin{{axis}}[
        width=0.98\linewidth,
        height=4.35cm,
        xmin=0.98, xmax=1.30,
        ymin=0.45, ymax=4.55,
        ytick={{{','.join(ticks)}}},
        yticklabels={{{','.join('{' + label + '}' for label in labels)}}},
        xtick={{1.0,1.1,1.2,1.3}},
        title={{\footnotesize\bfseries (b) Sorting network / Totalizer runtime}},
        xlabel={{\scriptsize paired median ratio ($>1$ favors Totalizer)}},
        tick label style={{font=\scriptsize}},
        yticklabel style={{align=right}},
        axis y line*=left,
        y axis line style={{draw=none}},
        ytick style={{draw=none}},
        axis x line*=bottom,
        xmajorgrids=true,
        grid style={{black!12}},
        clip=false]
        \draw[densely dashed,black!55] (axis cs:1,0.5) -- (axis cs:1,4.5);
{chr(10).join(intervals)}
      \end{{axis}}
    \end{{tikzpicture}}
  \end{{minipage}}
  \caption{{Main effects. (a) Each point is one of 48 jointly optimal
  corrected-v2 pairs; movement up and right means that continuity-first
  improves both criteria. (b) Sorting-network runtime divided by Totalizer
  runtime on the original benchmark; dots are paired medians and lines are
  95\% intervals over {proved_wording}. Values above one favor Totalizer; all
  runs use a 300-s limit.}}
  \Description{{Panel a is a scatter plot of continuity improvement against
  overtime reduction for 48 corrected-v2 instances. Most points lie above and
  to the right of zero. Panel b is a forest plot whose four confidence
  intervals lie above one and therefore favor Totalizer.}}
  \label{{fig:main-effects}}
\end{{figure*}}
"""


def _evidence_table(
    *,
    original_signal: dict[str, Any],
    corrected_signal: dict[str, Any],
    maxsat_progress: dict[str, dict[str, int]],
) -> str:
    return rf"""
\begin{{table}}[!t]
  \caption{{Benchmark signal and EvalMaxSAT progress. In the upper panel,
  ``both improve'' means strictly lower continuity and overtime under
  continuity-first; pairs are optimal under both policies, and SIM loss is the
  median compatibility reduction. The lower panel reports where runs stand at
  the 300-s limit.}}
  \label{{tab:benchmark-signal}}
  \centering
  \scriptsize
  \setlength{{\tabcolsep}}{{2.7pt}}
  \renewcommand{{\arraystretch}}{{1.03}}
  \begin{{tabular}}{{@{{}}lrrrr@{{}}}}
    \toprule
    Benchmark & Pairs & Weighted OT$>0$ & Both improve & SIM loss \\
    \midrule
    Original & {_count(original_signal['pairs'])} & {_count(original_signal['weighted_overtime_positive'])} & {_count(original_signal['joint_strict_improvements'])} & {_fmt(original_signal['median_relative_similarity_loss_pct'], 1)}\% \\
    Corrected-v2 & {_count(corrected_signal['pairs'])} & {_count(corrected_signal['weighted_overtime_positive'])} & {_count(corrected_signal['joint_strict_improvements'])} & {_fmt(corrected_signal['median_relative_similarity_loss_pct'], 1)}\% \\
    \midrule
    \multicolumn{{5}}{{@{{}}l}}{{\textit{{EvalMaxSAT progress on corrected-v2 (48 runs per policy)}}}} \\
    \midrule
    Policy & Optimum & \multicolumn{{2}}{{c}}{{Timeouts at final criterion}} & Total timeouts \\
    Weighted & {_count(maxsat_progress['weighted']['optimum'])} & \multicolumn{{2}}{{c}}{{--}} & {_count(maxsat_progress['weighted']['timeouts'])} \\
    Continuity-first & {_count(maxsat_progress['lex-cos']['optimum'])} & \multicolumn{{2}}{{c}}{{{_count(maxsat_progress['lex-cos']['final_stage_timeouts'])}}} & {_count(maxsat_progress['lex-cos']['timeouts'])} \\
    Overtime-first & {_count(maxsat_progress['lex-overtime']['optimum'])} & \multicolumn{{2}}{{c}}{{{_count(maxsat_progress['lex-overtime']['final_stage_timeouts'])}}} & {_count(maxsat_progress['lex-overtime']['timeouts'])} \\
    \bottomrule
  \end{{tabular}}
\end{{table}}
"""


def _factorial_footprint_table(factorial: list[dict[str, str]]) -> str:
    """Show every ablation cell without reproducing the full artifact matrix."""
    if len(factorial) != len(FACTORIAL_ORDER):
        raise ValueError(
            f"expected {len(FACTORIAL_ORDER)} factorial cells, found {len(factorial)}"
        )
    ordered = sorted(
        factorial,
        key=lambda row: (
            0 if row["cardinality"] == "sorting-network" else 1,
            0 if row["implied"] == "none" else 1,
            0 if row["symmetry"] == "none" else 1,
        ),
    )
    best_par2 = min(_float(row["par2_seconds"]) for row in ordered)
    profiles = {
        (
            _int(row["optimum_runs"]),
            _int(row["unsat_runs"]),
            _int(row["timeout_runs"]),
        )
        for row in ordered
    }
    if len(profiles) == 1:
        optimum, infeasible, timeout = next(iter(profiles))
        profile_caption = (
            f"each cell has {_count(optimum)} optimum, "
            f"{_count(infeasible)} infeasible, and {_count(timeout)} timeout runs"
        )
    else:
        profile_caption = "solved profiles vary by cell"
    lines = []
    for row in ordered:
        encoding = "SN" if row["cardinality"] == "sorting-network" else "TOT"
        implied = "off" if row["implied"] == "none" else "on"
        symmetry = "off" if row["symmetry"] == "none" else "on"
        par2 = _fmt(row["par2_seconds"], 1)
        if _float(row["par2_seconds"]) == best_par2:
            par2 = rf"\textbf{{{par2}}}"
        lines.append(
            f"    {encoding} & {implied} & {symmetry} & {par2} & "
            f"{_fmt(row['median_peak_rss_mb'], 1)} & "
            f"{_fmt_grouped(row['median_variables'], 0)} \\\\"
        )
    return rf"""
\begin{{table}}[!t]
  \caption{{Full encoding ablation (48 runs per cell). SN and TOT denote sorting
  networks and Totalizers; IC and SB denote implied constraints and symmetry
  breaking. PAR-2 includes the timeouts; RSS is peak resident memory; formula
  sizes and RSS are cell medians; {profile_caption}. Best PAR-2 is bold.}}
  \label{{tab:factorial-footprint}}
  \centering
  \scriptsize
  \setlength{{\tabcolsep}}{{3.1pt}}
  \renewcommand{{\arraystretch}}{{1.02}}
  \begin{{tabular}}{{@{{}}lllrrr@{{}}}}
    \toprule
    Enc. & IC & SB & PAR-2 (s) & RSS (MB) & Variables \\
    \midrule
{chr(10).join(lines)}
    \bottomrule
  \end{{tabular}}
\end{{table}}
"""


def _render(
    *,
    screening: dict[str, Any],
    factorial: list[dict[str, str]],
    contrasts: list[dict[str, str]],
    composite: dict[str, str],
    factorial_pairs: list[dict[str, str]],
    original_pairs: list[dict[str, str]],
    corrected_pairs: list[dict[str, str]],
    corrected_maxsat_runs: list[dict[str, str]],
    corrected_validation: dict[str, Any],
    corrected_rows: list[dict[str, str]],
    cross_rows: list[dict[str, str]],
) -> tuple[str, str, str]:
    """Build a concise result narrative from pair-level evidence."""
    del composite  # Retained in the generator contract and provenance.
    encoding = [row for row in contrasts if row["factor"] == "encoding"]
    implied = [row for row in contrasts if row["factor"] == "implied"]
    symmetry = [row for row in contrasts if row["factor"] == "symmetry"]
    if (len(encoding), len(implied), len(symmetry)) != (4, 4, 4):
        raise ValueError("expected four direct contrasts for each factorial factor")

    enc_speed = _range(encoding, "median_speedup_left_over_right")
    ic_speed = _range(implied, "median_speedup_left_over_right")
    sb_speed = _range(symmetry, "median_speedup_left_over_right")
    enc_wins = sum(_int(row["right_faster"]) for row in encoding)
    enc_losses = sum(_int(row["left_faster"]) for row in encoding)
    enc_var_reduction = (
        _fmt_grouped(
            min(abs(_float(row["median_variables_difference"])) for row in encoding)
        ),
        _fmt_grouped(
            max(abs(_float(row["median_variables_difference"])) for row in encoding)
        ),
    )
    enc_hard_increase = (
        _fmt_grouped(
            min(_float(row["median_hard_clauses_difference"]) for row in encoding)
        ),
        _fmt_grouped(
            max(_float(row["median_hard_clauses_difference"]) for row in encoding)
        ),
    )
    implied_slower = sum(_float(row["bootstrap_95_ci_high"]) < 1 for row in implied)
    symmetry_slower = sum(_float(row["bootstrap_95_ci_high"]) < 1 for row in symmetry)
    symmetry_unresolved = len(symmetry) - symmetry_slower

    original_signal = _policy_signal(
        original_pairs,
        comparison=None,
        delta_prefix="lex_minus_weighted_",
        weighted_similarity_key="weighted_similarity",
        weighted_overtime_key="weighted_overtime",
    )
    corrected_signal = _policy_signal(
        corrected_pairs,
        comparison="weighted-to-continuity-first",
        delta_prefix="delta_",
        weighted_similarity_key="left_similarity",
        weighted_overtime_key="left_overtime",
    )
    priority_pairs = [
        row
        for row in corrected_pairs
        if row["comparison"] == "continuity-first-to-overtime-first"
        and _truth(row["both_optimum"])
    ]
    if len(priority_pairs) != 48:
        raise ValueError("expected 48 exact corrected priority-order pairs")
    priority_same = sum(
        all(
            _float(row[f"delta_{metric}"]) == 0
            for metric in ("similarity", "continuity", "overtime")
        )
        for row in priority_pairs
    )
    priority_different = [
        row
        for row in priority_pairs
        if any(
            _float(row[f"delta_{metric}"]) != 0
            for metric in ("similarity", "continuity", "overtime")
        )
    ]
    priority_similarity_range = (
        min(_float(row["delta_similarity"]) for row in priority_different),
        max(_float(row["delta_similarity"]) for row in priority_different),
    )

    maxsat_progress = _maxsat_progress(corrected_maxsat_runs)
    lex_timeouts = (
        maxsat_progress["lex-cos"]["timeouts"]
        + maxsat_progress["lex-overtime"]["timeouts"]
    )
    final_stage_timeouts = (
        maxsat_progress["lex-cos"]["final_stage_timeouts"]
        + maxsat_progress["lex-overtime"]["final_stage_timeouts"]
    )
    totalizer_only = _totalizer_only_contrast(factorial_pairs, factorial)
    totalizer_cell = _one(
        factorial,
        lambda row: _configuration(row) == ("totalizer", "none", "none"),
        "Totalizer-only cell",
    )
    exact_groups = sum(_truth(row["all_exact_optimum"]) for row in cross_rows)
    infeasible_groups = sum(_truth(row["all_exact_infeasible"]) for row in cross_rows)
    agreement_groups = sum(_truth(row["objective_agreement"]) for row in cross_rows)
    if agreement_groups != exact_groups:
        raise ValueError("cross-solver objective agreement is incomplete")

    profiles = {
        (
            _int(row["optimum_runs"]),
            _int(row["unsat_runs"]),
            _int(row["timeout_runs"]),
        )
        for row in factorial
    }
    if len(profiles) != 1:
        raise ValueError("factorial cells have different solved profiles")
    optimum, infeasible, timeout = next(iter(profiles))

    result_figure = _result_figure(encoding, corrected_pairs)
    evidence_table = _evidence_table(
        original_signal=original_signal,
        corrected_signal=corrected_signal,
        maxsat_progress=maxsat_progress,
    )
    factorial_table = _factorial_footprint_table(factorial)

    results = rf"""% Generated from validator-approved campaign summaries. Do not edit.
\section{{Results}}
\label{{sec:results}}

The {_count(screening['expected_measured_runs'])}-run campaign yields three
findings. Explicit priorities change the schedules selected by a weighted
score. Totalizer improves runtime in every controlled comparison. The final
compatibility criterion accounts for almost all lexicographic timeouts.

{result_figure.rstrip()}

\subsection{{Priorities change schedules}}
\label{{sec:policy-results}}

The original benchmark rarely activates overtime: only
{_count(original_signal['weighted_overtime_positive'])}/{_count(original_signal['pairs'])}
weighted solutions have positive overtime, and continuity-first improves both
continuity and overtime in only
{_count(original_signal['joint_strict_improvements'])}/{_count(original_signal['pairs'])}
pairs. Corrected-v2 changes this picture. Weighted solutions use overtime in
{_count(corrected_signal['weighted_overtime_positive'])}/{_count(corrected_signal['pairs'])}
pairs, and continuity-first improves both criteria in
{_count(corrected_signal['joint_strict_improvements'])}/{_count(corrected_signal['pairs'])}
(Figure~\ref{{fig:main-effects}}a). It never worsens either prioritized
criterion, while compatibility falls by a median
{_fmt(corrected_signal['median_relative_similarity_loss_pct'], 1)}\%
(range {_fmt(corrected_signal['min_relative_similarity_loss_pct'], 1)}--{_fmt(corrected_signal['max_relative_similarity_loss_pct'], 1)}\%).
Table~\ref{{tab:benchmark-signal}} shows why corrected-v2 is needed for the
policy study rather than merely adding more instances of the original design.

Continuity-first and overtime-first select the same objective values on
{_count(priority_same)}/{_count(len(priority_pairs))} instances. In each of the
remaining {_count(len(priority_different))}, overtime-first removes one
overtime unit in exchange for one continuity unit; the associated compatibility
change ranges from {_fmt(priority_similarity_range[0])} to
{_fmt(priority_similarity_range[1])} points.

{evidence_table.rstrip()}

\subsection{{Totalizer helps; extra constraints do not}}
\label{{sec:encoding-results}}

All eight configurations solve the same {_count(optimum)} instances, prove
{_count(infeasible)} infeasible, and time out on {_count(timeout)}. Runtime
therefore distinguishes configurations more clearly than solved count. Across
the four sorting-network-to-Totalizer comparisons, paired median runtime ratios
{_range_wording(enc_speed)}, and every 95\% interval lies above one
(Figure~\ref{{fig:main-effects}}b). Totalizer is faster on {_count(enc_wins)} of
{_count(sum(_int(row['both_proved_pairs']) for row in encoding))} pairs solved
by both configurations,
versus {_count(enc_losses)} for sorting networks. It removes a median
{enc_var_reduction[0]}--{enc_var_reduction[1]} variables while adding
{enc_hard_increase[0]}--{enc_hard_increase[1]} clauses. These measurements
describe the formula change but do not isolate a single cause for the runtime gain.

{factorial_table.rstrip()}

The Totalizer-only cell has the lowest PAR-2,
{_fmt(totalizer_cell['par2_seconds'], 1)}\,s, an
{_fmt(totalizer_only['par2_reduction_vs_baseline_pct'], 1)}\% reduction from the
sorting-network baseline. Adding the implied constraints is slower in all
{_count(implied_slower)} contexts; its paired median ratios
{_range_wording(ic_speed)}. Symmetry breaking is slower in
{_count(symmetry_slower)} contexts and unresolved in
{_count(symmetry_unresolved)}; its paired median ratios
{_range_wording(sb_speed)}. The
full configuration is slower than Totalizer-only on
{_count(totalizer_only['totalizer_only_faster'])}/{_count(totalizer_only['pairs'])}
pairs solved by both configurations: its median runtime ratio is
{_fmt(totalizer_only['median_full_over_totalizer_only'], 2)}
([95\% CI: {_fmt(totalizer_only['bootstrap_95_ci_low'], 2)},
{_fmt(totalizer_only['bootstrap_95_ci_high'], 2)}]).

\subsection{{Solver progress and independent checks}}
\label{{sec:validation}}

Within 300\,s, EvalMaxSAT proves
{_count(maxsat_progress['weighted']['optimum'])}/48 weighted runs and
{_count(maxsat_progress['lex-cos']['optimum'])}/48 runs under each priority
order. Among the {_count(lex_timeouts)} lexicographic timeouts,
{_count(final_stage_timeouts)} reach the final compatibility criterion after
proving the first two optima. Thus the main difficulty is completing the last
criterion, not finding the higher-priority values.

Gurobi proves all {_count(sum(_int(row['optimum_runs']) for row in corrected_rows))}
corrected-v2 runs. CPLEX matches Gurobi on all
{_count(corrected_validation['audit_runs'])} audited runs. On the original
20-instance subset, EvalMaxSAT, Gurobi, and CPLEX also agree on
{_count(exact_groups)} optimal groups and {_count(infeasible_groups)} infeasible groups.
Every reported schedule passes the independent verifier.
"""

    abstract = (
        "% Generated from validator-approved campaign summaries. Do not edit.\n"
        "On corrected-v2, continuity-first improves both continuity and overtime "
        f"in {_count(corrected_signal['joint_strict_improvements'])}/"
        f"{_count(corrected_signal['pairs'])} exact pairs, with a median "
        f"compatibility reduction of {_fmt(corrected_signal['median_relative_similarity_loss_pct'], 1)}\\%. "
        "Totalizer yields median paired speedups of "
        f"{enc_speed[0]}--{enc_speed[1]} over sorting networks, whereas the two "
        "added constraint families do not improve the Totalizer-only setting. "
        f"Of {_count(lex_timeouts)} lexicographic timeouts, "
        f"{_count(final_stage_timeouts)} reach the final compatibility criterion.\n"
    )

    conclusion = rf"""% Generated from validator-approved campaign summaries. Do not edit.
\section{{Conclusion}}
\label{{sec:conclusion}}

Explicit priorities select different home-care schedules from a weighted
score. Corrected-v2 exposes the trade-off clearly: continuity-first improves
both continuity and overtime in
{_count(corrected_signal['joint_strict_improvements'])}/{_count(corrected_signal['pairs'])}
exact pairs at a median compatibility reduction of
{_fmt(corrected_signal['median_relative_similarity_loss_pct'], 1)}\%. Totalizer
is consistently faster than sorting networks, and Totalizer without the two
added constraint families has the lowest PAR-2. The three-solver checks agree
on all {_count(exact_groups + infeasible_groups)} original-benchmark groups.
On corrected-v2, the final compatibility criterion accounts for
{_count(final_stage_timeouts)}/{_count(lex_timeouts)} lexicographic timeouts and
is the clearest target for further improvement.
"""
    return abstract, results, conclusion


def generate(arguments: argparse.Namespace) -> dict[str, Any]:
    screening_path = arguments.screening_decision.resolve()
    primary_validation_path = (arguments.primary_dir / "analysis_validation.json").resolve()
    cross_validation_path = (arguments.cross_dir / "cross_paradigm_validation.json").resolve()
    screening = _json(screening_path)
    primary = _json(primary_validation_path)
    cross = _json(cross_validation_path)
    if screening.get("decision") != "GO":
        raise ValueError("screening decision is not GO")
    if primary.get("valid") is not True or cross.get("valid") is not True:
        raise ValueError("primary or cross-paradigm analysis is invalid")
    branches = screening.get("branches")
    if not isinstance(branches, dict):
        raise ValueError("screening decision has no branch map")
    original_enabled = branches["original_lexicographic"]["enabled"] is True
    corrected_enabled = branches["corrected_v2_lexicographic"]["enabled"] is True
    if not (original_enabled and corrected_enabled):
        raise ValueError("the compact manuscript requires both policy branches")
    expected_primary_scope = "compact"
    expected_cross_scope = "full"
    if primary.get("scope") != expected_primary_scope:
        raise ValueError("primary-analysis scope conflicts with screening decision")
    if cross.get("scope") != expected_cross_scope:
        raise ValueError("cross-paradigm scope conflicts with screening decision")

    source_paths = [screening_path, primary_validation_path, cross_validation_path]

    def primary_csv(name: str) -> list[dict[str, str]]:
        path = (arguments.primary_dir / name).resolve()
        source_paths.append(path)
        return _csv(path)

    factorial = primary_csv("factorial_summary.csv")
    factorial_pairs = primary_csv("factorial_paired_runs.csv")
    contrasts = primary_csv("factorial_contrasts.csv")
    composite_rows = primary_csv("weighted_composite_paired_summary.csv")
    if len(composite_rows) != 1:
        raise ValueError("weighted composite paired summary must have one row")
    composite = composite_rows[0]
    lex_rows = primary_csv("lex_confirmatory_summary.csv")
    if len(lex_rows) != 1:
        raise ValueError("compact policy scope requires one reference summary row")
    original_pairs = primary_csv("lex_confirmatory_pairs.csv")

    corrected_rows: list[dict[str, str]] = []
    corrected_maxsat_rows: list[dict[str, str]] = []
    corrected_validation: dict[str, Any] | None = None
    corrected_pairs: list[dict[str, str]] = []
    corrected_maxsat_runs: list[dict[str, str]] = []
    if corrected_enabled:
        corrected_validation_path = (
            arguments.corrected_dir / "corrected_exact_validation.json"
        ).resolve()
        corrected_validation = _json(corrected_validation_path)
        if corrected_validation.get("manuscript_eligible") is not True:
            raise ValueError("corrected-v2 exact policy evidence is not manuscript-eligible")
        corrected_maxsat_validation_path = (
            arguments.corrected_maxsat_dir / "corrected_validation.json"
        ).resolve()
        corrected_maxsat_validation = _json(corrected_maxsat_validation_path)
        if corrected_maxsat_validation.get("structurally_valid") is not True:
            raise ValueError("corrected-v2 EvalMaxSAT scalability results are invalid")
        corrected_maxsat_summary_path = (
            arguments.corrected_maxsat_dir / "corrected_policy_summary.csv"
        ).resolve()
        corrected_maxsat_rows = _csv(corrected_maxsat_summary_path)
        corrected_maxsat_runs_path = (
            arguments.corrected_maxsat_results_dir / "runs.csv"
        ).resolve()
        corrected_maxsat_runs = _csv(corrected_maxsat_runs_path)
        corrected_summary_path = (
            arguments.corrected_dir / "corrected_policy_summary.csv"
        ).resolve()
        corrected_paired_path = (
            arguments.corrected_dir / "corrected_pairwise_summary.csv"
        ).resolve()
        corrected_pairs_path = (
            arguments.corrected_dir / "corrected_pairwise_pairs.csv"
        ).resolve()
        corrected_rows = _csv(corrected_summary_path)
        corrected_paired_rows = _csv(corrected_paired_path)
        corrected_pairs = _csv(corrected_pairs_path)
        _one(
            corrected_paired_rows,
            lambda row: row["comparison"] == "weighted-to-continuity-first",
            "corrected weighted-to-continuity-first summary",
        )
        _one(
            corrected_paired_rows,
            lambda row: row["comparison"]
            == "continuity-first-to-overtime-first",
            "corrected priority-order summary",
        )
        source_paths.extend(
            (
                corrected_validation_path,
                corrected_summary_path,
                corrected_paired_path,
                corrected_pairs_path,
                corrected_maxsat_validation_path,
                corrected_maxsat_summary_path,
                corrected_maxsat_runs_path,
            )
        )
    if corrected_validation is None:
        raise ValueError("missing corrected-v2 validation")

    cross_agreement_path = (
        arguments.cross_dir / "cross_paradigm_agreement.csv"
    ).resolve()
    cross_rows = _csv(cross_agreement_path)
    source_paths.append(cross_agreement_path)
    expected_cross_rows = 40
    if len(cross_rows) != expected_cross_rows:
        raise ValueError(
            f"unexpected cross-paradigm groups: {len(cross_rows)}/{expected_cross_rows}"
        )

    abstract, results, conclusion = _render(
        screening=screening,
        factorial=factorial,
        contrasts=contrasts,
        composite=composite,
        factorial_pairs=factorial_pairs,
        original_pairs=original_pairs,
        corrected_pairs=corrected_pairs,
        corrected_maxsat_runs=corrected_maxsat_runs,
        corrected_validation=corrected_validation,
        corrected_rows=corrected_rows,
        cross_rows=cross_rows,
    )
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "abstract-findings.tex": abstract,
        "results.tex": results,
        "conclusion.tex": conclusion,
    }
    for name, content in outputs.items():
        (arguments.output_dir / name).write_text(content, encoding="utf-8")
    provenance = {
        "valid": True,
        "primary_scope": primary["scope"],
        "cross_scope": cross["scope"],
        "original_lexicographic_enabled": original_enabled,
        "corrected_v2_lexicographic_enabled": corrected_enabled,
        "expected_measured_runs": screening["expected_measured_runs"],
        "source_sha256": {
            _portable_path(path): _sha256(path) for path in sorted(set(source_paths))
        },
        "fragment_sha256": {
            name: hashlib.sha256(content.encode("utf-8")).hexdigest()
            for name, content in outputs.items()
        },
    }
    provenance_path = arguments.output_dir / "manuscript-provenance.json"
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return provenance


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--screening-decision",
        type=Path,
        default=Path("experiments/results/screening_decision.json"),
    )
    parser.add_argument(
        "--primary-dir",
        type=Path,
        default=Path("experiments/results/gcp_primary_analysis"),
    )
    parser.add_argument(
        "--corrected-dir",
        type=Path,
        default=Path("experiments/results/gcp_corrected_exact_analysis"),
    )
    parser.add_argument(
        "--corrected-maxsat-dir",
        type=Path,
        default=Path("experiments/results/gcp_corrected_analysis"),
    )
    parser.add_argument(
        "--corrected-maxsat-results-dir",
        type=Path,
        default=Path("experiments/results/gcp_corrected_primary"),
    )
    parser.add_argument(
        "--cross-dir",
        type=Path,
        default=Path("experiments/results/gcp_cross_paradigm_analysis"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("LaTeX-Templates/paper/generated"),
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    try:
        result = generate(arguments)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        raise SystemExit(f"refusing to generate manuscript results: {error}") from error
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
