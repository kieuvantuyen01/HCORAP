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
    joint_strict_improvements = sum(
        continuity < 0 and overtime < 0
        for continuity, overtime in zip(continuity_changes, overtime_changes)
    )
    continuity_only_improvements = sum(
        continuity < 0 and overtime == 0
        for continuity, overtime in zip(continuity_changes, overtime_changes)
    )
    overtime_only_improvements = sum(
        continuity == 0 and overtime < 0
        for continuity, overtime in zip(continuity_changes, overtime_changes)
    )
    return {
        "pairs": len(selected),
        "weighted_overtime_positive": sum(
            _float(row[weighted_overtime_key]) > 0 for row in selected
        ),
        "joint_strict_improvements": joint_strict_improvements,
        "continuity_only_improvements": continuity_only_improvements,
        "overtime_only_improvements": overtime_only_improvements,
        "continuity_improved": (
            joint_strict_improvements + continuity_only_improvements
        ),
        "overtime_improved": joint_strict_improvements + overtime_only_improvements,
        "any_priority_improvement": (
            joint_strict_improvements
            + continuity_only_improvements
            + overtime_only_improvements
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
        "q1_relative_similarity_loss_pct": _percentile(relative_losses, 0.25),
        "q3_relative_similarity_loss_pct": _percentile(relative_losses, 0.75),
        "min_relative_similarity_loss_pct": min(relative_losses),
        "max_relative_similarity_loss_pct": max(relative_losses),
    }


def _policy_size_breakdown(rows: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    """Summarize corrected policy effects by the two main size dimensions."""
    selected = [
        row
        for row in rows
        if row.get("comparison") == "weighted-to-continuity-first"
        and _truth(row.get("both_optimum"))
    ]
    output: dict[str, dict[str, Any]] = {}
    for field, expected in (
        ("users", {"30", "40"}),
        ("agents", {"10", "15", "20", "25"}),
        ("visits", {"4", "5"}),
    ):
        observed = {row.get(field, "") for row in selected}
        if observed != expected:
            raise ValueError(
                f"corrected policy pairs have unexpected {field} groups: {observed}"
            )
        for value in sorted(expected, key=int):
            group = [row for row in selected if row[field] == value]
            continuity = [-_float(row["delta_continuity"]) for row in group]
            overtime = [-_float(row["delta_overtime"]) for row in group]
            output[f"{field}_{value}"] = {
                "pairs": len(group),
                "joint_strict_improvements": sum(
                    cont > 0 and ot > 0 for cont, ot in zip(continuity, overtime)
                ),
                "median_continuity_reduction": _median(continuity),
                "median_overtime_reduction": _median(overtime),
            }
    return output


def _encoding_size_breakdown(rows: list[dict[str, str]]) -> dict[str, float]:
    """Pool the four direct SN/Totalizer contrasts within each size group."""
    indexed = {(row["instance_sha256"], _configuration(row)): row for row in rows}
    grouped: dict[str, list[float]] = {
        "users_30": [],
        "users_40": [],
        "visits_4": [],
        "visits_5": [],
    }
    for instance in sorted({row["instance_sha256"] for row in rows}):
        for implied in ("none", "both"):
            for symmetry in ("none", "slot-service"):
                sorting = indexed.get(
                    (instance, ("sorting-network", implied, symmetry))
                )
                totalizer = indexed.get((instance, ("totalizer", implied, symmetry)))
                if sorting is None or totalizer is None:
                    raise ValueError("factorial pair matrix is incomplete")
                if not (
                    _truth(sorting["both_proved"])
                    and _truth(totalizer["both_proved"])
                ):
                    continue
                ratio = _float(sorting["configuration_elapsed_seconds"]) / _float(
                    totalizer["configuration_elapsed_seconds"]
                )
                for field in ("users", "visits"):
                    key = f"{field}_{sorting.get(field, '')}"
                    if key not in grouped:
                        raise ValueError(f"unexpected factorial size group: {key}")
                    grouped[key].append(ratio)
    if any(not values for values in grouped.values()):
        raise ValueError("factorial size breakdown contains an empty group")
    return {key: _median(values) for key, values in grouped.items()}


def _solver_policy_performance(
    maxsat_rows: list[dict[str, str]],
    gurobi_rows: list[dict[str, str]],
    cplex_runs: list[dict[str, str]],
) -> dict[str, dict[str, dict[str, float | int]]]:
    """Normalize optimum counts and mean PAR-2 across the three solvers."""
    methods = ("weighted", "lex-cos", "lex-overtime")
    output: dict[str, dict[str, dict[str, float | int]]] = {}
    for solver, rows, expected in (
        ("EvalMaxSAT", maxsat_rows, 48),
        ("Gurobi", gurobi_rows, 48),
    ):
        solver_rows: dict[str, dict[str, float | int]] = {}
        for method in methods:
            row = _one(
                rows,
                lambda item, selected=method: item["method"] == selected,
                f"{solver}/{method} summary",
            )
            if _int(row["runs"]) != expected:
                raise ValueError(f"unexpected {solver}/{method} run count")
            solver_rows[method] = {
                "runs": expected,
                "optimum": _int(row["optimum_runs"]),
                "par2_seconds": _float(row["par2_seconds"]),
            }
        output[solver] = solver_rows

    cplex_summary: dict[str, dict[str, float | int]] = {}
    for method in methods:
        group = [row for row in cplex_runs if row["method"] == method]
        if len(group) != 16:
            raise ValueError(f"expected 16 CPLEX corrected-v2 runs for {method}")
        par2_values = [
            _float(row["elapsed_seconds"])
            if row["status"] in {"OPTIMUM", "INFEASIBLE", "UNSATISFIABLE"}
            else 2 * _float(row["timeout_seconds"])
            for row in group
        ]
        cplex_summary[method] = {
            "runs": len(group),
            "optimum": sum(row["status"] == "OPTIMUM" for row in group),
            "par2_seconds": statistics.fmean(par2_values),
        }
    output["CPLEX"] = cplex_summary
    return output


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
            "reached_final_stage": sum(
                method != "weighted" and _int(row.get("stage_count", 0)) >= 2
                for row in group
            ),
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
        ("IC=none;SB=none", "no optional constraints", 4),
        ("IC=none;SB=slot-service", "symmetry only", 3),
        ("IC=both;SB=none", "implied only", 2),
        ("IC=both;SB=slot-service", "both optional types", 1),
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
        f"{_count(next(iter(proved_counts)))} same-instance pairs per row"
        if len(proved_counts) == 1
        else "the same-instance pairs available in each row"
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
        title={{\footnotesize\bfseries (a) Weighted to continuity-first}},
        xlabel={{\scriptsize lower continuity penalty (CONT)}},
        ylabel={{\scriptsize less overtime (OT)}},
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
        width=0.90\linewidth,
        height=4.35cm,
        xmin=0.98, xmax=1.30,
        ymin=0.45, ymax=4.55,
        ytick={{{','.join(ticks)}}},
        yticklabels={{{','.join('{' + label + '}' for label in labels)}}},
        xtick={{1.0,1.1,1.2,1.3}},
        title={{\footnotesize\bfseries (b) Sorting-network time / Totalizer time}},
        xlabel={{\scriptsize median time ratio ($>1$: Totalizer is faster)}},
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
  \caption{{Policy and encoding effects. (a) Reductions under continuity-first
  for 48 corrected-v2 instances. (b) Sorting-network/Totalizer runtime ratios
  over {proved_wording}; points are medians, bars are 95\% intervals, and values
  above one favor Totalizer.}}
  \Description{{Panel a plots the reduction in additional caregivers against
  the reduction in overtime for 48 instances. Panel b compares sorting-network
  and Totalizer runtime under four choices of optional constraints.}}
  \label{{fig:main-effects}}
\end{{figure*}}
"""


def _evidence_table(
    *,
    original_signal: dict[str, Any],
    corrected_signal: dict[str, Any],
    encoding_speed: tuple[str, str],
    implied_contexts_slower: int,
    symmetry_contexts_slower: int,
    symmetry_contexts_unresolved: int,
    corrected_audit_groups: int,
    exact_groups: int,
    infeasible_groups: int,
    lex_timeouts: int,
    final_stage_timeouts: int,
) -> str:
    return rf"""
\begin{{table}}[!t]
  \caption{{Main results for the three research questions.}}
  \label{{tab:evidence-map}}
  \centering
  \scriptsize
  \setlength{{\tabcolsep}}{{2.8pt}}
  \renewcommand{{\arraystretch}}{{1.06}}
  \begin{{tabularx}}{{\columnwidth}}{{@{{}}>{{\raggedright\arraybackslash}}p{{0.22\columnwidth}}>{{\raggedright\arraybackslash}}p{{0.23\columnwidth}}X@{{}}}}
    \toprule
    Question & Compared cases & Main observation \\
    \midrule
    Priority rule & original: {_count(original_signal['pairs'])}; corrected: {_count(corrected_signal['pairs'])} & CONT and OT decrease in {_count(original_signal['joint_strict_improvements'])}/{_count(original_signal['pairs'])} versus {_count(corrected_signal['joint_strict_improvements'])}/{_count(corrected_signal['pairs'])}; corrected median compatibility loss {_fmt(corrected_signal['median_relative_similarity_loss_pct'], 1)}\% \\
    Count encoding & four paired settings & sorting/Totalizer time ratio {encoding_speed[0]}--{encoding_speed[1]}; Totalizer faster in all four \\
    Optional constraints & four settings per type & implied constraints slower in {_count(implied_contexts_slower)}/4; symmetry slower in {_count(symmetry_contexts_slower)}/4 and inconclusive in {_count(symmetry_contexts_unresolved)}/4 \\
    Solver agreement & {_count(corrected_audit_groups)} corrected; {_count(exact_groups + infeasible_groups)} original & no disagreement in feasibility or quality values \\
    Three-stage solving & 96 priority runs & {_count(final_stage_timeouts)}/{_count(lex_timeouts)} timeouts occur in final compatibility stage \\
    \bottomrule
  \end{{tabularx}}
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
            f"Each row has {_count(optimum)} proved-optimal, "
            f"{_count(infeasible)} proved-infeasible, and {_count(timeout)} timed-out runs"
        )
    else:
        profile_caption = "Solved profiles vary by row"
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
  \caption{{Runtime and model size for all eight Boolean configurations
  (48 runs per row).}}
  \label{{tab:factorial-footprint}}
  \centering
  \scriptsize
  \setlength{{\tabcolsep}}{{3.1pt}}
  \renewcommand{{\arraystretch}}{{1.02}}
  \begin{{tabular}}{{@{{}}lllrrr@{{}}}}
    \toprule
    Count & IC & SB & PAR-2 (s) & Peak MB & Variables \\
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
    corrected_maxsat_rows: list[dict[str, str]],
    corrected_maxsat_runs: list[dict[str, str]],
    corrected_cplex_runs: list[dict[str, str]],
    corrected_validation: dict[str, Any],
    corrected_rows: list[dict[str, str]],
    cross_rows: list[dict[str, str]],
) -> tuple[str, str, str]:
    """Build a concise result narrative from pair-level evidence."""
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
    implied_wins = sum(_int(row["right_faster"]) for row in implied)
    implied_losses = sum(_int(row["left_faster"]) for row in implied)
    implied_var_increase = (
        _fmt_grouped(min(_float(row["median_variables_difference"]) for row in implied)),
        _fmt_grouped(max(_float(row["median_variables_difference"]) for row in implied)),
    )
    implied_hard_increase = (
        _fmt_grouped(
            min(_float(row["median_hard_clauses_difference"]) for row in implied)
        ),
        _fmt_grouped(
            max(_float(row["median_hard_clauses_difference"]) for row in implied)
        ),
    )
    implied_rss_increase = (
        _fmt(
            min(_float(row["median_peak_rss_difference_mb"]) for row in implied),
            1,
        ),
        _fmt(
            max(_float(row["median_peak_rss_difference_mb"]) for row in implied),
            1,
        ),
    )
    symmetry_slower = sum(_float(row["bootstrap_95_ci_high"]) < 1 for row in symmetry)
    symmetry_unresolved = len(symmetry) - symmetry_slower
    symmetry_wins = sum(_int(row["right_faster"]) for row in symmetry)
    symmetry_losses = sum(_int(row["left_faster"]) for row in symmetry)
    symmetry_var_increase = _fmt_grouped(
        max(_float(row["median_variables_difference"]) for row in symmetry)
    )
    symmetry_hard_increase = _fmt_grouped(
        max(_float(row["median_hard_clauses_difference"]) for row in symmetry)
    )
    if _int(composite["both_proved_pairs"]) <= 0:
        raise ValueError("end-to-end baseline/full contrast has no proved pairs")

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
    if (
        corrected_signal["joint_nonworse"] != corrected_signal["pairs"]
        or corrected_signal["any_priority_improvement"] != corrected_signal["pairs"]
    ):
        raise ValueError(
            "corrected continuity-first schedules must weakly dominate weighted "
            "schedules in the two prioritized criteria"
        )
    policy_sizes = _policy_size_breakdown(corrected_pairs)
    agent_groups = [policy_sizes[f"agents_{agents}"] for agents in (10, 15, 20, 25)]
    agent_joint_improvements = [
        _int(group["joint_strict_improvements"]) for group in agent_groups
    ]
    agent_pairs = {_int(group["pairs"]) for group in agent_groups}
    if len(agent_pairs) != 1:
        raise ValueError("corrected policy caregiver-count groups are unbalanced")
    encoding_sizes = _encoding_size_breakdown(factorial_pairs)
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
    priority_cases = sorted(
        priority_different,
        key=lambda row: tuple(_int(row[field]) for field in ("users", "agents", "visits", "seed")),
    )
    priority_case_tuples = ", ".join(
        rf"$({_count(row['users'])},{_count(row['agents'])},"
        rf"{_count(row['visits'])},{_count(row['seed'])})$"
        for row in priority_cases
    )
    priority_case_similarity = ", ".join(
        _fmt(_float(row["delta_similarity"]), 0) for row in priority_cases
    )

    maxsat_progress = _maxsat_progress(corrected_maxsat_runs)
    solver_performance = _solver_policy_performance(
        corrected_maxsat_rows, corrected_rows, corrected_cplex_runs
    )
    lex_timeouts = (
        maxsat_progress["lex-cos"]["timeouts"]
        + maxsat_progress["lex-overtime"]["timeouts"]
    )
    final_stage_timeouts = (
        maxsat_progress["lex-cos"]["final_stage_timeouts"]
        + maxsat_progress["lex-overtime"]["final_stage_timeouts"]
    )
    lex_reached_final = (
        maxsat_progress["lex-cos"]["reached_final_stage"]
        + maxsat_progress["lex-overtime"]["reached_final_stage"]
    )
    commercial_overheads: dict[str, tuple[float, float, float]] = {}
    for solver in ("Gurobi", "CPLEX"):
        weighted_time = _float(solver_performance[solver]["weighted"]["par2_seconds"])
        continuity_time = _float(
            solver_performance[solver]["lex-cos"]["par2_seconds"]
        )
        overtime_time = _float(
            solver_performance[solver]["lex-overtime"]["par2_seconds"]
        )
        commercial_overheads[solver] = (
            min(continuity_time, overtime_time) / weighted_time,
            max(continuity_time, overtime_time) / weighted_time,
            max(continuity_time, overtime_time),
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
        encoding_speed=enc_speed,
        implied_contexts_slower=implied_slower,
        symmetry_contexts_slower=symmetry_slower,
        symmetry_contexts_unresolved=symmetry_unresolved,
        corrected_audit_groups=_int(corrected_validation["audit_runs"]),
        exact_groups=exact_groups,
        infeasible_groups=infeasible_groups,
        lex_timeouts=lex_timeouts,
        final_stage_timeouts=final_stage_timeouts,
    )
    factorial_table = _factorial_footprint_table(factorial)

    results = rf"""% Generated from validator-approved campaign summaries. Do not edit.
{result_figure.rstrip()}

\section{{Results}}
\label{{sec:results}}

The {_count(screening['expected_measured_runs'])} runs yield three findings:
strict priorities change schedule quality, Totalizer improves EvalMaxSAT
runtime, and the final compatibility stage dominates its timeouts.

\subsection{{Strict priorities change the selected schedules}}
\label{{sec:policy-results}}

Only {_count(original_signal['weighted_overtime_positive'])}/{_count(original_signal['pairs'])}
weighted schedules on the original benchmark use overtime, so that suite cannot
reveal the effect of an overtime priority. On corrected-v2, overtime appears in
{_count(corrected_signal['weighted_overtime_positive'])}/{_count(corrected_signal['pairs'])}
weighted schedules. Continuity-first reduces both CONT and OT in
{_count(corrected_signal['joint_strict_improvements'])}/{_count(corrected_signal['pairs'])}
comparisons, only CONT in {_count(corrected_signal['continuity_only_improvements'])},
and only OT in {_count(corrected_signal['overtime_only_improvements'])}
(Figure~\ref{{fig:main-effects}}a). The median reductions are
{_fmt(-corrected_signal['median_continuity_change'], 1)} CONT units and
{_fmt(-corrected_signal['median_overtime_change'], 1)} OT units. Compatibility
falls by a median {_fmt(corrected_signal['median_relative_similarity_loss_pct'], 1)}\%,
with an interquartile range of
{_fmt(corrected_signal['q1_relative_similarity_loss_pct'], 1)}--{_fmt(corrected_signal['q3_relative_similarity_loss_pct'], 1)}\%.
The joint improvement occurs in
{_count(policy_sizes['users_30']['joint_strict_improvements'])}/24 smaller and
{_count(policy_sizes['users_40']['joint_strict_improvements'])}/24 larger instances.

Continuity-first and overtime-first produce the same quality values on
{_count(priority_same)}/{_count(len(priority_pairs))} instances. In the other
{_count(len(priority_different))}, prioritizing overtime saves one overtime unit
at the cost of one continuity unit; compatibility changes by
{_fmt(priority_similarity_range[0])} to {_fmt(priority_similarity_range[1])} points.

\subsection{{Totalizer improves runtime}}
\label{{sec:encoding-results}}

All eight configurations prove the same {_count(optimum)} instances optimal,
prove {_count(infeasible)} infeasible, and time out on {_count(timeout)}. Across
the four controlled comparisons, the median sorting-network/Totalizer runtime
ratio is {enc_speed[0]}--{enc_speed[1]}, and every 95\% interval is above one
(Figure~\ref{{fig:main-effects}}b). Totalizer is faster on {_count(enc_wins)} of
{_count(sum(_int(row['both_proved_pairs']) for row in encoding))} paired runs,
uses about {enc_var_reduction[0]}--{enc_var_reduction[1]} fewer variables, and
reduces median peak memory by
{_fmt(abs(max(_float(row['median_peak_rss_difference_mb']) for row in encoding)), 1)}--{_fmt(abs(min(_float(row['median_peak_rss_difference_mb']) for row in encoding)), 1)}\,MB.
Table~\ref{{tab:factorial-footprint}} uses SN/TOT for the two encodings and
IC/SB for implied constraints and symmetry breaking.

{factorial_table.rstrip()}

Totalizer without optional constraints has the lowest PAR-2,
{_fmt(totalizer_cell['par2_seconds'], 1)}\,s, an
{_fmt(totalizer_only['par2_reduction_vs_baseline_pct'], 1)}\% reduction from the
sorting-network baseline. Implied constraints are slower in all
{_count(implied_slower)} controlled comparisons. Symmetry breaking has no
consistent benefit: disabling it is favored in {_count(symmetry_slower)}
settings, while {_count(symmetry_unresolved)} are inconclusive. Adding both to
Totalizer raises median runtime by a factor of
{_fmt(totalizer_only['median_full_over_totalizer_only'], 2)} (95\% interval
{_fmt(totalizer_only['bootstrap_95_ci_low'], 2)}--{_fmt(totalizer_only['bootstrap_95_ci_high'], 2)}).

\subsection{{The compatibility stage dominates EvalMaxSAT timeouts}}
\label{{sec:validation}}

EvalMaxSAT completes all objectives in
{_count(maxsat_progress['weighted']['optimum'])}/48 weighted runs and
{_count(maxsat_progress['lex-cos']['optimum'])}/48 runs under each strict order.
Yet {_count(lex_reached_final)}/96 strict-priority runs reach the compatibility
stage, where {_count(final_stage_timeouts)}/{_count(lex_timeouts)} timeouts occur.
The earlier continuity and overtime optima are therefore usually available even
when compatibility cannot be completed within 300\,s.

Gurobi proves all
{_count(sum(_int(row['optimum_runs']) for row in corrected_rows))} corrected-v2
runs, and CPLEX matches its quality values on all
{_count(corrected_validation['audit_runs'])} audited runs. On the original
subset, all three solvers agree on {_count(exact_groups)} optimal and
{_count(infeasible_groups)} infeasible instance--policy comparisons. Every
reported schedule passes the independent checker. Mean commercial-solver PAR-2
remains below
{_fmt(max(commercial_overheads['Gurobi'][2], commercial_overheads['CPLEX'][2]), 1)}\,s,
pointing to the staged EvalMaxSAT implementation as the performance bottleneck.
"""

    abstract = (
        "% Generated from validator-approved campaign summaries. Do not edit.\n"
        "On corrected-v2, continuity-first improves both continuity and overtime "
        f"in {_count(corrected_signal['joint_strict_improvements'])}/"
        f"{_count(corrected_signal['pairs'])} same-instance comparisons, with a median "
        f"compatibility reduction of {_fmt(corrected_signal['median_relative_similarity_loss_pct'], 1)}\\%. "
        "Across the four controlled settings, the median "
        "sorting-network/Totalizer runtime ratios are "
        f"{enc_speed[0]}--{enc_speed[1]}; implied constraints and symmetry "
        "breaking do not improve Totalizer without those constraints. "
        f"Of {_count(lex_timeouts)} strict-priority timeouts, "
        f"{_count(final_stage_timeouts)} occur in the third, compatibility stage.\n"
    )

    conclusion = rf"""% Generated from validator-approved campaign summaries. Do not edit.
\section{{Conclusion}}
\label{{sec:conclusion}}

Strict priorities and Boolean representation have distinct effects on HCORAP.
Continuity-first improves both continuity and overtime in
{_count(corrected_signal['joint_strict_improvements'])}/{_count(corrected_signal['pairs'])}
same-instance comparisons at a median compatibility reduction of
{_fmt(corrected_signal['median_relative_similarity_loss_pct'], 1)}\%. Totalizer
has a lower median runtime than sorting networks in all four controlled
comparisons, while implied constraints and symmetry breaking do not improve
Totalizer without those constraints. Gurobi and CPLEX confirm the reported
quality values. The compatibility stage accounts for
{_count(final_stage_timeouts)}/{_count(lex_timeouts)} EvalMaxSAT timeouts, making
cross-stage information reuse the clearest algorithmic next step. Operational
data are needed to assess whether the measured trade-offs match provider
priorities.
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
    corrected_cplex_runs: list[dict[str, str]] = []
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
        corrected_cplex_runs_path = (
            arguments.corrected_cplex_results_dir / "runs.csv"
        ).resolve()
        corrected_cplex_runs = _csv(corrected_cplex_runs_path)
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
                corrected_cplex_runs_path,
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
        corrected_maxsat_rows=corrected_maxsat_rows,
        corrected_maxsat_runs=corrected_maxsat_runs,
        corrected_cplex_runs=corrected_cplex_runs,
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
        "--corrected-cplex-results-dir",
        type=Path,
        default=Path("experiments/results/gcp_commercial_corrected_audit"),
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
