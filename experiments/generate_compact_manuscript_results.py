#!/usr/bin/env python3
"""Generate the result fragments for the locked two-study manuscript design.

The generator accepts only the validated HCORAP-LC policy analysis, the
validated Original-suite Policy x Encoding analysis, and the validated full
Gurobi/CPLEX comparison.  It deliberately refuses partial or failed analyses,
so draft numbers cannot silently enter the paper.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import statistics
from pathlib import Path
from typing import Any, Iterable


POLICIES = ("weighted", "lex-cos")
ENCODINGS = ("sorting-network", "totalizer")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _one(
    rows: Iterable[dict[str, str]], predicate: Any, description: str
) -> dict[str, str]:
    selected = [row for row in rows if predicate(row)]
    if len(selected) != 1:
        raise ValueError(f"expected one {description}, found {len(selected)}")
    return selected[0]


def _truth(value: Any) -> bool:
    return value is True or str(value).lower() == "true"


def _integer(value: Any) -> int:
    return int(float(str(value)))


def _number(value: Any, digits: int = 2) -> str:
    if value in (None, ""):
        return "--"
    rendered = f"{float(value):.{digits}f}"
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return "0" if rendered == "-0" else rendered


def _fixed(value: Any, digits: int = 1) -> str:
    if value in (None, ""):
        return "--"
    return f"{float(value):.{digits}f}"


def _percentile(values: Iterable[float], fraction: float) -> float:
    """Return a linearly interpolated sample percentile."""
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot compute a percentile of an empty sample")
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _grouped(value: Any) -> str:
    if value in (None, ""):
        return "--"
    return f"{round(float(value)):,}".replace(",", r"{,}")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _portable(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(resolved)


def _policy_label(method: str) -> str:
    return "Weighted" if method == "weighted" else "LEX-COS"


def _encoding_label(encoding: str) -> str:
    return "SN" if encoding == "sorting-network" else "TOT"


def _macro(name: str, value: Any) -> str:
    return rf"\newcommand{{\{name}}}{{{value}}}"


def _runtime_reduction_summary(rows: list[dict[str, str]]) -> dict[str, Any]:
    """Bootstrap the median of per-instance runtime reductions, excluding timeouts."""
    values = []
    for row in rows:
        if not _truth(row.get("both_proved")):
            continue
        sn = float(row["sorting_elapsed_seconds"])
        tot = float(row["totalizer_elapsed_seconds"])
        if not all(math.isfinite(value) and value > 0 for value in (sn, tot)):
            raise ValueError("paired runtimes must be finite and positive")
        values.append(100 * (sn - tot) / sn)
    if not values:
        raise ValueError("runtime figure needs at least one jointly proved pair")
    values.sort()
    rng = random.Random(20260908)
    medians = [statistics.median(rng.choices(values, k=len(values))) for _ in range(10000)]
    return {
        "pairs": len(values), "excluded_pairs": len(rows) - len(values),
        "median": statistics.median(values),
        "ci_low": _percentile(medians, 0.025),
        "ci_high": _percentile(medians, 0.975),
        "reductions_percent": values,
    }


def _figure_fragments(
    policy_values: list[list[float]], runtime: dict[str, dict[str, Any]]
) -> tuple[str, str]:
    policy = [r"% Generated; each point represents one paired instance.",
              r"\begin{figure*}[t]", r"\centering"]
    panels = (
        ("(a) Continuity", "Fewer continuity violations", "hcorapblue"),
        ("(b) Overtime", "Less overtime (service units)", "hcorapblue"),
        ("(c) Compatibility", r"Lower compatibility (\%)", "hcorapamber"),
    )
    for values, (title, ylabel, color) in zip(policy_values, panels):
        ordered = sorted(values)
        low, q1, median, q3, high = (
            _percentile(values, q) for q in (0, 0.25, 0.5, 0.75, 1)
        )
        # Small horizontal offsets expose coincident values without altering measurements.
        jitter = random.Random(48)
        coords = " ".join(f"({jitter.uniform(0.34, 0.76):.6f},{v:.8f})"
                          for v in ordered)
        span = max(high - min(0, low), 1)
        policy.extend([
            r"\begin{minipage}[t]{0.32\textwidth}\centering",
            r"\begin{tikzpicture}\begin{axis}[",
            r"width=\linewidth,height=3.8cm,xmin=0,xmax=1.55,",
            r"xtick={0.55,1.12},xticklabels={Paired cases,Summary},",
            f"ymin={min(0, low)-0.06*span:.8f},ymax={high+0.15*span:.8f},",
            f"title={{{title}}},ylabel={{{ylabel}}},",
            r"title style={font=\small\bfseries},label style={font=\footnotesize},",
            r"tick label style={font=\scriptsize},ymajorgrids,grid style={black!8},",
            r"axis line style={black!50}]",
            r"\draw[dashed,black!40] (axis cs:0,0)--(axis cs:1.55,0);",
            f"\\addplot[only marks,mark=*,mark size=1.4pt,{color},opacity=0.65] coordinates {{{coords}}};",
            f"\\draw[{color},semithick] (axis cs:1.12,{low})--(axis cs:1.12,{high});",
            f"\\draw[{color},semithick] (axis cs:1.04,{low})--(axis cs:1.20,{low});",
            f"\\draw[{color},semithick] (axis cs:1.04,{high})--(axis cs:1.20,{high});",
            f"\\filldraw[fill={color}!18,draw={color},thick] (axis cs:0.95,{q1}) rectangle (axis cs:1.29,{q3});",
            f"\\draw[{color},very thick] (axis cs:0.95,{median})--(axis cs:1.29,{median});",
            rf"\node[anchor=north,font=\scriptsize\bfseries,text={color}!80!black] at (rel axis cs:0.5,0.98) {{Median: {_number(median,1)}}};",
            r"\end{axis}\end{tikzpicture}\end{minipage}",
        ])
        if title != panels[-1][0]:
            policy.append(r"\hfill")
    policy.extend([
        r"\caption{Paired changes from Weighted to LEX-COS are shown for 48 HCORAP-LC instances.",
        r"Dots show instances, and box plots summarize their distributions.",
        r"Panels (a) and (b) show improvements; panel (c) shows compatibility reduction. Scales differ.}",
        r"\label{fig:policy-effect}",
        r"\Description{Three distributions show fewer continuity violations, less overtime,",
        r"and the percentage reduction in compatibility under LEX-COS.}",
        r"\end{figure*}",
    ])
    lower = min(0, *(s["ci_low"] for s in runtime.values()))
    upper = max(0, *(s["ci_high"] for s in runtime.values()))
    x_min = math.floor(lower / 5.0) * 5
    x_max = math.ceil((upper + 3.0) / 5.0) * 5
    ticks = list(range(int(x_min), int(x_max) + 1, 5))
    ticks_str = "{" + ",".join(str(t) for t in ticks) + "}"
    encoding = [r"% Generated from paired runtimes, not rounded speedup ratios.",
        r"\begin{figure}[tbp]\centering",
        r"\begin{tikzpicture}\begin{axis}[",
        r"width=0.96\linewidth,height=2.7cm,",
        f"xmin={x_min},xmax={x_max},ymin=0.45,ymax=2.65,",
        f"xtick={ticks_str},",
        r"ytick={1,2},yticklabels={LEX-COS,Weighted},",
        r"xlabel={Runtime reduction with Totalizer (\%)},",
        r"label style={font=\footnotesize},tick label style={font=\footnotesize},",
        r"xmajorgrids,grid style={black!8},axis line style={black!50}]",
        r"\draw[dashed,black!40,thick] (axis cs:0,0.45)--(axis cs:0,2.65);",
    ]
    for method, y in (("weighted",2), ("lex-cos",1)):
        s = runtime[method]
        med_val = s['median']
        fill_left = min(0.0, med_val)
        fill_right = max(0.0, med_val)
        if method == "weighted":
            label_anchor, label_shift = "north", "-3pt"
        else:
            label_anchor, label_shift = "south", "3pt"
        encoding.extend([
            f"\\fill[hcorapblue!25] (axis cs:{fill_left},{y-0.18}) rectangle (axis cs:{fill_right},{y+0.18});",
            f"\\draw[hcorapblue,very thick] (axis cs:{s['ci_low']},{y})--(axis cs:{s['ci_high']},{y});",
            f"\\draw[hcorapblue,thick] (axis cs:{s['ci_low']},{y-0.08})--(axis cs:{s['ci_low']},{y+0.08});",
            f"\\draw[hcorapblue,thick] (axis cs:{s['ci_high']},{y-0.08})--(axis cs:{s['ci_high']},{y+0.08});",
            f"\\addplot[mark=*,mark size=2.2pt,hcorapblue] coordinates {{({med_val},{y})}};",
            rf"\node[anchor={label_anchor},yshift={label_shift},font=\small\bfseries,hcorapblue] at (axis cs:{med_val},{y}) {{{_fixed(med_val)}\%}};",
        ])
    encoding.extend([
        r"\end{axis}\end{tikzpicture}",
        r"\caption{Bars show the median paired runtime reduction when Totalizer replaces SN,",
        r"with 95\% bootstrap confidence intervals.}",
        r"\label{fig:encoding-effect}",
        r"\Description{Horizontal bars show the median percentage runtime reduction",
        r"under each policy, with confidence intervals and a zero baseline.}",
        r"\end{figure}",
    ])
    return "\n".join(policy)+"\n", "\n".join(encoding)+"\n"


def generate(
    policy_analysis: Path,
    encoding_analysis: Path,
    commercial_analysis: Path,
    output: Path,
) -> dict[str, Any]:
    policy_validation_path = policy_analysis / "corrected_exact_validation.json"
    policy_pairs_path = policy_analysis / "corrected_pairwise_summary.csv"
    policy_pair_details_path = policy_analysis / "corrected_pairwise_pairs.csv"
    encoding_validation_path = encoding_analysis / "policy_encoding_validation.json"
    encoding_summary_path = encoding_analysis / "policy_encoding_summary.csv"
    encoding_pairs_path = encoding_analysis / "policy_encoding_pairs.csv"
    encoding_contrasts_path = encoding_analysis / "policy_encoding_contrasts.csv"
    commercial_validation_path = commercial_analysis / "full_exact_baseline_validation.json"
    commercial_summary_path = commercial_analysis / "exact_method_summary.csv"
    cross_solver_pairs_path = commercial_analysis / "cross_solver_pairs.csv"

    inputs = (
        policy_validation_path,
        policy_pairs_path,
        policy_pair_details_path,
        encoding_validation_path,
        encoding_summary_path,
        encoding_pairs_path,
        encoding_contrasts_path,
        commercial_validation_path,
        commercial_summary_path,
        cross_solver_pairs_path,
    )
    missing = [str(path) for path in inputs if not path.is_file()]
    if missing:
        raise ValueError("missing required analysis files: " + ", ".join(missing))

    policy_validation = _read_json(policy_validation_path)
    if policy_validation.get("manuscript_eligible") is not True:
        raise ValueError("Corrected-v2 policy analysis is not manuscript eligible")
    encoding_validation = _read_json(encoding_validation_path)
    if encoding_validation.get("evidence_valid") is not True:
        raise ValueError("Policy x Encoding analysis did not pass its evidence gate")
    commercial_validation = _read_json(commercial_validation_path)
    if not all(
        commercial_validation.get(field) is True
        for field in ("evidence_valid", "runtime_comparison_valid")
    ):
        raise ValueError("Full commercial baseline did not pass its evidence gate")
    if (
        commercial_validation.get("maxsat_runs") != 192
        or commercial_validation.get("gurobi_runs") != 96
        or commercial_validation.get("cplex_runs") != 96
    ):
        raise ValueError("full comparison must contain 192 MaxSAT, 96 Gurobi, and 96 CPLEX runs")

    policy_pairs = _read_csv(policy_pairs_path)
    policy_row = _one(
        policy_pairs,
        lambda row: row.get("solver") == "Gurobi"
        and row.get("comparison") == "weighted-to-continuity-first"
        and row.get("left_method") == "weighted"
        and row.get("right_method") == "lex-cos",
        "weighted-to-LEX-COS Gurobi policy comparison",
    )
    if (
        _integer(policy_row["pairs"]) != 48
        or _integer(policy_row["both_optimum_pairs"]) != 48
    ):
        raise ValueError("policy comparison must contain 48 jointly optimal pairs")
    policy_pair_details = [
        row
        for row in _read_csv(policy_pair_details_path)
        if row.get("comparison") == "weighted-to-continuity-first"
        and row.get("left_method") == "weighted"
        and row.get("right_method") == "lex-cos"
        and _truth(row.get("both_optimum"))
    ]
    if len(policy_pair_details) != 48:
        raise ValueError("policy detail file must contain 48 jointly optimal pairs")
    policy_instance_hashes = {
        row.get("instance_sha256") for row in policy_pair_details
    }
    if None in policy_instance_hashes or len(policy_instance_hashes) != 48:
        raise ValueError("policy detail file must contain 48 unique instance hashes")

    continuity_changes = [
        float(row["delta_continuity"]) for row in policy_pair_details
    ]
    overtime_changes = [float(row["delta_overtime"]) for row in policy_pair_details]
    similarity_changes = [
        float(row["delta_similarity"]) for row in policy_pair_details
    ]
    similarity_relative_changes = [
        100 * float(row["delta_similarity"]) / float(row["left_similarity"])
        for row in policy_pair_details
        if float(row["left_similarity"]) != 0
    ]
    if len(similarity_relative_changes) != 48:
        raise ValueError("policy detail file must contain 48 nonzero baseline similarities")
    summary_checks = {
        "continuity_improved": sum(value < 0 for value in continuity_changes)
        == _integer(policy_row["continuity_improved"]),
        "overtime_improved": sum(value < 0 for value in overtime_changes)
        == _integer(policy_row["overtime_decreased"]),
        "median_continuity": math.isclose(
            statistics.median(continuity_changes),
            float(policy_row["median_continuity_change"]),
        ),
        "median_overtime": math.isclose(
            statistics.median(overtime_changes),
            float(policy_row["median_overtime_change"]),
        ),
        "median_similarity": math.isclose(
            statistics.median(similarity_changes),
            float(policy_row["median_similarity_change"]),
        ),
    }
    if not all(summary_checks.values()):
        raise ValueError(f"policy summary/detail mismatch: {summary_checks}")

    encoding_rows = _read_csv(encoding_summary_path)
    contrasts = _read_csv(encoding_contrasts_path)
    cells: dict[tuple[str, str], dict[str, str]] = {}
    for method in POLICIES:
        for encoding in ENCODINGS:
            row = _one(
                encoding_rows,
                lambda candidate, method=method, encoding=encoding: (
                    candidate.get("method") == method
                    and candidate.get("cardinality") == encoding
                    and candidate.get("implied") == "none"
                    and candidate.get("symmetry") == "none"
                ),
                f"{method}/{encoding} summary",
            )
            if _integer(row["runs"]) != 48:
                raise ValueError(f"{method}/{encoding} must contain 48 runs")
            cells[(method, encoding)] = row
        status_fields = ("optimum_runs", "unsat_runs", "proved_runs", "timeout_runs")
        if any(
            _integer(cells[(method, ENCODINGS[0])][field])
            != _integer(cells[(method, ENCODINGS[1])][field])
            for field in status_fields
        ):
            raise ValueError(f"{method} encodings must have matching status counts")

    contrast_by_policy: dict[str, dict[str, str]] = {}
    for method in POLICIES:
        contrast = _one(
            contrasts,
            lambda row, method=method: row.get("method") == method,
            f"{method} encoding contrast",
        )
        if _integer(contrast["pairs"]) != 48:
            raise ValueError(f"{method} contrast must contain 48 pairs")
        contrast_by_policy[method] = contrast

    encoding_pairs = _read_csv(encoding_pairs_path)
    if len(encoding_pairs) != 96:
        raise ValueError("encoding pair file must contain 96 policy-instance pairs")
    runtime_reductions = {
        method: _runtime_reduction_summary([
            row for row in encoding_pairs if row.get("method") == method
        ]) for method in POLICIES
    }
    commercial_summary_rows = _read_csv(commercial_summary_path)
    expected_backends = ("evalmaxsat-totalizer", "gurobi-mip", "cplex-mip")
    commercial_cells: dict[tuple[str, str], dict[str, str]] = {}
    for backend in expected_backends:
        for method in POLICIES:
            row = _one(
                commercial_summary_rows,
                lambda item, backend=backend, method=method: (
                    item.get("backend") == backend and item.get("method") == method
                ),
                f"{backend}/{method} commercial summary",
            )
            if _integer(row["runs"]) != 48:
                raise ValueError(f"{backend}/{method} must contain 48 runs")
            commercial_cells[(backend, method)] = row
    if len(commercial_summary_rows) != 6:
        raise ValueError("commercial summary must contain exactly six solver-policy rows")

    cross_solver_pairs = _read_csv(cross_solver_pairs_path)
    if len(cross_solver_pairs) != 96:
        raise ValueError("full comparison must contain 96 cross-solver pairs")
    if not all(
        _truth(row.get("commercial_status_agreement"))
        and row.get("commercial_objective_agreement_when_optimal", "").lower()
        != "false"
        and row.get("maxsat_status_agreement_when_decided", "").lower() != "false"
        and row.get("maxsat_objective_agreement_when_optimal", "").lower() != "false"
        for row in cross_solver_pairs
    ):
        raise ValueError("cross-solver pair file contains an agreement failure")
    commercial_status_agreements = sum(
        _truth(row["commercial_status_agreement"]) for row in cross_solver_pairs
    )
    commercial_objective_agreements = sum(
        _truth(row.get("commercial_objective_agreement_when_optimal"))
        for row in cross_solver_pairs
    )
    commercial_optimal_cases = sum(
        row.get("gurobi_status") == "OPTIMUM" for row in cross_solver_pairs
    )
    commercial_infeasible_cases = sum(
        row.get("gurobi_status") in {"INFEASIBLE", "UNSAT", "UNSATISFIABLE"}
        for row in cross_solver_pairs
    )
    maxsat_decided_agreements = sum(
        _truth(row.get("maxsat_status_agreement_when_decided"))
        for row in cross_solver_pairs
    )
    maxsat_objective_agreements = sum(
        _truth(row.get("maxsat_objective_agreement_when_optimal"))
        for row in cross_solver_pairs
    )

    completed_medians: dict[tuple[str, str], float] = {}
    pairwise_slowdowns: dict[tuple[str, str], float] = {}
    for method in POLICIES:
        method_rows = [row for row in cross_solver_pairs if row["method"] == method]
        for backend, status_field, elapsed_field in (
            ("evalmaxsat-totalizer", "maxsat_status", "maxsat_elapsed_seconds"),
            ("gurobi-mip", "gurobi_status", "gurobi_elapsed_seconds"),
            ("cplex-mip", "cplex_status", "cplex_elapsed_seconds"),
        ):
            completed = [
                float(row[elapsed_field])
                for row in method_rows
                if row[status_field] in {"OPTIMUM", "UNSAT", "UNSATISFIABLE", "INFEASIBLE"}
            ]
            expected = _integer(commercial_cells[(backend, method)]["proved_runs"])
            if len(completed) != expected:
                raise ValueError(f"{backend}/{method} completed-runtime count disagrees")
            completed_medians[(backend, method)] = statistics.median(completed)
        proved_maxsat = [
            row for row in method_rows
            if row["maxsat_status"] in {"OPTIMUM", "UNSAT", "UNSATISFIABLE"}
        ]
        for backend, elapsed_field in (
            ("gurobi-mip", "gurobi_elapsed_seconds"),
            ("cplex-mip", "cplex_elapsed_seconds"),
        ):
            pairwise_slowdowns[(backend, method)] = statistics.median(
                float(row["maxsat_elapsed_seconds"]) / float(row[elapsed_field])
                for row in proved_maxsat
            )
    size_fields = ("users", "agents", "visits")
    size_strata_support: dict[str, bool] = {}
    for method in POLICIES:
        proved_pairs = [
            row
            for row in encoding_pairs
            if row.get("method") == method and _truth(row.get("both_proved"))
        ]
        if len(proved_pairs) != _integer(
            contrast_by_policy[method]["both_proved_pairs"]
        ):
            raise ValueError(f"{method} pair and contrast counts disagree")
        parsed: list[dict[str, Any]] = []
        for row in proved_pairs:
            match = re.search(
                r"instance_(\d+)_(\d+)_(\d+)_(\d+)\.txt$",
                row.get("instance", ""),
            )
            if match is None:
                raise ValueError(f"cannot parse Original instance size: {row.get('instance')}")
            enriched: dict[str, Any] = dict(row)
            enriched.update(zip((*size_fields, "seed"), match.groups()))
            parsed.append(enriched)
        stratum_medians = [
            statistics.median(
                float(row["speedup_sorting_over_totalizer"])
                for row in parsed
                if row[field] == value
            )
            for field in size_fields
            for value in sorted({row[field] for row in parsed}, key=int)
        ]
        size_strata_support[method] = all(value > 1 for value in stratum_medians)

    output.mkdir(parents=True, exist_ok=True)
    continuity_reduction = -statistics.median(continuity_changes)
    overtime_reduction = -statistics.median(overtime_changes)
    compatibility_change = statistics.median(similarity_changes)
    compatibility_relative_change = statistics.median(
        similarity_relative_changes
    )
    variable_reductions = {
        method: 100
        * (
            float(cells[(method, "sorting-network")]["median_variables"])
            - float(cells[(method, "totalizer")]["median_variables"])
        )
        / float(cells[(method, "sorting-network")]["median_variables"])
        for method in POLICIES
    }
    hard_clause_changes = {
        method: 100
        * (
            float(cells[(method, "totalizer")]["median_hard_clauses"])
            - float(cells[(method, "sorting-network")]["median_hard_clauses"])
        )
        / float(cells[(method, "sorting-network")]["median_hard_clauses"])
        for method in POLICIES
    }
    memory_multipliers = {
        method: float(cells[(method, "totalizer")]["median_peak_rss_mb"])
        / float(cells[(method, "sorting-network")]["median_peak_rss_mb"])
        for method in POLICIES
    }
    hard_clause_multipliers = {
        method: float(cells[(method, "totalizer")]["median_hard_clauses"])
        / float(cells[(method, "sorting-network")]["median_hard_clauses"])
        for method in POLICIES
    }
    par2_reductions = {
        method: 100
        * (
            float(cells[(method, "sorting-network")]["par2_seconds"])
            - float(cells[(method, "totalizer")]["par2_seconds"])
        )
        / float(cells[(method, "sorting-network")]["par2_seconds"])
        for method in POLICIES
    }
    both_improved = sum(
        float(row["delta_continuity"]) < 0 and float(row["delta_overtime"]) < 0
        for row in policy_pair_details
    )
    claim_support = {
        method: _truth(contrast_by_policy[method]["totalizer_faster_claim_supported"])
        for method in POLICIES
    }
    if claim_support["weighted"] and claim_support["lex-cos"]:
        cross_policy_conclusion = (
            "Under both policies, Totalizer proves the same number of runs as "
            "the sorting network and produces lower PAR-2.  Both 95\\% paired "
            "speedup intervals lie above one, so the runtime evidence consistently "
            "favors Totalizer."
        )
        cross_policy_conclusion_short = (
            "The runtime evidence favors Totalizer under both policies."
        )
    elif claim_support["weighted"]:
        cross_policy_conclusion = (
            "Totalizer improves the weighted formulation, but the LEX-COS "
            "evidence does not establish the same advantage.  The encoding "
            "effect is therefore policy-dependent rather than universal."
        )
        cross_policy_conclusion_short = (
            "The Totalizer advantage is limited to the weighted policy in this study."
        )
    elif claim_support["lex-cos"]:
        cross_policy_conclusion = (
            "Totalizer improves the LEX-COS formulation, but the weighted "
            "evidence does not establish the same advantage.  The encoding "
            "effect is therefore policy-dependent rather than universal."
        )
        cross_policy_conclusion_short = (
            "The Totalizer advantage is limited to LEX-COS in this study."
        )
    else:
        cross_policy_conclusion = (
            "Neither policy establishes a consistent Totalizer advantage under "
            "the completion, PAR-2, and paired-runtime criteria.  Encoding "
            "choice remains instance-dependent in this study."
        )
        cross_policy_conclusion_short = (
            "The experiment does not establish a policy-wide Totalizer advantage."
        )
    if all(size_strata_support.values()):
        size_strata_conclusion = (
            "For every grouping by patient count, caregiver count, or services per "
            "patient, the median SN/TOT runtime ratio is above one under both policies."
        )
    else:
        size_strata_conclusion = (
            "The descriptive size strata do not show a uniform encoding effect."
        )

    macro_lines = [
        "% Generated by experiments/generate_compact_manuscript_results.py.",
        "% Do not edit numerical values by hand.",
        _macro("PolicyPairCount", _integer(policy_row["pairs"])),
        _macro(
            "ContinuityImprovedCount",
            _integer(policy_row["continuity_improved"]),
        ),
        _macro("OvertimeImprovedCount", _integer(policy_row["overtime_decreased"])),
        _macro("BothPriorityMeasuresImprovedCount", both_improved),
        _macro("MedianContinuityReduction", _number(continuity_reduction, 1)),
        _macro("MedianOvertimeReduction", _number(overtime_reduction, 1)),
        _macro("MedianCompatibilityChange", _number(compatibility_change, 1)),
        _macro("MedianCompatibilityReduction", _number(-compatibility_change, 1)),
        _macro(
            "MedianCompatibilityRelativeChange",
            _number(compatibility_relative_change, 1),
        ),
        _macro(
            "MedianCompatibilityReductionPercent",
            _number(-compatibility_relative_change, 1),
        ),
        _macro("ContinuityEqualCount", sum(value == 0 for value in continuity_changes)),
        _macro("ContinuityWorsenedCount", sum(value > 0 for value in continuity_changes)),
        _macro("OvertimeEqualCount", sum(value == 0 for value in overtime_changes)),
        _macro("OvertimeWorsenedCount", sum(value > 0 for value in overtime_changes)),
        _macro("CompatibilityImprovedCount", sum(value > 0 for value in similarity_changes)),
        _macro("CompatibilityEqualCount", sum(value == 0 for value in similarity_changes)),
        _macro("CompatibilityWorsenedCount", sum(value < 0 for value in similarity_changes)),
        _macro("ContinuityIqrLow", _number(_percentile(continuity_changes, 0.25), 1)),
        _macro("ContinuityIqrHigh", _number(_percentile(continuity_changes, 0.75), 1)),
        _macro("OvertimeIqrLow", _number(_percentile(overtime_changes, 0.25), 1)),
        _macro("OvertimeIqrHigh", _number(_percentile(overtime_changes, 0.75), 1)),
        _macro("CompatibilityIqrLow", _number(_percentile(similarity_changes, 0.25), 1)),
        _macro("CompatibilityIqrHigh", _number(_percentile(similarity_changes, 0.75), 1)),
        _macro(
            "PolicyEffectCoordinates",
            " ".join(
                "({},{})".format(
                    _number(-float(row["delta_continuity"]), 1),
                    _number(-float(row["delta_overtime"]), 1),
                )
                for row in policy_pair_details
            ),
        ),
        _macro("CommercialComparisonPairCount", len(cross_solver_pairs)),
        _macro("CommercialStatusAgreementCount", commercial_status_agreements),
        _macro("CommercialObjectiveAgreementCount", commercial_objective_agreements),
        _macro("CommercialOptimalCaseCount", commercial_optimal_cases),
        _macro("CommercialInfeasibleCaseCount", commercial_infeasible_cases),
        _macro("MaxsatDecidedAgreementCount", maxsat_decided_agreements),
        _macro("MaxsatObjectiveAgreementCount", maxsat_objective_agreements),
        _macro(
            "WeightedMaxsatOverGurobiMedian",
            _number(pairwise_slowdowns[("gurobi-mip", "weighted")], 0),
        ),
        _macro(
            "WeightedMaxsatOverCplexMedian",
            _number(pairwise_slowdowns[("cplex-mip", "weighted")], 0),
        ),
        _macro(
            "LexCosMaxsatOverGurobiMedian",
            _number(pairwise_slowdowns[("gurobi-mip", "lex-cos")], 0),
        ),
        _macro(
            "LexCosMaxsatOverCplexMedian",
            _number(pairwise_slowdowns[("cplex-mip", "lex-cos")], 0),
        ),
    ]
    for method in POLICIES:
        prefix = "Weighted" if method == "weighted" else "LexCos"
        for suffix, field in (("RuntimeReductionPercent", "median"),
                              ("RuntimeReductionLow", "ci_low"),
                              ("RuntimeReductionHigh", "ci_high")):
            macro_lines.append(_macro(prefix + suffix, _fixed(runtime_reductions[method][field])))
        contrast = contrast_by_policy[method]
        status_row = cells[(method, "totalizer")]
        macro_lines.extend(
            (
                _macro(f"{prefix}OptimalCount", _integer(status_row["optimum_runs"])),
                _macro(f"{prefix}InfeasibleCount", _integer(status_row["unsat_runs"])),
                _macro(f"{prefix}TimeoutCount", _integer(status_row["timeout_runs"])),
                _macro(
                    f"{prefix}EncodingPairCount",
                    _integer(contrast["both_proved_pairs"]),
                ),
                _macro(
                    f"{prefix}TotalizerFasterCount",
                    _integer(contrast["totalizer_faster"]),
                ),
                _macro(
                    f"{prefix}MedianEncodingSpeedup",
                    _number(contrast["median_speedup_sorting_over_totalizer"]),
                ),
                _macro(
                    f"{prefix}EncodingSpeedupLow",
                    _number(contrast["bootstrap_95_ci_low"]),
                ),
                _macro(
                    f"{prefix}EncodingSpeedupHigh",
                    _number(contrast["bootstrap_95_ci_high"]),
                ),
                _macro(
                    f"{prefix}VariableReductionPercent",
                    _number(variable_reductions[method], 0),
                ),
                _macro(
                    f"{prefix}HardClauseChangePercent",
                    _number(hard_clause_changes[method], 0),
                ),
                _macro(
                    f"{prefix}HardClauseMultiplier",
                    _number(hard_clause_multipliers[method], 1),
                ),
                _macro(
                    f"{prefix}MemoryMultiplier",
                    _number(memory_multipliers[method], 1),
                ),
                _macro(
                    f"{prefix}ParTwoReductionPercent",
                    _number(par2_reductions[method], 1),
                ),
                _macro(
                    f"{prefix}RuntimeIncreasePercent",
                    _number(max(0.0, -runtime_reductions[method]["median"]), 1),
                ),
                _macro(
                    f"{prefix}ParTwoIncreasePercent",
                    _number(max(0.0, -par2_reductions[method]), 1),
                ),
            )
        )
    macro_lines.append(
        _macro("CrossPolicyEncodingConclusion", cross_policy_conclusion)
    )
    macro_lines.append(
        _macro("CrossPolicyEncodingConclusionShort", cross_policy_conclusion_short)
    )
    macro_lines.append(_macro("EncodingSizeStrataConclusion", size_strata_conclusion))
    macros_path = output / "compact_result_macros.tex"
    macros_path.write_text("\n".join(macro_lines) + "\n", encoding="utf-8")

    policy_table_lines = [
        "% Generated by experiments/generate_compact_manuscript_results.py.",
        r"\begin{table}[tbp]",
        rf"\caption{{Care outcomes under LEX-COS compared with Weighted on {len(policy_pair_details)} "
        r"HCORAP-LC instances.}",
        r"\label{tab:policy}",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{7pt}",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r" & & \multicolumn{3}{c}{Instances} & \multicolumn{2}{c}{Reduction from Weighted}\\",
        r"\cmidrule(lr){3-5}\cmidrule(l){6-7}",
        r"Care outcome & Preferred value & Better & Unchanged & Worse & Median & Middle 50\%\\",
        r"\midrule",
    ]
    policy_table_rows = (
        ("Continuity violations", "Lower", continuity_changes,
         [-value for value in continuity_changes]),
        ("Overtime (service units)", "Lower", overtime_changes,
         [-value for value in overtime_changes]),
        (r"Compatibility (\%)", "Higher", similarity_changes,
         [-value for value in similarity_relative_changes]),
    )
    for label, preferred, changes, reductions in policy_table_rows:
        better = sum(value < 0 if preferred == "Lower" else value > 0 for value in changes)
        worse = sum(value > 0 if preferred == "Lower" else value < 0 for value in changes)
        policy_table_lines.append(
            f"{label} & {preferred} & {better} & {sum(value == 0 for value in changes)} & {worse} & "
            f"{_fixed(statistics.median(reductions))} & "
            f"[{_fixed(_percentile(reductions, 0.25))}, {_fixed(_percentile(reductions, 0.75))}]"
            + r"\\"
        )
    policy_table_lines.extend((
        r"\bottomrule", r"\end{tabular}",
        r"\par\smallskip\begin{minipage}{0.92\linewidth}\footnotesize",
        r"Compatibility reductions are relative to each instance's Weighted SIM score. "
        r"The middle 50\% spans the 25th to 75th percentiles.",
        r"\end{minipage}",
        r"\end{table}",
    ))
    policy_table_path = output / "compact_policy_table.tex"
    policy_table_path.write_text(
        "\n".join(policy_table_lines) + "\n", encoding="utf-8"
    )

    table_lines = [
        "% Generated by experiments/generate_compact_manuscript_results.py.",
        r"\begin{table}[tbp]",
        r"\caption{EvalMaxSAT performance on 48 Original instances per configuration.}",
        r"\label{tab:policy-encoding}",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}llccc@{}}",
        r"\toprule",
        r" & & Result counts & \multicolumn{2}{c}{Runtime (s)}\\",
        r"\cmidrule(lr){3-3}\cmidrule(l){4-5}",
        r"Policy & Encoding & Opt / Inf / TO & PAR-2 & Median\\",
        r"\midrule",
    ]
    for method in POLICIES:
        sn_row = cells[(method, "sorting-network")]
        tot_row = cells[(method, "totalizer")]

        def _pair(val_sn: float, val_tot: float, fmt_fn: Any, lower_is_better: bool = True) -> tuple[str, str]:
            str_sn = fmt_fn(val_sn)
            str_tot = fmt_fn(val_tot)
            if math.isclose(float(val_sn), float(val_tot), rel_tol=1e-7, abs_tol=1e-7):
                return str_sn, str_tot
            sn_better = (val_sn < val_tot) if lower_is_better else (val_sn > val_tot)
            if sn_better:
                return rf"\textbf{{{str_sn}}}", str_tot
            return str_sn, rf"\textbf{{{str_tot}}}"

        opt_s, opt_t = _pair(_integer(sn_row["optimum_runs"]), _integer(tot_row["optimum_runs"]), str, False)
        unsat_s, unsat_t = _pair(_integer(sn_row["unsat_runs"]), _integer(tot_row["unsat_runs"]), str, False)
        to_s, to_t = _pair(_integer(sn_row["timeout_runs"]), _integer(tot_row["timeout_runs"]), str, True)
        par2_s, par2_t = _pair(float(sn_row["par2_seconds"]), float(tot_row["par2_seconds"]), _fixed, True)
        med_s, med_t = _pair(float(sn_row["median_proved_seconds"]), float(tot_row["median_proved_seconds"]), _fixed, True)
        table_lines.append(
            f"{_policy_label(method)} & SN & {opt_s} / {unsat_s} / {to_s} & "
            f"{par2_s} & {med_s}\\\\"
        )
        table_lines.append(
            f"{_policy_label(method)} & TOT & {opt_t} / {unsat_t} / {to_t} & "
            f"{par2_t} & {med_t}\\\\"
        )
        if method != POLICIES[-1]:
            table_lines.append(r"\addlinespace")
    table_lines.extend((
        r"\bottomrule", r"\end{tabular*}",
        r"\par\smallskip\begin{minipage}{\linewidth}\footnotesize",
        r"Opt, Inf, and TO denote optimal, infeasible, and timed-out runs, respectively. "
        r"Median runtime includes proved runs; PAR-2 includes all 48 runs. "
        r"Bold values mark the lower runtime within each policy.",
        r"\end{minipage}", r"\end{table}",
    ))
    table_path = output / "compact_encoding_table.tex"
    table_path.write_text("\n".join(table_lines) + "\n", encoding="utf-8")

    backend_labels = {
        "evalmaxsat-totalizer": "EvalMaxSAT",
        "gurobi-mip": "Gurobi",
        "cplex-mip": "CPLEX",
    }
    commercial_table_lines = [
        "% Generated by experiments/generate_compact_manuscript_results.py.",
        r"\begin{table}[tbp]",
        r"\caption{Exact-solver performance on 48 Original instances per policy.}",
        r"\label{tab:exact-solvers}",
        r"\centering", r"\small", r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{1.12}",
        r"\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}llccc@{}}",
        r"\toprule",
        r" & & Result counts & \multicolumn{2}{c}{Runtime (s)}\\",
        r"\cmidrule(lr){3-3}\cmidrule(l){4-5}",
        r"Solver & Policy & Opt / Inf / TO & PAR-2 & Median\\",
        r"\midrule",
    ]
    best_par2_by_policy = {
        method: min(float(commercial_cells[(b, method)]["par2_seconds"]) for b in expected_backends)
        for method in POLICIES
    }
    best_med_by_policy = {
        method: min(float(completed_medians[(b, method)]) for b in expected_backends)
        for method in POLICIES
    }
    for backend in expected_backends:
        for method in POLICIES:
            row = commercial_cells[(backend, method)]
            par2_val = float(row["par2_seconds"])
            med_val = completed_medians[(backend, method)]
            par2_str = f"{par2_val:.1f}" if par2_val >= 10.0 else f"{par2_val:.2f}"
            med_str = f"{med_val:.1f}" if med_val >= 10.0 else f"{med_val:.2f}"
            if math.isclose(par2_val, best_par2_by_policy[method], rel_tol=1e-5, abs_tol=1e-5):
                par2_str = rf"\textbf{{{par2_str}}}"
            if math.isclose(med_val, best_med_by_policy[method], rel_tol=1e-5, abs_tol=1e-5):
                med_str = rf"\textbf{{{med_str}}}"
            commercial_table_lines.append(
                "{} & {} & {} / {} / {} & {} & {}\\\\".format(
                    backend_labels[backend], _policy_label(method),
                    _integer(row["optimal_runs"]), _integer(row["infeasible_runs"]),
                    _integer(row["timeout_runs"]), par2_str, med_str,
                )
            )
        if backend != expected_backends[-1]:
            commercial_table_lines.append(r"\addlinespace")
    commercial_table_lines.extend((
        r"\bottomrule", r"\end{tabular*}",
        r"\par\smallskip\begin{minipage}{\linewidth}\footnotesize",
        r"Opt, Inf, and TO denote optimal, infeasible, and timed-out runs, respectively. "
        r"EvalMaxSAT uses Totalizer. Median runtime includes proved runs; "
        r"PAR-2 includes all 48 runs. Bold values mark the fastest runtime within each policy.",
        r"\end{minipage}", r"\end{table}",
    ))
    commercial_table_path = output / "compact_commercial_table.tex"
    commercial_table_path.write_text(
        "\n".join(commercial_table_lines) + "\n", encoding="utf-8"
    )

    policy_figure, encoding_figure = _figure_fragments(
        [[-v for v in continuity_changes], [-v for v in overtime_changes],
         [-v for v in similarity_relative_changes]], runtime_reductions
    )
    figure_paths = []
    for name, content in (("compact_policy_figure.tex", policy_figure),
                          ("compact_encoding_figure.tex", encoding_figure)):
        path = output / name
        path.write_text(content, encoding="utf-8")
        figure_paths.append(path)
    figure_statistics = output / "compact_figure_statistics.json"
    figure_statistics.write_text(json.dumps({
        "runtime_reduction_formula": "100 * (sorting_elapsed_seconds - totalizer_elapsed_seconds) / sorting_elapsed_seconds",
        "bootstrap_samples": 10000, "bootstrap_seed": 20260908,
        "bootstrap_unit": "jointly proved instance pair within each policy",
        "confidence_interval": "percentile 95% for median runtime reduction",
        "runtime_reductions": runtime_reductions,
        "policy_box_whiskers": "minimum and maximum; box is 25th to 75th percentile",
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    figure_paths.append(figure_statistics)
    provenance = {
        "schema_version": 1,
        "scope": "two-study-compact-manuscript-results",
        "policy_gate": True,
        "encoding_gate": True,
        "commercial_gate": True,
        "totalizer_claim_supported": claim_support,
        "inputs": {
            _portable(path): _sha256(path)
            for path in inputs
        },
        "outputs": {
            macros_path.name: _sha256(macros_path),
            policy_table_path.name: _sha256(policy_table_path),
            table_path.name: _sha256(table_path),
            commercial_table_path.name: _sha256(commercial_table_path),
            **{path.name: _sha256(path) for path in figure_paths},
        },
    }
    provenance_path = output / "compact_result_provenance.json"
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return provenance


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-analysis", type=Path, required=True)
    parser.add_argument("--encoding-analysis", type=Path, required=True)
    parser.add_argument("--commercial-analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        report = generate(
            arguments.policy_analysis.resolve(),
            arguments.encoding_analysis.resolve(),
            arguments.commercial_analysis.resolve(),
            arguments.output.resolve(),
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
