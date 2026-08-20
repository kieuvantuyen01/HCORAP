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
from pathlib import Path
from typing import Any, Iterable


BASELINE = ("sorting-network", "none", "none")
REFERENCE = ("totalizer", "both", "slot-service")
CONFIG_KEYS = ("cardinality", "implied", "symmetry")
FACTORIAL_ORDER = (
    ("sorting-network", "none", "none"),
    ("sorting-network", "none", "slot-service"),
    ("sorting-network", "both", "none"),
    ("sorting-network", "both", "slot-service"),
    ("totalizer", "none", "none"),
    ("totalizer", "none", "slot-service"),
    ("totalizer", "both", "none"),
    ("totalizer", "both", "slot-service"),
)
SELECTED_CONTRASTS = (
    ("encoding", "IC=none;SB=none", "SN $\\to$ TOT", "IC off, SB off"),
    (
        "encoding",
        "IC=both;SB=slot-service",
        "SN $\\to$ TOT",
        "IC on, SB on",
    ),
    ("implied", "Enc=totalizer;SB=none", "IC off $\\to$ on", "TOT, SB off"),
    ("symmetry", "Enc=totalizer;IC=both", "SB off $\\to$ on", "TOT, IC on"),
)


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
    rendered = f"{number:.{digits}f}".rstrip("0").rstrip(".")
    return "0" if rendered == "-0" else rendered


def _count(value: Any) -> str:
    return f"{_int(value):,}".replace(",", r"{,}")


def _configuration(row: dict[str, str]) -> tuple[str, str, str]:
    return tuple(row[key] for key in CONFIG_KEYS)  # type: ignore[return-value]


def _config_label(configuration: tuple[str, str, str]) -> str:
    if configuration == BASELINE:
        return "Baseline"
    if configuration == REFERENCE:
        return "Enhanced"
    raise ValueError(f"unexpected policy configuration: {configuration}")


def _factorial_label(configuration: tuple[str, str, str]) -> str:
    encoding = "SN" if configuration[0] == "sorting-network" else "TOT"
    implied = "on" if configuration[1] == "both" else "off"
    symmetry = "on" if configuration[2] == "slot-service" else "off"
    return f"{encoding}-{implied}-{symmetry}"


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


def _factorial_table(
    factorial: list[dict[str, str]],
    contrasts: list[dict[str, str]],
    composite: dict[str, str],
) -> str:
    indexed = {_configuration(row): row for row in factorial}
    if set(indexed) != set(FACTORIAL_ORDER):
        raise ValueError("factorial summary does not contain exactly eight cells")
    cell_lines = []
    for configuration in FACTORIAL_ORDER:
        row = indexed[configuration]
        encoding = "SN" if configuration[0] == "sorting-network" else "TOT"
        implied = "on" if configuration[1] == "both" else "off"
        symmetry = "on" if configuration[2] == "slot-service" else "off"
        cell_lines.append(
            "    "
            + " & ".join(
                (
                    _factorial_label(configuration),
                    encoding,
                    implied,
                    symmetry,
                    _count(row["optimum_runs"]),
                    _count(row["unsat_runs"]),
                    _count(row["timeout_runs"]),
                    _fmt(row["par2_seconds"]),
                    _fmt(row["median_peak_rss_mb"]),
                )
            )
            + r" \\"
        )

    contrast_lines = []
    for factor, condition, label, display_condition in SELECTED_CONTRASTS:
        row = _one(
            contrasts,
            lambda item, f=factor, c=condition: item["factor"] == f
            and item["condition"] == c,
            f"{factor}/{condition} contrast",
        )
        wins = "/".join(
            _count(row[key]) for key in ("right_faster", "ties", "left_faster")
        )
        interval = (
            f"{_fmt(row['median_speedup_left_over_right'])} "
            f"[{_fmt(row['bootstrap_95_ci_low'])}, "
            f"{_fmt(row['bootstrap_95_ci_high'])}]"
        )
        contrast_lines.append(
            "    "
            + " & ".join(
                (
                    label,
                    rf"\multicolumn{{2}}{{l}}{{{display_condition}}}",
                    _count(row["both_proved_pairs"]),
                    wins,
                    rf"\multicolumn{{2}}{{c}}{{{interval}}}",
                    _fmt(row["median_variables_difference"]),
                    _fmt(row["median_hard_clauses_difference"]),
                )
            )
            + r" \\"
        )
    composite_wins = "/".join(
        _count(composite[key])
        for key in ("reference_faster", "ties", "baseline_faster")
    )
    composite_interval = (
        f"{_fmt(composite['median_speedup_baseline_over_reference'])} "
        f"[{_fmt(composite['bootstrap_95_ci_low'])}, "
        f"{_fmt(composite['bootstrap_95_ci_high'])}]"
    )
    contrast_lines.append(
        "    "
        + " & ".join(
            (
                r"Baseline $\to$ enhanced",
                r"\multicolumn{2}{l}{end-to-end, $n=48$}",
                _count(composite["both_proved_pairs"]),
                composite_wins,
                rf"\multicolumn{{2}}{{c}}{{{composite_interval}}}",
                "--",
                "--",
            )
        )
        + r" \\"
    )

    return rf"""
\begin{{table*}}[t]
  \caption{{Factorial and end-to-end encoding results on 48 original instances
  ($T=300$\,s). Panel~A reports all eight configurations; Panel~B reports the
  direct factor and baseline--enhanced comparisons. SN denotes sorting network,
  TOT Totalizer, IC implied constraints, SB symmetry breaking, W/T/L
  wins/ties/losses, and CI confidence interval. \textsc{{Opt}},
  \textsc{{Unsat}}, and TO denote optimal, infeasible, and timeout runs. Speedup
  is left runtime divided by right runtime; PAR-2 and RSS are lower-is-better.}}
  \label{{tab:factorial}}
  \centering
  \scriptsize
  \setlength{{\tabcolsep}}{{3.5pt}}
  \renewcommand{{\arraystretch}}{{0.94}}
  \begin{{tabularx}}{{\textwidth}}{{@{{}}Xlllrrrrr@{{}}}}
    \toprule
    \multicolumn{{9}}{{@{{}}l}}{{\textit{{Panel A: factorial cells and all-run performance}}}} \\
    Configuration & Counting & IC & SB & \textsc{{Opt}} & \textsc{{Unsat}} & TO & PAR-2 (s) & RSS (MB) \\
    \midrule
{chr(10).join(cell_lines)}
    \midrule
    \multicolumn{{9}}{{@{{}}l}}{{\textit{{Panel B: paired factor comparisons and baseline--enhanced validation}}}} \\
    Contrast (left $\to$ right) & \multicolumn{{2}}{{c}}{{Condition}} & Pairs & W/T/L & \multicolumn{{2}}{{c}}{{Median speedup [95\% CI]}} & $\Delta$variables & $\Delta$hard clauses \\
    \midrule
{chr(10).join(contrast_lines)}
    \bottomrule
  \end{{tabularx}}
\end{{table*}}
"""


def _policy_table(
    *,
    lex_rows: list[dict[str, str]],
    sensitivity_rows: list[dict[str, str]],
    corrected_rows: list[dict[str, str]],
    cross_rows: list[dict[str, str]],
    original_enabled: bool,
    corrected_enabled: bool,
) -> str:
    lines = [
        r"\begin{table*}[t]",
        r"  \caption{Objective-rule and cross-solver results. Panels~A and B",
        r"  compare objective rules on the original and corrected benchmarks;",
        r"  deltas are second rule minus first, with lower CONT/OT and higher SIM",
        r"  preferred. Panel~C reports agreement where EvalMaxSAT, Gurobi, and",
        r"  CPLEX all prove optimality. PAR-2 includes timeouts and is lower-is-better.}",
        r"  \label{tab:policy-validation}",
        r"  \centering",
        r"  \scriptsize",
        r"  \setlength{\tabcolsep}{3.5pt}",
        r"  \renewcommand{\arraystretch}{0.94}",
        r"  \begin{tabularx}{\textwidth}{@{}Xlrrrrrrr@{}}",
        r"    \toprule",
    ]
    if original_enabled:
        lines.extend(
            (
                r"    \multicolumn{9}{@{}l}{\textit{Panel A: paired objective-rule effects}} \\",
                r"    Comparison & Encoding & $n$ & Proved 1/2 & Both optimal & $\Delta$SIM & $\Delta$CONT & $\Delta$OT & PAR-2 1/2 (s) \\",
                r"    \midrule",
            )
        )
        for row in sorted(lex_rows, key=lambda item: _config_label(_configuration(item))):
            lines.append(
                "    "
                + " & ".join(
                    (
                        r"Weighted $\to$ continuity-first (original)",
                        _config_label(_configuration(row)),
                        _count(row["pairs"]),
                        f"{_count(row['weighted_proved_runs'])}/"
                        f"{_count(row['lex_cos_proved_runs'])}",
                        _count(row["both_optimum_pairs"]),
                        _fmt(row["median_similarity_change"]),
                        _fmt(row["median_continuity_change"]),
                        _fmt(row["median_overtime_change"]),
                        f"{_fmt(row['weighted_par2_seconds'])}/"
                        f"{_fmt(row['lex_cos_par2_seconds'])}",
                    )
                )
                + r" \\"
            )
        for row in sorted(
            sensitivity_rows,
            key=lambda item: _config_label(_configuration(item)),
        ):
            lines.append(
                "    "
                + " & ".join(
                    (
                        r"Continuity-first $\to$ overtime-first (corrected)",
                        _config_label(_configuration(row)),
                        _count(row["pairs"]),
                        f"{_count(row['lex_cos_proved_runs'])}/"
                        f"{_count(row['lex_ocs_proved_runs'])}",
                        _count(row["both_optimum_pairs"]),
                        _fmt(row["median_similarity_change"]),
                        _fmt(row["median_continuity_change"]),
                        _fmt(row["median_overtime_change"]),
                        f"{_fmt(row['lex_cos_par2_seconds'])}/"
                        f"{_fmt(row['lex_ocs_par2_seconds'])}",
                    )
                )
                + r" \\"
            )
        lines.append(r"    \midrule")

    if corrected_enabled:
        lines.extend(
            (
                r"    \multicolumn{9}{@{}l}{\textit{Panel B: corrected-benchmark confirmatory set, enhanced encoding}} \\",
                r"    Objective rule & Encoding & Runs & \textsc{Opt} & TO & Median SIM & Median CONT & Median OT & PAR-2 (s) \\",
                r"    \midrule",
            )
        )
        method_labels = {"weighted": "Weighted", "lex-cos": "Continuity-first"}
        for method in ("weighted", "lex-cos"):
            row = _one(
                corrected_rows,
                lambda item, selected=method: item["method"] == selected,
                f"corrected-v2 {method} row",
            )
            lines.append(
                "    "
                + " & ".join(
                    (
                        method_labels[method],
                        "Enhanced",
                        _count(row["runs"]),
                        _count(row["optimum_runs"]),
                        _count(row["timeout_runs"]),
                        _fmt(row["median_similarity"]),
                        _fmt(row["median_continuity"]),
                        _fmt(row["median_overtime"]),
                        _fmt(row["par2_seconds"]),
                    )
                )
                + r" \\"
            )
        lines.append(r"    \midrule")

    lines.extend(
        (
            r"    \multicolumn{9}{@{}l}{\textit{Panel C: objective agreement on the 20-instance commercial subset}} \\",
            r"    Three-solver comparison & Objective rule & Groups & All optimal & Agree & Disagree & \multicolumn{3}{c}{Compared measures} \\",
            r"    \midrule",
        )
    )
    method_labels = {"weighted": "Weighted", "lex-cos": "Continuity-first"}
    objective_labels = {
        "weighted": "coverage, weighted score",
        "lex-cos": "coverage, SIM, CONT, OT",
    }
    methods = ("weighted", "lex-cos") if original_enabled else ("weighted",)
    for method in methods:
        group = [row for row in cross_rows if row["method"] == method]
        if len(group) != 20:
            raise ValueError(f"expected 20 cross-paradigm {method} groups")
        all_exact = sum(_truth(row["all_exact_optimum"]) for row in group)
        agree = sum(_truth(row["objective_agreement"]) for row in group)
        disagree = sum(
            str(row["objective_agreement"]).lower() == "false" for row in group
        )
        lines.append(
            "    "
            + " & ".join(
                (
                    "EvalMaxSAT / Gurobi / CPLEX",
                    method_labels[method],
                    _count(len(group)),
                    _count(all_exact),
                    _count(agree),
                    _count(disagree),
                    rf"\multicolumn{{3}}{{c}}{{{objective_labels[method]}}}",
                )
            )
            + r" \\"
        )
    lines.extend((r"    \bottomrule", r"  \end{tabularx}", r"\end{table*}"))
    return "\n".join(lines) + "\n"


def _policy_prose(
    lex_rows: list[dict[str, str]],
    sensitivity_rows: list[dict[str, str]],
    original_enabled: bool,
) -> str:
    if not original_enabled:
        return (
            "The original-benchmark objective comparison was not run; RQ1 "
            "therefore uses the corrected benchmark.\n"
        )
    sentences = []
    for configuration in (REFERENCE,):
        label = _config_label(configuration)
        row = _one(
            lex_rows,
            lambda item, selected=configuration: _configuration(item) == selected,
            f"LEX-COS configuration {label}",
        )
        sentences.append(
            (
                f"Under ${label}$, {_count(row['both_optimum_pairs'])}/"
                f"{_count(row['pairs'])} pairs are jointly optimum; the "
                f"continuity-first objective changes median SIM, CONT, and OT by "
                f"{_fmt(row['median_similarity_change'])}, "
                f"{_fmt(row['median_continuity_change'])}, and "
                f"{_fmt(row['median_overtime_change'])}, respectively."
                if _int(row["both_optimum_pairs"]) > 0
                else (
                    f"Under ${label}$, no matched pair completes both objective "
                    "rules; paired objective deltas are unavailable."
                )
            )
        )
    same = sum(_int(row["same_objective_vector_pairs"]) for row in sensitivity_rows)
    both = sum(_int(row["both_optimum_pairs"]) for row in sensitivity_rows)
    sentences.append(
        (
            f"On the corrected benchmark, the continuity-first and overtime-first objectives yield the same "
            f"objective vector on "
            f"{_count(same)}/{_count(both)} jointly optimum sensitivity pairs."
            if both
            else "No sensitivity pair completes both lexicographic orders."
        )
    )
    return "  ".join(sentences) + "\n"


def _render(
    *,
    screening: dict[str, Any],
    factorial: list[dict[str, str]],
    contrasts: list[dict[str, str]],
    composite: dict[str, str],
    lex_rows: list[dict[str, str]],
    sensitivity_rows: list[dict[str, str]],
    corrected_rows: list[dict[str, str]],
    corrected_paired: dict[str, str] | None,
    cross_rows: list[dict[str, str]],
    original_enabled: bool,
    corrected_enabled: bool,
) -> tuple[str, str, str]:
    encoding = [row for row in contrasts if row["factor"] == "encoding"]
    implied = [row for row in contrasts if row["factor"] == "implied"]
    symmetry = [row for row in contrasts if row["factor"] == "symmetry"]
    if (len(encoding), len(implied), len(symmetry)) != (4, 4, 4):
        raise ValueError("expected four direct contrasts for each factorial factor")
    enc_speed = _range(encoding, "median_speedup_left_over_right")
    enc_vars = _range(encoding, "median_variables_difference")
    enc_hard = _range(encoding, "median_hard_clauses_difference")
    ic_speed = _range(implied, "median_speedup_left_over_right")
    sb_speed = _range(symmetry, "median_speedup_left_over_right")
    enc_wins = sum(_int(row["right_faster"]) for row in encoding)
    enc_losses = sum(_int(row["left_faster"]) for row in encoding)

    exact_groups = sum(_truth(row["all_exact_optimum"]) for row in cross_rows)
    agreement_groups = sum(_truth(row["objective_agreement"]) for row in cross_rows)
    disagreements = sum(
        str(row["objective_agreement"]).lower() == "false" for row in cross_rows
    )
    policy_prose = _policy_prose(lex_rows, sensitivity_rows, original_enabled)
    corrected_prose = ""
    if corrected_enabled:
        if corrected_paired is None:
            raise ValueError("enabled corrected-v2 branch lacks paired summary")
        if _int(corrected_paired["both_optimum_pairs"]) > 0:
            corrected_prose = (
                "On the corrected benchmark, the continuity-first objective "
                "changes median SIM, CONT, and OT by "
                f"{_fmt(corrected_paired['median_similarity_change'])}, "
                f"{_fmt(corrected_paired['median_continuity_change'])}, and "
                f"{_fmt(corrected_paired['median_overtime_change'])} over "
                f"{_count(corrected_paired['both_optimum_pairs'])} jointly "
                "optimum pairs.  "
            )
        else:
            corrected_prose = (
                "No corrected-benchmark pair completes both objective rules; "
                "paired objective deltas are unavailable.  "
            )
    else:
        corrected_prose = (
            "The corrected-benchmark comparison was not run.  "
        )

    factorial_table = _factorial_table(factorial, contrasts, composite)
    policy_table = _policy_table(
        lex_rows=lex_rows,
        sensitivity_rows=sensitivity_rows,
        corrected_rows=corrected_rows,
        cross_rows=cross_rows,
        original_enabled=original_enabled,
        corrected_enabled=corrected_enabled,
    )

    results = rf"""% Generated from validator-approved campaign summaries. Do not edit.
\section{{Results}}
\label{{sec:results}}

The compact campaign contains {_count(screening['expected_measured_runs'])}
measured runs.  Every reported optimal assignment passes independent
verification.

\subsection{{RQ1: Objective-rule effects}}
\label{{sec:rq1}}

{policy_prose.rstrip()}

\subsection{{RQ2: Totalizer encoding}}
\label{{sec:rq2}}

Across the four direct sorting-network-to-Totalizer comparisons, median runtime
ratios among pairs proved by both configurations {_range_wording(enc_speed)};
Totalizer is faster on {_count(enc_wins)} such pairs and sorting
networks on {_count(enc_losses)}.  Median formula-size changes range from {enc_vars[0]} to
{enc_vars[1]} variables and from {enc_hard[0]} to {enc_hard[1]} hard clauses.
The conditional contrasts and their uncertainty intervals are reported in
Table~\ref{{tab:factorial}}B.

\subsection{{RQ3: Constraint strengthening and interactions}}
\label{{sec:rq3}}

The four direct implied-constraint comparisons have median paired runtime ratios
that {_range_wording(ic_speed)}, while the four symmetry-breaking comparisons
{_range_wording(sb_speed)}.  The 48-instance end-to-end
baseline--enhanced comparison has a median ratio of
{_fmt(composite['median_speedup_baseline_over_reference'])}
(95\% bootstrap confidence interval [{_fmt(composite['bootstrap_95_ci_low'])},
{_fmt(composite['bootstrap_95_ci_high'])}]) over
{_count(composite['both_proved_pairs'])} pairs proved by both configurations.
Table~\ref{{tab:factorial}}B separates the three component effects.

{factorial_table.rstrip()}

\subsection{{Validation and scope}}
\label{{sec:validation}}

{corrected_prose}Among {_count(exact_groups)} commercial-subset groups where
all three solvers prove optimum, {_count(agreement_groups)} objective vectors
agree and {_count(disagreements)} disagree.

{policy_table.rstrip()}
"""

    if original_enabled:
        reference_lex = _one(
            lex_rows,
            lambda row: _configuration(row) == REFERENCE,
            "reference LEX-COS summary",
        )
        if _int(reference_lex["both_optimum_pairs"]) > 0:
            policy_finding = (
                "Under the enhanced encoding, the continuity-first objective "
                "changes median CONT and "
                f"OT by {_fmt(reference_lex['median_continuity_change'])} and "
                f"{_fmt(reference_lex['median_overtime_change'])} over "
                f"{_count(reference_lex['both_optimum_pairs'])} jointly optimum "
                "pairs."
            )
        else:
            policy_finding = (
                "Under the enhanced encoding, no matched pair completes both "
                "objective rules; paired differences are unavailable."
            )
    elif corrected_enabled and corrected_paired is not None:
        if _int(corrected_paired["both_optimum_pairs"]) > 0:
            policy_finding = (
                "On the corrected benchmark, the continuity-first objective "
                "changes median CONT and OT by "
                f"{_fmt(corrected_paired['median_continuity_change'])} and "
                f"{_fmt(corrected_paired['median_overtime_change'])} over "
                f"{_count(corrected_paired['both_optimum_pairs'])} jointly "
                "optimum pairs."
            )
        else:
            policy_finding = (
                "No corrected-benchmark pair completes both objective rules; "
                "paired differences are unavailable."
            )
    else:
        policy_finding = (
            "Objective-rule results are unavailable because neither evaluation "
            "branch was run."
        )
    abstract = (
        "% Generated from validator-approved campaign summaries. Do not edit.\n"
        "Across four direct encoding contrasts, the median paired "
        "sorting-network-to-Totalizer runtime ratios "
        f"{_range_wording(enc_speed)}.  {policy_finding}  Among "
        f"{_count(exact_groups)} "
        "commercial-subset groups where EvalMaxSAT, Gurobi, and CPLEX all prove "
        f"optimum, {_count(agreement_groups)} objective vectors agree and "
        f"{_count(disagreements)} disagree.\n"
    )

    eligible_encoding = [
        row
        for row in encoding
        if row.get("bootstrap_95_ci_low") not in (None, "")
        and row.get("bootstrap_95_ci_high") not in (None, "")
    ]
    if len(eligible_encoding) == 4 and all(
        _float(row["bootstrap_95_ci_low"]) > 1 for row in eligible_encoding
    ):
        effect = "consistently favors Totalizer in the direct encoding contrasts"
    elif len(eligible_encoding) == 4 and all(
        _float(row["bootstrap_95_ci_high"]) < 1 for row in eligible_encoding
    ):
        effect = "consistently favors sorting networks in the direct encoding contrasts"
    elif eligible_encoding:
        effect = "is mixed or only partially resolved across the direct contrasts"
    else:
        effect = "is unavailable because no direct contrast completes both configurations"
    conclusion = rf"""% Generated from validator-approved campaign summaries. Do not edit.
\section{{Conclusion}}
\label{{sec:conclusion}}

We evaluated lexicographic MaxSAT objectives together with Totalizer encoding,
implied constraints, and symmetry breaking for HCORAP.  The encoding effect
{effect}: median paired ratios
{_range_wording(enc_speed)}, and the implied-constraint and symmetry effects
also vary by context.  {policy_finding}  Objective vectors agree in
{_count(agreement_groups)}/{_count(exact_groups)} groups in which EvalMaxSAT,
Gurobi, and CPLEX all prove optimum.  Future work will extend the model to
routing and uncertainty and evaluate it on operational data.
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
    contrasts = primary_csv("factorial_contrasts.csv")
    composite_rows = primary_csv("weighted_composite_paired_summary.csv")
    if len(composite_rows) != 1:
        raise ValueError("weighted composite paired summary must have one row")
    composite = composite_rows[0]
    lex_rows: list[dict[str, str]] = []
    sensitivity_rows: list[dict[str, str]] = []
    if original_enabled:
        lex_rows = primary_csv("lex_confirmatory_summary.csv")
        sensitivity_rows = primary_csv("lex_policy_sensitivity_summary.csv")
        if len(lex_rows) != 1 or len(sensitivity_rows) != 1:
            raise ValueError("compact policy scope requires one reference summary row")

    corrected_rows: list[dict[str, str]] = []
    corrected_paired: dict[str, str] | None = None
    if corrected_enabled:
        corrected_validation_path = (
            arguments.corrected_dir / "corrected_validation.json"
        ).resolve()
        corrected = _json(corrected_validation_path)
        if corrected.get("valid") is not True:
            raise ValueError("corrected-v2 analysis is invalid")
        corrected_summary_path = (
            arguments.corrected_dir / "corrected_policy_summary.csv"
        ).resolve()
        corrected_paired_path = (
            arguments.corrected_dir / "corrected_paired_summary.csv"
        ).resolve()
        corrected_rows = _csv(corrected_summary_path)
        corrected_paired_rows = _csv(corrected_paired_path)
        if len(corrected_paired_rows) != 1:
            raise ValueError("corrected-v2 paired summary must have one row")
        corrected_paired = corrected_paired_rows[0]
        source_paths.extend(
            (corrected_validation_path, corrected_summary_path, corrected_paired_path)
        )

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
        lex_rows=lex_rows,
        sensitivity_rows=sensitivity_rows,
        corrected_rows=corrected_rows,
        corrected_paired=corrected_paired,
        cross_rows=cross_rows,
        original_enabled=original_enabled,
        corrected_enabled=corrected_enabled,
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
        default=Path("experiments/results/gcp_corrected_analysis"),
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
