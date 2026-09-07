#!/usr/bin/env python3
"""Generate the result fragments for the locked two-study manuscript design.

The generator accepts only the validated Corrected-v2 policy analysis and the
validated Original-suite Policy x Encoding analysis.  It deliberately refuses
partial or failed analyses, so draft numbers cannot silently enter the paper.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
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


def generate(
    policy_analysis: Path, encoding_analysis: Path, output: Path
) -> dict[str, Any]:
    policy_validation_path = policy_analysis / "corrected_exact_validation.json"
    policy_pairs_path = policy_analysis / "corrected_pairwise_summary.csv"
    policy_pair_details_path = policy_analysis / "corrected_pairwise_pairs.csv"
    encoding_validation_path = encoding_analysis / "policy_encoding_validation.json"
    encoding_summary_path = encoding_analysis / "policy_encoding_summary.csv"
    encoding_pairs_path = encoding_analysis / "policy_encoding_pairs.csv"
    encoding_contrasts_path = encoding_analysis / "policy_encoding_contrasts.csv"

    inputs = (
        policy_validation_path,
        policy_pairs_path,
        policy_pair_details_path,
        encoding_validation_path,
        encoding_summary_path,
        encoding_pairs_path,
        encoding_contrasts_path,
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
    ]
    for method in POLICIES:
        prefix = "Weighted" if method == "weighted" else "LexCos"
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
                    f"{prefix}ParTwoReductionPercent",
                    _number(par2_reductions[method], 1),
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
        r"\begin{table}[H]",
        rf"\caption{{The table compares LEX-COS with Weighted on {len(policy_pair_details)} "
        r"HCORAP-LC instances. The second column reports whether LEX-COS is better, "
        r"the same, or worse. Both solutions in every pair are proved optimal.}",
        r"\label{tab:policy}",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3.8pt}",
        r"\renewcommand{\arraystretch}{1.08}",
        r"\begin{tabular}{@{}llll@{}}",
        r"\toprule",
        r"Measure & Better / same / worse & Median difference & Middle 50\%\\",
        r"\midrule",
        rf"Continuity ($\CONT\downarrow$) & {sum(value < 0 for value in continuity_changes)} / "
        rf"{sum(value == 0 for value in continuity_changes)} / {sum(value > 0 for value in continuity_changes)} & "
        rf"${_fixed(continuity_reduction)}$ fewer & "
        rf"[${_fixed(_percentile((-value for value in continuity_changes), 0.25))}$, "
        rf"${_fixed(_percentile((-value for value in continuity_changes), 0.75))}$] fewer\\",
        rf"Overtime ($\OT\downarrow$) & {sum(value < 0 for value in overtime_changes)} / "
        rf"{sum(value == 0 for value in overtime_changes)} / {sum(value > 0 for value in overtime_changes)} & "
        rf"${_fixed(overtime_reduction)}$ fewer & "
        rf"[${_fixed(_percentile((-value for value in overtime_changes), 0.25))}$, "
        rf"${_fixed(_percentile((-value for value in overtime_changes), 0.75))}$] fewer\\",
        rf"Compatibility ($\SIM\uparrow$) & {sum(value > 0 for value in similarity_changes)} / "
        rf"{sum(value == 0 for value in similarity_changes)} / {sum(value < 0 for value in similarity_changes)} & "
        rf"${_fixed(-compatibility_change)}$ lower (${_fixed(-compatibility_relative_change)}\%$) & "
        rf"[${_fixed(_percentile((-value for value in similarity_changes), 0.25))}$, "
        rf"${_fixed(_percentile((-value for value in similarity_changes), 0.75))}$] lower\\",
        r"\midrule",
        rf"\multicolumn{{4}}{{@{{}}l}}{{\textit{{Continuity and overtime both improve in {both_improved} of {len(policy_pair_details)} pairs.}}}}\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    policy_table_path = output / "compact_policy_table.tex"
    policy_table_path.write_text(
        "\n".join(policy_table_lines) + "\n", encoding="utf-8"
    )

    table_lines = [
        "% Generated by experiments/generate_compact_manuscript_results.py.",
        r"\begin{table}[H]",
        r"\caption{EvalMaxSAT results on 48 Original instances. Each policy is "
        r"tested with a sorting network (SN) and a Totalizer (TOT) under a "
        r"3600\,s limit. Median runtime is calculated over proved runs.}",
        r"\label{tab:policy-encoding}",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3.4pt}",
        r"\begin{tabular}{@{}llrrrrrr@{}}",
        r"\toprule",
        r"Policy & Enc. & Proved & PAR-2 & Median & Memory & Variables & Constraint clauses\\",
        r" & & (/48) & (s) & (s) & (MB) & & \\",
        r"\midrule",
    ]
    for method in POLICIES:
        for encoding in ENCODINGS:
            row = cells[(method, encoding)]
            table_lines.append(
                "{} & {} & {} & {} & {} & {} & {} & {}\\\\".format(
                    _policy_label(method),
                    _encoding_label(encoding),
                    _integer(row["proved_runs"]),
                    _fixed(row["par2_seconds"]),
                    _fixed(row["median_proved_seconds"]),
                    _fixed(row["median_peak_rss_mb"]),
                    _grouped(row["median_variables"]),
                    _grouped(row["median_hard_clauses"]),
                )
            )
        if method != POLICIES[-1]:
            table_lines.append(r"\addlinespace")
    table_lines.extend((r"\bottomrule", r"\end{tabular}", r"\end{table}"))
    table_path = output / "compact_encoding_table.tex"
    table_path.write_text("\n".join(table_lines) + "\n", encoding="utf-8")

    provenance = {
        "schema_version": 1,
        "scope": "two-study-compact-manuscript-results",
        "policy_gate": True,
        "encoding_gate": True,
        "totalizer_claim_supported": claim_support,
        "inputs": {
            _portable(path): _sha256(path)
            for path in inputs
        },
        "outputs": {
            macros_path.name: _sha256(macros_path),
            policy_table_path.name: _sha256(policy_table_path),
            table_path.name: _sha256(table_path),
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
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        report = generate(
            arguments.policy_analysis.resolve(),
            arguments.encoding_analysis.resolve(),
            arguments.output.resolve(),
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
