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
    rendered = rendered.rstrip("0").rstrip(".")
    return "0" if rendered == "-0" else rendered


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
    encoding_contrasts_path = encoding_analysis / "policy_encoding_contrasts.csv"

    inputs = (
        policy_validation_path,
        policy_pairs_path,
        policy_pair_details_path,
        encoding_validation_path,
        encoding_summary_path,
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

    output.mkdir(parents=True, exist_ok=True)
    continuity_reduction = -statistics.median(continuity_changes)
    overtime_reduction = -statistics.median(overtime_changes)
    compatibility_change = statistics.median(similarity_changes)
    both_improved = sum(
        float(row["delta_continuity"]) < 0 and float(row["delta_overtime"]) < 0
        for row in policy_pair_details
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
    ]
    for method in POLICIES:
        prefix = "Weighted" if method == "weighted" else "LexCos"
        contrast = contrast_by_policy[method]
        macro_lines.extend(
            (
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
            )
        )
    macros_path = output / "compact_result_macros.tex"
    macros_path.write_text("\n".join(macro_lines) + "\n", encoding="utf-8")

    table_lines = [
        "% Generated by experiments/generate_compact_manuscript_results.py.",
        r"\begin{table}[t]",
        r"\caption{EvalMaxSAT results for the fixed Policy $\times$ Encoding "
        r"design on 48 Original instances. All runs use a 3600\,s limit; "
        r"IC and SB are disabled.}",
        r"\label{tab:policy-encoding}",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3.4pt}",
        r"\begin{tabular}{@{}llrrrrrr@{}}",
        r"\toprule",
        r"Policy & Enc. & Completed & PAR-2 & Med. time & RSS & Variables & Hard clauses\\",
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
                    _number(row["par2_seconds"], 1),
                    _number(row["median_proved_seconds"], 1),
                    _number(row["median_peak_rss_mb"], 1),
                    _grouped(row["median_variables"]),
                    _grouped(row["median_hard_clauses"]),
                )
            )
        if method != POLICIES[-1]:
            table_lines.append(r"\addlinespace")
    table_lines.extend((r"\bottomrule", r"\end{tabular}", r"\end{table}"))
    table_path = output / "compact_encoding_table.tex"
    table_path.write_text("\n".join(table_lines) + "\n", encoding="utf-8")

    claim_support = {
        method: _truth(contrast_by_policy[method]["totalizer_faster_claim_supported"])
        for method in POLICIES
    }
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
