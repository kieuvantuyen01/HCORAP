#!/usr/bin/env python3
"""Analyze the 3600-second Lex-COS MaxSAT pilot or confirmation campaign."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


EXPECTED_VARIANTS = {
    "staged-aligned": (
        "lex-cos", True, False, "sequential-stages",
    ),
    "staged-incumbent-bound": (
        "lex-cos", True, True, "sequential-stages",
    ),
    "single-call-dominance": (
        "lex-cos-one-shot", True, False, "single-call-dominance-weights",
    ),
}
BASELINE = "staged-aligned"
CANDIDATE_ORDER = ("staged-incumbent-bound", "single-call-dominance")
VALID_STATUSES = {"OPTIMUM", "TIMEOUT_FEASIBLE", "TIMEOUT"}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def _bool(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def _float(value: Any) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _int(value: Any) -> int | None:
    number = _float(value)
    return int(number) if number is not None else None


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _two_sided_sign_p(wins: int, losses: int) -> float | None:
    discordant = wins + losses
    if discordant == 0:
        return None
    tail = sum(
        math.comb(discordant, index)
        for index in range(min(wins, losses) + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * tail)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(
    maxsat_results: Path,
    exact_results: Path,
    output_dir: Path,
    *,
    expected_instances: int,
    required_variants: set[str] | None = None,
) -> dict[str, Any]:
    maxsat_results = Path(maxsat_results)
    exact_results = Path(exact_results)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    maxsat_validation = json.loads(
        (maxsat_results / "validation.json").read_text(encoding="utf-8")
    )
    exact_validation = json.loads(
        (exact_results / "validation.json").read_text(encoding="utf-8")
    )
    maxsat_rows = _read_csv(maxsat_results / "runs.csv")
    exact_rows = _read_csv(exact_results / "runs.csv")

    variants = sorted({row.get("variant", "") for row in maxsat_rows})
    required_variants = (
        set(variants) if required_variants is None else set(required_variants)
    )
    exact_candidates = [
        row for row in exact_rows
        if row.get("backend") == "gurobi-mip" and row.get("method") == "lex-cos"
    ]
    exact_by_instance = {
        row["instance_sha256"]: row for row in exact_candidates
    }
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in maxsat_rows:
        grouped[row.get("variant", "")].append(row)

    observed_variant_set = set(variants)
    known_variants = observed_variant_set <= set(EXPECTED_VARIANTS)
    required_variant_set = observed_variant_set == required_variants
    complete_variant_blocks = bool(required_variants) and all(
        len(grouped[variant]) == expected_instances
        for variant in required_variants
    )
    unique_variant_instances = all(
        len({row["instance_sha256"] for row in grouped[variant]})
        == expected_instances
        for variant in required_variants
    )
    variant_instance_sets = {
        variant: {row["instance_sha256"] for row in grouped[variant]}
        for variant in required_variants
    }
    common_instances = (
        len({frozenset(values) for values in variant_instance_sets.values()}) == 1
        if variant_instance_sets else False
    )
    maxsat_instance_set = (
        next(iter(variant_instance_sets.values())) if common_instances else set()
    )
    exact_complete = (
        len(exact_rows) == expected_instances
        and len(exact_candidates) == expected_instances
        and len(exact_by_instance) == expected_instances
        and all(row.get("status") == "OPTIMUM" for row in exact_by_instance.values())
        and all(_bool(row.get("verified")) for row in exact_by_instance.values())
        and all(not row.get("validation_errors") for row in exact_by_instance.values())
        and all(
            _float(row.get("timeout_seconds")) == 3600
            for row in exact_by_instance.values()
        )
        and all(
            row.get("formulation") == "mip-e"
            and _int(row.get("schema_version")) == 1
            and _int(row.get("threads")) == 1
            and _float(row.get("mip_gap")) == 0
            and _float(row.get("absolute_mip_gap")) == 0
            and not _bool(row.get("hard_timeout"))
            and (_int(row.get("assignment_count")) or 0) > 0
            for row in exact_by_instance.values()
        )
    )
    exact_instance_match = set(exact_by_instance) == maxsat_instance_set

    pairs: list[dict[str, Any]] = []
    row_contract = True
    optimum_disagreements = 0
    invalid_incumbents = 0
    for variant in variants:
        (
            expected_method,
            expected_tct,
            expected_bound,
            expected_implementation,
        ) = EXPECTED_VARIANTS.get(variant, (None, None, None, None))
        for row in grouped[variant]:
            status = row.get("status", "")
            verified = _bool(row.get("verified"))
            row_ok = (
                _int(row.get("schema_version")) == 3
                and row.get("method") == expected_method
                and _bool(row.get("align_evalmaxsat_tct")) is expected_tct
                and _bool(row.get("stage3_incumbent_bound")) is expected_bound
                and row.get("lexicographic_implementation")
                == expected_implementation
                and row.get("cardinality") == "totalizer"
                and row.get("implied") == "none"
                and row.get("symmetry") == "none"
                and _float(row.get("timeout_seconds")) == 3600
                and status in VALID_STATUSES
                and not row.get("validation_errors")
                and not _bool(row.get("hard_timeout"))
            )
            solver_calls = _int(row.get("solver_calls"))
            prefix = _int(row.get("certified_lexicographic_prefix"))
            if expected_method == "lex-cos-one-shot":
                row_ok = (
                    row_ok
                    and solver_calls is not None
                    and 0 <= solver_calls <= 1
                    and prefix in {0, 3}
                )
            else:
                row_ok = (
                    row_ok
                    and solver_calls is not None
                    and 0 <= solver_calls <= 3
                    and prefix is not None
                    and 0 <= prefix <= 3
                )
            if status == "OPTIMUM":
                row_ok = row_ok and prefix == 3 and solver_calls in {1, 3}
            if status in {"OPTIMUM", "TIMEOUT_FEASIBLE"}:
                row_ok = row_ok and verified and (
                    (_int(row.get("assignment_count")) or 0) > 0
                )
            elif status == "TIMEOUT":
                row_ok = (
                    row_ok
                    and not verified
                    and _int(row.get("similarity")) is None
                    and _int(row.get("continuity")) is None
                    and _int(row.get("overtime")) is None
                )
            row_contract = row_contract and row_ok
            if status in {"OPTIMUM", "TIMEOUT_FEASIBLE"} and not verified:
                invalid_incumbents += 1

            exact = exact_by_instance.get(row["instance_sha256"])
            similarity = _int(row.get("similarity"))
            continuity = _int(row.get("continuity"))
            overtime = _int(row.get("overtime"))
            exact_similarity = _int(exact.get("similarity")) if exact else None
            exact_continuity = _int(exact.get("continuity")) if exact else None
            exact_overtime = _int(exact.get("overtime")) if exact else None
            has_incumbent = (
                verified
                and similarity is not None
                and continuity is not None
                and overtime is not None
            )
            vector_match = has_incumbent and (
                similarity,
                continuity,
                overtime,
            ) == (exact_similarity, exact_continuity, exact_overtime)
            primary_match = has_incumbent and (
                continuity,
                overtime,
            ) == (exact_continuity, exact_overtime)
            similarity_gap = (
                exact_similarity - similarity
                if has_incumbent and exact_similarity is not None
                else None
            )
            similarity_gap_pct = (
                100.0 * similarity_gap / exact_similarity
                if similarity_gap is not None and exact_similarity not in (None, 0)
                else (0.0 if similarity_gap == 0 else None)
            )
            if status == "OPTIMUM" and not vector_match:
                optimum_disagreements += 1
            pairs.append(
                {
                    "variant": variant,
                    "instance_sha256": row["instance_sha256"],
                    "instance": row.get("instance"),
                    "status": status,
                    "verified_incumbent": has_incumbent,
                    "certified_lexicographic_prefix": _int(
                        row.get("certified_lexicographic_prefix")
                    ),
                    "similarity": similarity,
                    "continuity": continuity,
                    "overtime": overtime,
                    "exact_similarity": exact_similarity,
                    "exact_continuity": exact_continuity,
                    "exact_overtime": exact_overtime,
                    "primary_objectives_match": primary_match,
                    "exact_vector_match": vector_match,
                    "similarity_gap": similarity_gap,
                    "similarity_gap_pct": similarity_gap_pct,
                    "elapsed_seconds": _float(row.get("elapsed_seconds")),
                }
            )

    summaries: list[dict[str, Any]] = []
    for variant in variants:
        rows = [row for row in pairs if row["variant"] == variant]
        raw = grouped[variant]
        par2_values = [
            (_float(item.get("elapsed_seconds")) or 0.0)
            if item.get("status") == "OPTIMUM"
            else 7200.0
            for item in raw
        ]
        gaps = [
            float(row["similarity_gap_pct"])
            for row in rows
            if row["primary_objectives_match"]
            and row["similarity_gap_pct"] is not None
            and row["similarity_gap_pct"] >= 0
        ]
        summaries.append(
            {
                "variant": variant,
                "runs": len(rows),
                "optimum_runs": sum(row["status"] == "OPTIMUM" for row in rows),
                "timeout_feasible_runs": sum(
                    row["status"] == "TIMEOUT_FEASIBLE" for row in rows
                ),
                "timeout_without_model_runs": sum(
                    row["status"] == "TIMEOUT" for row in rows
                ),
                "verified_incumbent_runs": sum(
                    row["verified_incumbent"] for row in rows
                ),
                "primary_objective_matches": sum(
                    row["primary_objectives_match"] for row in rows
                ),
                "exact_vector_matches": sum(row["exact_vector_match"] for row in rows),
                "median_similarity_gap_pct_after_primary_match": _median(gaps),
                "par2_seconds": statistics.fmean(par2_values),
            }
        )

    summary_by_variant = {row["variant"]: row for row in summaries}
    pairs_by_variant = {
        variant: {
            row["instance_sha256"]: row
            for row in pairs
            if row["variant"] == variant
        }
        for variant in variants
    }
    baseline = summary_by_variant.get(BASELINE)
    candidates: list[dict[str, Any]] = []
    if baseline is not None:
        minimum_count_gain = max(2, math.ceil(0.10 * expected_instances))
        for order, variant in enumerate(CANDIDATE_ORDER):
            candidate = summary_by_variant.get(variant)
            if candidate is None:
                continue
            optimum_gain = candidate["optimum_runs"] - baseline["optimum_runs"]
            no_optimum_loss = candidate["optimum_runs"] >= baseline["optimum_runs"]
            par2_improvement = (
                1.0 - candidate["par2_seconds"] / baseline["par2_seconds"]
                if baseline["par2_seconds"]
                else 0.0
            )
            primary_gain = (
                candidate["primary_objective_matches"]
                - baseline["primary_objective_matches"]
            )
            exact_vector_gain = (
                candidate["exact_vector_matches"]
                - baseline["exact_vector_matches"]
            )
            baseline_pairs = pairs_by_variant[BASELINE]
            candidate_pairs = pairs_by_variant[variant]
            optimum_wins = optimum_losses = 0
            primary_wins = primary_losses = 0
            vector_wins = vector_losses = 0
            prefix_wins = prefix_losses = 0
            similarity_wins = similarity_losses = 0
            for identity, baseline_pair in baseline_pairs.items():
                candidate_pair = candidate_pairs.get(identity)
                if candidate_pair is None:
                    continue
                baseline_optimum = baseline_pair["status"] == "OPTIMUM"
                candidate_optimum = candidate_pair["status"] == "OPTIMUM"
                optimum_wins += candidate_optimum and not baseline_optimum
                optimum_losses += baseline_optimum and not candidate_optimum
                primary_wins += (
                    candidate_pair["primary_objectives_match"]
                    and not baseline_pair["primary_objectives_match"]
                )
                primary_losses += (
                    baseline_pair["primary_objectives_match"]
                    and not candidate_pair["primary_objectives_match"]
                )
                vector_wins += (
                    candidate_pair["exact_vector_match"]
                    and not baseline_pair["exact_vector_match"]
                )
                vector_losses += (
                    baseline_pair["exact_vector_match"]
                    and not candidate_pair["exact_vector_match"]
                )
                baseline_prefix = baseline_pair["certified_lexicographic_prefix"]
                candidate_prefix = candidate_pair["certified_lexicographic_prefix"]
                if candidate_prefix is not None and baseline_prefix is not None:
                    prefix_wins += candidate_prefix > baseline_prefix
                    prefix_losses += candidate_prefix < baseline_prefix
                baseline_gap = baseline_pair["similarity_gap"]
                candidate_gap = candidate_pair["similarity_gap"]
                if (
                    baseline_pair["primary_objectives_match"]
                    and candidate_pair["primary_objectives_match"]
                    and baseline_gap is not None
                    and candidate_gap is not None
                ):
                    similarity_wins += candidate_gap < baseline_gap
                    similarity_losses += candidate_gap > baseline_gap
            no_paired_quality_regression = (
                optimum_losses == 0
                and primary_losses == 0
                and vector_losses == 0
            )
            passes = no_paired_quality_regression and (
                optimum_gain >= minimum_count_gain
                or (no_optimum_loss and par2_improvement >= 0.20)
                or (no_optimum_loss and primary_gain >= minimum_count_gain)
                or (no_optimum_loss and exact_vector_gain >= minimum_count_gain)
            )
            candidates.append(
                {
                    **candidate,
                    "candidate_order": order,
                    "optimum_gain_over_baseline": optimum_gain,
                    "primary_match_gain_over_baseline": primary_gain,
                    "exact_vector_gain_over_baseline": exact_vector_gain,
                    "par2_improvement_fraction": par2_improvement,
                    "minimum_count_gain": minimum_count_gain,
                    "paired_optimum_wins": optimum_wins,
                    "paired_optimum_losses": optimum_losses,
                    "paired_optimum_sign_p": _two_sided_sign_p(
                        optimum_wins, optimum_losses
                    ),
                    "paired_primary_match_wins": primary_wins,
                    "paired_primary_match_losses": primary_losses,
                    "paired_primary_match_sign_p": _two_sided_sign_p(
                        primary_wins, primary_losses
                    ),
                    "paired_exact_vector_wins": vector_wins,
                    "paired_exact_vector_losses": vector_losses,
                    "paired_exact_vector_sign_p": _two_sided_sign_p(
                        vector_wins, vector_losses
                    ),
                    "paired_certified_prefix_wins": prefix_wins,
                    "paired_certified_prefix_losses": prefix_losses,
                    "paired_similarity_gap_wins": similarity_wins,
                    "paired_similarity_gap_losses": similarity_losses,
                    "no_paired_quality_regression": no_paired_quality_regression,
                    "passes_gate": passes,
                }
            )

    structural_checks = {
        "maxsat_collection_complete": maxsat_validation.get("complete") is True,
        "exact_collection_complete": exact_validation.get("complete") is True,
        "known_variants": known_variants,
        "required_variants_present_exactly": required_variant_set,
        "complete_variant_blocks": complete_variant_blocks,
        "unique_instances_within_variants": unique_variant_instances,
        "common_instances": common_instances,
        "exact_gurobi_reference": exact_complete,
        "gurobi_and_maxsat_instance_sets_match": exact_instance_match,
        "row_contract": row_contract,
        "verified_timeout_incumbents": invalid_incumbents == 0,
        "maxsat_optimum_matches_gurobi": optimum_disagreements == 0,
    }
    structurally_valid = all(structural_checks.values())
    qualifying = [item for item in candidates if item["passes_gate"]]
    qualifying.sort(
        key=lambda item: (
            -item["optimum_runs"],
            -item["exact_vector_matches"],
            -item["primary_objective_matches"],
            item["median_similarity_gap_pct_after_primary_match"]
            if item["median_similarity_gap_pct_after_primary_match"] is not None
            else float("inf"),
            item["par2_seconds"],
            item["candidate_order"],
        )
    )
    selected = qualifying[0]["variant"] if structurally_valid and qualifying else None
    decision = "GO" if selected else ("STOP" if structurally_valid else "INVALID")
    report = {
        "scope": "maxsat-lex-cos-3600",
        "expected_instances": expected_instances,
        "required_variants": sorted(required_variants),
        "structurally_valid": structurally_valid,
        "structural_checks": structural_checks,
        "variants": variants,
        "summary": summaries,
        "candidate_gates": candidates,
        "decision": decision,
        "selected_variant": selected,
        "confirmation_required": decision == "GO",
        "confirmation_config": (
            "gcp_maxsat_lex_3600_confirm_bound.json"
            if selected == "staged-incumbent-bound"
            else "gcp_maxsat_lex_3600_confirm_one_shot.json"
            if selected == "single-call-dominance"
            else None
        ),
    }
    _write_csv(output_dir / "maxsat_lex_3600_pairs.csv", pairs)
    _write_csv(output_dir / "maxsat_lex_3600_summary.csv", summaries)
    _write_csv(output_dir / "maxsat_lex_3600_candidate_gates.csv", candidates)
    (output_dir / "maxsat_lex_3600_decision.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maxsat-results", type=Path, required=True)
    parser.add_argument("--exact-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-instances", type=int, required=True)
    parser.add_argument(
        "--required-variants",
        required=True,
        help="comma-separated variants that must appear exactly",
    )
    arguments = parser.parse_args()
    try:
        report = analyze(
            arguments.maxsat_results,
            arguments.exact_results,
            arguments.output,
            expected_instances=arguments.expected_instances,
            required_variants={
                value.strip()
                for value in arguments.required_variants.split(",")
                if value.strip()
            },
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["structurally_valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
