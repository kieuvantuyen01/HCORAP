from __future__ import annotations

import csv
import json
from pathlib import Path

from experiments.analyze_maxsat_lex_3600 import analyze


VARIANTS = {
    "staged-aligned": (
        "lex-cos", "False", "sequential-stages",
    ),
    "staged-incumbent-bound": (
        "lex-cos", "True", "sequential-stages",
    ),
    "single-call-dominance": (
        "lex-cos-one-shot", "False", "single-call-dominance-weights",
    ),
}


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _exact_row(index: int) -> dict[str, object]:
    return {
        "schema_version": 1,
        "instance_sha256": f"sha-{index}",
        "instance": f"instance-{index}.txt",
        "backend": "gurobi-mip",
        "formulation": "mip-e",
        "method": "lex-cos",
        "status": "OPTIMUM",
        "verified": "True",
        "validation_errors": "",
        "timeout_seconds": 3600,
        "threads": 1,
        "mip_gap": 0,
        "absolute_mip_gap": 0,
        "hard_timeout": "False",
        "assignment_count": 10,
        "similarity": 100 + index,
        "continuity": index % 3,
        "overtime": index % 2,
    }


def _maxsat_row(
    index: int,
    variant: str,
    *,
    optimum: bool,
    exact_similarity: bool,
) -> dict[str, object]:
    method, incumbent_bound, implementation = VARIANTS[variant]
    return {
        "schema_version": 3,
        "instance_sha256": f"sha-{index}",
        "instance": f"instance-{index}.txt",
        "variant": variant,
        "method": method,
        "align_evalmaxsat_tct": "True",
        "stage3_incumbent_bound": incumbent_bound,
        "lexicographic_implementation": implementation,
        "cardinality": "totalizer",
        "implied": "none",
        "symmetry": "none",
        "timeout_seconds": 3600,
        "status": "OPTIMUM" if optimum else "TIMEOUT_FEASIBLE",
        "validation_errors": "",
        "hard_timeout": "False",
        "verified": "True",
        "assignment_count": 10,
        "solver_calls": 1 if method.endswith("one-shot") else 3,
        "certified_lexicographic_prefix": (
            3 if optimum else (0 if method.endswith("one-shot") else 2)
        ),
        "similarity": 100 + index if exact_similarity else 99 + index,
        "continuity": index % 3,
        "overtime": index % 2,
        "elapsed_seconds": 10 + index if optimum else 3600,
    }


def _campaigns(
    tmp_path: Path,
    *,
    bound_optimum_runs: int,
) -> tuple[Path, Path, Path]:
    maxsat = tmp_path / "maxsat"
    exact = tmp_path / "exact"
    output = tmp_path / "analysis"
    maxsat.mkdir()
    exact.mkdir()
    (maxsat / "validation.json").write_text(
        json.dumps({"complete": True}), encoding="utf-8"
    )
    (exact / "validation.json").write_text(
        json.dumps({"complete": True}), encoding="utf-8"
    )
    maxsat_rows = []
    for index in range(16):
        maxsat_rows.append(
            _maxsat_row(
                index,
                "staged-aligned",
                optimum=index < 2,
                exact_similarity=index < 2,
            )
        )
        maxsat_rows.append(
            _maxsat_row(
                index,
                "staged-incumbent-bound",
                optimum=index < bound_optimum_runs,
                exact_similarity=index < bound_optimum_runs,
            )
        )
        maxsat_rows.append(
            _maxsat_row(
                index,
                "single-call-dominance",
                optimum=index < 2,
                exact_similarity=index < 2,
            )
        )
    _write_csv(maxsat / "runs.csv", maxsat_rows)
    _write_csv(exact / "runs.csv", [_exact_row(index) for index in range(16)])
    return maxsat, exact, output


def test_analyzer_selects_only_a_candidate_with_a_real_gain(tmp_path: Path) -> None:
    maxsat, exact, output = _campaigns(tmp_path, bound_optimum_runs=4)
    report = analyze(
        maxsat,
        exact,
        output,
        expected_instances=16,
        required_variants=set(VARIANTS),
    )

    assert report["structurally_valid"] is True
    assert report["decision"] == "GO"
    assert report["selected_variant"] == "staged-incumbent-bound"
    gate = next(
        item
        for item in report["candidate_gates"]
        if item["variant"] == "staged-incumbent-bound"
    )
    assert gate["minimum_count_gain"] == 2
    assert gate["optimum_gain_over_baseline"] == 2
    assert gate["passes_gate"] is True
    assert (output / "maxsat_lex_3600_pairs.csv").is_file()


def test_analyzer_stops_when_candidates_do_not_improve(tmp_path: Path) -> None:
    maxsat, exact, output = _campaigns(tmp_path, bound_optimum_runs=2)
    report = analyze(
        maxsat,
        exact,
        output,
        expected_instances=16,
        required_variants=set(VARIANTS),
    )

    assert report["structurally_valid"] is True
    assert report["decision"] == "STOP"
    assert report["selected_variant"] is None
    assert report["confirmation_required"] is False
    assert report["confirmation_config"] is None


def test_analyzer_rejects_a_different_gurobi_instance_set(tmp_path: Path) -> None:
    maxsat, exact, output = _campaigns(tmp_path, bound_optimum_runs=4)
    rows = list(csv.DictReader((exact / "runs.csv").open(newline="", encoding="utf-8")))
    rows[-1]["instance_sha256"] = "wrong-sha"
    _write_csv(exact / "runs.csv", rows)

    report = analyze(
        maxsat,
        exact,
        output,
        expected_instances=16,
        required_variants=set(VARIANTS),
    )

    assert report["structurally_valid"] is False
    assert report["decision"] == "INVALID"
    assert not report["structural_checks"]["gurobi_and_maxsat_instance_sets_match"]


def test_analyzer_does_not_select_a_candidate_with_paired_regression(
    tmp_path: Path,
) -> None:
    maxsat, exact, output = _campaigns(tmp_path, bound_optimum_runs=4)
    rows = list(csv.DictReader((maxsat / "runs.csv").open(newline="", encoding="utf-8")))
    for row in rows:
        if row["variant"] != "staged-incumbent-bound":
            continue
        index = int(row["instance_sha256"].split("-")[1])
        if index == 0:
            row.update(
                status="TIMEOUT_FEASIBLE",
                similarity=str(99 + index),
                certified_lexicographic_prefix="2",
                elapsed_seconds="3600",
            )
        elif index == 4:
            row.update(
                status="OPTIMUM",
                similarity=str(100 + index),
                certified_lexicographic_prefix="3",
                elapsed_seconds=str(10 + index),
            )
    _write_csv(maxsat / "runs.csv", rows)

    report = analyze(
        maxsat,
        exact,
        output,
        expected_instances=16,
        required_variants=set(VARIANTS),
    )

    gate = next(
        item
        for item in report["candidate_gates"]
        if item["variant"] == "staged-incumbent-bound"
    )
    assert gate["optimum_gain_over_baseline"] == 2
    assert gate["paired_optimum_losses"] == 1
    assert gate["no_paired_quality_regression"] is False
    assert gate["passes_gate"] is False
