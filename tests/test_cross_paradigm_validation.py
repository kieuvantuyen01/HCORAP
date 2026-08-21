from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from experiments.analyze_cross_paradigm_validation import analyze


def _write(path: Path, rows: list[dict[str, str]]) -> None:
    path.mkdir()
    (path / "validation.json").write_text(
        json.dumps({"complete": True}), encoding="utf-8"
    )
    with (path / "runs.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _metrics(index: int) -> dict[str, str]:
    return {
        "coverage": "20",
        "similarity": str(100 + index),
        "continuity": "1",
        "overtime": "2",
        "weighted_reference_score": str(90 + index),
        "status": "OPTIMUM",
        "verified": "True",
    }


def test_cross_paradigm_analysis_detects_exact_objective_disagreement(
    tmp_path: Path,
) -> None:
    weighted = []
    lex = []
    commercial = []
    for index in range(20):
        common = {
            "instance_sha256": f"sha-{index}",
            "instance": f"instance-{index}.txt",
            **_metrics(index),
        }
        weighted.append(
            {
                **common,
                "method": "weighted",
                "cardinality": "totalizer",
                "implied": "both",
                "symmetry": "slot-service",
            }
        )
        lex.append(
            {
                **common,
                "method": "lex-cos",
                "cardinality": "totalizer",
                "implied": "both",
                "symmetry": "slot-service",
            }
        )
        for backend in ("gurobi-mip", "cplex-mip"):
            for method in ("weighted", "lex-cos"):
                commercial.append({**common, "method": method, "backend": backend})

    for row in weighted + lex:
        if row["instance_sha256"] == "sha-19":
            row["status"] = "UNSAT"
            row["verified"] = "False"
    for row in commercial:
        if row["instance_sha256"] == "sha-19":
            row["status"] = "INFEASIBLE"
            row["verified"] = "False"

    weighted_dir = tmp_path / "weighted"
    lex_dir = tmp_path / "lex"
    commercial_dir = tmp_path / "commercial"
    _write(weighted_dir, weighted)
    _write(lex_dir, lex)
    _write(commercial_dir, commercial)
    arguments = argparse.Namespace(
        weighted_maxsat_results=weighted_dir,
        lex_maxsat_results=lex_dir,
        commercial_results=commercial_dir,
        output_dir=tmp_path / "analysis",
        scope="full",
    )

    result = analyze(arguments)
    assert result["valid"]
    assert result["complete_groups"] == 40
    assert result["objective_disagreements"] == 0
    assert result["all_infeasible_groups"] == 2
    assert result["unresolved_groups"] == 0
    assert result["status_disagreements"] == 0

    arguments.scope = "weighted-only"
    arguments.lex_maxsat_results = tmp_path / "intentionally-absent"
    weighted_only = analyze(arguments)
    assert weighted_only["valid"]
    assert weighted_only["complete_groups"] == 20
    arguments.scope = "full"
    arguments.lex_maxsat_results = lex_dir

    for row in commercial:
        if (
            row["instance_sha256"] == "sha-0"
            and row["backend"] == "cplex-mip"
            and row["method"] == "lex-cos"
        ):
            row["similarity"] = "999"
    with (commercial_dir / "runs.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(commercial[0]))
        writer.writeheader()
        writer.writerows(commercial)

    mismatched = analyze(arguments)
    assert not mismatched["valid"]
    assert mismatched["objective_disagreements"] == 1
