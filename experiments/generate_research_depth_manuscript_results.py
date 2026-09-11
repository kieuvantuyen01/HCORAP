#!/usr/bin/env python3
"""Generate compact LaTeX evidence from validated research-depth full results."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path


CAMPAIGNS = {
    "diagnostics": (
        "research_depth_diagnostics_full",
        "research_depth_diagnostics_full.json",
        480,
    ),
    "weights": (
        "research_depth_weights_full",
        "research_depth_weights_full.json",
        432,
    ),
    "load": (
        "research_depth_load_full",
        "research_depth_load_full.json",
        1296,
    ),
}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str):
    if not condition:
        raise ValueError(message)


def format_number(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.2f}".rstrip("0").rstrip(".")


def generate(results_root: Path, output_dir: Path):
    repository_root = Path(__file__).resolve().parents[1]
    campaign_dirs = {
        name: results_root / values[0] for name, values in CAMPAIGNS.items()
    }
    commits = set()
    binary_hashes = set()
    source_files = []
    total_runs = 0
    for name, directory in campaign_dirs.items():
        _, config_name, expected = CAMPAIGNS[name]
        validation_path = directory / "validation.json"
        environment_path = directory / "environment.json"
        config_path = repository_root / "experiments" / "configs" / config_name
        validation = read_json(validation_path)
        environment = read_json(environment_path)
        require(validation.get("complete") is True, f"incomplete campaign: {name}")
        require(validation.get("expected_runs") == expected, f"unexpected run count: {name}")
        require(validation.get("complete_runs") == expected, f"missing completed runs: {name}")
        require(not validation.get("invalid_run_ids"), f"invalid runs: {name}")
        require(environment.get("git", {}).get("dirty") is False, f"dirty measured source: {name}")
        require(environment.get("campaign_config_sha256") == sha256(config_path),
                f"configuration hash mismatch: {name}")
        commits.add(environment["git"]["commit"])
        binary_hashes.add(environment["binary_sha256"])
        total_runs += expected
        source_files.extend((validation_path, environment_path, config_path))
    require(len(commits) == 1, "full campaigns used different source commits")
    require(len(binary_hashes) == 1, "full campaigns used different commercial binaries")

    diagnostics_dir = campaign_dirs["diagnostics"] / "analysis"
    diagnostics = read_json(diagnostics_dir / "policy_diagnostics_validation.json")
    require(diagnostics.get("all_runs_optimum") is True, "diagnostic runs are not all optimal")
    require(diagnostics.get("blocks") == diagnostics.get("complete_face_blocks") == 48,
            "incomplete Weighted optimal-face analysis")
    faces_path = diagnostics_dir / "weighted_face_intervals.csv"
    curves_path = diagnostics_dir / "continuity_budget_curves.csv"
    faces = read_csv(faces_path)
    curves = read_csv(curves_path)
    source_files.extend((diagnostics_dir / "policy_diagnostics_validation.json", faces_path, curves_path))

    cont_widths = [int(row["continuity_max"]) - int(row["continuity_min"]) for row in faces]
    overtime_widths = [int(row["overtime_max"]) - int(row["overtime_min"]) for row in faces]
    budget = defaultdict(dict)
    for row in curves:
        require(row["status"] == "OPTIMUM", "non-optimal continuity-budget row")
        budget[row["instance_sha256"]][int(row["slack"])] = row
    require(len(budget) == 48 and all(set(rows) == {0, 1, 2} for rows in budget.values()),
            "continuity-budget grid is incomplete")

    budget_stats = {}
    for left, right in ((0, 1), (1, 2)):
        overtime_changes = []
        similarity_changes = []
        for rows in budget.values():
            overtime_changes.append(int(rows[right]["overtime"]) - int(rows[left]["overtime"]))
            similarity_changes.append(int(rows[right]["similarity"]) - int(rows[left]["similarity"]))
        budget_stats[(left, right)] = {
            "overtime_reduction_instances": sum(value < 0 for value in overtime_changes),
            "total_overtime_reduction": -sum(value for value in overtime_changes if value < 0),
            "median_similarity_change": statistics.median(similarity_changes),
        }

    weights_dir = campaign_dirs["weights"] / "analysis"
    weights_summary_path = weights_dir / "analysis.json"
    weights_runs_path = weights_dir / "weight_optimum_runs.csv"
    weights_stability_path = weights_dir / "weight_instance_stability.csv"
    weights_summary = read_json(weights_summary_path)
    require(weights_summary.get("valid") is True and weights_summary.get("optimum_runs") == 432,
            "weight analysis is incomplete or invalid")
    weights = read_csv(weights_runs_path)
    stability = read_csv(weights_stability_path)
    source_files.extend((weights_summary_path, weights_runs_path, weights_stability_path))

    cos_vectors = {
        row["instance_sha256"]: (int(row["continuity"]), int(row["overtime"]), int(row["similarity"]))
        for row in curves if int(row["slack"]) == 0
    }
    diagonal = {}
    for scale in (1, 4, 8):
        selected = [row for row in weights if int(row["wc"]) == scale and int(row["wo"]) == scale]
        require(len(selected) == 48, f"missing diagonal weight rows for {scale}")
        diagonal[scale] = {
            "cos_matches": sum(
                (int(row["continuity"]), int(row["overtime"]), int(row["similarity"]))
                == cos_vectors[row["instance_sha256"]]
                for row in selected
            ),
            "vectors": {
                row["instance_sha256"]: (row["continuity"], row["overtime"], row["similarity"])
                for row in selected
            },
        }

    load_dir = campaign_dirs["load"] / "analysis"
    load_summary_path = load_dir / "load_sweep_analysis.json"
    load_cells_path = load_dir / "load_cell_summary.csv"
    load_summary = read_json(load_summary_path)
    require(load_summary.get("verified", {}).get("independently_verified_optima") == 1296,
            "load schedules were not all independently verified")
    load_cells = read_csv(load_cells_path)
    weighted_cells = [row for row in load_cells if row["left"] == "lex-cos" and row["right"] == "weighted"]
    order_cells = [row for row in load_cells if row["left"] == "lex-cos" and row["right"] == "lex-overtime"]
    require(len(weighted_cells) == len(order_cells) == 9, "load grid is incomplete")
    source_files.extend((load_summary_path, load_cells_path))

    weight_changes_1_4 = sum(
        diagonal[1]["vectors"][key] != diagonal[4]["vectors"][key]
        for key in diagonal[1]["vectors"]
    )
    weight_changes_4_8 = sum(
        diagonal[4]["vectors"][key] != diagonal[8]["vectors"][key]
        for key in diagonal[4]["vectors"]
    )
    unique_vectors = [int(row["unique_objective_vectors"]) for row in stability]
    load_weighted_conflicts = [int(row["conflicts"]) for row in weighted_cells]
    load_order_conflicts = [int(row["conflicts"]) for row in order_cells]

    statistics_payload = {
        "source_commit": next(iter(commits)),
        "total_runs": total_runs,
        "optimal_face": {
            "instances": len(faces),
            "unavoidable_continuity_loss": diagnostics["certified_unavoidable_continuity_losses"],
            "unavoidable_overtime_loss": diagnostics["certified_unavoidable_overtime_losses"],
            "unavoidable_both_losses": diagnostics["certified_unavoidable_both_losses"],
            "continuity_variation": diagnostics["weighted_faces_with_continuity_variation"],
            "overtime_variation": diagnostics["weighted_faces_with_overtime_variation"],
            "median_continuity_width": statistics.median(cont_widths),
            "median_overtime_width": statistics.median(overtime_widths),
        },
        "continuity_budget": {
            "k0_to_k1": budget_stats[(0, 1)],
            "k1_to_k2": budget_stats[(1, 2)],
        },
        "weight_sensitivity": {
            "instances_with_multiple_vectors": weights_summary["instances_with_multiple_vectors"],
            "minimum_vectors_per_instance": min(unique_vectors),
            "maximum_vectors_per_instance": max(unique_vectors),
            "cos_matches": {str(scale): diagonal[scale]["cos_matches"] for scale in (1, 4, 8)},
            "changed_vectors_1_to_4": weight_changes_1_4,
            "changed_vectors_4_to_8": weight_changes_4_8,
        },
        "load_sensitivity": {
            "capacity_cells": len(weighted_cells),
            "parent_families": max(int(row["parent_families"]) for row in weighted_cells),
            "weighted_difference_min": min(load_weighted_conflicts),
            "weighted_difference_max": max(load_weighted_conflicts),
            "cos_ocs_conflicts": sum(load_order_conflicts),
            "cos_ocs_conflict_cells": sum(value > 0 for value in load_order_conflicts),
        },
    }

    macros = {
        "ResearchDepthRunCount": total_runs,
        "WeightedFaceInstanceCount": len(faces),
        "WeightedFaceContinuityLossCount": diagnostics["certified_unavoidable_continuity_losses"],
        "WeightedFaceOvertimeLossCount": diagnostics["certified_unavoidable_overtime_losses"],
        "WeightedFaceBothLossCount": diagnostics["certified_unavoidable_both_losses"],
        "WeightedFaceContinuityVariationCount": diagnostics["weighted_faces_with_continuity_variation"],
        "WeightedFaceOvertimeVariationCount": diagnostics["weighted_faces_with_overtime_variation"],
        "WeightedFaceMedianContinuityWidth": format_number(statistics.median(cont_widths)),
        "WeightedFaceMedianOvertimeWidth": format_number(statistics.median(overtime_widths)),
        "BudgetFirstOvertimeInstanceCount": budget_stats[(0, 1)]["overtime_reduction_instances"],
        "BudgetFirstTotalOvertimeReduction": budget_stats[(0, 1)]["total_overtime_reduction"],
        "BudgetFirstMedianSimilarityGain": format_number(budget_stats[(0, 1)]["median_similarity_change"]),
        "BudgetSecondOvertimeInstanceCount": budget_stats[(1, 2)]["overtime_reduction_instances"],
        "BudgetSecondMedianSimilarityGain": format_number(budget_stats[(1, 2)]["median_similarity_change"]),
        "WeightSensitiveInstanceCount": weights_summary["instances_with_multiple_vectors"],
        "WeightMinimumVectorCount": min(unique_vectors),
        "WeightMaximumVectorCount": max(unique_vectors),
        "WeightOneCosMatchCount": diagonal[1]["cos_matches"],
        "WeightFourCosMatchCount": diagonal[4]["cos_matches"],
        "WeightEightCosMatchCount": diagonal[8]["cos_matches"],
        "WeightOneToFourChangeCount": weight_changes_1_4,
        "WeightFourToEightChangeCount": weight_changes_4_8,
        "LoadCapacityCellCount": len(weighted_cells),
        "LoadParentFamilyCount": max(int(row["parent_families"]) for row in weighted_cells),
        "LoadWeightedDifferenceMin": min(load_weighted_conflicts),
        "LoadWeightedDifferenceMax": max(load_weighted_conflicts),
        "LoadOrderConflictCount": sum(load_order_conflicts),
        "LoadOrderConflictCellCount": sum(value > 0 for value in load_order_conflicts),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    macro_path = output_dir / "research_depth_macros.tex"
    macro_lines = ["% Generated by experiments/generate_research_depth_manuscript_results.py."]
    macro_lines.extend(f"\\newcommand{{\\{name}}}{{{value}}}" for name, value in macros.items())
    macro_path.write_text("\n".join(macro_lines) + "\n", encoding="utf-8")

    source_hashes = {}
    for path in sorted(set(source_files)):
        try:
            display_path = path.relative_to(repository_root)
        except ValueError:
            display_path = path
        source_hashes[str(display_path)] = sha256(path)
    provenance = {
        "generator": "experiments/generate_research_depth_manuscript_results.py",
        "results_root": str(results_root),
        "statistics": statistics_payload,
        "source_files": source_hashes,
    }
    (output_dir / "research_depth_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return statistics_payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = generate(args.results_root, args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
