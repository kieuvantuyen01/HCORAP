#!/usr/bin/env python3
"""Validate the compact campaign's factorial hard gate and freeze its scope.

The compact design no longer spends measured runs on epsilon, weight, or
lexicographic calibration screens.  The complete 8-cell factorial is run first;
technical validity, weighted-objective agreement, and peak memory are hard
stops.  A valid factorial releases the fixed 732-run publication matrix.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


SUCCESS = {"OPTIMUM", "UNSAT", "UNSATISFIABLE"}


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _read_rows(result_dir: Path) -> list[dict[str, str]]:
    validation_path = result_dir / "validation.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if not validation.get("complete"):
        raise ValueError(f"campaign is incomplete: {validation_path}")
    with (result_dir / "runs.csv").open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _hard_errors(rows: list[dict[str, str]]) -> int:
    return sum(
        row["status"] not in SUCCESS and not row["status"].startswith("TIMEOUT")
        for row in rows
    )


def _matches(row: dict[str, str], specification: dict[str, str]) -> bool:
    return all(row.get(key) == str(value) for key, value in specification.items())


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _encoding_metrics(
    rows: list[dict[str, str]], gate: dict[str, Any]
) -> dict[str, Any]:
    baseline = [row for row in rows if _matches(row, gate["baseline"])]
    reference = [
        row for row in rows if _matches(row, gate["reference_composite"])
    ]
    if not baseline or not reference:
        raise ValueError("could not identify both encoding configurations")

    identity = ("instance_sha256", "method", "delta", "wc", "wo")
    indexed: dict[str, dict[tuple[str, ...], dict[str, str]]] = {
        "baseline": {tuple(row[key] for key in identity): row for row in baseline},
        "reference_composite": {
            tuple(row[key] for key in identity): row for row in reference
        },
    }
    common = sorted(
        set(indexed["baseline"]) & set(indexed["reference_composite"])
    )
    paired_optimum = []
    mismatches = []
    for key in common:
        left = indexed["baseline"][key]
        right = indexed["reference_composite"][key]
        if left["status"] == right["status"] == "OPTIMUM":
            paired_optimum.append(key)
            # Different assignments/objective vectors may be legitimate weighted
            # ties. Coverage and the scalar weighted objective must still agree.
            if (left["coverage"], left["weighted_reference_score"]) != (
                right["coverage"], right["weighted_reference_score"]
            ):
                mismatches.append(
                    {
                        "identity": list(key),
                        "baseline": {
                            "coverage": left["coverage"],
                            "weighted_reference_score": left[
                                "weighted_reference_score"
                            ],
                        },
                        "reference_composite": {
                            "coverage": right["coverage"],
                            "weighted_reference_score": right[
                                "weighted_reference_score"
                            ],
                        },
                    }
                )

    baseline_optimum = sum(row["status"] == "OPTIMUM" for row in baseline)
    reference_optimum = sum(row["status"] == "OPTIMUM" for row in reference)
    optimum_ratio = (
        reference_optimum / baseline_optimum
        if baseline_optimum
        else (1.0 if reference_optimum == 0 else float("inf"))
    )
    hard_checks = {
        "objective_mismatches": len(mismatches)
        <= int(gate["maximum_objective_mismatches"]),
    }
    evidence_checks = {
        "paired_optimum_runs": len(paired_optimum)
        >= int(gate["minimum_paired_optimum_runs"]),
        "reference_to_baseline_optimum_ratio": optimum_ratio
        >= float(gate["minimum_reference_to_baseline_optimum_ratio"]),
    }
    return {
        "baseline_runs": len(baseline),
        "reference_composite_runs": len(reference),
        "baseline_optimum_runs": baseline_optimum,
        "reference_composite_optimum_runs": reference_optimum,
        "reference_to_baseline_optimum_ratio": optimum_ratio,
        "paired_optimum_runs": len(paired_optimum),
        "objective_mismatches": mismatches,
        "hard_checks": hard_checks,
        "evidence_checks": evidence_checks,
        "hard_pass": all(hard_checks.values()),
        "evidence_pass": all(evidence_checks.values()),
    }


def _multiobjective_metrics(
    rows: list[dict[str, str]], gate: dict[str, Any]
) -> dict[str, Any]:
    lex = [row for row in rows if row["method"] == "lex-cos"]
    epsilon = [row for row in rows if row["method"] == "epsilon"]
    lex_optimum = sum(row["status"] == "OPTIMUM" for row in lex)
    epsilon_optimum = sum(row["status"] == "OPTIMUM" for row in epsilon)
    lex_rate = _rate(lex_optimum, len(lex))
    epsilon_rate = _rate(epsilon_optimum, len(epsilon))
    lex_checks = {
        "lex_cos_optimum_rate": lex_rate
        >= float(gate["minimum_lex_cos_optimum_rate"]),
    }
    epsilon_checks = {
        "epsilon_optimum_rate": epsilon_rate
        >= float(gate["minimum_epsilon_optimum_rate"]),
    }
    return {
        "lex_cos_runs": len(lex),
        "lex_cos_optimum_runs": lex_optimum,
        "lex_cos_optimum_rate": lex_rate,
        "epsilon_runs": len(epsilon),
        "epsilon_optimum_runs": epsilon_optimum,
        "epsilon_optimum_rate": epsilon_rate,
        "lex_evidence_checks": lex_checks,
        "epsilon_evidence_checks": epsilon_checks,
        "lex_evidence_pass": all(lex_checks.values()),
        "epsilon_evidence_pass": all(epsilon_checks.values()),
    }


def _weight_metrics(
    rows: list[dict[str, str]], gate: dict[str, Any]
) -> dict[str, Any]:
    weighted = [row for row in rows if row["method"] == "weighted"]
    optimum = [row for row in weighted if row["status"] == "OPTIMUM"]
    vectors_by_instance: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    for row in optimum:
        vectors_by_instance[row["instance_sha256"]].add(
            tuple(
                row[key]
                for key in ("coverage", "similarity", "continuity", "overtime")
            )
        )
    unique_vectors = {
        vector for vectors in vectors_by_instance.values() for vector in vectors
    }
    instances_with_multiple_vectors = sum(
        len(vectors) >= 2 for vectors in vectors_by_instance.values()
    )
    optimum_rate = _rate(len(optimum), len(weighted))
    evidence_checks = {
        "optimum_rate": optimum_rate >= float(gate["minimum_optimum_rate"]),
        "instances_with_multiple_vectors": instances_with_multiple_vectors
        >= int(gate["minimum_instances_with_multiple_vectors"]),
    }
    return {
        "runs": len(weighted),
        "optimum_runs": len(optimum),
        "optimum_rate": optimum_rate,
        "instances_with_multiple_vectors": instances_with_multiple_vectors,
        "unique_objective_vectors": len(unique_vectors),
        "evidence_checks": evidence_checks,
        "evidence_pass": all(evidence_checks.values()),
    }


def _lex_scalability_metrics(
    rows: list[dict[str, str]], gate: dict[str, Any]
) -> dict[str, Any]:
    configurations: dict[tuple[str, str, str], dict[str, dict[str, str]]] = (
        defaultdict(dict)
    )
    for row in rows:
        if row["method"] not in {"weighted", "lex-cos"}:
            continue
        configuration = (row["cardinality"], row["implied"], row["symmetry"])
        configurations[configuration][
            f"{row['instance_sha256']}:{row['method']}"
        ] = row

    rates = {}
    b0_optimum_total = 0
    for configuration, indexed in sorted(configurations.items()):
        b0_instances = {
            key.rsplit(":", 1)[0]
            for key, row in indexed.items()
            if key.endswith(":weighted") and row["status"] == "OPTIMUM"
        }
        lex_optimum = sum(
            indexed.get(f"{instance}:lex-cos", {}).get("status") == "OPTIMUM"
            for instance in b0_instances
        )
        b0_optimum_total += len(b0_instances)
        label = "/".join(configuration)
        rates[label] = {
            "b0_optimum_runs": len(b0_instances),
            "lex_cos_optimum_runs": lex_optimum,
            "completion_rate_on_b0_optimum": _rate(
                lex_optimum, len(b0_instances)
            ),
        }
    best_rate = max(
        (item["completion_rate_on_b0_optimum"] for item in rates.values()),
        default=0.0,
    )
    memory_values = [
        float(row["peak_rss_mb"])
        for row in rows
        if row.get("peak_rss_mb") not in (None, "")
    ]
    maximum_memory = max(memory_values) if memory_values else None
    evidence_checks = {
        "b0_optimum_runs": b0_optimum_total
        >= int(gate["minimum_b0_optimum_runs"]),
        "best_configuration_completion_rate": best_rate
        >= float(gate["minimum_best_configuration_completion_rate"]),
    }
    hard_checks = {
        "peak_rss_mb": maximum_memory is not None
        and maximum_memory <= float(gate["maximum_peak_rss_mb"]),
    }
    return {
        "configurations": rates,
        "b0_optimum_runs": b0_optimum_total,
        "best_configuration_completion_rate": best_rate,
        "maximum_peak_rss_mb": maximum_memory,
        "hard_checks": hard_checks,
        "evidence_checks": evidence_checks,
        "hard_pass": all(hard_checks.values()),
        "evidence_pass": all(evidence_checks.values()),
    }


def _branch(passed: bool, pass_action: str, fail_action: str) -> dict[str, Any]:
    """Return a machine-readable, auditable decision for one claim branch."""
    return {
        "enabled": passed,
        "decision": "GO" if passed else "DROP_OR_REFRAME",
        "action": pass_action if passed else fail_action,
    }


def evaluate(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    base = config_path.parent
    directories = {"encoding": _resolve(base, config["encoding_result_dir"])}
    rows = {"encoding": _read_rows(directories["encoding"])}
    maximum_errors = int(config["maximum_hard_errors_per_campaign"])
    hard_errors = {name: _hard_errors(value) for name, value in rows.items()}
    encoding = _encoding_metrics(rows["encoding"], config["encoding"])
    memory_values = [
        float(row["peak_rss_mb"])
        for row in rows["encoding"]
        if row.get("peak_rss_mb") not in (None, "")
    ]
    maximum_memory = max(memory_values) if memory_values else None
    memory_pass = maximum_memory is not None and maximum_memory <= float(
        config["maximum_peak_rss_mb"]
    )
    result = {
        "config": str(config_path),
        "thresholds": {
            key: value
            for key, value in config.items()
            if key
            not in {
                "encoding_result_dir",
                "output",
            }
        },
        "hard_errors": hard_errors,
        "encoding": encoding,
        "maximum_peak_rss_mb": maximum_memory,
    }
    technical_pass = all(count <= maximum_errors for count in hard_errors.values())
    result["hard_checks"] = {
        "technical_and_validation_errors": technical_pass,
        "encoding_objective_equivalence": encoding["hard_pass"],
        "memory_limit": memory_pass,
    }
    result["hard_stop_pass"] = all(result["hard_checks"].values())
    result["branches"] = {
        "reference_composite": _branch(
            encoding["evidence_pass"],
            "retain the reference composite for the predeclared paired study",
            "retain factorial evidence but do not call the composite preferred or superior",
        ),
        "original_lexicographic": _branch(
            result["hard_stop_pass"],
            "run the original weighted/LEX-COS comparison under the fixed reference configuration",
            "stop the publication campaign and repair the factorial hard failure",
        ),
        "corrected_v2_lexicographic": _branch(
            result["hard_stop_pass"],
            "run the 48-instance corrected-v2 policy and priority-order validation",
            "stop the publication campaign and repair the factorial hard failure",
        ),
    }
    result["publication_scope"] = "COMPACT" if result["hard_stop_pass"] else "STOP"
    result["expected_measured_runs"] = int(config["expected_measured_runs"])
    result["decision"] = "GO" if result["hard_stop_pass"] else "NO-GO"
    output = _resolve(base, config["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    result["output"] = str(output)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config",
        nargs="?",
        type=Path,
        default=Path("experiments/configs/screening_gates.json"),
    )
    arguments = parser.parse_args()
    try:
        result = evaluate(arguments.config)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["decision"] == "GO" else 2


if __name__ == "__main__":
    raise SystemExit(main())
