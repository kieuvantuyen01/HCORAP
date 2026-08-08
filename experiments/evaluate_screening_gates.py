#!/usr/bin/env python3
"""Apply predeclared GO/NO-GO gates to the four reduced screening campaigns."""

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
    proposed = [row for row in rows if _matches(row, gate["proposed"])]
    if not baseline or not proposed:
        raise ValueError("could not identify both encoding configurations")

    identity = ("instance_sha256", "method", "delta", "wc", "wo")
    indexed: dict[str, dict[tuple[str, ...], dict[str, str]]] = {
        "baseline": {tuple(row[key] for key in identity): row for row in baseline},
        "proposed": {tuple(row[key] for key in identity): row for row in proposed},
    }
    common = sorted(set(indexed["baseline"]) & set(indexed["proposed"]))
    paired_optimum = []
    mismatches = []
    for key in common:
        left = indexed["baseline"][key]
        right = indexed["proposed"][key]
        if left["status"] == right["status"] == "OPTIMUM":
            paired_optimum.append(key)
            # Different assignments/objective vectors may be legitimate weighted
            # ties.  Coverage and the scalar weighted objective must still agree.
            if (left["coverage"], left["weighted_reference_score"]) != (
                right["coverage"], right["weighted_reference_score"]
            ):
                mismatches.append(
                    {
                        "identity": list(key),
                        "baseline": {
                            "coverage": left["coverage"],
                            "weighted_reference_score": left["weighted_reference_score"],
                        },
                        "proposed": {
                            "coverage": right["coverage"],
                            "weighted_reference_score": right["weighted_reference_score"],
                        },
                    }
                )

    baseline_optimum = sum(row["status"] == "OPTIMUM" for row in baseline)
    proposed_optimum = sum(row["status"] == "OPTIMUM" for row in proposed)
    optimum_ratio = (
        proposed_optimum / baseline_optimum
        if baseline_optimum
        else (1.0 if proposed_optimum == 0 else float("inf"))
    )
    checks = {
        "objective_mismatches": len(mismatches)
        <= int(gate["maximum_objective_mismatches"]),
        "paired_optimum_runs": len(paired_optimum)
        >= int(gate["minimum_paired_optimum_runs"]),
        "proposed_to_baseline_optimum_ratio": optimum_ratio
        >= float(gate["minimum_proposed_to_baseline_optimum_ratio"]),
    }
    return {
        "baseline_runs": len(baseline),
        "proposed_runs": len(proposed),
        "baseline_optimum_runs": baseline_optimum,
        "proposed_optimum_runs": proposed_optimum,
        "proposed_to_baseline_optimum_ratio": optimum_ratio,
        "paired_optimum_runs": len(paired_optimum),
        "objective_mismatches": mismatches,
        "checks": checks,
        "pass": all(checks.values()),
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
    checks = {
        "lex_cos_optimum_rate": lex_rate
        >= float(gate["minimum_lex_cos_optimum_rate"]),
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
        "checks": checks,
        "pass": all(checks.values()),
    }


def _weight_metrics(
    rows: list[dict[str, str]], gate: dict[str, Any]
) -> dict[str, Any]:
    weighted = [row for row in rows if row["method"] == "weighted"]
    optimum = [row for row in weighted if row["status"] == "OPTIMUM"]
    vectors_by_instance: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    for row in optimum:
        vectors_by_instance[row["instance_sha256"]].add(
            tuple(row[key] for key in ("coverage", "similarity", "continuity", "overtime"))
        )
    unique_vectors = {
        vector for vectors in vectors_by_instance.values() for vector in vectors
    }
    instances_with_multiple_vectors = sum(
        len(vectors) >= 2 for vectors in vectors_by_instance.values()
    )
    optimum_rate = _rate(len(optimum), len(weighted))
    checks = {
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
        "checks": checks,
        "pass": all(checks.values()),
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
        configuration = (
            row["cardinality"], row["implied"], row["symmetry"]
        )
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
    checks = {
        "b0_optimum_runs": b0_optimum_total
        >= int(gate["minimum_b0_optimum_runs"]),
        "best_configuration_completion_rate": best_rate
        >= float(gate["minimum_best_configuration_completion_rate"]),
        "peak_rss_mb": maximum_memory is not None
        and maximum_memory <= float(gate["maximum_peak_rss_mb"]),
    }
    return {
        "configurations": rates,
        "b0_optimum_runs": b0_optimum_total,
        "best_configuration_completion_rate": best_rate,
        "maximum_peak_rss_mb": maximum_memory,
        "checks": checks,
        "pass": all(checks.values()),
    }


def evaluate(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    base = config_path.parent
    directories = {
        name: _resolve(base, config[f"{name}_result_dir"])
        for name in ("encoding", "multiobjective", "weight", "lex_scalability")
    }
    rows = {name: _read_rows(path) for name, path in directories.items()}
    maximum_errors = int(config["maximum_hard_errors_per_campaign"])
    hard_errors = {name: _hard_errors(value) for name, value in rows.items()}
    result = {
        "config": str(config_path),
        "thresholds": {
            key: value
            for key, value in config.items()
            if key not in {
                "encoding_result_dir", "multiobjective_result_dir",
                "weight_result_dir", "lex_scalability_result_dir", "output",
            }
        },
        "hard_errors": hard_errors,
        "encoding": _encoding_metrics(rows["encoding"], config["encoding"]),
        "multiobjective": _multiobjective_metrics(
            rows["multiobjective"], config["multiobjective"]
        ),
        "weights": _weight_metrics(rows["weight"], config["weights"]),
        "lex_scalability": _lex_scalability_metrics(
            rows["lex_scalability"], config["lex_scalability"]
        ),
    }
    result["hard_error_gate"] = all(
        count <= maximum_errors for count in hard_errors.values()
    )
    result["decision"] = (
        "GO"
        if result["hard_error_gate"]
        and result["encoding"]["pass"]
        and result["multiobjective"]["pass"]
        and result["weights"]["pass"]
        and result["lex_scalability"]["pass"]
        else "NO-GO"
    )
    output = _resolve(base, config["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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
