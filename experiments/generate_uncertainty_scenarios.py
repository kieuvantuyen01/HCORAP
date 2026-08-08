#!/usr/bin/env python3
"""Generate paired, nested agent-day absence scenarios for HCORAP."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import replace
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable

from hcorap.io import read_instance, write_instance


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _probability(value: str) -> Decimal:
    try:
        parsed = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError(f"invalid absence probability: {value!r}") from exc
    if parsed < 0 or parsed > 1:
        raise ValueError("absence probabilities must lie in [0,1]")
    return parsed


def _uniform(base_hash: str, scenario_seed: int, agent: int, day: int) -> Decimal:
    digest = hashlib.sha256(
        f"{base_hash}:{scenario_seed}:{agent}:{day}".encode("utf-8")
    ).digest()
    integer = int.from_bytes(digest[:8], "big")
    return Decimal(integer) / Decimal(2**64)


def _sidecar(path: Path) -> dict[str, Any]:
    sidecar = path.with_suffix(path.suffix + ".json")
    if not sidecar.is_file():
        raise ValueError(f"corrected-v2 sidecar is required: {sidecar}")
    return json.loads(sidecar.read_text(encoding="utf-8"))


def generate_scenarios(
    instances: Iterable[Path],
    *,
    probabilities: Iterable[str],
    scenario_seeds: Iterable[int],
    output_dir: Path,
) -> dict[str, Any]:
    bases = sorted({Path(path).resolve() for path in instances})
    if not bases:
        raise ValueError("no base instances were provided")
    probability_values = tuple(sorted({_probability(value) for value in probabilities}))
    seeds = tuple(sorted(set(int(seed) for seed in scenario_seeds)))
    if not probability_values or not seeds:
        raise ValueError("probabilities and scenario_seeds cannot be empty")
    if any(seed < 0 for seed in seeds):
        raise ValueError("scenario seeds must be non-negative")
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for base_path in bases:
        instance = read_instance(base_path)
        source_sidecar = _sidecar(base_path)
        generator_config = source_sidecar["metadata"].get("config") or {}
        days = int(generator_config.get("days", 0))
        slots_per_day = int(generator_config.get("slots_per_day", 0))
        if days <= 0 or slots_per_day <= 0 or days * slots_per_day != instance.time_slots:
            raise ValueError(f"invalid corrected-v2 horizon metadata: {base_path}")
        base_hash = _sha256(base_path)
        for scenario_seed in seeds:
            uniforms = {
                (agent, day): _uniform(base_hash, scenario_seed, agent, day)
                for agent in range(instance.agents)
                for day in range(days)
            }
            previous_absences: set[tuple[int, int]] = set()
            for probability in probability_values:
                absences = {
                    key for key, value in uniforms.items() if value < probability
                }
                if not previous_absences <= absences:
                    raise RuntimeError("nested uncertainty scenario construction failed")
                previous_absences = absences
                availability = [list(row) for row in instance.agent_availability]
                removed_slots = 0
                for agent, day in absences:
                    start = day * slots_per_day
                    for slot in range(start, start + slots_per_day):
                        removed_slots += availability[agent][slot]
                        availability[agent][slot] = 0
                normal = []
                extra = []
                for agent, row in enumerate(availability):
                    available = sum(row)
                    total_cap = min(
                        available,
                        instance.normal_hours[agent] + instance.extra_hours[agent],
                    )
                    regular = min(instance.normal_hours[agent], total_cap)
                    normal.append(regular)
                    extra.append(total_cap - regular)
                scenario_metadata = {
                    "schema_version": 1,
                    "type": "agent-day-absence",
                    "recourse": "evaluated separately as fixed-schedule and full-reoptimization",
                    "base_instance": str(base_path),
                    "base_instance_sha256": base_hash,
                    "scenario_seed": scenario_seed,
                    "absence_probability": str(probability),
                    "days": days,
                    "slots_per_day": slots_per_day,
                    "absent_agent_days": [list(item) for item in sorted(absences)],
                    "removed_available_slots": removed_slots,
                    "common_random_numbers": True,
                    "nested_across_probabilities": True,
                }
                scenario = replace(
                    instance,
                    agent_availability=tuple(tuple(row) for row in availability),
                    normal_hours=tuple(normal),
                    extra_hours=tuple(extra),
                    source=None,
                    metadata={"uncertainty": scenario_metadata},
                )
                probability_tag = str(probability).replace(".", "p")
                name = (
                    f"{base_path.stem}_unc_p{probability_tag}"
                    f"_s{scenario_seed}.txt"
                )
                scenario_path = output_dir / name
                write_instance(scenario, scenario_path)
                metadata_path = scenario_path.with_suffix(".txt.json")
                metadata_path.write_text(
                    json.dumps(
                        {
                            "instance": scenario.to_summary(),
                            "metadata": {"uncertainty": scenario_metadata},
                        },
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                rows.append(
                    {
                        "base_instance": str(base_path),
                        "base_instance_sha256": base_hash,
                        "scenario_instance": str(scenario_path),
                        "scenario_sha256": _sha256(scenario_path),
                        "metadata": str(metadata_path),
                        "scenario_seed": scenario_seed,
                        "absence_probability": str(probability),
                        "absent_agent_days": len(absences),
                        "removed_available_slots": removed_slots,
                        "capacity": sum(normal) + sum(extra),
                        "rho": scenario.to_summary()["rho"],
                    }
                )

    diagnostics = output_dir / "scenarios.csv"
    with diagnostics.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "schema_version": 1,
        "uncertainty_model": "agent-day-absence",
        "recourse_modes": ["fixed-schedule", "full-reoptimization"],
        "base_instances": len(bases),
        "base_instance_sha256": sorted({_sha256(path) for path in bases}),
        "probabilities": [str(value) for value in probability_values],
        "scenario_seeds": list(seeds),
        "scenarios": len(rows),
        "diagnostics": str(diagnostics),
        "diagnostics_sha256": _sha256(diagnostics),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {**manifest, "manifest": str(manifest_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", type=Path, nargs="+", required=True)
    parser.add_argument("--probabilities", nargs="+", required=True)
    parser.add_argument("--scenario-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        result = generate_scenarios(
            arguments.instances,
            probabilities=arguments.probabilities,
            scenario_seeds=arguments.scenario_seeds,
            output_dir=arguments.output_dir,
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
