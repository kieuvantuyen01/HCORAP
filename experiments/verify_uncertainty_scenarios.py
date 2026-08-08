#!/usr/bin/env python3
"""Verify paired/nested uncertainty scenarios against their base instances."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from decimal import Decimal
from pathlib import Path
from typing import Any

from hcorap.io import read_instance


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def verify(root: Path) -> dict[str, Any]:
    root = Path(root).resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    diagnostics = root / "scenarios.csv"
    if _sha256(diagnostics) != manifest["diagnostics_sha256"]:
        raise ValueError("uncertainty scenarios.csv hash mismatch")
    with diagnostics.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    failures = []
    if len(rows) != int(manifest["scenarios"]):
        failures.append("scenario row count differs from manifest")
    nested: dict[tuple[str, int], list[tuple[Decimal, set[tuple[int, int]]]]] = defaultdict(list)
    for row in rows:
        scenario_path = Path(row["scenario_instance"])
        if not scenario_path.is_file():
            matches = list(root.glob(f"**/{scenario_path.name}"))
            scenario_path = matches[0] if len(matches) == 1 else scenario_path
        base_path = Path(row["base_instance"])
        if not base_path.is_file():
            failures.append(f"missing base instance: {base_path}")
            continue
        if not scenario_path.is_file():
            failures.append(f"missing scenario instance: {scenario_path}")
            continue
        if _sha256(scenario_path) != row["scenario_sha256"]:
            failures.append(f"scenario hash mismatch: {scenario_path}")
        base = read_instance(base_path)
        scenario = read_instance(scenario_path)
        if (
            base.users != scenario.users
            or base.services != scenario.services
            or base.agents != scenario.agents
            or base.time_slots != scenario.time_slots
            or base.services_by_user != scenario.services_by_user
            or base.sequences != scenario.sequences
            or base.service_availability != scenario.service_availability
            or base.rewards != scenario.rewards
        ):
            failures.append(f"scenario changes a non-uncertain model field: {scenario_path}")
        if any(
            scenario.agent_availability[a][t] > base.agent_availability[a][t]
            for a in range(base.agents)
            for t in range(base.time_slots)
        ):
            failures.append(f"scenario adds agent availability: {scenario_path}")
        if any(
            scenario.normal_hours[a] + scenario.extra_hours[a]
            > sum(scenario.agent_availability[a])
            for a in range(scenario.agents)
        ):
            failures.append(f"scenario capacity exceeds availability: {scenario_path}")
        sidecar = json.loads(
            scenario_path.with_suffix(".txt.json").read_text(encoding="utf-8")
        )
        uncertainty = sidecar["metadata"]["uncertainty"]
        if uncertainty["base_instance_sha256"] != _sha256(base_path):
            failures.append(f"base hash mismatch in sidecar: {scenario_path}")
        absence_set = {
            (int(item[0]), int(item[1]))
            for item in uncertainty["absent_agent_days"]
        }
        nested[(row["base_instance_sha256"], int(row["scenario_seed"]))].append(
            (Decimal(row["absence_probability"]), absence_set)
        )
    for key, values in nested.items():
        previous: set[tuple[int, int]] = set()
        for _probability, absences in sorted(values):
            if not previous <= absences:
                failures.append(f"absence sets are not nested for {key}")
            previous = absences
    result = {
        "schema_version": 1,
        "scenario_root": str(root),
        "scenarios": len(rows),
        "nested_groups": len(nested),
        "failures": failures,
        "valid": not failures,
    }
    (root / "validation.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario_root", type=Path)
    arguments = parser.parse_args()
    try:
        result = verify(arguments.scenario_root)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
