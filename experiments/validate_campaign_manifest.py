#!/usr/bin/env python3
"""Validate the locked reduced-campaign budget against its JSON configs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def validate(manifest_path: Path) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    config_root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    measured_runs = 0
    worst_case_seconds = 0
    result_directories: set[str] = set()
    rows: list[dict[str, Any]] = []

    groups = (
        ("measured", manifest.get("measured_campaigns", [])),
        ("non-measured", manifest.get("non_measured_campaigns", [])),
    )
    for role, campaigns in groups:
        for campaign in campaigns:
            config_path = config_root / campaign["config"]
            try:
                config = json.loads(config_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                errors.append(f"{campaign['name']}: cannot read {config_path}: {exc}")
                continue
            for key in ("expected_runs", "timeout_seconds"):
                if config.get(key) != campaign.get(key):
                    errors.append(
                        f"{campaign['name']}: {key} is {config.get(key)!r} in config "
                        f"but {campaign.get(key)!r} in manifest"
                    )
            result_dir = str(config.get("result_dir", ""))
            if result_dir in result_directories:
                errors.append(f"duplicate result_dir in campaign: {result_dir}")
            result_directories.add(result_dir)
            runs = int(campaign["expected_runs"])
            timeout = int(campaign["timeout_seconds"])
            if role == "measured":
                measured_runs += runs
                worst_case_seconds += runs * timeout
            rows.append(
                {
                    "role": role,
                    "name": campaign["name"],
                    "runs": runs,
                    "timeout_seconds": timeout,
                    "worst_case_core_hours": runs * timeout / 3600,
                }
            )

    if measured_runs != manifest.get("expected_measured_runs"):
        errors.append(
            f"measured run total is {measured_runs}, expected "
            f"{manifest.get('expected_measured_runs')}"
        )
    if worst_case_seconds != manifest.get("expected_worst_case_seconds"):
        errors.append(
            f"worst-case seconds total is {worst_case_seconds}, expected "
            f"{manifest.get('expected_worst_case_seconds')}"
        )
    declared_hours = float(manifest.get("expected_worst_case_core_hours", -1))
    computed_hours = worst_case_seconds / 3600
    if abs(computed_hours - declared_hours) > 1e-9:
        errors.append(
            f"worst-case core-hour total is {computed_hours}, expected {declared_hours}"
        )

    return {
        "valid": not errors,
        "manifest": str(manifest_path),
        "measured_runs": measured_runs,
        "worst_case_seconds": worst_case_seconds,
        "worst_case_core_hours": computed_hours,
        "worst_case_sequential_days": computed_hours / 24,
        "campaigns": rows,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "manifest",
        nargs="?",
        type=Path,
        default=Path("experiments/configs/reduced_campaign_manifest.json"),
    )
    arguments = parser.parse_args()
    try:
        report = validate(arguments.manifest)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
