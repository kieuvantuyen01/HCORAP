#!/usr/bin/env python3
"""Validate a completed Gurobi campaign before cross-commit reuse."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path
from typing import Any

try:
    from .analyze_policy_encoding_matrix import (
        EXACT_PROVED,
        METHODS,
        _exact_row_valid,
        _resolved_exact_config_valid,
        _validation_valid,
    )
    from .source_provenance import (
        GUROBI_SOURCE_PATHS,
        ROOT,
        git_paths_equivalent,
    )
except ImportError:
    from analyze_policy_encoding_matrix import (
        EXACT_PROVED,
        METHODS,
        _exact_row_valid,
        _resolved_exact_config_valid,
        _validation_valid,
    )
    from source_provenance import (
        GUROBI_SOURCE_PATHS,
        ROOT,
        git_paths_equivalent,
    )


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _current_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def validate(
    result_dir: Path,
    *,
    expected_instances: int = 48,
    cpu_core: int | None = None,
    current_commit: str | None = None,
) -> dict[str, Any]:
    result_dir = Path(result_dir).resolve()
    validation = _read_object(result_dir / "validation.json")
    environment = _read_object(result_dir / "environment.json")
    resolved = _read_object(result_dir / "resolved_campaign.json")
    with (result_dir / "runs.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    expected_runs = expected_instances * len(METHODS)
    keys = {
        (row.get("instance_sha256", ""), row.get("method", ""))
        for row in rows
    }
    instances = {identity for identity, _ in keys if identity}
    git = environment.get("git") or {}
    reference_commit = str(git.get("commit") or "")
    target_commit = current_commit or _current_commit()
    affinity = environment.get("process_cpu_affinity") or []

    checks = {
        "collection_complete": _validation_valid(validation, expected_runs),
        "run_count": len(rows) == expected_runs,
        "unique_complete_matrix": len(keys) == expected_runs
        and len(instances) == expected_instances
        and all(
            (identity, method) in keys
            for identity in instances
            for method in METHODS
        ),
        "resolved_config": _resolved_exact_config_valid(
            resolved.get("config") or {}, expected_instances
        ),
        "rows_valid": all(_exact_row_valid(row) for row in rows),
        "all_runs_proved": all(
            row.get("status") in EXACT_PROVED for row in rows
        ),
        "reference_source_clean": git.get("dirty") is False,
        "gurobi_source_equivalent": git_paths_equivalent(
            reference_commit,
            target_commit,
            GUROBI_SOURCE_PATHS,
        ),
        "linux_x86_64": environment.get("machine") == "x86_64"
        and str(environment.get("platform", "")).startswith("Linux"),
        "eight_logical_cpus": environment.get("logical_cpu_count") == 8,
        "single_cpu_affinity": len(affinity) == 1,
        "requested_cpu_affinity": cpu_core is None or affinity == [cpu_core],
    }
    return {
        "checks": checks,
        "current_commit": target_commit,
        "reference_commit": reference_commit or None,
        "result_dir": str(result_dir),
        "reusable": all(checks.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--expected-instances", type=int, default=48)
    parser.add_argument("--cpu-core", type=int)
    arguments = parser.parse_args()
    try:
        report = validate(
            arguments.result_dir,
            expected_instances=arguments.expected_instances,
            cpu_core=arguments.cpu_core,
        )
    except (
        OSError,
        ValueError,
        KeyError,
        json.JSONDecodeError,
        subprocess.SubprocessError,
    ) as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["reusable"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
