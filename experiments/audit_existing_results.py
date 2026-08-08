#!/usr/bin/env python3
"""Audit completeness and provenance of the historical HCORAP result folders."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


INSTANCE_CLASS = re.compile(r"instance_(?P<users>\d+)_(?P<agents>\d+)_(?P<visits>\d+)_(?P<seed>\d+)")
WEIGHTED_FILES = (
    "ORIGINAL.csv", "SN-none-ss.csv", "SN-both-none.csv", "SN-both-ss.csv",
    "TOT-none-none.csv", "TOT-none-ss.csv", "TOT-both-none.csv", "TOT-both-ss.csv",
)


def _read(path: Path) -> list[dict[str, str]]:
    first = path.read_text(encoding="utf-8-sig").splitlines()[0]
    delimiter = ";" if first.count(";") > first.count(",") else ","
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream, delimiter=delimiter))


def _instance_name(row: dict[str, str]) -> str:
    value = row.get("instance_name") or row.get("instance") or ""
    return Path(value).stem


def audit(workspace: Path) -> dict[str, Any]:
    results_dir = workspace / "results"
    expected_paths = sorted((workspace / "instances" / "paperInstances").glob("**/*.txt"))
    expected = {_instance_name({"instance": str(path)}) for path in expected_paths}
    weighted: dict[str, Any] = {}
    optimum_vectors: dict[str, dict[str, tuple[str, ...]]] = {}
    for name in WEIGHTED_FILES:
        path = results_dir / name
        rows = _read(path) if path.is_file() else []
        names = [_instance_name(row) for row in rows]
        counts = Counter(names)
        duplicates = sorted(item for item, count in counts.items() if count > 1)
        missing = sorted(expected - set(names))
        statuses = Counter(row.get("status", "") for row in rows)
        weighted[name] = {
            "rows": len(rows),
            "unique_instances": len(set(names)),
            "missing_count": len(missing),
            "missing_instances": missing,
            "duplicate_instances": duplicates,
            "statuses": dict(sorted(statuses.items())),
        }
        optimum_vectors[name] = {
            _instance_name(row): (
                row.get("best_value", ""), row.get("similarity", ""),
                row.get("continuity", ""), row.get("overtime", ""),
            )
            for row in rows
            if row.get("status") == "OPTIMUM"
        }

    baseline = optimum_vectors.get("ORIGINAL.csv", {})
    optimum_disagreements = {}
    for name, vectors in optimum_vectors.items():
        if name == "ORIGINAL.csv":
            continue
        common = sorted(set(baseline) & set(vectors))
        score_mismatches = [
            instance for instance in common if baseline[instance][0] != vectors[instance][0]
        ]
        optimum_disagreements[name] = {
            "paired_optimum_instances": len(common),
            "weighted_score_mismatches": len(score_mismatches),
            "mismatch_instances": score_mismatches,
        }

    commercial_path = results_dir / "results_per_instance.csv"
    commercial_rows = _read(commercial_path) if commercial_path.is_file() else []
    commercial_groups: dict[str, Counter[str]] = {}
    for row in commercial_rows:
        key = f"{row.get('backend')}:{row.get('formulation')}:{row.get('method')}:{row.get('delta')}"
        commercial_groups.setdefault(key, Counter())[row.get("status", "")] += 1
    parse_error_instances = sorted(
        {_instance_name(row) for row in commercial_rows if row.get("status") == "PARSE_ERROR"}
    )

    # Only a run-level raw/ directory counts as historical raw provenance.
    # Exclude this audit's own JSON and ignored pilot campaigns under
    # experiments/results so the report is stable on a fresh clone.
    raw_full = list(results_dir.glob("**/raw/*.json"))
    manifests = list(results_dir.glob("**/manifest.*"))
    return {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "expected_official_instances": len(expected),
        "weighted": weighted,
        "weighted_optimum_crosscheck": optimum_disagreements,
        "weighted_missing_run_records_total": sum(
            item["missing_count"] for item in weighted.values()
        ),
        "commercial": {
            "rows": len(commercial_rows),
            "groups": {key: dict(sorted(value.items())) for key, value in sorted(commercial_groups.items())},
            "parse_error_unique_instances": len(parse_error_instances),
            "parse_error_instances": parse_error_instances,
        },
        "provenance": {
            "raw_json_files_under_historical_results": len(raw_full),
            "manifest_files": [str(path.relative_to(workspace)) for path in manifests],
            "historical_weighted_csv_has_full_raw_json": bool(raw_full),
            "ignored_experiment_results_excluded": True,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    payload = audit(arguments.workspace.resolve())
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
