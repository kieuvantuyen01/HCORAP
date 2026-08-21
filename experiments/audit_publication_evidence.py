#!/usr/bin/env python3
"""Audit campaign completeness, provenance, and manuscript evidence gates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from .publication_contract import DEFAULT_MANIFEST, ROOT
except ImportError:
    from publication_contract import DEFAULT_MANIFEST, ROOT


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolves(commit: str) -> bool:
    if not commit:
        return False
    return subprocess.run(
        ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0


def _campaign(
    *, role: str, declaration: dict[str, Any], config_root: Path, results_root: Path
) -> dict[str, Any]:
    config_path = config_root / declaration["config"]
    config = _json(config_path)
    result_name = Path(str(config["result_dir"])).name
    result_dir = results_root / result_name
    runs_path = result_dir / "runs.csv"
    validation_path = result_dir / "validation.json"
    environment_path = result_dir / "environment.json"
    observed_rows = 0
    statuses: Counter[str] = Counter()
    if runs_path.is_file():
        with runs_path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        observed_rows = len(rows)
        statuses.update(row.get("status", "") for row in rows)
    validation = _json(validation_path) if validation_path.is_file() else {}
    environment = _json(environment_path) if environment_path.is_file() else {}
    source = environment.get("git") if isinstance(environment.get("git"), dict) else {}
    commit = str(source.get("commit") or "")
    checks = {
        "runs_csv_present": runs_path.is_file(),
        "row_count": observed_rows == int(declaration["expected_runs"]),
        "collector_complete": validation.get("complete") is True,
        "environment_present": environment_path.is_file(),
        "config_hash": environment.get("campaign_config_sha256") == _sha256(config_path),
        "source_clean": source.get("dirty") is False,
        "source_commit_resolves": _resolves(commit),
    }
    return {
        "role": role,
        "name": declaration["name"],
        "config": declaration["config"],
        "result_dir": str(result_dir),
        "expected_rows": declaration["expected_runs"],
        "observed_rows": observed_rows,
        "status_counts": dict(sorted(statuses.items())),
        "source_commit": commit or None,
        "checks": checks,
        "complete": all(checks.values()),
    }


def _analysis_check(path: Path, key: str) -> dict[str, Any]:
    payload = _json(path) if path.is_file() else {}
    return {
        "path": str(path),
        "present": path.is_file(),
        "required_key": key,
        "pass": payload.get(key) is True,
    }


def audit(results_root: Path, manifest_path: Path) -> dict[str, Any]:
    manifest = _json(manifest_path)
    campaigns = []
    for role, declarations in (
        ("measured", manifest["measured_campaigns"]),
        ("non-measured", manifest["non_measured_campaigns"]),
    ):
        campaigns.extend(
            _campaign(
                role=role,
                declaration=declaration,
                config_root=manifest_path.parent,
                results_root=results_root,
            )
            for declaration in declarations
        )

    measured = [row for row in campaigns if row["role"] == "measured"]
    observed_measured_rows = sum(int(row["observed_rows"]) for row in measured)
    commits = sorted(
        {str(row["source_commit"]) for row in campaigns if row["source_commit"]}
    )
    analyses = {
        "evalmaxsat_calibration": _analysis_check(
            results_root
            / "gcp_evalmaxsat_lex_calibration/calibration_decision.json",
            "pass",
        ),
        "primary": _analysis_check(
            results_root / "gcp_primary_analysis/analysis_validation.json", "valid"
        ),
        "corrected_maxsat_scalability": _analysis_check(
            results_root / "gcp_corrected_analysis/corrected_validation.json",
            "structurally_valid",
        ),
        "corrected_exact_policy": _analysis_check(
            results_root
            / "gcp_corrected_exact_analysis/corrected_exact_validation.json",
            "manuscript_eligible",
        ),
        "corrected_commercial_calibration": _analysis_check(
            results_root
            / "gcp_commercial_corrected_calibration/calibration_decision.json",
            "pass",
        ),
        "commercial_correctness_smoke": _analysis_check(
            results_root / "gcp_commercial_correctness_smoke/smoke_decision.json",
            "pass",
        ),
        "cross_paradigm": _analysis_check(
            results_root
            / "gcp_cross_paradigm_analysis/cross_paradigm_validation.json",
            "valid",
        ),
        "screening": _analysis_check(
            results_root / "screening_decision.json", "hard_stop_pass"
        ),
    }
    campaign_matrix_complete = all(row["complete"] for row in campaigns)
    analyses_complete = all(row["pass"] for row in analyses.values())
    expected_measured = int(manifest["expected_measured_runs"])
    return {
        "scope": "ICIIT-2027-publication-evidence",
        "results_root": str(results_root),
        "expected_measured_rows": expected_measured,
        "observed_measured_rows": observed_measured_rows,
        "missing_measured_rows": max(0, expected_measured - observed_measured_rows),
        "campaigns": campaigns,
        "source_commits": commits,
        "single_source_commit": len(commits) == 1,
        "source_commits_resolve": bool(commits) and all(_resolves(item) for item in commits),
        "analyses": analyses,
        "campaign_matrix_complete": campaign_matrix_complete,
        "analyses_complete": analyses_complete,
        "publication_ready": campaign_matrix_complete
        and analyses_complete
        and bool(commits)
        and all(_resolves(item) for item in commits),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root", type=Path, default=Path("experiments/results")
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    try:
        report = audit(arguments.results_root.resolve(), arguments.manifest.resolve())
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        parser.error(str(error))
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["publication_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
