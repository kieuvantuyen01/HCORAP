from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

from experiments.source_provenance import git_paths_equivalent
from experiments.validate_reusable_gurobi_reference import validate


ROOT = Path(__file__).resolve().parents[1]


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def test_git_provenance_ignores_unrelated_changes_but_rejects_model_changes(
    tmp_path: Path,
) -> None:
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.email", "test@example.com")
    _git(tmp_path, "config", "user.name", "Test")
    model = tmp_path / "model.cpp"
    notes = tmp_path / "notes.md"
    model.write_text("model-v1\n", encoding="utf-8")
    notes.write_text("notes-v1\n", encoding="utf-8")
    _git(tmp_path, "add", "model.cpp", "notes.md")
    _git(tmp_path, "commit", "-m", "initial")
    initial = _git(tmp_path, "rev-parse", "HEAD")

    notes.write_text("notes-v2\n", encoding="utf-8")
    _git(tmp_path, "add", "notes.md")
    _git(tmp_path, "commit", "-m", "notes")
    notes_only = _git(tmp_path, "rev-parse", "HEAD")
    assert git_paths_equivalent(
        initial, notes_only, ("model.cpp",), root=tmp_path
    )

    model.write_text("model-v2\n", encoding="utf-8")
    _git(tmp_path, "add", "model.cpp")
    _git(tmp_path, "commit", "-m", "model")
    model_changed = _git(tmp_path, "rev-parse", "HEAD")
    assert not git_paths_equivalent(
        initial, model_changed, ("model.cpp",), root=tmp_path
    )


def _write_reusable_campaign(path: Path, commit: str) -> None:
    path.mkdir()
    rows = []
    for index in range(2):
        for method in ("weighted", "lex-cos"):
            rows.append(
                {
                    "schema_version": 1,
                    "instance": f"instance-{index}.txt",
                    "instance_sha256": f"sha-{index}",
                    "method": method,
                    "backend": "gurobi-mip",
                    "formulation": "mip-e",
                    "timeout_seconds": 3600,
                    "threads": 1,
                    "solver_seed": 0,
                    "mip_gap": 0,
                    "absolute_mip_gap": 0,
                    "status": "OPTIMUM",
                    "validation_errors": "",
                    "hard_timeout": "False",
                    "verified": "True",
                }
            )
    with (path / "runs.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (path / "validation.json").write_text(
        json.dumps(
            {
                "complete": True,
                "expected_runs": 4,
                "complete_runs": 4,
                "manifest_runs": 4,
                "workers": 1,
                "invalid_run_ids": [],
                "missing_run_ids": [],
                "unexpected_run_ids": [],
            }
        ),
        encoding="utf-8",
    )
    (path / "environment.json").write_text(
        json.dumps(
            {
                "machine": "x86_64",
                "platform": "Linux-test",
                "logical_cpu_count": 8,
                "process_cpu_affinity": [0],
                "git": {"commit": commit, "dirty": False},
            }
        ),
        encoding="utf-8",
    )
    (path / "resolved_campaign.json").write_text(
        json.dumps(
            {
                "config": {
                    "expected_instances": 2,
                    "expected_runs": 4,
                    "timeout_seconds": 3600,
                    "workers": 1,
                    "threads": 1,
                    "seed": 0,
                    "mip_gap": 0,
                    "absolute_mip_gap": 0,
                    "order_strategy": "blocked-instance",
                    "order_seed": 20270906,
                    "instances": ["../../instances/paperInstances/**/*.txt"],
                    "instance_filters": {"seeds": [1, 2, 3]},
                    "commercial_configurations": [
                        {"backend": "gurobi-mip", "formulation": "mip-e"}
                    ],
                    "runs": [
                        {"method": "weighted"},
                        {"method": "lex-cos"},
                    ],
                }
            }
        ),
        encoding="utf-8",
    )


def test_reuse_validator_accepts_a_complete_same_source_campaign(
    tmp_path: Path,
) -> None:
    commit = _git(ROOT, "rev-parse", "HEAD")
    result_dir = tmp_path / "gurobi"
    _write_reusable_campaign(result_dir, commit)

    report = validate(
        result_dir,
        expected_instances=2,
        cpu_core=0,
        current_commit=commit,
    )

    assert report["reusable"] is True
    assert all(report["checks"].values())


def test_reuse_validator_rejects_an_incomplete_campaign(tmp_path: Path) -> None:
    commit = _git(ROOT, "rev-parse", "HEAD")
    result_dir = tmp_path / "gurobi"
    _write_reusable_campaign(result_dir, commit)
    validation = json.loads((result_dir / "validation.json").read_text())
    validation["complete"] = False
    (result_dir / "validation.json").write_text(
        json.dumps(validation), encoding="utf-8"
    )

    report = validate(
        result_dir,
        expected_instances=2,
        cpu_core=0,
        current_commit=commit,
    )

    assert report["reusable"] is False
    assert report["checks"]["collection_complete"] is False
