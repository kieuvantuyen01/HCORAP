"""Reproducible JSON-configured experiment runner."""

from __future__ import annotations

import glob
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

from .cpsat import (
    solve_cpsat_epsilon_constraint,
    solve_cpsat_lexicographic,
    solve_cpsat_weighted,
)
from .io import read_instance
from .solvers import (
    solve_epsilon_constraint,
    solve_lexicographic,
    solve_weighted,
)


def _git_metadata(workdir: Path) -> Mapping[str, Any]:
    def run(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=workdir,
            check=False,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip() if completed.returncode == 0 else ""

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {"commit": commit or None, "dirty": bool(status)}


def environment_metadata(workdir: Path) -> Mapping[str, Any]:
    """Capture the minimum environment needed to interpret runtime results."""

    try:
        import pysat

        pysat_version = getattr(pysat, "__version__", None)
    except ImportError:
        pysat_version = None
    try:
        import ortools

        ortools_version = getattr(ortools, "__version__", None)
    except ImportError:
        ortools_version = None
    implementation_files = sorted(Path(__file__).resolve().parent.glob("*.py"))
    implementation_digest = hashlib.sha256()
    for path in implementation_files:
        implementation_digest.update(path.name.encode("utf-8"))
        implementation_digest.update(path.read_bytes())
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "pysat_version": pysat_version,
        "ortools_version": ortools_version,
        "implementation_sha256": implementation_digest.hexdigest(),
        "git": _git_metadata(workdir),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _expand_instances(patterns: Iterable[str], base_dir: Path) -> List[Path]:
    paths = set()
    for pattern in patterns:
        expanded_pattern = Path(pattern)
        if not expanded_pattern.is_absolute():
            expanded_pattern = base_dir / expanded_pattern
        matches = glob.glob(str(expanded_pattern), recursive=True)
        paths.update(Path(match).resolve() for match in matches if Path(match).is_file())
    return sorted(paths)


def _dispatch(instance: Any, specification: Mapping[str, Any]) -> Any:
    params = dict(specification)
    method = params.pop("method")
    dispatch = {
        "weighted": solve_weighted,
        "lexicographic": solve_lexicographic,
        "epsilon-constraint": solve_epsilon_constraint,
        "cpsat-weighted": solve_cpsat_weighted,
        "cpsat-lexicographic": solve_cpsat_lexicographic,
        "cpsat-epsilon-constraint": solve_cpsat_epsilon_constraint,
    }
    try:
        solver = dispatch[method]
    except KeyError as exc:
        raise ValueError(f"unknown experiment method {method!r}") from exc
    return solver(instance, **params)


def _run_id(
    instance_hash: str,
    specification: Mapping[str, Any],
    implementation_hash: str,
) -> str:
    payload = json.dumps(
        {
            "instance_sha256": instance_hash,
            "specification": specification,
            "implementation_sha256": implementation_hash,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:20]


def run_experiment_config(
    config_path: Path,
    *,
    output_path: Path | None = None,
    resume: bool = False,
) -> Mapping[str, Any]:
    """Execute a JSON grid and write one self-contained record per run."""

    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config.get("instances"), list) or not config["instances"]:
        raise ValueError("config.instances must be a non-empty list of glob patterns")
    if not isinstance(config.get("methods"), list) or not config["methods"]:
        raise ValueError("config.methods must be a non-empty list")

    base_dir = config_path.parent
    instances = _expand_instances(config["instances"], base_dir)
    if not instances:
        raise ValueError("instance glob patterns matched no files")
    if output_path is None:
        configured = config.get("output", "results.jsonl")
        output_path = Path(configured)
        if not output_path.is_absolute():
            output_path = base_dir / output_path
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    defaults = dict(config.get("defaults", {}))
    specifications = []
    for item in config["methods"]:
        if not isinstance(item, dict) or "method" not in item:
            raise ValueError("each method entry must be an object containing 'method'")
        specifications.append({**defaults, **item})

    completed_ids = set()
    if resume and output_path.exists():
        for line in output_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                completed_ids.add(json.loads(line)["run_id"])

    environment = environment_metadata(Path.cwd())
    mode = "a" if resume else "w"
    executed = 0
    skipped = 0
    errors = 0
    with output_path.open(mode, encoding="utf-8") as output:
        for instance_path in instances:
            instance_hash = _sha256(instance_path)
            instance = read_instance(instance_path)
            for specification in specifications:
                run_id = _run_id(
                    instance_hash,
                    specification,
                    environment["implementation_sha256"],
                )
                if run_id in completed_ids:
                    skipped += 1
                    continue
                record: Dict[str, Any] = {
                    "schema_version": 1,
                    "run_id": run_id,
                    "instance": {
                        "path": str(instance_path),
                        "sha256": instance_hash,
                        "summary": instance.to_summary(),
                    },
                    "specification": specification,
                    "environment": environment,
                }
                try:
                    result = _dispatch(instance, specification)
                    record["result"] = result.as_dict()
                except Exception as exc:  # keep the remaining grid auditable
                    errors += 1
                    record["result"] = {
                        "status": "ERROR",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                output.write(json.dumps(record, sort_keys=True) + "\n")
                output.flush()
                executed += 1

    return {
        "output": str(output_path),
        "instances": len(instances),
        "specifications": len(specifications),
        "executed": executed,
        "skipped": skipped,
        "errors": errors,
    }
