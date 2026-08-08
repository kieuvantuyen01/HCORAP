#!/usr/bin/env python3
"""Run an auditable, resumable C++ MaxSAT campaign from one JSON specification."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import platform
import random
import re
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
ALLOWED_METHODS = {"weighted", "lex-continuity", "lex-cos", "lex-overtime", "epsilon"}
ALLOWED_CARDINALITY = {"sorting-network", "totalizer"}
ALLOWED_IMPLIED = {"none", "user-slots", "slot-capacity", "both", "both-plus"}
ALLOWED_SYMMETRY = {"none", "slots", "services", "slot-service", "all"}
CORRECTED_NAME = re.compile(
    r"instance_u(?P<users>\d+)_a(?P<agents>\d+)_v(?P<visits>\d+)"
    r"_seed(?P<seed>\d+)_(?P<load>relaxed|critical|saturated)"
    r"(?:_unc_.*)?$"
)
ORIGINAL_NAME = re.compile(
    r"instance_(?P<users>\d+)_(?P<agents>\d+)_(?P<visits>\d+)_(?P<seed>\d+)$"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _resolve(base: Path, value: str) -> Path:
    expanded = os.path.expandvars(value)
    if "$" in expanded:
        raise ValueError(f"unresolved environment variable in path: {value}")
    path = Path(expanded)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _expand_instances(base: Path, patterns: Iterable[str]) -> list[Path]:
    paths: set[Path] = set()
    for pattern in patterns:
        resolved = Path(pattern)
        text = str(resolved if resolved.is_absolute() else base / resolved)
        paths.update(Path(match).resolve() for match in glob.glob(text, recursive=True))
    return sorted(path for path in paths if path.is_file())


def _instance_dimensions(path: Path) -> dict[str, Any]:
    match = CORRECTED_NAME.fullmatch(path.stem) or ORIGINAL_NAME.fullmatch(path.stem)
    if not match:
        raise ValueError(f"cannot parse HCORAP dimensions from filename: {path.name}")
    fields: dict[str, Any] = {
        key: int(value) if key != "load" and value is not None else value
        for key, value in match.groupdict().items()
    }
    fields.setdefault("load", None)
    return fields


def _filter_instances(paths: list[Path], filters: dict[str, Any]) -> list[Path]:
    if not filters:
        return paths
    filters = dict(filters)
    excluded_classes = filters.pop("exclude_classes", [])
    if not isinstance(excluded_classes, list) or any(
        not isinstance(item, dict) or not item for item in excluded_classes
    ):
        raise ValueError("exclude_classes must be a list of non-empty objects")
    mapping = {
        "users": "users",
        "agents": "agents",
        "visits": "visits",
        "seeds": "seed",
        "load_profiles": "load",
    }
    unknown = sorted(set(filters) - set(mapping))
    if unknown:
        raise ValueError(f"unknown instance filter fields: {unknown}")
    for item in excluded_classes:
        unknown_exclusion = sorted(set(item) - set(mapping))
        if unknown_exclusion:
            raise ValueError(
                f"unknown exclude_classes fields: {unknown_exclusion}"
            )
    accepted = {key: set(values) for key, values in filters.items()}
    selected = []
    for path in paths:
        dimensions = _instance_dimensions(path)
        included = all(
            dimensions[mapping[key]] in values for key, values in accepted.items()
        )
        excluded = any(
            all(dimensions[mapping[key]] == value for key, value in item.items())
            for item in excluded_classes
        )
        if included and not excluded:
            selected.append(path)
    return selected


def _order_tasks(
    tasks: list[dict[str, Any]], *, seed: int, strategy: str
) -> list[dict[str, Any]]:
    randomizer = random.Random(seed)
    if strategy == "global-shuffle":
        randomizer.shuffle(tasks)
        return tasks
    if strategy != "blocked-instance":
        raise ValueError(f"unsupported order strategy: {strategy}")
    blocks: dict[str, list[dict[str, Any]]] = {}
    for task in tasks:
        blocks.setdefault(task["instance_sha256"], []).append(task)
    identities = sorted(blocks)
    randomizer.shuffle(identities)
    ordered = []
    for identity in identities:
        block = sorted(blocks[identity], key=lambda item: item["run_id"])
        randomizer.shuffle(block)
        ordered.extend(block)
    return ordered


def _git_environment() -> dict[str, Any]:
    def run(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", *arguments], cwd=ROOT, capture_output=True, text=True, check=False
        )
        return completed.stdout.strip() if completed.returncode == 0 else ""

    status = run("status", "--porcelain")
    diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        check=False,
    ).stdout
    return {
        "commit": run("rev-parse", "HEAD") or None,
        "dirty": bool(status),
        "dirty_files": status.splitlines(),
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
    }


def _environment(binary: Path, solver: Path, config_sha256: str) -> dict[str, Any]:
    affinity = None
    if hasattr(os, "sched_getaffinity"):
        affinity = sorted(os.sched_getaffinity(0))
    return {
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "process_cpu_affinity": affinity,
        "python": sys.version,
        "binary": str(binary),
        "binary_sha256": _sha256(binary),
        "solver": str(solver),
        "solver_sha256": _sha256(solver),
        "campaign_config_sha256": config_sha256,
        "git": _git_environment(),
    }


def _validate_specification(specification: dict[str, Any]) -> None:
    method = specification.get("method")
    if method not in ALLOWED_METHODS:
        raise ValueError(f"unsupported method: {method!r}")
    if method == "epsilon":
        delta = float(specification.get("delta", -1))
        if not 0 <= delta <= 1:
            raise ValueError(f"epsilon delta must lie in [0,1]: {delta}")
    for key in ("wc", "wo"):
        value = int(specification.get(key, 1))
        if value < 0:
            raise ValueError(f"{key} must be non-negative")


def _build_tasks(
    config: dict[str, Any],
    *,
    base: Path,
    environment: dict[str, Any],
) -> list[dict[str, Any]]:
    instances = _filter_instances(
        _expand_instances(base, config.get("instances", [])),
        dict(config.get("instance_filters", {})),
    )
    if not instances:
        raise ValueError("campaign instance patterns matched no files")
    expected_instances = config.get("expected_instances")
    if expected_instances is not None and len(instances) != int(expected_instances):
        raise ValueError(
            f"expected {expected_instances} instances but selected {len(instances)}"
        )
    configurations = config.get("configurations") or [
        {"cardinality": "totalizer", "implied": "none", "symmetry": "none"}
    ]
    runs = config.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError("campaign.runs must be a non-empty list")
    for run in runs:
        _validate_specification(run)

    tasks = []
    seen = set()
    for instance in instances:
        instance_sha = _sha256(instance)
        for configuration in configurations:
            cardinality = configuration.get("cardinality", "totalizer")
            implied = configuration.get("implied", "none")
            symmetry = configuration.get("symmetry", "none")
            if cardinality not in ALLOWED_CARDINALITY:
                raise ValueError(f"unsupported cardinality encoding: {cardinality}")
            if implied not in ALLOWED_IMPLIED:
                raise ValueError(f"unsupported implied constraints: {implied}")
            if symmetry not in ALLOWED_SYMMETRY:
                raise ValueError(f"unsupported symmetry breaking: {symmetry}")
            for requested in runs:
                specification = {
                    "method": requested["method"],
                    "delta": str(requested.get("delta", "-")),
                    "wc": int(requested.get("wc", 1)),
                    "wo": int(requested.get("wo", 1)),
                    "cardinality": cardinality,
                    "implied": implied,
                    "symmetry": symmetry,
                    "soft_coverage": bool(requested.get("soft_coverage", False)),
                    "print_assignments": bool(requested.get("print_assignments", False)),
                }
                identity = {
                    "instance_sha256": instance_sha,
                    "specification": specification,
                    "binary_sha256": environment["binary_sha256"],
                    "solver_sha256": environment["solver_sha256"],
                    "timeout_seconds": float(config.get("timeout_seconds", 300)),
                }
                run_id = _json_hash(identity)[:24]
                if run_id in seen:
                    raise ValueError(f"duplicate campaign task: {identity}")
                seen.add(run_id)
                tasks.append(
                    {
                        "run_id": run_id,
                        "instance": str(instance),
                        "instance_sha256": instance_sha,
                        "specification": specification,
                    }
                )
    tasks = _order_tasks(
        tasks,
        seed=int(config.get("order_seed", 0)),
        strategy=str(config.get("order_strategy", "blocked-instance")),
    )
    expected_runs = config.get("expected_runs")
    if expected_runs is not None and len(tasks) != int(expected_runs):
        raise ValueError(f"expected {expected_runs} runs but resolved {len(tasks)}")
    return tasks


def _sample_rss(process: subprocess.Popen[Any]) -> int | None:
    try:
        import psutil  # type: ignore

        root = psutil.Process(process.pid)
        processes = [root, *root.children(recursive=True)]
        return sum(item.memory_info().rss for item in processes if item.is_running())
    except (ImportError, OSError, ProcessLookupError):
        pass
    # Dependency-free fallback for the GCP/Linux and macOS execution hosts.
    try:
        completed = subprocess.run(
            ["ps", "-axo", "pid=,ppid=,rss="],
            capture_output=True,
            text=True,
            check=True,
        )
        parents: dict[int, int] = {}
        rss_kib: dict[int, int] = {}
        for line in completed.stdout.splitlines():
            pid_text, parent_text, rss_text = line.split()
            pid = int(pid_text)
            parents[pid] = int(parent_text)
            rss_kib[pid] = int(rss_text)
        descendants = {process.pid}
        changed = True
        while changed:
            changed = False
            for pid, parent in parents.items():
                if parent in descendants and pid not in descendants:
                    descendants.add(pid)
                    changed = True
        return sum(rss_kib.get(pid, 0) for pid in descendants) * 1024
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def _run_task(
    task: dict[str, Any],
    *,
    binary: Path,
    solver: Path,
    result_dir: Path,
    timeout_seconds: float,
    hard_grace_seconds: float,
) -> dict[str, Any]:
    specification = task["specification"]
    raw_dir = result_dir / "raw"
    log_dir = result_dir / "logs"
    raw_dir.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)
    final_result = raw_dir / f"{task['run_id']}.json"
    temporary_result = raw_dir / f".{task['run_id']}.tmp.json"
    stderr_path = log_dir / f"{task['run_id']}.stderr.log"
    # A VM interruption may leave a temporary JSON.  It must never be mistaken
    # for output from the resumed attempt.
    temporary_result.unlink(missing_ok=True)
    command = [
        str(binary),
        task["instance"],
        "--solver",
        str(solver),
        "--timeout",
        str(timeout_seconds),
        "--method",
        specification["method"],
        "--wc",
        str(specification["wc"]),
        "--wo",
        str(specification["wo"]),
        "--cardinality-encoding",
        specification["cardinality"],
        "--implied-constraints",
        specification["implied"],
        "--symmetry-breaking",
        specification["symmetry"],
        "--output",
        str(temporary_result),
    ]
    if specification["method"] == "epsilon":
        command.extend(["--delta", specification["delta"]])
    if specification["soft_coverage"]:
        command.append("--soft-coverage")
    if specification["print_assignments"]:
        command.append("--print-assignments")

    started_utc = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    peak_rss = None
    hard_timeout = False
    with stderr_path.open("wb") as stderr:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=stderr,
            start_new_session=True,
        )
        deadline = started + timeout_seconds + hard_grace_seconds
        while process.poll() is None:
            rss = _sample_rss(process)
            if rss is not None:
                peak_rss = max(peak_rss or 0, rss)
            if time.monotonic() > deadline:
                hard_timeout = True
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                break
            time.sleep(0.1)
        exit_code = process.wait()
    wall_seconds = time.monotonic() - started

    validation_errors = []
    payload: dict[str, Any] = {}
    if temporary_result.is_file():
        try:
            payload = json.loads(temporary_result.read_text(encoding="utf-8"))
            if payload.get("method") != specification["method"]:
                validation_errors.append("method mismatch")
            for key, result_key in (
                ("cardinality", "cardinality_encoding"),
                ("implied", "implied_constraints"),
                ("symmetry", "symmetry_breaking"),
            ):
                if payload.get(result_key) != specification[key]:
                    validation_errors.append(f"{result_key} mismatch")
            if payload.get("status") == "OPTIMUM" and not (
                payload.get("metrics") or {}
            ).get("verified"):
                validation_errors.append("OPTIMUM result is not independently verified")
        except (OSError, json.JSONDecodeError) as exc:
            validation_errors.append(f"invalid result JSON: {exc}")
        os.replace(temporary_result, final_result)
    else:
        validation_errors.append("result JSON was not created")

    return {
        **task,
        "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "exit_code": exit_code,
        "hard_timeout": hard_timeout,
        "wall_seconds": wall_seconds,
        "peak_rss_bytes": peak_rss,
        "result": str(final_result),
        "stderr_log": str(stderr_path),
        "result_status": payload.get("status", "RUNNER_ERROR"),
        "validation_errors": validation_errors,
    }


def _load_completed(manifest: Path) -> dict[str, dict[str, Any]]:
    completed = {}
    if not manifest.exists():
        return completed
    for line_number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        run_id = record["run_id"]
        if run_id in completed:
            raise ValueError(f"duplicate run_id in manifest line {line_number}: {run_id}")
        completed[run_id] = record
    return completed


def _record_is_complete(record: dict[str, Any]) -> bool:
    path = Path(record.get("result", ""))
    if not path.is_file() or record.get("validation_errors"):
        return False
    try:
        json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return True


def _validate_campaign(tasks: list[dict[str, Any]], records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    expected = {task["run_id"] for task in tasks}
    observed = set(records)
    invalid = sorted(run_id for run_id in expected & observed if not _record_is_complete(records[run_id]))
    return {
        "expected_runs": len(expected),
        "manifest_runs": len(observed),
        "complete_runs": len(expected & observed) - len(invalid),
        "missing_run_ids": sorted(expected - observed),
        "unexpected_run_ids": sorted(observed - expected),
        "invalid_run_ids": invalid,
        "complete": expected == observed and not invalid,
    }


def run_campaign(
    config_path: Path,
    *,
    resume: bool = False,
    workers: int | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    base = config_path.parent
    binary = _resolve(base, config.get("binary", "../../bin/release/hcorap_multi"))
    solver = _resolve(base, config["solver"])
    result_dir = _resolve(base, config["result_dir"])
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise ValueError(f"missing executable HCORAP binary: {binary}")
    if not solver.is_file() or not os.access(solver, os.X_OK):
        raise ValueError(f"missing executable MaxSAT solver: {solver}")
    environment = _environment(binary, solver, hashlib.sha256(config_bytes).hexdigest())
    tasks = _build_tasks(config, base=base, environment=environment)
    if dry_run:
        return {
            "valid": True,
            "instances": len({task["instance_sha256"] for task in tasks}),
            "tasks": len(tasks),
            "result_dir": str(result_dir),
        }

    result_dir.mkdir(parents=True, exist_ok=True)
    manifest = result_dir / "manifest.jsonl"

    existing = _load_completed(manifest)
    if existing and not resume:
        raise ValueError(f"manifest already exists; use --resume or a new directory: {manifest}")
    pending = [task for task in tasks if task["run_id"] not in existing or not _record_is_complete(existing[task["run_id"]])]
    if resume and pending:
        # Keep a single authoritative record per run id.
        retained = [record for run_id, record in existing.items() if run_id not in {task['run_id'] for task in pending}]
        manifest.write_text(
            "".join(json.dumps(record, sort_keys=True) + "\n" for record in retained),
            encoding="utf-8",
        )
        existing = {record["run_id"]: record for record in retained}

    (result_dir / "environment.json").write_text(
        json.dumps(environment, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (result_dir / "resolved_campaign.json").write_text(
        json.dumps({"config": config, "tasks": tasks}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    timeout_seconds = float(config.get("timeout_seconds", 300))
    hard_grace_seconds = float(config.get("hard_grace_seconds", 60))
    worker_count = int(workers if workers is not None else config.get("workers", 1))
    if worker_count <= 0:
        raise ValueError("workers must be positive")
    lock = threading.Lock()
    with manifest.open("a", encoding="utf-8") as output:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(
                    _run_task,
                    task,
                    binary=binary,
                    solver=solver,
                    result_dir=result_dir,
                    timeout_seconds=timeout_seconds,
                    hard_grace_seconds=hard_grace_seconds,
                ): task
                for task in pending
            }
            for future in as_completed(futures):
                record = future.result()
                with lock:
                    output.write(json.dumps(record, sort_keys=True) + "\n")
                    output.flush()
                existing[record["run_id"]] = record
                print(
                    f"[{len(existing)}/{len(tasks)}] {record['run_id']} "
                    f"{record['result_status']} {record['wall_seconds']:.3f}s",
                    flush=True,
                )

    validation = _validate_campaign(tasks, _load_completed(manifest))
    validation["result_dir"] = str(result_dir)
    validation["workers"] = worker_count
    (result_dir / "validation.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return validation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()
    try:
        result = run_campaign(
            arguments.config,
            resume=arguments.resume,
            workers=arguments.workers,
            dry_run=arguments.dry_run,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("complete", result.get("valid", False)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
