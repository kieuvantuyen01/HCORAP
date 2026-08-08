#!/usr/bin/env python3
"""Run a validated, resumable Gurobi/CPLEX HCORAP campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
from run_reproducible_campaign import (  # noqa: E402
    _expand_instances,
    _filter_instances,
    _git_environment,
    _json_hash,
    _load_completed,
    _order_tasks,
    _record_is_complete,
    _resolve,
    _sample_rss,
    _sha256,
)


ALLOWED_CONFIGS = {
    ("gurobi-mip", "mip-e"),
    ("cplex-mip", "mip-e"),
    ("cplex-cp", "cp-t"),
    ("cplex-cp", "cp-i"),
    ("reference-enumerator", "direct-schedule-enumeration"),
}
ALLOWED_METHODS = {"weighted", "lex-continuity", "lex-cos", "lex-overtime", "epsilon"}


def _inventory(binary: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [str(binary), "--list-backends"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(completed.stdout)


def _parameter_file(base: Path, value: str | None) -> Path | None:
    if not value:
        return None
    return _resolve(base, value)


def _environment(
    binary: Path,
    config_hash: str,
    inventory: dict[str, Any],
    commercial_configs: list[dict[str, Any]],
) -> dict[str, Any]:
    affinity = None
    if hasattr(os, "sched_getaffinity"):
        affinity = sorted(os.sched_getaffinity(0))
    parameter_hashes = {}
    for configuration in commercial_configs:
        path = configuration.get("resolved_parameter_file")
        if path:
            parameter_hashes[path] = _sha256(Path(path))
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
        "campaign_config_sha256": config_hash,
        "backend_inventory": inventory,
        "parameter_file_sha256": parameter_hashes,
        "git": _git_environment(),
    }


def _validate_configuration(
    configuration: dict[str, Any], base: Path, inventory: dict[str, Any]
) -> dict[str, Any]:
    backend = configuration.get("backend")
    formulation = configuration.get("formulation")
    if (backend, formulation) not in ALLOWED_CONFIGS:
        raise ValueError(f"unsupported commercial configuration: {backend}:{formulation}")
    available = {item["name"]: item for item in inventory.get("backends", [])}
    if backend not in available or not available[backend].get("compiled"):
        raise ValueError(f"commercial backend is not compiled: {backend}")
    parameter = _parameter_file(base, configuration.get("parameter_file"))
    if parameter is not None and not parameter.is_file():
        raise ValueError(f"missing parameter file for {backend}: {parameter}")
    return {
        "backend": backend,
        "formulation": formulation,
        "resolved_parameter_file": str(parameter) if parameter else None,
    }


def _validate_run(run: dict[str, Any]) -> None:
    method = run.get("method")
    if method not in ALLOWED_METHODS:
        raise ValueError(f"unsupported commercial method: {method}")
    if method == "epsilon" and not 0 <= float(run.get("delta", -1)) <= 1:
        raise ValueError("epsilon delta must lie in [0,1]")
    if int(run.get("wc", 1)) < 0 or int(run.get("wo", 1)) < 0:
        raise ValueError("commercial objective weights must be non-negative")


def _build_tasks(
    config: dict[str, Any],
    *,
    base: Path,
    binary_hash: str,
    commercial_configs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    instances = _filter_instances(
        _expand_instances(base, config.get("instances", [])),
        dict(config.get("instance_filters", {})),
    )
    if not instances:
        raise ValueError("commercial campaign matched no instances")
    if config.get("expected_instances") is not None and len(instances) != int(config["expected_instances"]):
        raise ValueError(
            f"expected {config['expected_instances']} instances but selected {len(instances)}"
        )
    runs = config.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError("commercial campaign.runs must be a non-empty list")
    for run in runs:
        _validate_run(run)

    timeout = float(config.get("timeout_seconds", 300))
    threads = int(config.get("threads", 1))
    seed = int(config.get("seed", 0))
    mip_gap = float(config.get("mip_gap", 0))
    absolute_mip_gap = float(config.get("absolute_mip_gap", 0))
    if timeout <= 0 or threads <= 0 or seed < 0:
        raise ValueError("timeout/threads must be positive and seed non-negative")
    if mip_gap != 0 or absolute_mip_gap != 0:
        raise ValueError("certified commercial campaign requires both MIP gaps to be zero")

    tasks = []
    seen = set()
    for instance in instances:
        instance_hash = _sha256(instance)
        for commercial in commercial_configs:
            for requested in runs:
                specification = {
                    "backend": commercial["backend"],
                    "formulation": commercial["formulation"],
                    "parameter_file": commercial["resolved_parameter_file"],
                    "method": requested["method"],
                    "delta": str(requested.get("delta", "-")),
                    "wc": int(requested.get("wc", 1)),
                    "wo": int(requested.get("wo", 1)),
                    "soft_coverage": bool(requested.get("soft_coverage", False)),
                    "print_assignments": bool(requested.get("print_assignments", True)),
                    "native_log": bool(requested.get("native_log", True)),
                    "threads": threads,
                    "seed": seed,
                    "mip_gap": mip_gap,
                    "absolute_mip_gap": absolute_mip_gap,
                }
                identity = {
                    "instance_sha256": instance_hash,
                    "specification": specification,
                    "binary_sha256": binary_hash,
                    "timeout_seconds": timeout,
                }
                run_id = _json_hash(identity)[:24]
                if run_id in seen:
                    raise ValueError(f"duplicate commercial task: {identity}")
                seen.add(run_id)
                tasks.append(
                    {
                        "run_id": run_id,
                        "instance": str(instance),
                        "instance_sha256": instance_hash,
                        "specification": specification,
                    }
                )
    tasks = _order_tasks(
        tasks,
        seed=int(config.get("order_seed", 0)),
        strategy=str(config.get("order_strategy", "blocked-instance")),
    )
    if config.get("expected_runs") is not None and len(tasks) != int(config["expected_runs"]):
        raise ValueError(f"expected {config['expected_runs']} runs but resolved {len(tasks)}")
    return tasks


def _command(
    task: dict[str, Any], binary: Path, temporary_result: Path, solver_log: Path
) -> list[str]:
    item = task["specification"]
    command = [
        str(binary), task["instance"],
        "--backend", item["backend"],
        "--formulation", item["formulation"],
        "--method", item["method"],
        "--timeout", str(task["timeout_seconds"]),
        "--threads", str(item["threads"]),
        "--seed", str(item["seed"]),
        "--wc", str(item["wc"]),
        "--wo", str(item["wo"]),
        "--mip-gap", str(item["mip_gap"]),
        "--absolute-mip-gap", str(item["absolute_mip_gap"]),
        "--output", str(temporary_result),
    ]
    if item["method"] == "epsilon":
        command.extend(["--delta", item["delta"]])
    if item["soft_coverage"]:
        command.append("--soft-coverage")
    if item["print_assignments"]:
        command.append("--print-assignments")
    if item["parameter_file"]:
        command.extend(["--parameter-file", item["parameter_file"]])
    if item["native_log"] and item["backend"] != "reference-enumerator":
        command.extend(["--solver-log", str(solver_log)])
    return command


def _validate_payload(payload: dict[str, Any], specification: dict[str, Any]) -> list[str]:
    errors = []
    for key in ("backend", "formulation", "method"):
        if payload.get(key) != specification[key]:
            errors.append(f"{key} mismatch")
    if payload.get("threads") != specification["threads"]:
        errors.append("thread count mismatch")
    if payload.get("seed") != specification["seed"]:
        errors.append("seed mismatch")
    if payload.get("mip_gap") != 0 or payload.get("absolute_mip_gap") != 0:
        errors.append("non-zero optimality gap in certified campaign")
    if payload.get("status") == "OPTIMUM" and not (payload.get("metrics") or {}).get("verified"):
        errors.append("OPTIMUM incumbent is not independently verified")
    return errors


def _run_task(
    task: dict[str, Any], *, binary: Path, result_dir: Path, hard_grace: float
) -> dict[str, Any]:
    raw_dir = result_dir / "raw"
    stderr_dir = result_dir / "stderr"
    native_dir = result_dir / "native_logs"
    for directory in (raw_dir, stderr_dir, native_dir):
        directory.mkdir(exist_ok=True)
    final_result = raw_dir / f"{task['run_id']}.json"
    temporary_result = raw_dir / f".{task['run_id']}.tmp.json"
    stderr_path = stderr_dir / f"{task['run_id']}.log"
    solver_log = native_dir / f"{task['run_id']}.log"
    temporary_result.unlink(missing_ok=True)
    # Native APIs append to their log files.  Keep exactly one attempt in the
    # authoritative log after a resumed invalid/interrupted task.
    solver_log.unlink(missing_ok=True)
    command = _command(task, binary, temporary_result, solver_log)
    started_utc = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    peak_rss = None
    hard_timeout = False
    with stderr_path.open("wb") as stderr:
        process = subprocess.Popen(
            command, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=stderr,
            start_new_session=True,
        )
        deadline = started + task["timeout_seconds"] + hard_grace
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
    payload: dict[str, Any] = {}
    validation_errors = []
    if temporary_result.is_file():
        try:
            payload = json.loads(temporary_result.read_text(encoding="utf-8"))
            validation_errors.extend(_validate_payload(payload, task["specification"]))
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
        "wall_seconds": time.monotonic() - started,
        "peak_rss_bytes": peak_rss,
        "result": str(final_result),
        "stderr_log": str(stderr_path),
        "native_log": str(solver_log) if solver_log.exists() else None,
        "result_status": payload.get("status", "RUNNER_ERROR"),
        "validation_errors": validation_errors,
    }


def _validation(tasks: list[dict[str, Any]], records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    expected = {task["run_id"] for task in tasks}
    observed = set(records)
    invalid = sorted(
        run_id for run_id in expected & observed if not _record_is_complete(records[run_id])
    )
    return {
        "expected_runs": len(expected),
        "manifest_runs": len(observed),
        "complete_runs": len(expected & observed) - len(invalid),
        "missing_run_ids": sorted(expected - observed),
        "unexpected_run_ids": sorted(observed - expected),
        "invalid_run_ids": invalid,
        "complete": expected == observed and not invalid,
    }


def _preflight_backend(
    binary: Path,
    configuration: dict[str, Any],
    *,
    timeout: float,
    threads: int,
    seed: int,
) -> dict[str, Any]:
    output = Path(f"/tmp/hcorap_commercial_preflight_{os.getpid()}_{configuration['backend']}_{configuration['formulation']}.json")
    command = [
        str(binary), str(ROOT / "tests" / "instances" / "tradeoff.txt"),
        "--backend", configuration["backend"],
        "--formulation", configuration["formulation"],
        "--method", "weighted", "--timeout", str(timeout),
        "--threads", str(threads), "--seed", str(seed),
        "--mip-gap", "0", "--absolute-mip-gap", "0",
        "--output", str(output),
    ]
    if configuration.get("resolved_parameter_file"):
        command.extend(["--parameter-file", configuration["resolved_parameter_file"]])
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    payload = json.loads(output.read_text(encoding="utf-8")) if output.is_file() else {}
    if output.is_file():
        output.unlink()
    valid = (
        completed.returncode == 0
        and payload.get("status") == "OPTIMUM"
        and (payload.get("metrics") or {}).get("verified") is True
    )
    return {
        "backend": configuration["backend"],
        "formulation": configuration["formulation"],
        "returncode": completed.returncode,
        "status": payload.get("status"),
        "solver_version": payload.get("solver_version"),
        "valid": valid,
        "stderr": completed.stderr[-2000:],
    }


def run_campaign(
    config_path: Path,
    *,
    resume: bool = False,
    workers: int | None = None,
    preflight_only: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    base = config_path.parent
    binary = _resolve(base, config.get("binary", "../../bin/release/hcorap_commercial"))
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise ValueError(f"missing commercial executable: {binary}")
    inventory = _inventory(binary)
    requested_configs = config.get("commercial_configurations")
    if not isinstance(requested_configs, list) or not requested_configs:
        raise ValueError("commercial_configurations must be a non-empty list")
    commercial_configs = [
        _validate_configuration(item, base, inventory) for item in requested_configs
    ]
    preflight = [
        _preflight_backend(
            binary,
            item,
            timeout=float(config.get("preflight_timeout_seconds", 30)),
            threads=int(config.get("threads", 1)),
            seed=int(config.get("seed", 0)),
        )
        for item in commercial_configs
    ]
    if not all(item["valid"] for item in preflight):
        raise ValueError(f"commercial backend/license preflight failed: {preflight}")
    if preflight_only:
        return {"preflight": preflight, "valid": True}

    result_dir = _resolve(base, config["result_dir"])
    environment = _environment(
        binary, hashlib.sha256(config_bytes).hexdigest(), inventory, commercial_configs
    )
    environment["preflight"] = preflight
    tasks = _build_tasks(
        config,
        base=base,
        binary_hash=environment["binary_sha256"],
        commercial_configs=commercial_configs,
    )
    timeout = float(config.get("timeout_seconds", 300))
    for task in tasks:
        task["timeout_seconds"] = timeout
    if dry_run:
        return {
            "valid": True,
            "preflight": preflight,
            "instances": len({task["instance_sha256"] for task in tasks}),
            "tasks": len(tasks),
            "result_dir": str(result_dir),
        }

    result_dir.mkdir(parents=True, exist_ok=True)
    manifest = result_dir / "manifest.jsonl"
    existing = _load_completed(manifest)
    if existing and not resume:
        raise ValueError(f"manifest already exists; use --resume: {manifest}")
    pending_ids = {
        task["run_id"]
        for task in tasks
        if task["run_id"] not in existing or not _record_is_complete(existing[task["run_id"]])
    }
    if resume and pending_ids:
        retained = [record for run_id, record in existing.items() if run_id not in pending_ids]
        manifest.write_text(
            "".join(json.dumps(record, sort_keys=True) + "\n" for record in retained),
            encoding="utf-8",
        )
        existing = {record["run_id"]: record for record in retained}
    pending = [task for task in tasks if task["run_id"] in pending_ids]
    (result_dir / "environment.json").write_text(
        json.dumps(environment, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (result_dir / "resolved_campaign.json").write_text(
        json.dumps({"config": config, "tasks": tasks}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    worker_count = int(workers if workers is not None else config.get("workers", 1))
    if worker_count <= 0:
        raise ValueError("workers must be positive")
    hard_grace = float(config.get("hard_grace_seconds", 60))
    lock = threading.Lock()
    with manifest.open("a", encoding="utf-8") as stream:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(
                    _run_task, task, binary=binary, result_dir=result_dir,
                    hard_grace=hard_grace,
                ): task
                for task in pending
            }
            for future in as_completed(futures):
                record = future.result()
                with lock:
                    stream.write(json.dumps(record, sort_keys=True) + "\n")
                    stream.flush()
                existing[record["run_id"]] = record
                print(
                    f"[{len(existing)}/{len(tasks)}] {record['run_id']} "
                    f"{record['specification']['backend']} {record['result_status']} "
                    f"{record['wall_seconds']:.3f}s",
                    flush=True,
                )
    result = _validation(tasks, _load_completed(manifest))
    result.update({"result_dir": str(result_dir), "workers": worker_count, "preflight": preflight})
    (result_dir / "validation.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()
    try:
        result = run_campaign(
            arguments.config,
            resume=arguments.resume,
            workers=arguments.workers,
            preflight_only=arguments.preflight_only,
            dry_run=arguments.dry_run,
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError, subprocess.SubprocessError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("complete", result.get("valid", False)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
