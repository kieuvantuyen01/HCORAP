#!/usr/bin/env python3
"""Verify hashes, split isolation, structure, and serialized witnesses of v2."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from hcorap.io import read_instance
from hcorap.metrics import verify_assignments
from hcorap.model import Assignment


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_record_path(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_file():
        return path.resolve()
    matches = list(root.glob(f"**/{path.name}"))
    if len(matches) != 1:
        raise ValueError(f"cannot resolve unique batch file for {value!r}")
    return matches[0].resolve()


def verify_batch(root: Path) -> dict[str, Any]:
    root = Path(root).resolve()
    manifest_path = root / "manifest.json"
    diagnostics_path = root / "diagnostics.csv"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if _sha256(diagnostics_path) != manifest["diagnostics_sha256"]:
        raise ValueError("diagnostics.csv SHA-256 differs from manifest")
    with diagnostics_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != int(manifest["instances"]):
        raise ValueError("diagnostics row count differs from manifest.instances")

    calibration = set(int(seed) for seed in manifest["calibration_seeds"])
    evaluation = set(int(seed) for seed in manifest["evaluation_seeds"])
    if calibration & evaluation:
        raise ValueError("calibration and evaluation seeds overlap")
    hashes = set()
    failures = []
    for row in rows:
        instance_path = _resolve_record_path(root, row["instance"])
        metadata_path = _resolve_record_path(root, row["metadata"])
        digest = _sha256(instance_path)
        if digest != row["sha256"]:
            failures.append(f"hash mismatch: {instance_path}")
        hashes.add(digest)
        instance = read_instance(instance_path)
        sidecar = json.loads(metadata_path.read_text(encoding="utf-8"))
        witness = tuple(
            Assignment(
                agent=int(item["agent"]),
                service=int(item["service"]),
                time_slot=int(item["time_slot"]),
            )
            for item in sidecar["metadata"]["witness"]
        )
        checked = verify_assignments(instance, witness)
        if not checked.valid or len(witness) != instance.services:
            failures.append(f"invalid full-coverage witness: {instance_path}")
        split = row["split"]
        seed = int(row["seed"])
        if split == "calibration" and seed not in calibration:
            failures.append(f"undeclared calibration seed {seed}: {instance_path}")
        elif split == "evaluation" and seed not in evaluation:
            failures.append(f"undeclared evaluation seed {seed}: {instance_path}")
        elif split not in {"calibration", "evaluation"}:
            failures.append(f"unknown split {split!r}: {instance_path}")
        if row.get("witness_verified") != "True":
            failures.append(f"generator did not mark witness verified: {instance_path}")

    result = {
        "schema_version": 1,
        "batch": str(root),
        "instances": len(rows),
        "unique_instance_hashes": len(hashes),
        "calibration_seeds": sorted(calibration),
        "evaluation_seeds": sorted(evaluation),
        "failures": failures,
        "valid": not failures and len(hashes) == len(rows),
    }
    if len(hashes) != len(rows):
        result["failures"].append("duplicate serialized instance content detected")
        result["valid"] = False
    (root / "validation.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch", type=Path)
    arguments = parser.parse_args()
    try:
        result = verify_batch(arguments.batch)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
