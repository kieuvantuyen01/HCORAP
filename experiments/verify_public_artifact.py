#!/usr/bin/env python3
"""Verify the versioned paper artifact and its HCORAP-LC inputs."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "artifact" / "results"
MANIFEST = RESULTS / "manifest.json"
POLICY_PAIRS = RESULTS / "policy" / "corrected_pairwise_pairs.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def csv_rows(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        return sum(1 for _ in reader)


def verify_result_snapshot() -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    files = manifest.get("files", {})
    if not files:
        raise ValueError("artifact manifest contains no files")
    for relative, metadata in files.items():
        path = RESULTS / relative
        if not path.is_file():
            raise FileNotFoundError(f"missing artifact file: {relative}")
        observed = sha256(path)
        if observed != metadata["sha256"]:
            raise ValueError(f"artifact hash mismatch: {relative}")
        if "rows" in metadata:
            observed_rows = csv_rows(path)
            if observed_rows != metadata["rows"]:
                raise ValueError(
                    f"row-count mismatch for {relative}: "
                    f"expected {metadata['rows']}, found {observed_rows}"
                )
    return len(files)


def verify_hcorap_lc_inputs() -> int:
    parents: dict[Path, str] = {}
    with POLICY_PAIRS.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            relative = Path(row["instance"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"non-relocatable instance path: {relative}")
            path = ROOT / relative
            digest = row["instance_sha256"]
            previous = parents.setdefault(path, digest)
            if previous != digest:
                raise ValueError(f"inconsistent hashes for {relative}")

    if len(parents) != 48:
        raise ValueError(f"expected 48 HCORAP-LC inputs, found {len(parents)}")

    for path, expected in parents.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing HCORAP-LC input: {path.relative_to(ROOT)}")
        if sha256(path) != expected:
            raise ValueError(f"instance hash mismatch: {path.relative_to(ROOT)}")
        sidecar = path.with_suffix(path.suffix + ".json")
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        if not isinstance(payload.get("metadata"), dict):
            raise ValueError(f"invalid generation sidecar: {sidecar.relative_to(ROOT)}")
    return len(parents)


def main() -> int:
    file_count = verify_result_snapshot()
    instance_count = verify_hcorap_lc_inputs()
    print(
        f"Public artifact verified: {file_count} result files and "
        f"{instance_count} HCORAP-LC inputs."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
