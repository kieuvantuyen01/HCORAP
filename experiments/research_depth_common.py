"""Read archived campaigns without changing their evidence or provenance."""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_instance(value: str, expected_hash: str | None = None) -> Path:
    path = Path(value)
    if not path.is_file():
        # Preserve the repository-relative path; never guess by basename.
        parts = path.parts
        marker = 'tests' if 'tests' in parts else 'instances'
        if marker not in parts:
            raise FileNotFoundError(value)
        path = ROOT.joinpath(*parts[parts.index(marker):])
    if expected_hash and sha256(path) != expected_hash:
        raise ValueError(f'instance hash mismatch: {path}')
    return path


def campaign_records(directory: Path):
    validation = json.loads((directory / "validation.json").read_text())
    if not validation.get("complete"):
        raise ValueError(f"incomplete campaign: {directory}")
    records = {}
    for line in (directory / 'manifest.jsonl').read_text().splitlines():
        row = json.loads(line)
        records[row['run_id']] = row  # resume may append a replacement attempt
    if len(records) != validation["expected_runs"]:
        raise ValueError(f"manifest count does not match validation: {directory}")
    for run_id, row in sorted(records.items()):
        if row.get('validation_errors') or row.get('hard_timeout'):
            raise ValueError(f'invalid campaign record: {run_id}')
        payload = json.loads((directory / 'raw' / f'{run_id}.json').read_text())
        if payload['status'] != row['result_status']:
            raise ValueError(f'status mismatch: {run_id}')
        yield row, payload


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
