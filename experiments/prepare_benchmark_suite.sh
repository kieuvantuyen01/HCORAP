#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src/proposed${PYTHONPATH:+:$PYTHONPATH}"

SUITE_ROOT=${BENCHMARK_SUITE_ROOT:-instances/corrected_v2_reduced_suite}
CALIBRATION_ROOT="$SUITE_ROOT/calibration"
EVALUATION_ROOT="$SUITE_ROOT/evaluation_critical"

generate_if_absent() {
    target=$1
    shift
    if [ ! -f "$target/manifest.json" ]; then
        python3 -m hcorap generate-benchmark "$@" --output-dir "$target"
    fi
    python3 experiments/verify_benchmark_batch.py "$target"
}

generate_if_absent "$CALIBRATION_ROOT" \
    --users 30 40 --agents 10 15 20 25 --visits 4 5 \
    --calibration-seeds 1 2 3 4 5 6 7 8 9 10 \
    --load-profiles critical \
    --normal-fraction 0.85

generate_if_absent "$EVALUATION_ROOT" \
    --users 30 40 --agents 10 15 20 25 --visits 4 5 \
    --evaluation-seeds \
        1001 1002 1003 1004 1005 1006 1007 1008 1009 1010 \
    --load-profiles critical \
    --normal-fraction 0.85

python3 - "$SUITE_ROOT" <<'PY'
import csv
import hashlib
import itertools
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
parts = {
    "calibration": root / "calibration",
    "evaluation_critical": root / "evaluation_critical",
}
payload = {"schema_version": 1, "parts": {}}
all_seeds = {}
expected = {"calibration": 160, "evaluation_critical": 160}
specifications = {
    "calibration": {
        "split": "calibration", "seeds": range(1, 11),
        "loads": ("critical",),
    },
    "evaluation_critical": {
        "split": "evaluation", "seeds": range(1001, 1011),
        "loads": ("critical",),
    },
}
for name, path in parts.items():
    manifest_path = path / "manifest.json"
    validation_path = path / "validation.json"
    manifest = json.loads(manifest_path.read_text())
    validation = json.loads(validation_path.read_text())
    if not validation.get("valid"):
        raise SystemExit(f"invalid benchmark part: {path}")
    if manifest["instances"] != expected[name]:
        raise SystemExit(
            f"unexpected instance count for {name}: {manifest['instances']} != {expected[name]}"
        )
    expected_profiles = {
        profile: {"relaxed": 0.55, "critical": 0.85, "saturated": 0.98}[profile]
        for profile in specifications[name]["loads"]
    }
    if (
        manifest.get("generator") != "hcorap-corrected-v2"
        or manifest.get("schema_version") != 1
        or manifest.get("normal_fraction") != 0.85
        or manifest.get("load_profiles") != expected_profiles
    ):
        raise SystemExit(f"stale or unexpected generator parameters in {manifest_path}")
    with (path / "diagnostics.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    observed_matrix = {
        (
            row["split"], int(row["seed"]), int(row["users"]),
            int(row["agents"]), int(row["visits"]), row["load_profile"],
        )
        for row in rows
    }
    specification = specifications[name]
    expected_matrix = set(itertools.product(
        (specification["split"],), specification["seeds"], (30, 40),
        (10, 15, 20, 25), (4, 5), specification["loads"],
    ))
    if observed_matrix != expected_matrix or len(rows) != len(expected_matrix):
        raise SystemExit(f"benchmark matrix mismatch in {path}")
    seeds = set(manifest["calibration_seeds"]) | set(manifest["evaluation_seeds"])
    for previous_name, previous in all_seeds.items():
        overlap = seeds & previous
        if overlap:
            raise SystemExit(f"seed overlap {name}/{previous_name}: {sorted(overlap)}")
    all_seeds[name] = seeds
    payload["parts"][name] = {
        "path": str(path),
        "instances": manifest["instances"],
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "diagnostics_sha256": manifest["diagnostics_sha256"],
        "validation_sha256": hashlib.sha256(validation_path.read_bytes()).hexdigest(),
    }
payload["instances"] = sum(item["instances"] for item in payload["parts"].values())
(root / "suite_manifest.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(json.dumps(payload, indent=2, sort_keys=True))
PY
