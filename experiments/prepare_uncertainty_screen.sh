#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src/proposed${PYTHONPATH:+:$PYTHONPATH}"

BASE_ROOT=${CORRECTED_EVALUATION_ROOT:-instances/corrected_v2_reduced_suite/evaluation_critical/evaluation/critical}
OUTPUT_ROOT=${UNCERTAINTY_ROOT:-instances/uncertainty_screen}

instances=()
while IFS= read -r instance; do
    instances+=("$instance")
done < <(
    find "$BASE_ROOT" -maxdepth 1 -type f -name 'instance_u*_seed1001_critical.txt' \
        | awk '/_a(10|25)_v(4|5)_/' \
        | sort
)
if [ "${#instances[@]}" -ne 8 ]; then
    echo "Expected 8 boundary-class base instances; found ${#instances[@]}." >&2
    exit 2
fi

if [ ! -f "$OUTPUT_ROOT/manifest.json" ]; then
    python3 experiments/generate_uncertainty_scenarios.py \
        --instances "${instances[@]}" \
        --probabilities 0.05 0.10 0.20 \
        --scenario-seeds 3001 3002 3003 3004 3005 \
        --output-dir "$OUTPUT_ROOT"
fi
python3 experiments/verify_uncertainty_scenarios.py "$OUTPUT_ROOT"

python3 - "$OUTPUT_ROOT/manifest.json" <<'PY'
import json
import sys
manifest = json.load(open(sys.argv[1]))
if (
    manifest["base_instances"] != 8
    or manifest["scenarios"] != 120
    or manifest["uncertainty_model"] != "agent-day-absence"
    or manifest["probabilities"] != ["0.05", "0.10", "0.20"]
    or manifest["scenario_seeds"] != [3001, 3002, 3003, 3004, 3005]
):
    raise SystemExit(f"unexpected uncertainty matrix: {manifest}")
print("Uncertainty screening matrix verified: 8 bases, 120 scenarios")
PY
