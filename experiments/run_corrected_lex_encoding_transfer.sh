#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$PROJECT_ROOT"

MODE=${1:-all}
case "$MODE" in
    all|pilot|full|--check-only|-h|--help) ;;
    *)
        echo "Usage: experiments/run_corrected_lex_encoding_transfer.sh [all|pilot|full|--check-only]" >&2
        exit 2
        ;;
esac

if [ "$MODE" = "-h" ] || [ "$MODE" = "--help" ]; then
    cat <<'EOF'
Usage: experiments/run_corrected_lex_encoding_transfer.sh MODE

  pilot         16 strata x 2 configurations x LEX-COS = 32 runs (2.67 h max)
  full          48 instances x 2 configurations x LEX-COS = 96 runs (8 h max)
  all           Run the pilot, then full only if the pilot gate says GO
  --check-only  Validate the fixed matrix without starting a solver

The paired comparison is Totalizer-only versus Totalizer plus the two added
constraint families. Every top-level run has a 300-second end-to-end limit.
EOF
    exit 0
fi

python3 - <<'PY'
import json
from pathlib import Path

root = Path("experiments/configs")
expected = {
    "gcp_corrected_lex_encoding_transfer_pilot.json": (16, 32, [1002]),
    "gcp_corrected_lex_encoding_transfer_full.json": (48, 96, [1001, 1002, 1003]),
}
configurations = {
    ("totalizer", "none", "none"),
    ("totalizer", "both", "slot-service"),
}
for name, (instances, runs, seeds) in expected.items():
    config = json.loads((root / name).read_text(encoding="utf-8"))
    observed = {
        (item["cardinality"], item["implied"], item["symmetry"])
        for item in config["configurations"]
    }
    checks = {
        "instances": config["expected_instances"] == instances,
        "runs": config["expected_runs"] == runs,
        "seeds": config["instance_filters"]["seeds"] == seeds,
        "critical": config["instance_filters"]["load_profiles"] == ["critical"],
        "timeout": config["timeout_seconds"] == 300,
        "workers": config["workers"] == 1,
        "configurations": observed == configurations,
        "policy": config["runs"] == [{"method": "lex-cos", "print_assignments": True}],
    }
    if not all(checks.values()):
        raise SystemExit(f"invalid transfer matrix {name}: {checks}")
print("Transfer matrix valid: pilot=32 runs; conditional full=96 runs; timeout=300 s.")
PY

if [ "$MODE" = "--check-only" ]; then
    exit 0
fi
if [ "${CONFIRM_LEX_TRANSFER:-}" != "YES" ]; then
    echo "Set CONFIRM_LEX_TRANSFER=YES after reviewing the 32+96 run budget." >&2
    exit 2
fi

export WORKERS=1
export CONFIRM_FULL_CAMPAIGN=YES
export HCORAP_MAX_PEAK_RSS_MB=12288

run_pilot() {
    bash experiments/gcp_prepare_and_run.sh lex-transfer-pilot
}

run_full() {
    bash experiments/gcp_prepare_and_run.sh lex-transfer-full
}

case "$MODE" in
    pilot)
        run_pilot
        ;;
    full)
        run_full
        ;;
    all)
        run_pilot
        decision=$(python3 - <<'PY'
import json
from pathlib import Path
path = Path("experiments/results/gcp_corrected_lex_encoding_transfer_pilot_analysis/lex_encoding_transfer_validation.json")
print(json.loads(path.read_text(encoding="utf-8"))["decision"])
PY
        )
        if [ "$decision" = "GO" ]; then
            run_full
        else
            echo "Pilot decision is $decision; no full confirmation was started."
        fi
        ;;
esac
