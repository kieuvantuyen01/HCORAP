#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$PROJECT_ROOT"

MODE=${1:-run}
if [ "$MODE" = "-h" ] || [ "$MODE" = "--help" ]; then
    cat <<'EOF'
Usage: experiments/run_remaining_corrected_evidence.sh [run|--check-only]

Runs only the missing corrected-v2 exact-policy supplement:
  48 non-measured calibration rows (Gurobi + CPLEX), then
  144 measured Gurobi primary rows and 48 measured CPLEX audit rows.

Measured budget: 192 x 300 s = 16 core-hours.  Including calibration, the
sequential worst case is 20 hours.  Existing 732 measured rows are not rerun.
EOF
    exit 0
fi
if [ "$MODE" != "run" ] && [ "$MODE" != "--check-only" ]; then
    echo "Usage: experiments/run_remaining_corrected_evidence.sh [run|--check-only]" >&2
    exit 2
fi

python3 experiments/validate_campaign_manifest.py
python3 experiments/validate_publication_campaign.py

python3 - <<'PY'
import csv
import json
import subprocess
from pathlib import Path

from experiments.publication_contract import CORRECTED_EXACT_MEASURED_CAMPAIGNS

root = Path("experiments/results")
config_root = Path("experiments/configs")
manifest = json.loads((config_root / "reduced_campaign_manifest.json").read_text())
existing = [
    declaration
    for declaration in manifest["measured_campaigns"]
    if declaration["name"] not in CORRECTED_EXACT_MEASURED_CAMPAIGNS
]
supplement = [
    declaration
    for declaration in manifest["measured_campaigns"]
    if declaration["name"] in CORRECTED_EXACT_MEASURED_CAMPAIGNS
]
for declaration in existing:
    config = json.loads((config_root / declaration["config"]).read_text())
    directory = root / Path(config["result_dir"]).name
    expected = int(declaration["expected_runs"])
    validation = directory / "validation.json"
    runs = directory / "runs.csv"
    if not validation.is_file() or not json.loads(validation.read_text()).get("complete"):
        raise SystemExit(f"existing campaign is absent or incomplete: {directory}")
    with runs.open(newline="", encoding="utf-8") as stream:
        observed = sum(1 for _ in csv.DictReader(stream))
    if observed != expected:
        raise SystemExit(
            f"existing row count differs for {declaration['name']}: "
            f"{observed}/{expected}"
        )
    environment_path = directory / "environment.json"
    if not environment_path.is_file():
        raise SystemExit(f"missing campaign provenance: {environment_path}")
    commit = json.loads(environment_path.read_text()).get("git", {}).get("commit", "")
    if not commit or subprocess.run(
        ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode != 0:
        raise SystemExit(
            f"source commit for {declaration['name']} does not resolve locally: "
            f"{commit}; "
            "restore/push the GCP campaign commit before running new evidence"
        )
existing_rows = sum(int(item["expected_runs"]) for item in existing)
supplement_rows = sum(int(item["expected_runs"]) for item in supplement)
print(
    f"Existing publication evidence verified: {existing_rows} measured rows; "
    f"starting only {supplement_rows} new rows."
)
PY

if [ "$MODE" = "--check-only" ]; then
    echo "Supplement contract and existing 732-row evidence are valid; no solver was started."
    exit 0
fi

python3 experiments/evaluate_screening_gates.py \
    experiments/configs/screening_gates.json
python3 experiments/evaluate_evalmaxsat_calibration.py \
    --results experiments/results/gcp_evalmaxsat_lex_calibration
python3 experiments/evaluate_commercial_correctness_smoke.py \
    --results experiments/results/gcp_commercial_correctness_smoke

if [ "${CONFIRM_PUBLICATION_CAMPAIGN:-}" != "YES" ]; then
    echo "Set CONFIRM_PUBLICATION_CAMPAIGN=YES after reading docs/GCP_EXPERIMENT_RUNBOOK.md." >&2
    exit 2
fi
export WORKERS=1
export CONFIRM_FULL_CAMPAIGN=YES
export HCORAP_MAX_PEAK_RSS_MB=12288

LOG_ROOT=${HCORAP_VM_LOG_ROOT:-vm-logs}
mkdir -p "$LOG_ROOT"
STAMP=$(date -u '+%Y%m%dT%H%M%SZ')
LOG_PATH="$LOG_ROOT/corrected-exact-supplement-$STAMP.log"
{
    echo "Corrected-v2 exact-policy supplement"
    echo "UTC start: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    echo "Source revision: $(git rev-parse HEAD)"
    echo "Measured rows: 192; measured worst case: 16 core-hours"
    echo "Non-measured calibration rows: 48"
    bash experiments/gcp_prepare_and_run.sh corrected-commercial-evidence
    python3 experiments/analyze_primary_campaigns.py
    python3 experiments/analyze_corrected_validation.py
    python3 experiments/analyze_cross_paradigm_validation.py --scope full
    python3 experiments/analyze_corrected_exact_evidence.py
    python3 experiments/audit_publication_evidence.py \
        --results-root experiments/results \
        --output experiments/results/publication_evidence_audit.json
} 2>&1 | tee "$LOG_PATH"
