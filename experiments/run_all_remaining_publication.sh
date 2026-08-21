#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$PROJECT_ROOT"

contract_values=$(python3 - <<'PY'
import json
from pathlib import Path
manifest = json.loads(
    Path("experiments/configs/reduced_campaign_manifest.json").read_text()
)
runs = int(manifest["expected_measured_runs"])
seconds = int(manifest["expected_worst_case_seconds"])
print(runs, seconds, f"{seconds / 3600:g}", f"{seconds / 86400:.4f}")
PY
)
read -r measured_runs worst_seconds worst_hours worst_days <<< "$contract_values"

MODE=${1:-run}
case "$MODE" in
    run|--check-only|-h|--help) ;;
    *)
        echo "Usage: experiments/run_all_remaining_publication.sh [run|--check-only]" >&2
        exit 2
        ;;
esac
if [ "$MODE" = "-h" ] || [ "$MODE" = "--help" ]; then
    cat <<EOF
Usage: experiments/run_all_remaining_publication.sh [run|--check-only]

  run           Run/resume all $measured_runs measured publication rows (default).
  --check-only  Validate the frozen campaign contract without running solvers.

The measured timeout is locked to 300 seconds per top-level run.  Smoke tests
use 30 seconds and are excluded from publication timing.
EOF
    exit 0
fi

python3 experiments/validate_campaign_manifest.py
python3 experiments/validate_publication_campaign.py
if [ "$MODE" = "--check-only" ]; then
    echo "Publication campaign contract is valid; no solver was started."
    exit 0
fi

if [ "$(uname -s)" != "Linux" ]; then
    echo "Measured publication runs are restricted to the designated Linux GCP VM." >&2
    exit 2
fi
if [ "${CONFIRM_PUBLICATION_CAMPAIGN:-}" != "YES" ]; then
    echo "Set CONFIRM_PUBLICATION_CAMPAIGN=YES after reading docs/GCP_EXPERIMENT_RUNBOOK.md." >&2
    exit 2
fi

required_commands=(flock git python3 realpath rsync sha256sum tee)
for command_name in "${required_commands[@]}"; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
        echo "Required command is unavailable: $command_name" >&2
        exit 2
    fi
done

required_variables=(
    EVALMAXSAT_BIN
    HCORAP_EXPECTED_COMMIT
    HCORAP_BACKUP_DIR
    GUROBI_HOME
    CPLEX_STUDIO_DIR
)
for variable in "${required_variables[@]}"; do
    if [ -z "${!variable:-}" ]; then
        echo "Required environment variable is unset: $variable" >&2
        exit 2
    fi
done

project_root=$(realpath "$PROJECT_ROOT")
backup_root=$(realpath -m "$HCORAP_BACKUP_DIR")
if [ "$backup_root" = "/" ] || [ "$backup_root" = "$project_root" ]; then
    echo "HCORAP_BACKUP_DIR must not be / or the project root." >&2
    exit 2
fi
case "$backup_root" in
    "$project_root"/*)
        echo "HCORAP_BACKUP_DIR must be outside the project worktree." >&2
        exit 2
        ;;
esac
mkdir -p "$backup_root"

export WORKERS=1
export HCORAP_CPU_CORE=${HCORAP_CPU_CORE:-0}
export HCORAP_MAX_PEAK_RSS_MB=12288
export CONFIRM_FULL_CAMPAIGN=YES

LOG_ROOT=${HCORAP_VM_LOG_ROOT:-vm-logs}
mkdir -p "$LOG_ROOT"
STAMP=$(date -u '+%Y%m%dT%H%M%SZ')
LOG_PATH="$LOG_ROOT/publication-campaign-$STAMP.log"
LOCK_PATH="$LOG_ROOT/.publication-campaign.lock"

exec 9>"$LOCK_PATH"
if ! flock -n 9; then
    echo "Another publication-campaign wrapper already holds $LOCK_PATH." >&2
    exit 2
fi

checkpoint_on_exit() {
    status=$?
    trap - EXIT
    set +e
    checkpoint_root="$backup_root/hcorap_iciit2027_checkpoint"
    mkdir -p "$checkpoint_root/results" "$checkpoint_root/vm-logs"
    if [ -d experiments/results ]; then
        rsync -a experiments/results/ "$checkpoint_root/results/"
    fi
    if [ -d "$LOG_ROOT" ]; then
        rsync -a "$LOG_ROOT"/ "$checkpoint_root/vm-logs/"
    fi
    {
        date -u '+checkpoint_utc=%Y-%m-%dT%H:%M:%SZ'
        echo "phase=wrapper-exit"
        echo "exit_status=$status"
        echo "source_commit=$(git rev-parse HEAD 2>/dev/null)"
    } > "$checkpoint_root/wrapper-checkpoint.txt"
    sync
    if [ "$status" -eq 0 ]; then
        echo "Campaign completed; final checkpoint: $checkpoint_root"
    else
        echo "Campaign stopped with status $status; resumable data checkpointed to $checkpoint_root" >&2
    fi
    exit "$status"
}
trap checkpoint_on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

{
    echo "ICIIT 2027 publication campaign"
    echo "UTC start: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    echo "Source revision: $(git rev-parse HEAD)"
    echo "EvalMaxSAT SHA-256: $(sha256sum "$EVALMAXSAT_BIN" | awk '{print $1}')"
    echo "Expected measured rows: $measured_runs"
    echo "Non-measured gates: 4 EvalMaxSAT calibration + 18 smoke + 48 corrected commercial calibration rows"
    echo "Measured timeout: 300 seconds per top-level run"
    echo "Worst-case solver budget: $worst_seconds seconds = $worst_hours core-hours = $worst_days sequential days"
    echo "Worker policy: one worker, one pinned vCPU, 12 GB peak-RSS gate"
    echo "Log: $LOG_PATH"
    python3 experiments/validate_publication_campaign.py
    bash experiments/gcp_prepare_and_run.sh all
} 2>&1 | tee "$LOG_PATH"
