#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$PROJECT_ROOT"

if [ "${CONFIRM_REDUCED_CAMPAIGN:-}" != "YES" ]; then
    echo "Set CONFIRM_REDUCED_CAMPAIGN=YES after reading docs/GCP_EXPERIMENT_RUNBOOK.md." >&2
    exit 2
fi

required_variables=(
    OPEN_WBO_SOURCE_DIR
    OPEN_WBO_BIN
    OPEN_WBO_COMMIT
    GUROBI_HOME
    CPLEX_STUDIO_DIR
)
for variable in "${required_variables[@]}"; do
    if [ -z "${!variable:-}" ]; then
        echo "Required environment variable is unset: $variable" >&2
        exit 2
    fi
done

export WORKERS=1
export HCORAP_CPU_CORE=${HCORAP_CPU_CORE:-0}
export CONFIRM_FULL_CAMPAIGN=YES

LOG_ROOT=${HCORAP_VM_LOG_ROOT:-vm-logs}
mkdir -p "$LOG_ROOT"
STAMP=$(date -u '+%Y%m%dT%H%M%SZ')
LOG_PATH="$LOG_ROOT/reduced-campaign-$STAMP.log"

echo "Running the reduced ICIIT campaign with one pinned worker."
echo "Log: $LOG_PATH"
bash experiments/gcp_prepare_and_run.sh all 2>&1 | tee "$LOG_PATH"
