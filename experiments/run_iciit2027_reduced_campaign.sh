#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$PROJECT_ROOT"

if [ "${CONFIRM_REDUCED_CAMPAIGN:-}" != "YES" ]; then
    echo "Set CONFIRM_REDUCED_CAMPAIGN=YES after reading docs/GCP_EXPERIMENT_RUNBOOK.md." >&2
    exit 2
fi

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

export WORKERS=1
export HCORAP_CPU_CORE=${HCORAP_CPU_CORE:-0}
export CONFIRM_PUBLICATION_CAMPAIGN=YES

echo "Compatibility wrapper: delegating to run_all_remaining_publication.sh."
exec bash experiments/run_all_remaining_publication.sh run
