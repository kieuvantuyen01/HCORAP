#!/usr/bin/env bash
# Launch the approved 2,208-run research-depth full campaign on the GCP VM.
#
# Usage with an environment file outside the repository:
#   nohup experiments/launch_research_depth_full_gcp.sh \
#     /path/to/research_depth_gcp.env > research-depth-full.log 2>&1 &
#
# If the required variables are already exported, omit the argument.
set -Eeuo pipefail

PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$PROJECT_ROOT"

ENV_FILE=${1:-${HCORAP_ENV_FILE:-}}
if [ -n "$ENV_FILE" ]; then
    [ -f "$ENV_FILE" ] || {
        printf 'ERROR: environment file does not exist: %s\n' "$ENV_FILE" >&2
        exit 2
    }
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
fi

CURRENT_COMMIT=$(git rev-parse HEAD)
export HCORAP_EXPECTED_COMMIT=$CURRENT_COMMIT
export CONFIRM_RESEARCH_DEPTH_FULL=YES

printf 'Research-depth full campaign\n'
printf '  source commit: %s\n' "$CURRENT_COMMIT"
printf '  diagnostics:     480 runs\n'
printf '  weights:         432 runs\n'
printf '  load:          1,296 runs\n'
printf '  total:         2,208 runs\n'
printf 'The runner will validate the machine, source, inputs, licenses, binaries,\n'
printf 'campaign matrices and completed run IDs before running or resuming.\n'

exec experiments/run_research_depth_gcp.sh full
