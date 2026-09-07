#!/usr/bin/env bash
set -Eeuo pipefail

PHASE=${1:-help}
PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$PROJECT_ROOT"

usage() {
    cat <<'EOF'
Usage: experiments/run_cardinality_aligned_full_campaign.sh PHASE

Phases:
  preflight       Build and validate the EvalMaxSAT, Gurobi, and CPLEX campaigns
  policy-encoding Run/resume 96 Gurobi and 192 EvalMaxSAT rows, then analyze them
  cplex           Run/resume 96 CPLEX rows, then analyze all three solvers
  all             Run/resume policy-encoding first and CPLEX second

Required for every phase:
  EVALMAXSAT_BIN
  GUROBI_HOME
  CPLEX_STUDIO_DIR

Required for measured phases:
  HCORAP_EXPECTED_COMMIT=<full commit currently checked out>
  CONFIRM_COMPACT_POLICY_ENCODING=YES
  CONFIRM_FULL_CPLEX_BASELINE=YES

All underlying runners are resumable. Completed valid run IDs are retained;
only missing or invalid tasks in the new cardinality_aligned_3600 directories
are executed again.
EOF
}

preflight() {
    ./experiments/run_compact_policy_encoding.sh preflight
    ./experiments/run_full_cplex_baseline.sh preflight
}

run_policy_encoding() {
    ./experiments/run_compact_policy_encoding.sh all
}

run_cplex() {
    ./experiments/run_full_cplex_baseline.sh all
}

case "$PHASE" in
    help|-h|--help)
        usage
        ;;
    preflight)
        preflight
        ;;
    policy-encoding)
        run_policy_encoding
        ;;
    cplex)
        run_cplex
        ;;
    all)
        run_policy_encoding
        run_cplex
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac
