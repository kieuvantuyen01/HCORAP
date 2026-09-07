#!/usr/bin/env bash
set -Eeuo pipefail

PHASE=${1:-help}
PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$PROJECT_ROOT"

CPLEX_CONFIG=experiments/configs/gcp_original_cplex_reference_3600.json
CPLEX_RESULTS=experiments/results/gcp_original_cplex_reference_cardinality_aligned_3600
MAXSAT_RESULTS=${HCORAP_MAXSAT_RESULTS:-experiments/results/gcp_original_policy_encoding_cardinality_aligned_3600}
GUROBI_RESULTS=${HCORAP_GUROBI_RESULTS:-experiments/results/gcp_original_policy_reference_cardinality_aligned_3600}
ANALYSIS_RESULTS=${HCORAP_COMMERCIAL_ANALYSIS:-experiments/results/gcp_original_commercial_baseline_cardinality_aligned_3600_analysis}
BUILD_JOBS=${HCORAP_BUILD_JOBS:-8}
CPU_CORE=${HCORAP_CPU_CORE:-}
RUNNER_PREFIX=()

usage() {
    cat <<'EOF'
Usage: experiments/run_full_cplex_baseline.sh PHASE

Phases:
  preflight  Build and license-test CPLEX, run tests, and resolve 96 tasks
  run        Run/resume 48 instances x 2 policies = 96 CPLEX MIP-E tasks
  analyze    Check the full CPLEX results against the existing Gurobi results
  all        Run preflight, the CPLEX campaign, collection, and analysis

Required:
  CPLEX_STUDIO_DIR   IBM ILOG CPLEX Optimization Studio installation

Required for measured phases (run and all):
  CONFIRM_FULL_CPLEX_BASELINE=YES
  HCORAP_EXPECTED_COMMIT=<full frozen commit or tag>

Optional:
  HCORAP_CPU_CORE=<allowed logical CPU>  default: first allowed CPU
  HCORAP_BUILD_JOBS=<positive integer>   default: 8
  HCORAP_BACKUP_DIR=<external directory> checkpoint after run and analysis
  HCORAP_MAXSAT_RESULTS=<directory>      full 192-row EvalMaxSAT campaign
  HCORAP_GUROBI_RESULTS=<directory>      full 96-row Gurobi reference campaign
  HCORAP_COMMERCIAL_ANALYSIS=<directory> validation and summary output

All measured runs use one thread and a 3,600-second limit. The runner is
resumable and retains every completed task. The older 300-second, 20-instance
CPLEX audit is not reused because it does not follow this runtime protocol.
EOF
}

die() {
    echo "$*" >&2
    exit 2
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || die "Required command is missing: $1"
}

check_positive_integer() {
    [[ "$1" =~ ^[1-9][0-9]*$ ]] || die "$2 must be a positive integer."
}

check_machine_and_tools() {
    [ "$(uname -s)" = "Linux" ] || die "Measured runs require Linux."
    [ "$(uname -m)" = "x86_64" ] || die "Measured runs require x86_64."
    for command_name in python3 git make sha256sum taskset; do
        require_command "$command_name"
    done
    check_positive_integer "$BUILD_JOBS" HCORAP_BUILD_JOBS
    cpu_count=$(getconf _NPROCESSORS_ONLN)
    memory_kib=$(awk '/MemTotal/ {print $2}' /proc/meminfo)
    free_disk_kib=$(df -Pk . | awk 'NR==2 {print $4}')
    [ "$cpu_count" -eq 8 ] || die "Expected exactly 8 vCPUs; found $cpu_count."
    [ "$memory_kib" -ge 15000000 ] || \
        die "Expected at least 15,000,000 KiB RAM; found $memory_kib."
    [ "$free_disk_kib" -ge 10000000 ] || \
        die "Expected at least 10 GB free disk; found $free_disk_kib KiB."
}

check_cplex_installation() {
    [ -n "${CPLEX_STUDIO_DIR:-}" ] || \
        die "Set CPLEX_STUDIO_DIR to IBM ILOG CPLEX Optimization Studio."
    [ -f "$CPLEX_STUDIO_DIR/cplex/include/ilcplex/ilocplex.h" ] || \
        die "CPLEX_STUDIO_DIR does not contain cplex/include/ilcplex/ilocplex.h."
    [ -f "$CPLEX_STUDIO_DIR/concert/include/ilconcert/iloenv.h" ] || \
        die "CPLEX_STUDIO_DIR does not contain concert/include/ilconcert/iloenv.h."
}

configure_affinity() {
    if [ -z "$CPU_CORE" ]; then
        CPU_CORE=$(python3 -c 'import os; print(min(os.sched_getaffinity(0)))')
    fi
    [[ "$CPU_CORE" =~ ^[0-9]+$ ]] || die "HCORAP_CPU_CORE must be an integer."
    python3 - "$CPU_CORE" <<'PY'
import os
import sys

core = int(sys.argv[1])
if core not in os.sched_getaffinity(0):
    raise SystemExit(
        f"HCORAP_CPU_CORE={core} is outside allowed CPUs "
        f"{sorted(os.sched_getaffinity(0))}"
    )
PY
    RUNNER_PREFIX=(taskset --cpu-list "$CPU_CORE")
}

check_measured_authorization() {
    [ "${CONFIRM_FULL_CPLEX_BASELINE:-}" = "YES" ] || \
        die "Set CONFIRM_FULL_CPLEX_BASELINE=YES after reviewing the 96-run matrix."
    [ -n "${HCORAP_EXPECTED_COMMIT:-}" ] || \
        die "Set HCORAP_EXPECTED_COMMIT to the frozen commit or tag."
    expected=$(git rev-parse --verify "${HCORAP_EXPECTED_COMMIT}^{commit}" 2>/dev/null) || \
        die "HCORAP_EXPECTED_COMMIT does not resolve to a commit."
    observed=$(git rev-parse HEAD)
    [ "$observed" = "$expected" ] || \
        die "Repository is at $observed, not expected commit $expected."
    [ -z "$(git status --porcelain)" ] || \
        die "Refusing to collect measured results from a dirty worktree."
}

build_cplex_backend() {
    # Make does not track CPLEX feature macros as dependencies.
    make -B -j"$BUILD_JOBS" YICES=0 CPLEX=1 hcorap_commercial
}

validate_result_dir() {
    python3 - "$1" <<'PY'
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
validation = json.loads((root / "validation.json").read_text(encoding="utf-8"))
if validation.get("complete") is not True:
    raise SystemExit(f"Incomplete campaign: {root}")
if validation.get("expected_runs") != 96 or validation.get("complete_runs") != 96:
    raise SystemExit(f"Expected 96 complete CPLEX rows: {root}")
with (root / "runs.csv").open(newline="", encoding="utf-8") as stream:
    rows = list(csv.DictReader(stream))
if len(rows) != 96:
    raise SystemExit(f"Expected 96 collected rows, found {len(rows)}: {root}")
invalid = [row.get("run_id") for row in rows if row.get("validation_errors")]
if invalid:
    raise SystemExit(f"Validation errors in {root}: {invalid[:5]}")
PY
}

checkpoint() {
    label=$1
    [ -n "${HCORAP_BACKUP_DIR:-}" ] || return 0
    require_command rsync
    backup_root=$(realpath -m "$HCORAP_BACKUP_DIR")
    project_root=$(realpath "$PROJECT_ROOT")
    [ "$backup_root" != "/" ] && [ "$backup_root" != "$project_root" ] || \
        die "HCORAP_BACKUP_DIR must be outside the project root."
    case "$backup_root" in
        "$project_root"/*) die "HCORAP_BACKUP_DIR must be outside the worktree." ;;
    esac
    destination=$backup_root/hcorap_full_cplex_baseline
    mkdir -p "$destination"
    if [ -d "$CPLEX_RESULTS" ]; then
        rsync -a "$CPLEX_RESULTS/" "$destination/cplex-results/"
    fi
    if [ -d "$ANALYSIS_RESULTS" ]; then
        rsync -a "$ANALYSIS_RESULTS/" "$destination/analysis/"
    fi
    {
        date -u '+checkpoint_utc=%Y-%m-%dT%H:%M:%SZ'
        echo "phase=$label"
        echo "source_commit=$(git rev-parse HEAD)"
    } > "$destination/checkpoint.txt"
    sync
}

preflight() {
    check_machine_and_tools
    check_cplex_installation
    configure_affinity
    build_cplex_backend
    python3 -m pytest -q
    "${RUNNER_PREFIX[@]}" python3 experiments/run_commercial_campaign.py \
        "$CPLEX_CONFIG" --dry-run
    echo "Preflight passed on CPU $CPU_CORE for 48 instances x 2 policies = 96 CPLEX runs."
}

run_cplex() {
    build_cplex_backend
    "${RUNNER_PREFIX[@]}" python3 experiments/run_commercial_campaign.py \
        "$CPLEX_CONFIG" --resume --workers 1
    python3 experiments/collect_commercial_campaign.py "$CPLEX_RESULTS"
    validate_result_dir "$CPLEX_RESULTS"
    checkpoint original-cplex-reference-3600
}

analyze_results() {
    [ -d "$MAXSAT_RESULTS" ] || \
        die "EvalMaxSAT result directory is missing: $MAXSAT_RESULTS"
    [ -d "$GUROBI_RESULTS" ] || \
        die "Gurobi result directory is missing: $GUROBI_RESULTS"
    python3 experiments/analyze_full_commercial_baseline.py \
        --maxsat-results "$MAXSAT_RESULTS" \
        --gurobi-results "$GUROBI_RESULTS" \
        --cplex-results "$CPLEX_RESULTS" \
        --output "$ANALYSIS_RESULTS" \
        --expected-instances 48
    checkpoint original-commercial-baseline-3600-analysis
}

case "$PHASE" in
    help|-h|--help)
        usage
        ;;
    preflight)
        preflight
        ;;
    run)
        check_machine_and_tools
        check_cplex_installation
        configure_affinity
        check_measured_authorization
        run_cplex
        ;;
    analyze)
        analyze_results
        ;;
    all)
        preflight
        check_measured_authorization
        run_cplex
        analyze_results
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac
