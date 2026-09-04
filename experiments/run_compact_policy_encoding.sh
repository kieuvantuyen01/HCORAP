#!/usr/bin/env bash
set -Eeuo pipefail

PHASE=${1:-help}
PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$PROJECT_ROOT"

MAXSAT_CONFIG=experiments/configs/gcp_original_policy_encoding_3600.json
REFERENCE_CONFIG=experiments/configs/gcp_original_policy_reference_3600.json
MAXSAT_RESULTS=experiments/results/gcp_original_policy_encoding_3600
REFERENCE_RESULTS=experiments/results/gcp_original_policy_reference_3600
ANALYSIS_RESULTS=experiments/results/gcp_original_policy_encoding_3600_analysis
POLICY_ANALYSIS=${HCORAP_POLICY_ANALYSIS:-results_v2/gcp_corrected_exact_analysis}
MANUSCRIPT_RESULTS=${HCORAP_MANUSCRIPT_RESULTS:-LaTeX-Templates/paper/generated_compact}
PINNED_EVALMAXSAT_SHA256=$(python3 -c \
    'import json; print(json.load(open("experiments/configs/reduced_campaign_manifest.json"))["maxsat_solver"]["sha256"])')
BUILD_JOBS=${HCORAP_BUILD_JOBS:-8}
CPU_CORE=${HCORAP_CPU_CORE:-}
RUNNER_PREFIX=()

usage() {
    cat <<'EOF'
Usage: experiments/run_compact_policy_encoding.sh PHASE

Phases:
  preflight  Build, test, verify tools, and resolve the two fixed matrices
  reference  Run/resume 96 Gurobi reference rows
  maxsat     Run/resume 192 EvalMaxSAT Policy x Encoding rows
  analyze    Validate and analyze the two completed campaigns
  manuscript Generate gated LaTeX result fragments from both validated studies
  all        Run preflight, reference, MaxSAT, and analysis

Required:
  EVALMAXSAT_BIN    pinned Linux x86-64 EvalMaxSAT executable
  GUROBI_HOME       Gurobi installation containing include/ and lib/

Required for measured phases (reference, maxsat, and all):
  CONFIRM_COMPACT_POLICY_ENCODING=YES
  HCORAP_EXPECTED_COMMIT=<full frozen commit or tag>

Optional:
  HCORAP_CPU_CORE=<allowed logical CPU>  default: first allowed CPU
  HCORAP_BUILD_JOBS=<positive integer>   default: 8
  HCORAP_BACKUP_DIR=<external directory> checkpoint after each phase
  HCORAP_POLICY_ANALYSIS=<directory>      validated Corrected-v2 analysis
  HCORAP_MANUSCRIPT_RESULTS=<directory>   generated LaTeX fragments

The MaxSAT matrix is 48 Original instances x 2 policies x 2 encodings, with
Totalizer and sorting network both using no implied constraints and no symmetry
breaking. Every top-level run has one cumulative 3,600-second budget.
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
    [ "$(uname -m)" = "x86_64" ] || die "EvalMaxSAT requires Linux x86-64."
    for command_name in python3 git make sha256sum taskset; do
        require_command "$command_name"
    done
    check_positive_integer "$BUILD_JOBS" HCORAP_BUILD_JOBS
    cpu_count=$(getconf _NPROCESSORS_ONLN)
    memory_kib=$(awk '/MemTotal/ {print $2}' /proc/meminfo)
    free_disk_kib=$(df -Pk . | awk 'NR==2 {print $4}')
    [ "$cpu_count" -ge 8 ] || die "Expected at least 8 vCPUs; found $cpu_count."
    [ "$memory_kib" -ge 15000000 ] || \
        die "Expected at least 15,000,000 KiB RAM; found $memory_kib."
    [ "$free_disk_kib" -ge 10000000 ] || \
        die "Expected at least 10 GB free disk; found $free_disk_kib KiB."
}

check_solvers() {
    [ -n "${EVALMAXSAT_BIN:-}" ] && [ -x "$EVALMAXSAT_BIN" ] || \
        die "Set EVALMAXSAT_BIN to the executable EvalMaxSAT binary."
    observed_hash=$(sha256sum "$EVALMAXSAT_BIN" | awk '{print $1}')
    [ "$observed_hash" = "$PINNED_EVALMAXSAT_SHA256" ] || \
        die "EvalMaxSAT SHA-256 mismatch. Expected $PINNED_EVALMAXSAT_SHA256; found $observed_hash."
    [ -n "${GUROBI_HOME:-}" ] || die "Set GUROBI_HOME before running this campaign."
    [ -f "$GUROBI_HOME/include/gurobi_c++.h" ] || \
        die "GUROBI_HOME does not contain include/gurobi_c++.h."
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
    [ "${CONFIRM_COMPACT_POLICY_ENCODING:-}" = "YES" ] || \
        die "Set CONFIRM_COMPACT_POLICY_ENCODING=YES after reviewing docs/COMPACT_RESULTS_RUNBOOK.md."
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

build_and_test() {
    make -j"$BUILD_JOBS" YICES=0 hcorap_multi
    make -B -j"$BUILD_JOBS" YICES=0 GUROBI=1 hcorap_commercial
    python3 -m pytest -q
}

validate_configs() {
    python3 experiments/run_reproducible_campaign.py "$MAXSAT_CONFIG" --dry-run
    python3 experiments/run_commercial_campaign.py "$REFERENCE_CONFIG" --dry-run
}

preflight() {
    check_machine_and_tools
    check_solvers
    configure_affinity
    build_and_test
    validate_configs
    echo "Preflight passed on CPU $CPU_CORE for the fixed 192 + 96 row matrix."
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
    destination=$backup_root/hcorap_compact_policy_encoding
    mkdir -p "$destination"
    rsync -a experiments/results/ "$destination/results/"
    {
        date -u '+checkpoint_utc=%Y-%m-%dT%H:%M:%SZ'
        echo "phase=$label"
        echo "source_commit=$(git rev-parse HEAD)"
        echo "evalmaxsat_sha256=$PINNED_EVALMAXSAT_SHA256"
    } > "$destination/checkpoint.txt"
    sync
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
with (root / "runs.csv").open(newline="", encoding="utf-8") as stream:
    rows = list(csv.DictReader(stream))
invalid = [row.get("run_id") for row in rows if row.get("validation_errors")]
if invalid:
    raise SystemExit(f"Validation errors in {root}: {invalid[:5]}")
PY
}

run_reference() {
    "${RUNNER_PREFIX[@]}" python3 experiments/run_commercial_campaign.py \
        "$REFERENCE_CONFIG" --resume --workers 1
    python3 experiments/collect_commercial_campaign.py "$REFERENCE_RESULTS"
    validate_result_dir "$REFERENCE_RESULTS"
    checkpoint original-policy-reference-3600
}

run_maxsat() {
    "${RUNNER_PREFIX[@]}" python3 experiments/run_reproducible_campaign.py \
        "$MAXSAT_CONFIG" --resume --workers 1
    python3 experiments/collect_reproducible_campaign.py "$MAXSAT_RESULTS"
    validate_result_dir "$MAXSAT_RESULTS"
    checkpoint original-policy-encoding-3600
}

analyze_results() {
    python3 experiments/analyze_policy_encoding_matrix.py \
        --maxsat-results "$MAXSAT_RESULTS" \
        --exact-results "$REFERENCE_RESULTS" \
        --output "$ANALYSIS_RESULTS" \
        --expected-instances 48
    checkpoint original-policy-encoding-3600-analysis
}

generate_manuscript_results() {
    python3 experiments/generate_compact_manuscript_results.py \
        --policy-analysis "$POLICY_ANALYSIS" \
        --encoding-analysis "$ANALYSIS_RESULTS" \
        --output "$MANUSCRIPT_RESULTS"
}

case "$PHASE" in
    help|-h|--help)
        usage
        ;;
    preflight)
        preflight
        ;;
    reference|maxsat)
        check_machine_and_tools
        check_solvers
        configure_affinity
        check_measured_authorization
        if [ "$PHASE" = "reference" ]; then
            run_reference
        else
            run_maxsat
        fi
        ;;
    analyze)
        analyze_results
        ;;
    manuscript)
        generate_manuscript_results
        ;;
    all)
        preflight
        check_measured_authorization
        run_reference
        run_maxsat
        analyze_results
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac
