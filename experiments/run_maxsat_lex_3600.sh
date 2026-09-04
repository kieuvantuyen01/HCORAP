#!/usr/bin/env bash
set -euo pipefail

PHASE=${1:-help}
PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$PROJECT_ROOT"

PINNED_EVALMAXSAT_SHA256=$(python3 -c \
    'import json; print(json.load(open("experiments/configs/reduced_campaign_manifest.json"))["maxsat_solver"]["sha256"])')
BUILD_JOBS=${HCORAP_BUILD_JOBS:-8}
CPU_CORE=${HCORAP_CPU_CORE:-}
RUNNER_PREFIX=()

PILOT_MAXSAT=experiments/results/gcp_maxsat_lex_3600_pilot
PILOT_GUROBI=experiments/results/gcp_maxsat_lex_3600_pilot_gurobi
PILOT_ANALYSIS=experiments/results/gcp_maxsat_lex_3600_pilot_analysis
PILOT_DECISION=$PILOT_ANALYSIS/maxsat_lex_3600_decision.json
CONFIRM_GUROBI=experiments/results/gcp_maxsat_lex_3600_confirm_gurobi
CONFIRM_ANALYSIS=experiments/results/gcp_maxsat_lex_3600_confirmation_analysis

usage() {
    cat <<'EOF'
Usage: experiments/run_maxsat_lex_3600.sh PHASE

Phases:
  preflight       Build, test, validate EvalMaxSAT/Gurobi, and dry-run configs
  pilot           Run the 16-instance pilot and select at most one candidate
  confirm         Run the locked 48-instance confirmation selected by the pilot
  all             Run preflight, pilot, and confirmation
  analyze-pilot   Rebuild pilot tables without running solvers
  analyze-confirm Rebuild confirmation tables without running solvers

Required for preflight and measured runs:
  EVALMAXSAT_BIN       pinned Linux x86-64 EvalMaxSAT executable
  GUROBI_HOME          Gurobi installation containing include/ and lib/

Required for pilot, confirm, and all:
  CONFIRM_MAXSAT_LEX_3600=YES
  HCORAP_EXPECTED_COMMIT=<full commit or tag>

Optional:
  HCORAP_CPU_CORE=<allowed logical CPU>  default: first allowed CPU
  HCORAP_BUILD_JOBS=<positive integer>   default: 8
  HCORAP_BACKUP_DIR=<external directory> checkpoint after each campaign

Worst-case MaxSAT budget is 48 core-hours when the pilot stops, or 144
core-hours when one candidate advances to paired confirmation. Runs are
resumable and are intentionally sequential.
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
    for command in python3 git make sha256sum taskset; do
        require_command "$command"
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
    [ "${CONFIRM_MAXSAT_LEX_3600:-}" = "YES" ] || \
        die "Set CONFIRM_MAXSAT_LEX_3600=YES to authorize this measured campaign."
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
    # Feature flags are not dependency-tracked, so force a commercial rebuild.
    make -B -j"$BUILD_JOBS" YICES=0 GUROBI=1 hcorap_commercial
    python3 -m pytest -q
}

validate_smoke() {
    smoke_dir=$(mktemp -d)
    trap 'rm -r "$smoke_dir"' EXIT
    for method in lex-cos lex-cos-one-shot; do
        "${RUNNER_PREFIX[@]}" bin/release/hcorap_multi \
            tests/instances/lex_cos_tie.txt \
            --solver "$EVALMAXSAT_BIN" \
            --timeout 30 \
            --solver-shutdown-grace 5 \
            --align-evalmaxsat-tct \
            --method "$method" \
            --cardinality-encoding totalizer \
            --implied-constraints none \
            --symmetry-breaking none \
            --print-assignments \
            --output "$smoke_dir/$method.json" >/dev/null
    done
    python3 - "$smoke_dir" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
for method, calls in (("lex-cos", 3), ("lex-cos-one-shot", 1)):
    payload = json.loads((root / f"{method}.json").read_text(encoding="utf-8"))
    if payload.get("status") != "OPTIMUM":
        raise SystemExit(f"EvalMaxSAT smoke failed for {method}: {payload}")
    if payload.get("solver_calls") != calls:
        raise SystemExit(f"Unexpected solver-call count for {method}: {payload}")
    if payload.get("certified_lexicographic_prefix") != 3:
        raise SystemExit(f"Uncertified smoke result for {method}: {payload}")
    if payload.get("metrics", {}).get("verified") is not True:
        raise SystemExit(f"Verifier rejected smoke result for {method}: {payload}")
PY
    rm -r "$smoke_dir"
    trap - EXIT
}

validate_configs() {
    maxsat_configs=(
        gcp_maxsat_lex_3600_pilot
        gcp_maxsat_lex_3600_confirm_bound
        gcp_maxsat_lex_3600_confirm_one_shot
    )
    for name in "${maxsat_configs[@]}"; do
        python3 experiments/run_reproducible_campaign.py \
            "experiments/configs/$name.json" --dry-run
    done
    for name in \
        gcp_maxsat_lex_3600_pilot_gurobi \
        gcp_maxsat_lex_3600_confirm_gurobi; do
        python3 experiments/run_commercial_campaign.py \
            "experiments/configs/$name.json" --dry-run
    done
    validate_smoke
}

preflight() {
    check_machine_and_tools
    check_solvers
    configure_affinity
    build_and_test
    validate_configs
    echo "Preflight passed on CPU $CPU_CORE with the pinned EvalMaxSAT binary and Gurobi."
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
    destination=$backup_root/hcorap_maxsat_lex_3600
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

run_maxsat() {
    config=$1
    result_dir=$2
    "${RUNNER_PREFIX[@]}" python3 experiments/run_reproducible_campaign.py \
        "$config" --resume --workers 1
    python3 experiments/collect_reproducible_campaign.py "$result_dir"
    validate_result_dir "$result_dir"
    checkpoint "$(basename "$result_dir")"
}

run_gurobi() {
    config=$1
    result_dir=$2
    "${RUNNER_PREFIX[@]}" python3 experiments/run_commercial_campaign.py \
        "$config" --resume --workers 1
    python3 experiments/collect_commercial_campaign.py "$result_dir"
    validate_result_dir "$result_dir"
    checkpoint "$(basename "$result_dir")"
}

analyze_pilot() {
    python3 experiments/analyze_maxsat_lex_3600.py \
        --maxsat-results "$PILOT_MAXSAT" \
        --exact-results "$PILOT_GUROBI" \
        --output "$PILOT_ANALYSIS" \
        --expected-instances 16 \
        --required-variants \
        staged-aligned,staged-incumbent-bound,single-call-dominance
}

run_pilot() {
    run_gurobi \
        experiments/configs/gcp_maxsat_lex_3600_pilot_gurobi.json \
        "$PILOT_GUROBI"
    run_maxsat \
        experiments/configs/gcp_maxsat_lex_3600_pilot.json \
        "$PILOT_MAXSAT"
    analyze_pilot
    checkpoint maxsat-lex-3600-pilot-analysis
}

confirmation_selection() {
    [ -f "$PILOT_DECISION" ] || die "Missing pilot decision: $PILOT_DECISION"
    python3 - "$PILOT_DECISION" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if report.get("structurally_valid") is not True:
    raise SystemExit("Pilot evidence is structurally invalid.")
decision = report.get("decision")
if decision not in {"GO", "STOP"}:
    raise SystemExit(f"Unsupported pilot decision: {decision}")
if decision == "STOP":
    print("STOP|staged-aligned")
    raise SystemExit(0)
print(
    report["confirmation_config"]
    + "|"
    + (report.get("selected_variant") or "staged-aligned")
)
PY
}

analyze_confirmation() {
    selection=$(confirmation_selection)
    IFS='|' read -r config_name selected <<< "$selection"
    if [ "$config_name" = "STOP" ]; then
        echo "Pilot decision is STOP; no confirmation campaign is authorized."
        return 0
    fi
    result_dir=experiments/results/${config_name%.json}
    required=staged-aligned
    if [ "$selected" != "staged-aligned" ]; then
        required=$required,$selected
    fi
    python3 experiments/analyze_maxsat_lex_3600.py \
        --maxsat-results "$result_dir" \
        --exact-results "$CONFIRM_GUROBI" \
        --output "$CONFIRM_ANALYSIS" \
        --expected-instances 48 \
        --required-variants "$required"
    python3 - "$PILOT_DECISION" \
        "$CONFIRM_ANALYSIS/maxsat_lex_3600_decision.json" <<'PY'
import json
import sys
from pathlib import Path

pilot = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
confirmation = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
selected = pilot.get("selected_variant")
if selected and confirmation.get("selected_variant") != selected:
    raise SystemExit(
        f"Pilot candidate {selected} was not confirmed; retain the results but "
        "do not claim an improvement."
    )
PY
}

run_confirmation() {
    selection=$(confirmation_selection)
    IFS='|' read -r config_name selected <<< "$selection"
    if [ "$config_name" = "STOP" ]; then
        echo "Pilot decision is STOP; skipping baseline-only confirmation."
        checkpoint maxsat-lex-3600-pilot-stop
        return 0
    fi
    config=experiments/configs/$config_name
    result_dir=experiments/results/${config_name%.json}
    run_gurobi \
        experiments/configs/gcp_maxsat_lex_3600_confirm_gurobi.json \
        "$CONFIRM_GUROBI"
    run_maxsat "$config" "$result_dir"
    analyze_confirmation
    checkpoint maxsat-lex-3600-confirmation-analysis
}

if [ "$PHASE" = "help" ] || [ "$PHASE" = "--help" ] || [ "$PHASE" = "-h" ]; then
    usage
    exit 0
fi

case "$PHASE" in
    analyze-pilot)
        analyze_pilot
        ;;
    analyze-confirm)
        analyze_confirmation
        ;;
    preflight)
        preflight
        ;;
    pilot)
        preflight
        check_measured_authorization
        run_pilot
        ;;
    confirm)
        preflight
        check_measured_authorization
        run_confirmation
        ;;
    all)
        preflight
        check_measured_authorization
        run_pilot
        run_confirmation
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac
