#!/usr/bin/env bash
set -Eeuo pipefail

PHASE=${1:-help}
PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$PROJECT_ROOT"

BUILD_JOBS=${HCORAP_BUILD_JOBS:-8}
CPU_CORE=${HCORAP_CPU_CORE:-}
EXPECTED_VCPUS=${HCORAP_EXPECTED_VCPUS:-8}
CORRECTED_SOURCE_RESULTS=${HCORAP_CORRECTED_SOURCE_RESULTS:-results_v2/gcp_commercial_corrected_primary}
LOAD_MANIFEST=instances/research_depth_load_sweep/load_sweep_manifest.json
BACKUP_ROOT=${HCORAP_BACKUP_DIR:-}
RUNNER_PREFIX=()

DIAGNOSTICS_PILOT_CONFIG=experiments/configs/research_depth_diagnostics_pilot.json
CPLEX_AUDIT_CONFIG=experiments/configs/research_depth_diagnostics_cplex_audit.json
WEIGHTS_PILOT_CONFIG=experiments/configs/research_depth_weights_pilot.json
LOAD_PILOT_CONFIG=experiments/configs/research_depth_load_pilot.json
MAXSAT_PILOT_CONFIG=experiments/configs/research_depth_zero_continuity_pilot.json

DIAGNOSTICS_FULL_CONFIG=experiments/configs/research_depth_diagnostics_full.json
WEIGHTS_FULL_CONFIG=experiments/configs/research_depth_weights_full.json
LOAD_FULL_CONFIG=experiments/configs/research_depth_load_full.json

DIAGNOSTICS_PILOT_RESULTS=experiments/results/research_depth_diagnostics_pilot
CPLEX_AUDIT_RESULTS=experiments/results/research_depth_diagnostics_cplex_audit
WEIGHTS_PILOT_RESULTS=experiments/results/research_depth_weights_pilot
LOAD_PILOT_RESULTS=experiments/results/research_depth_load_pilot
MAXSAT_PILOT_RESULTS=experiments/results/research_depth_zero_continuity_pilot
AUDIT_RESULTS=experiments/results/research_depth_diagnostic_audit

DIAGNOSTICS_FULL_RESULTS=experiments/results/research_depth_diagnostics_full
WEIGHTS_FULL_RESULTS=experiments/results/research_depth_weights_full
LOAD_FULL_RESULTS=experiments/results/research_depth_load_full

usage() {
    cat <<'EOF'
Usage: experiments/run_research_depth_gcp.sh PHASE

Phases:
  preflight         Check GCP, inputs, licenses and solvers; build and test
  pilot-commercial Run/resume Gurobi diagnostics, CPLEX audit, weights and load
  pilot-maxsat      Run/resume the 64-run CONT=0 MaxSAT comparison
  pilot             Run both pilot parts and all pilot analyses
  full              Run/resume the three full Gurobi campaigns and analyses
  analyze-pilot     Recollect and analyze existing pilot artifacts only
  analyze-full      Recollect and analyze existing full artifacts only
  status            Print validation/status counts from existing result folders
  all               Preflight, pilot, then full

Required for preflight and measured runs:
  GUROBI_HOME            Gurobi installation containing include/ and lib/
  CPLEX_STUDIO_DIR       IBM ILOG CPLEX Optimization Studio installation
  EVALMAXSAT_BIN         Linux EvalMaxSAT executable

Required for measured phases:
  HCORAP_EXPECTED_COMMIT=<full frozen commit or tag>
  CONFIRM_RESEARCH_DEPTH_PILOT=YES  for pilot and pilot-* phases
  CONFIRM_RESEARCH_DEPTH_FULL=YES   for full and all

Required once when the generated load sweep is absent:
  HCORAP_CORRECTED_SOURCE_RESULTS=<archived 144-run corrected campaign>
  Default: results_v2/gcp_commercial_corrected_primary

Optional:
  HCORAP_CPU_CORE=<allowed logical CPU>   default: first allowed CPU
  HCORAP_BUILD_JOBS=<positive integer>    default: 8
  HCORAP_EXPECTED_VCPUS=<positive integer> default: 8
  HCORAP_BACKUP_DIR=<external directory>  checkpoint after each component

Recommended GCP invocation:
  nohup experiments/run_research_depth_gcp.sh pilot \
    > research-depth-pilot.log 2>&1 &

The pilot contains 960 driver runs: 320 diagnostics (Gurobi + CPLEX),
144 weight runs, 432 load runs, and 64 MaxSAT runs. The full phase contains
2,208 Gurobi runs and deliberately repeats the pilot cells in separate result
directories. Every runner is resumable and keeps valid completed run IDs.
EOF
}

log() {
    printf '[%s] %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*"
}

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 2
}

on_error() {
    local code=$?
    printf 'FAILED: phase=%s line=%s exit=%s\n' "$PHASE" "$1" "$code" >&2
    printf 'Rerun the same phase to resume completed tasks.\n' >&2
    exit "$code"
}
trap 'on_error $LINENO' ERR

require_command() {
    command -v "$1" >/dev/null 2>&1 || die "Required command is missing: $1"
}

require_positive_integer() {
    [[ "$1" =~ ^[1-9][0-9]*$ ]] || die "$2 must be a positive integer."
}

check_machine() {
    [ "$(uname -s)" = Linux ] || die "Measured experiments require Linux."
    [ "$(uname -m)" = x86_64 ] || die "Measured experiments require x86_64."
    for name in python3 git make sha256sum taskset getconf awk df find wc \
        realpath mktemp; do
        require_command "$name"
    done
    require_positive_integer "$BUILD_JOBS" HCORAP_BUILD_JOBS
    require_positive_integer "$EXPECTED_VCPUS" HCORAP_EXPECTED_VCPUS
    local observed_vcpus memory_kib free_disk_kib
    observed_vcpus=$(getconf _NPROCESSORS_ONLN)
    memory_kib=$(awk '/MemTotal/ {print $2}' /proc/meminfo)
    free_disk_kib=$(df -Pk . | awk 'NR==2 {print $4}')
    [ "$observed_vcpus" -eq "$EXPECTED_VCPUS" ] || \
        die "Expected $EXPECTED_VCPUS vCPUs; found $observed_vcpus."
    [ "$memory_kib" -ge 15000000 ] || \
        die "Expected at least 15,000,000 KiB RAM; found $memory_kib."
    [ "$free_disk_kib" -ge 10000000 ] || \
        die "Expected at least 10 GB free disk; found $free_disk_kib KiB."
}

check_solver_installations() {
    [ -n "${GUROBI_HOME:-}" ] || die "Set GUROBI_HOME."
    [ -f "$GUROBI_HOME/include/gurobi_c++.h" ] || \
        die "GUROBI_HOME does not contain include/gurobi_c++.h."
    compgen -G "$GUROBI_HOME/lib/libgurobi[0-9]*.so" >/dev/null || \
        die "No versioned Gurobi shared library found under GUROBI_HOME/lib."

    [ -n "${CPLEX_STUDIO_DIR:-}" ] || die "Set CPLEX_STUDIO_DIR."
    [ -f "$CPLEX_STUDIO_DIR/cplex/include/ilcplex/ilocplex.h" ] || \
        die "CPLEX headers are missing under CPLEX_STUDIO_DIR."
    [ -f "$CPLEX_STUDIO_DIR/concert/include/ilconcert/iloenv.h" ] || \
        die "Concert headers are missing under CPLEX_STUDIO_DIR."

    [ -n "${EVALMAXSAT_BIN:-}" ] || die "Set EVALMAXSAT_BIN."
    [ -f "$EVALMAXSAT_BIN" ] || die "EvalMaxSAT does not exist: $EVALMAXSAT_BIN"
    [ -x "$EVALMAXSAT_BIN" ] || die "EvalMaxSAT is not executable: $EVALMAXSAT_BIN"
    EVALMAXSAT_BIN=$(realpath "$EVALMAXSAT_BIN")
    export EVALMAXSAT_BIN
    export LD_LIBRARY_PATH="$GUROBI_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
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
allowed = sorted(os.sched_getaffinity(0))
if core not in allowed:
    raise SystemExit(f"HCORAP_CPU_CORE={core} is outside allowed CPUs {allowed}")
PY
    RUNNER_PREFIX=(taskset --cpu-list "$CPU_CORE")
}

check_measured_source() {
    [ -n "${HCORAP_EXPECTED_COMMIT:-}" ] || \
        die "Set HCORAP_EXPECTED_COMMIT to the frozen commit or tag."
    local expected observed
    expected=$(git rev-parse --verify "${HCORAP_EXPECTED_COMMIT}^{commit}" 2>/dev/null) || \
        die "HCORAP_EXPECTED_COMMIT does not resolve to a commit."
    observed=$(git rev-parse HEAD)
    [ "$observed" = "$expected" ] || \
        die "Repository is at $observed, not expected commit $expected."
    [ -z "$(git status --porcelain)" ] || \
        die "Refusing measured runs from a dirty worktree. Commit the implementation first."
}

check_pilot_authorization() {
    [ "${CONFIRM_RESEARCH_DEPTH_PILOT:-}" = YES ] || \
        die "Set CONFIRM_RESEARCH_DEPTH_PILOT=YES after reviewing the 960-run pilot."
}

check_full_authorization() {
    [ "${CONFIRM_RESEARCH_DEPTH_FULL:-}" = YES ] || \
        die "Set CONFIRM_RESEARCH_DEPTH_FULL=YES after reviewing pilot results and the 2,208-run matrix."
}

validate_load_sweep() {
    python3 - "$LOAD_MANIFEST" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
payload = json.loads(manifest_path.read_text(encoding="utf-8"))
rows = payload.get("instances", [])
if payload.get("parents") != 48 or payload.get("variants") != 432 or len(rows) != 432:
    raise SystemExit("load sweep must contain 48 parents and 432 variants")
if sum(row.get("anchor") is True for row in rows) != 48:
    raise SystemExit("load sweep must contain exactly 48 anchor variants")
for row in rows:
    path = Path(row["instance"])
    if not path.is_file():
        parts = path.parts
        if "instances" not in parts:
            raise SystemExit(f"cannot relocate load instance: {path}")
        path = Path.cwd().joinpath(*parts[parts.index("instances"):])
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != row["instance_sha256"]:
        raise SystemExit(f"load instance hash mismatch: {path}")
print(f"Validated {len(rows)} load variants and 48 anchors")
PY
}

prepare_inputs() {
    local corrected_instances=instances/corrected_v2_reduced_suite/evaluation_critical/evaluation/critical
    [ -d "$corrected_instances" ] || \
        die "Missing corrected instance suite: $corrected_instances"
    [ "$(find "$corrected_instances" -maxdepth 1 -name '*.txt' | wc -l)" -ge 48 ] || \
        die "Corrected instance suite contains fewer than 48 TXT files."
    [ -d instances/paperInstances ] || die "Missing Original suite: instances/paperInstances"

    if [ ! -f "$LOAD_MANIFEST" ]; then
        [ -d "$CORRECTED_SOURCE_RESULTS" ] || \
            die "Load sweep is absent and source campaign is missing: $CORRECTED_SOURCE_RESULTS"
        log "Generating the paired 432-instance load sweep"
        python3 experiments/generate_load_sweep.py \
            --source "$CORRECTED_SOURCE_RESULTS" \
            --output instances/research_depth_load_sweep
    fi
    validate_load_sweep
}

build_binaries() {
    log "Building Gurobi, CPLEX and MaxSAT drivers from scratch"
    make -B -j"$BUILD_JOBS" YICES=0 GUROBI=1 CPLEX=1 \
        hcorap_commercial hcorap_multi
    python3 - <<'PY'
import json
import subprocess

payload = json.loads(subprocess.run(
    ["bin/release/hcorap_commercial", "--list-backends"],
    check=True, capture_output=True, text=True,
).stdout)
backends = {row["name"]: row for row in payload["backends"]}
for name in ("gurobi-mip", "cplex-mip", "reference-enumerator"):
    if not backends.get(name, {}).get("compiled"):
        raise SystemExit(f"backend was not compiled: {name}")
print("Commercial inventory:", ", ".join(
    f"{name}={backends[name]['compiled']}" for name in sorted(backends)
))
PY
}

run_unit_tests() {
    log "Running implementation tests with the deterministic test solver"
    env -u EVALMAXSAT_BIN -u OPEN_WBO_BIN python3 -m pytest \
        tests/test_research_depth.py \
        tests/test_cpp_multiobjective.py \
        tests/test_commercial_backends.py \
        tests/test_commercial_campaign.py \
        tests/test_reproducible_campaign.py \
        tests/test_weight_analysis.py -q
}

test_evalmaxsat() {
    local result_file
    result_file=$(mktemp "${TMPDIR:-/tmp}/hcorap-evalmaxsat-preflight.XXXXXX.json")
    if "${RUNNER_PREFIX[@]}" bin/release/hcorap_multi \
        tests/instances/lex_cos_tie.txt \
        --solver "$EVALMAXSAT_BIN" \
        --timeout 60 \
        --method lex-cos \
        --cardinality-encoding totalizer \
        --zero-continuity-local \
        --print-assignments > "$result_file"; then
        :
    else
        local code=$?
        printf 'EvalMaxSAT preflight output retained at %s\n' "$result_file" >&2
        return "$code"
    fi
    python3 - "$result_file" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
if payload.get("status") != "OPTIMUM":
    raise SystemExit(f"EvalMaxSAT preflight status: {payload.get('status')}")
if not payload.get("metrics", {}).get("verified"):
    raise SystemExit("EvalMaxSAT preflight incumbent was not verified")
if payload.get("zero_continuity_local") is not True:
    raise SystemExit("EvalMaxSAT preflight did not enable local CONT=0 encoding")
print("EvalMaxSAT preflight: OPTIMUM and independently verified")
PY
    rm -f -- "$result_file"
}

dry_run_matrices() {
    local config
    for config in \
        "$DIAGNOSTICS_PILOT_CONFIG" \
        "$CPLEX_AUDIT_CONFIG" \
        "$WEIGHTS_PILOT_CONFIG" \
        "$LOAD_PILOT_CONFIG" \
        "$DIAGNOSTICS_FULL_CONFIG" \
        "$WEIGHTS_FULL_CONFIG" \
        "$LOAD_FULL_CONFIG"; do
        "${RUNNER_PREFIX[@]}" python3 experiments/run_commercial_campaign.py \
            "$config" --dry-run --workers 1
    done
    "${RUNNER_PREFIX[@]}" python3 experiments/run_reproducible_campaign.py \
        "$MAXSAT_PILOT_CONFIG" --dry-run --workers 1
}

license_preflight() {
    log "Testing Gurobi and CPLEX licenses on a campaign instance"
    "${RUNNER_PREFIX[@]}" python3 experiments/run_commercial_campaign.py \
        "$DIAGNOSTICS_PILOT_CONFIG" --preflight-only --workers 1
    "${RUNNER_PREFIX[@]}" python3 experiments/run_commercial_campaign.py \
        "$CPLEX_AUDIT_CONFIG" --preflight-only --workers 1
}

preflight() {
    check_machine
    check_solver_installations
    configure_affinity
    prepare_inputs
    build_binaries
    run_unit_tests
    dry_run_matrices
    license_preflight
    test_evalmaxsat
    log "Preflight passed; measured runs will use CPU $CPU_CORE and one solver thread"
}

run_commercial() {
    local config=$1 result_dir=$2 label=$3
    log "Running/resuming $label"
    "${RUNNER_PREFIX[@]}" python3 experiments/run_commercial_campaign.py \
        "$config" --resume --workers 1
    python3 experiments/collect_commercial_campaign.py "$result_dir"
    checkpoint "$label"
}

run_maxsat() {
    log "Running/resuming MaxSAT zero-continuity pilot"
    "${RUNNER_PREFIX[@]}" python3 experiments/run_reproducible_campaign.py \
        "$MAXSAT_PILOT_CONFIG" --resume --workers 1
    python3 experiments/collect_reproducible_campaign.py "$MAXSAT_PILOT_RESULTS"
    python3 experiments/analyze_zero_continuity.py \
        "$MAXSAT_PILOT_RESULTS" "$MAXSAT_PILOT_RESULTS/analysis"
    checkpoint maxsat-zero-continuity-pilot
}

checkpoint() {
    local label=$1
    [ -n "$BACKUP_ROOT" ] || return 0
    require_command rsync
    local backup_abs project_abs destination
    backup_abs=$(realpath -m "$BACKUP_ROOT")
    project_abs=$(realpath "$PROJECT_ROOT")
    [ "$backup_abs" != / ] && [ "$backup_abs" != "$project_abs" ] || \
        die "HCORAP_BACKUP_DIR must be outside the project root."
    case "$backup_abs" in
        "$project_abs"/*) die "HCORAP_BACKUP_DIR must be outside the worktree." ;;
    esac
    destination=$backup_abs/hcorap_research_depth
    mkdir -p "$destination/results"
    local directory
    for directory in experiments/results/research_depth_*; do
        [ -d "$directory" ] || continue
        rsync -a "$directory/" "$destination/results/$(basename "$directory")/"
    done
    rsync -a "$LOAD_MANIFEST" "$destination/load_sweep_manifest.json"
    {
        date -u '+checkpoint_utc=%Y-%m-%dT%H:%M:%SZ'
        printf 'phase=%s\n' "$label"
        printf 'source_commit=%s\n' "$(git rev-parse HEAD)"
        printf 'cpu_core=%s\n' "$CPU_CORE"
        if [ -n "${EVALMAXSAT_BIN:-}" ] && [ -f "$EVALMAXSAT_BIN" ]; then
            printf 'evalmaxsat_sha256=%s\n' \
                "$(sha256sum "$EVALMAXSAT_BIN" | awk '{print $1}')"
        else
            printf 'evalmaxsat_sha256=not-available-during-analysis\n'
        fi
    } > "$destination/checkpoint.txt"
    sync
    log "Checkpoint written to $destination"
}

analyze_pilot() {
    python3 experiments/collect_commercial_campaign.py "$DIAGNOSTICS_PILOT_RESULTS"
    python3 experiments/analyze_policy_diagnostics.py \
        "$DIAGNOSTICS_PILOT_RESULTS" "$DIAGNOSTICS_PILOT_RESULTS/analysis"

    python3 experiments/collect_commercial_campaign.py "$CPLEX_AUDIT_RESULTS"
    python3 experiments/compare_policy_diagnostics.py \
        "$DIAGNOSTICS_PILOT_RESULTS" "$CPLEX_AUDIT_RESULTS" "$AUDIT_RESULTS"

    python3 experiments/collect_commercial_campaign.py "$WEIGHTS_PILOT_RESULTS"
    python3 experiments/analyze_weight_sensitivity.py \
        --results "$WEIGHTS_PILOT_RESULTS" \
        --output-dir "$WEIGHTS_PILOT_RESULTS/analysis"

    python3 experiments/collect_commercial_campaign.py "$LOAD_PILOT_RESULTS"
    python3 experiments/analyze_load_sweep.py \
        "$LOAD_PILOT_RESULTS" "$LOAD_MANIFEST" "$LOAD_PILOT_RESULTS/analysis"

    python3 experiments/collect_reproducible_campaign.py "$MAXSAT_PILOT_RESULTS"
    python3 experiments/analyze_zero_continuity.py \
        "$MAXSAT_PILOT_RESULTS" "$MAXSAT_PILOT_RESULTS/analysis"
    checkpoint pilot-analysis
}

analyze_full() {
    python3 experiments/collect_commercial_campaign.py "$DIAGNOSTICS_FULL_RESULTS"
    python3 experiments/analyze_policy_diagnostics.py \
        "$DIAGNOSTICS_FULL_RESULTS" "$DIAGNOSTICS_FULL_RESULTS/analysis"

    python3 experiments/collect_commercial_campaign.py "$WEIGHTS_FULL_RESULTS"
    python3 experiments/analyze_weight_sensitivity.py \
        --results "$WEIGHTS_FULL_RESULTS" \
        --output-dir "$WEIGHTS_FULL_RESULTS/analysis"

    python3 experiments/collect_commercial_campaign.py "$LOAD_FULL_RESULTS"
    python3 experiments/analyze_load_sweep.py \
        "$LOAD_FULL_RESULTS" "$LOAD_MANIFEST" "$LOAD_FULL_RESULTS/analysis"
    checkpoint full-analysis
}

run_pilot_commercial() {
    run_commercial "$DIAGNOSTICS_PILOT_CONFIG" "$DIAGNOSTICS_PILOT_RESULTS" \
        gurobi-diagnostics-pilot
    python3 experiments/analyze_policy_diagnostics.py \
        "$DIAGNOSTICS_PILOT_RESULTS" "$DIAGNOSTICS_PILOT_RESULTS/analysis"

    run_commercial "$CPLEX_AUDIT_CONFIG" "$CPLEX_AUDIT_RESULTS" \
        cplex-diagnostics-audit
    python3 experiments/compare_policy_diagnostics.py \
        "$DIAGNOSTICS_PILOT_RESULTS" "$CPLEX_AUDIT_RESULTS" "$AUDIT_RESULTS"

    run_commercial "$WEIGHTS_PILOT_CONFIG" "$WEIGHTS_PILOT_RESULTS" \
        gurobi-weights-pilot
    python3 experiments/analyze_weight_sensitivity.py \
        --results "$WEIGHTS_PILOT_RESULTS" \
        --output-dir "$WEIGHTS_PILOT_RESULTS/analysis"

    run_commercial "$LOAD_PILOT_CONFIG" "$LOAD_PILOT_RESULTS" \
        gurobi-load-pilot
    python3 experiments/analyze_load_sweep.py \
        "$LOAD_PILOT_RESULTS" "$LOAD_MANIFEST" "$LOAD_PILOT_RESULTS/analysis"
    checkpoint pilot-commercial-analysis
}

run_full() {
    run_commercial "$DIAGNOSTICS_FULL_CONFIG" "$DIAGNOSTICS_FULL_RESULTS" \
        gurobi-diagnostics-full
    run_commercial "$WEIGHTS_FULL_CONFIG" "$WEIGHTS_FULL_RESULTS" \
        gurobi-weights-full
    run_commercial "$LOAD_FULL_CONFIG" "$LOAD_FULL_RESULTS" \
        gurobi-load-full
    analyze_full
}

status() {
    python3 - <<'PY'
import json
from collections import Counter
from pathlib import Path

roots = sorted(Path("experiments/results").glob("research_depth_*"))
if not roots:
    raise SystemExit("No research-depth result directories found")
for root in roots:
    validation_path = root / "validation.json"
    if not validation_path.is_file():
        continue
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    statuses = Counter()
    raw = root / "raw"
    if raw.is_dir():
        for path in raw.glob("*.json"):
            try:
                statuses[json.loads(path.read_text(encoding="utf-8")).get("status", "MISSING")] += 1
            except (OSError, json.JSONDecodeError):
                statuses["INVALID_JSON"] += 1
    print(
        f"{root.name}: complete={validation.get('complete')} "
        f"runs={validation.get('complete_runs')}/{validation.get('expected_runs')} "
        f"statuses={dict(sorted(statuses.items()))}"
    )
PY
}

initialize_measured_run() {
    local authorization=$1
    check_machine
    check_solver_installations
    configure_affinity
    check_measured_source
    if [ "$authorization" = pilot ]; then
        check_pilot_authorization
    else
        check_full_authorization
    fi
    prepare_inputs
    build_binaries
    dry_run_matrices
    license_preflight
    test_evalmaxsat
}

case "$PHASE" in
    help|-h|--help)
        usage
        ;;
    preflight)
        preflight
        ;;
    pilot-commercial)
        initialize_measured_run pilot
        run_pilot_commercial
        ;;
    pilot-maxsat)
        initialize_measured_run pilot
        run_maxsat
        ;;
    pilot)
        initialize_measured_run pilot
        run_pilot_commercial
        run_maxsat
        ;;
    full)
        initialize_measured_run full
        run_full
        ;;
    analyze-pilot)
        prepare_inputs
        analyze_pilot
        ;;
    analyze-full)
        prepare_inputs
        analyze_full
        ;;
    status)
        status
        ;;
    all)
        check_pilot_authorization
        check_full_authorization
        initialize_measured_run full
        run_pilot_commercial
        run_maxsat
        run_full
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac
