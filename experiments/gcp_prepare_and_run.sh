#!/usr/bin/env bash
set -euo pipefail

PHASE=${1:-help}
WORKERS=${WORKERS:-1}
PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src/proposed${PYTHONPATH:+:$PYTHONPATH}"

PINNED_OPEN_WBO_COMMIT=80f3073e41028b219b0b0ad7c61fba28351f88e6
RUNNER_PREFIX=()

usage() {
    cat <<'EOF'
Usage: experiments/gcp_prepare_and_run.sh PHASE

Safe phases:
  preflight             Validate VM, solver, build, tests, and benchmark files
  prepare               Generate and verify the corrected-v2 benchmark suite
  screen                 Factorial, lex, small epsilon, and 4-weight screens
  commercial-preflight  Build and license-test Gurobi/CPLEX backends

Full manuscript phases (require CONFIRM_FULL_CAMPAIGN=YES):
  original-primary      Full weighted pair + held-out LEX-COS + LEX-OCS
  corrected-primary     Reduced corrected-v2 critical evaluation campaign
  commercial            Gurobi/CPLEX weighted + LEX-COS on original subset
  all                   Run the complete reduced ICIIT campaign

Deferred (not called by all):
  pareto, weight-confirmation, uncertainty

Post-processing:
  analyze               Rebuild confirmatory factorial/lex manuscript tables
  package               Create a checksummed reproducibility archive
EOF
}

require_maxsat_environment() {
    if [ -z "${OPEN_WBO_BIN:-}" ] || [ ! -x "$OPEN_WBO_BIN" ]; then
        echo "Set OPEN_WBO_BIN to the pinned Linux Open-WBO executable." >&2
        exit 2
    fi
    if [ "${OPEN_WBO_COMMIT:-}" != "$PINNED_OPEN_WBO_COMMIT" ]; then
        echo "Set OPEN_WBO_COMMIT=$PINNED_OPEN_WBO_COMMIT after checking the solver source." >&2
        exit 2
    fi
    if [ -z "${OPEN_WBO_SOURCE_DIR:-}" ] || [ ! -d "$OPEN_WBO_SOURCE_DIR/.git" ]; then
        echo "Set OPEN_WBO_SOURCE_DIR to the pinned Open-WBO source checkout." >&2
        exit 2
    fi
    observed_commit=$(git -C "$OPEN_WBO_SOURCE_DIR" rev-parse HEAD)
    if [ "$observed_commit" != "$PINNED_OPEN_WBO_COMMIT" ]; then
        echo "Open-WBO checkout is at $observed_commit, not the pinned commit." >&2
        exit 2
    fi
    solver_path=$(realpath "$OPEN_WBO_BIN")
    solver_root=$(realpath "$OPEN_WBO_SOURCE_DIR")
    case "$solver_path" in
        "$solver_root"/*) ;;
        *)
            echo "OPEN_WBO_BIN must be built inside OPEN_WBO_SOURCE_DIR." >&2
            exit 2
            ;;
    esac
}

check_machine() {
    if [ "$(uname -s)" != "Linux" ]; then
        echo "The manuscript campaign must run on Linux; found $(uname -s)." >&2
        exit 2
    fi
    cpu_count=$(getconf _NPROCESSORS_ONLN)
    memory_kib=$(awk '/MemTotal/ {print $2}' /proc/meminfo)
    available_kib=$(df -Pk . | awk 'NR==2 {print $4}')
    if [ "$cpu_count" -lt 8 ] || [ "$memory_kib" -lt 15000000 ]; then
        echo "Expected C4 high-CPU resources: >=8 vCPU and >=15,000,000 KiB RAM." >&2
        echo "Observed: ${cpu_count} CPU, ${memory_kib} KiB RAM." >&2
        exit 2
    fi
    if [ "$available_kib" -lt 20000000 ]; then
        echo "At least 20 GB free disk is required; found ${available_kib} KiB." >&2
        exit 2
    fi
    if [ "$WORKERS" -lt 1 ] || [ "$WORKERS" -gt 4 ]; then
        echo "WORKERS must be in [1,4] on the 16 GB VM; publication default is 1." >&2
        exit 2
    fi
}

configure_execution() {
    if ! command -v timeout >/dev/null 2>&1; then
        echo "GNU timeout is required for bounded warm-up runs." >&2
        exit 2
    fi
    if [ "$WORKERS" -eq 1 ]; then
        if ! command -v taskset >/dev/null 2>&1; then
            echo "taskset is required to pin publication runs to one vCPU." >&2
            exit 2
        fi
        cpu_core=${HCORAP_CPU_CORE:-0}
        if ! [[ "$cpu_core" =~ ^[0-9]+$ ]] || [ "$cpu_core" -ge "$cpu_count" ]; then
            echo "HCORAP_CPU_CORE must identify one available vCPU." >&2
            exit 2
        fi
        RUNNER_PREFIX=(taskset --cpu-list "$cpu_core")
    else
        echo "Warning: WORKERS=$WORKERS is for screening only; runs are not CPU-pinned." >&2
    fi
}

check_clean_for_full() {
    if [ "${CONFIRM_FULL_CAMPAIGN:-}" != "YES" ]; then
        echo "Set CONFIRM_FULL_CAMPAIGN=YES to authorize a full-cost campaign." >&2
        exit 2
    fi
    if [ "$WORKERS" -ne 1 ]; then
        echo "Full publication phases require WORKERS=1 for uncontended timing." >&2
        exit 2
    fi
    if [ -n "$(git status --porcelain)" ] && [ "${ALLOW_DIRTY_WORKTREE:-0}" != "1" ]; then
        echo "Refusing a full campaign from a dirty worktree. Commit/tag it first." >&2
        exit 2
    fi
}

build_and_test() {
    make -j8 YICES=0 hcorap_multi hcorap_commercial
    python3 -m pytest -q
}

prepare_suite() {
    bash experiments/prepare_benchmark_suite.sh
}

run_warmup() {
    warmup_dir=$(mktemp -d)
    trap 'rm -r "$warmup_dir"' EXIT
    warmup_instances=()
    while IFS= read -r instance; do
        warmup_instances+=("$instance")
    done < <(
        find instances/corrected_v2_reduced_suite/calibration/calibration/critical \
            -maxdepth 1 -type f -name '*.txt' | sort | head -10
    )
    if [ "${#warmup_instances[@]}" -ne 10 ]; then
        echo "Expected 10 non-evaluation warm-up instances." >&2
        exit 2
    fi
    for index in "${!warmup_instances[@]}"; do
        if "${RUNNER_PREFIX[@]}" timeout --signal=TERM --kill-after=5 45 \
                bin/release/hcorap_multi "${warmup_instances[$index]}" \
                --solver "$OPEN_WBO_BIN" --timeout 30 --method weighted \
                --cardinality-encoding totalizer --implied-constraints both \
                --symmetry-breaking slot-service \
                --output "$warmup_dir/$index.json" >/dev/null; then
            :
        else
            warmup_exit=$?
            # hcorap_multi uses 2 for a valid, internally bounded TIMEOUT.
            if [ "$warmup_exit" -ne 2 ]; then
                echo "Warm-up process failed with exit code $warmup_exit." >&2
                exit 2
            fi
        fi
    done
    python3 - "$warmup_dir" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
paths = sorted(root.glob("*.json"))
if len(paths) != 10:
    raise SystemExit(f"warm-up created {len(paths)}/10 JSON outputs")
allowed = {"OPTIMUM", "UNSAT", "UNSATISFIABLE", "TIMEOUT", "TIMEOUT_FEASIBLE"}
for path in paths:
    payload = json.loads(path.read_text())
    if payload.get("status") not in allowed:
        raise SystemExit(f"warm-up technical failure: {path}: {payload.get('status')}")
PY
    rm -r "$warmup_dir"
    trap - EXIT
    echo "Completed 10 non-measured calibration warm-up runs."
}

validate_maxsat_configs() {
    python3 experiments/validate_campaign_manifest.py
    configs=(
        gcp_original_ablation
        gcp_multiobjective_screen
        gcp_weight_screen
        gcp_lex_scalability_screen
        gcp_original_weighted_primary
        gcp_original_lex_primary
        gcp_original_lex_sensitivity
        gcp_corrected_primary
    )
    for name in "${configs[@]}"; do
        python3 experiments/run_reproducible_campaign.py \
            "experiments/configs/$name.json" --dry-run
    done
}

run_maxsat() {
    config=$1
    result_dir=$2
    "${RUNNER_PREFIX[@]}" python3 experiments/run_reproducible_campaign.py "$config" \
        --resume --workers "$WORKERS"
    python3 experiments/collect_reproducible_campaign.py "$result_dir"
    validate_result_integrity "$result_dir"
}

validate_result_integrity() {
    result_dir=$1
    python3 - "$result_dir" <<'PY'
import csv, json, sys
from pathlib import Path
root = Path(sys.argv[1])
validation = json.loads((root / "validation.json").read_text())
if not validation.get("complete"):
    raise SystemExit(f"incomplete campaign: {root}")
allowed = {
    "OPTIMUM", "UNSAT", "UNSATISFIABLE", "INFEASIBLE",
    "TIMEOUT", "TIMEOUT_FEASIBLE",
}
with (root / "runs.csv").open(newline="", encoding="utf-8") as stream:
    rows = list(csv.DictReader(stream))
bad = [row["run_id"] for row in rows if row["status"] not in allowed]
invalid = [row["run_id"] for row in rows if row.get("validation_errors")]
if bad or invalid:
    raise SystemExit(
        f"technical/validation errors in {root}: statuses={len(bad)}, validation={len(invalid)}"
    )
PY
}

run_commercial() {
    config=$1
    result_dir=$2
    "${RUNNER_PREFIX[@]}" python3 experiments/run_commercial_campaign.py "$config" \
        --resume --workers "$WORKERS"
    python3 experiments/collect_commercial_campaign.py "$result_dir"
    validate_result_integrity "$result_dir"
    validate_commercial_agreement "$result_dir"
}

validate_commercial_agreement() {
    result_dir=$1
    python3 - "$result_dir/backend_agreement.csv" <<'PY'
import csv, sys
from pathlib import Path
path = Path(sys.argv[1])
with path.open(newline="", encoding="utf-8") as stream:
    rows = list(csv.DictReader(stream))
bad_scores = [row for row in rows if row["weighted_score_agreement"] != "True"]
bad_lex_vectors = [
    row for row in rows
    if row["method"] != "weighted" and row["objective_vector_agreement"] != "True"
]
if bad_scores or bad_lex_vectors:
    raise SystemExit(
        f"commercial optimum disagreement: score={len(bad_scores)}, "
        f"lex/epsilon vector={len(bad_lex_vectors)}"
    )
PY
}

validate_commercial_smoke() {
    result_dir=$1
    python3 - "$result_dir" <<'PY'
import csv, sys
from pathlib import Path

root = Path(sys.argv[1])
with (root / "runs.csv").open(newline="", encoding="utf-8") as stream:
    runs = list(csv.DictReader(stream))
with (root / "backend_agreement.csv").open(newline="", encoding="utf-8") as stream:
    agreements = list(csv.DictReader(stream))

expected_backends = {"gurobi-mip", "cplex-mip", "reference-enumerator"}
problems = []
if len(runs) != 36:
    problems.append(f"runs={len(runs)}/36")
if any(row["status"] != "OPTIMUM" for row in runs):
    problems.append("not every smoke run reached OPTIMUM")
if any(row["verified"] != "True" for row in runs):
    problems.append("not every smoke optimum was verified")
if len(agreements) != 12:
    problems.append(f"agreement_groups={len(agreements)}/12")
for row in agreements:
    observed = {item.strip() for item in row["backends"].split("|")}
    if observed != expected_backends:
        problems.append(f"incomplete backend group: {sorted(observed)}")
    if row["weighted_score_agreement"] != "True":
        problems.append("weighted-score disagreement")
    if row["method"] != "weighted" and row["objective_vector_agreement"] != "True":
        problems.append("lex/epsilon vector disagreement")
if problems:
    raise SystemExit("commercial correctness smoke failed: " + "; ".join(problems))
PY
}

run_screen() {
    run_maxsat experiments/configs/gcp_original_ablation.json \
        experiments/results/gcp_original_ablation
    run_maxsat experiments/configs/gcp_multiobjective_screen.json \
        experiments/results/gcp_multiobjective_screen
    python3 experiments/analyze_pareto_results.py \
        --results experiments/results/gcp_multiobjective_screen \
        --output-dir experiments/results/gcp_multiobjective_screen_analysis
    run_maxsat experiments/configs/gcp_weight_screen.json \
        experiments/results/gcp_weight_screen
    python3 experiments/analyze_weight_sensitivity.py \
        --results experiments/results/gcp_weight_screen \
        --output-dir experiments/results/gcp_weight_screen_analysis
    run_maxsat experiments/configs/gcp_lex_scalability_screen.json \
        experiments/results/gcp_lex_scalability_screen
    python3 experiments/evaluate_screening_gates.py \
        experiments/configs/screening_gates.json
}

run_original_primary() {
    run_maxsat experiments/configs/gcp_original_weighted_primary.json \
        experiments/results/gcp_original_weighted_primary
    run_maxsat experiments/configs/gcp_original_lex_primary.json \
        experiments/results/gcp_original_lex_primary
    run_maxsat experiments/configs/gcp_original_lex_sensitivity.json \
        experiments/results/gcp_original_lex_sensitivity
    python3 experiments/analyze_primary_campaigns.py
}

run_corrected_primary() {
    run_maxsat experiments/configs/gcp_corrected_primary.json \
        experiments/results/gcp_corrected_primary
}

run_uncertainty() {
    bash experiments/prepare_uncertainty_screen.sh
    run_maxsat experiments/configs/gcp_uncertainty_nominal.json \
        experiments/results/gcp_uncertainty_nominal
    run_maxsat experiments/configs/gcp_uncertainty_scenarios.json \
        experiments/results/gcp_uncertainty_scenarios
    python3 experiments/analyze_uncertainty_campaign.py \
        --nominal-results experiments/results/gcp_uncertainty_nominal \
        --scenario-results experiments/results/gcp_uncertainty_scenarios \
        --output-dir experiments/results/gcp_uncertainty_analysis
}

commercial_preflight() {
    if [ -z "${GUROBI_HOME:-}" ] || [ -z "${CPLEX_STUDIO_DIR:-}" ]; then
        echo "GUROBI_HOME and CPLEX_STUDIO_DIR must be set." >&2
        exit 2
    fi
    # Make does not track feature macros as dependencies.  Force a rebuild so a
    # prior reference-only binary cannot masquerade as a commercial build.
    make -B -j8 YICES=0 GUROBI=1 CPLEX=1 hcorap_commercial
    python3 experiments/run_commercial_campaign.py \
        experiments/configs/gcp_commercial_original.json --preflight-only
    run_commercial experiments/configs/gcp_commercial_correctness_smoke.json \
        experiments/results/gcp_commercial_correctness_smoke
    validate_commercial_smoke experiments/results/gcp_commercial_correctness_smoke
}

run_all_commercial() {
    commercial_preflight
    run_commercial experiments/configs/gcp_commercial_original.json \
        experiments/results/gcp_commercial_original
}

if [ "$PHASE" = "help" ] || [ "$PHASE" = "--help" ] || [ "$PHASE" = "-h" ]; then
    usage
    exit 0
fi

require_maxsat_environment
check_machine
configure_execution

case "$PHASE" in
    preflight)
        build_and_test
        prepare_suite
        validate_maxsat_configs
        ;;
    prepare)
        prepare_suite
        ;;
    screen)
        build_and_test
        prepare_suite
        validate_maxsat_configs
        run_warmup
        run_screen
        ;;
    original-primary)
        check_clean_for_full
        build_and_test
        prepare_suite
        run_warmup
        run_original_primary
        ;;
    corrected-primary)
        check_clean_for_full
        build_and_test
        prepare_suite
        run_warmup
        run_corrected_primary
        ;;
    pareto)
        check_clean_for_full
        build_and_test
        prepare_suite
        run_warmup
        run_maxsat experiments/configs/gcp_corrected_pareto.json \
            experiments/results/gcp_corrected_pareto
        python3 experiments/analyze_pareto_results.py \
            --results experiments/results/gcp_corrected_pareto \
            --output-dir experiments/results/gcp_corrected_pareto_analysis
        ;;
    weight-confirmation)
        check_clean_for_full
        build_and_test
        prepare_suite
        run_warmup
        run_maxsat experiments/configs/gcp_weight_confirmation.json \
            experiments/results/gcp_weight_confirmation
        python3 experiments/analyze_weight_sensitivity.py \
            --results experiments/results/gcp_weight_confirmation \
            --output-dir experiments/results/gcp_weight_confirmation_analysis
        ;;
    uncertainty)
        build_and_test
        prepare_suite
        run_warmup
        run_uncertainty
        ;;
    commercial-preflight)
        commercial_preflight
        ;;
    commercial)
        check_clean_for_full
        prepare_suite
        run_all_commercial
        ;;
    analyze)
        python3 experiments/analyze_primary_campaigns.py
        ;;
    package)
        bash experiments/package_experiment_artifacts.sh
        ;;
    all)
        check_clean_for_full
        build_and_test
        prepare_suite
        validate_maxsat_configs
        run_warmup
        run_screen
        run_original_primary
        run_corrected_primary
        run_all_commercial
        bash experiments/package_experiment_artifacts.sh
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac
