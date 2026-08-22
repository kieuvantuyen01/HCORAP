#!/usr/bin/env bash
set -euo pipefail

PHASE=${1:-help}
WORKERS=${WORKERS:-1}
PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src/proposed${PYTHONPATH:+:$PYTHONPATH}"

PINNED_EVALMAXSAT_SHA256=$(python3 -c \
    'import json; print(json.load(open("experiments/configs/reduced_campaign_manifest.json"))["maxsat_solver"]["sha256"])')
RUNNER_PREFIX=()
SCREEN_DECISION=experiments/results/screening_decision.json

usage() {
    cat <<'EOF'
Usage: experiments/gcp_prepare_and_run.sh PHASE

Safe phases:
  preflight             Validate VM, EvalMaxSAT, build, tests, and benchmarks
  prepare               Generate and verify the corrected-v2 benchmark suite
  solver-calibration    Run four non-measured EvalMaxSAT LEX-COS gate rows
  screen                Run the 384-row factorial primary/hard gate
  commercial-preflight  Build and license-test Gurobi/CPLEX backends

Full manuscript phases (require CONFIRM_FULL_CAMPAIGN=YES):
  original-primary      Run 42 weighted + 42 LEX-COS rows under R
  corrected-primary     Run 48 weighted + 48 LEX-COS + 48 LEX-OCS rows
  commercial            Run 80 MIP + 40 MaxSAT original-benchmark validation rows
  corrected-commercial-evidence
                        Gate and run 144 Gurobi + 48 CPLEX corrected-v2 rows
  all                   Run the complete manifest-locked ICIIT campaign

Deferred research phases (outside the publication manifest; not called by all):
  lex-transfer-pilot   Compare Totalizer-only with the full configuration on
                       one corrected-v2 seed per stratum (32 runs)
  lex-transfer-full    Run the 96-row paired confirmation after a GO pilot
  pareto, weight-confirmation, uncertainty

Post-processing:
  analyze               Rebuild compact factorial/policy/validation tables
  package               Create a checksummed reproducibility archive
EOF
}

require_maxsat_environment() {
    if [ -z "${EVALMAXSAT_BIN:-}" ] || [ ! -x "$EVALMAXSAT_BIN" ]; then
        echo "Set EVALMAXSAT_BIN to the pinned Linux x86-64 EvalMaxSAT executable." >&2
        exit 2
    fi
    if ! command -v sha256sum >/dev/null 2>&1; then
        echo "sha256sum is required to verify the EvalMaxSAT binary." >&2
        exit 2
    fi
    observed_solver_hash=$(sha256sum "$EVALMAXSAT_BIN" | awk '{print $1}')
    if [ "$observed_solver_hash" != "$PINNED_EVALMAXSAT_SHA256" ]; then
        echo "EvalMaxSAT SHA-256 is $observed_solver_hash, not the pinned Linux binary." >&2
        echo "Expected: $PINNED_EVALMAXSAT_SHA256" >&2
        exit 2
    fi
}

check_machine() {
    if [ "$(uname -s)" != "Linux" ]; then
        echo "The manuscript campaign must run on Linux; found $(uname -s)." >&2
        exit 2
    fi
    if [ "$(uname -m)" != "x86_64" ]; then
        echo "The pinned EvalMaxSAT binary requires Linux x86-64; found $(uname -m)." >&2
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
    check_frozen_revision
}

check_frozen_revision() {
    if [ -z "${HCORAP_EXPECTED_COMMIT:-}" ]; then
        echo "Set HCORAP_EXPECTED_COMMIT to the clean publication commit or tag." >&2
        exit 2
    fi
    if ! expected_commit=$(git rev-parse --verify "${HCORAP_EXPECTED_COMMIT}^{commit}" 2>/dev/null); then
        echo "HCORAP_EXPECTED_COMMIT does not resolve to a commit: $HCORAP_EXPECTED_COMMIT" >&2
        exit 2
    fi
    observed_commit=$(git rev-parse HEAD)
    if [ "$observed_commit" != "$expected_commit" ]; then
        echo "Repository is at $observed_commit, not expected $expected_commit." >&2
        exit 2
    fi
    if [ -n "$(git status --porcelain)" ]; then
        echo "Refusing a measured campaign from a dirty worktree. Commit/tag it first." >&2
        exit 2
    fi
    validate_backup_destination
}

validate_backup_destination() {
    if [ -z "${HCORAP_BACKUP_DIR:-}" ]; then
        echo "Set HCORAP_BACKUP_DIR to a persistent mounted backup directory." >&2
        exit 2
    fi
    if ! command -v rsync >/dev/null 2>&1; then
        echo "rsync is required for phase checkpoints." >&2
        exit 2
    fi
    backup_root=$(realpath -m "$HCORAP_BACKUP_DIR")
    project_root=$(realpath "$PROJECT_ROOT")
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
}

checkpoint_results() {
    label=$1
    if [ -z "${HCORAP_BACKUP_DIR:-}" ]; then
        return
    fi
    validate_backup_destination
    backup_root=$(realpath -m "$HCORAP_BACKUP_DIR")
    checkpoint_root="$backup_root/hcorap_iciit2027_checkpoint"
    mkdir -p "$checkpoint_root/results" "$checkpoint_root/vm-logs"
    rsync -a experiments/results/ "$checkpoint_root/results/"
    if [ -d vm-logs ]; then
        rsync -a vm-logs/ "$checkpoint_root/vm-logs/"
    fi
    if [ -d artifacts ]; then
        mkdir -p "$checkpoint_root/artifacts"
        rsync -a artifacts/ "$checkpoint_root/artifacts/"
    fi
    {
        date -u '+checkpoint_utc=%Y-%m-%dT%H:%M:%SZ'
        echo "phase=$label"
        echo "source_commit=$(git rev-parse HEAD)"
    } > "$checkpoint_root/checkpoint.txt"
    sync
    echo "Checkpointed phase '$label' to $checkpoint_root."
}

build_and_test() {
    make -j8 YICES=0 hcorap_multi hcorap_commercial
    python3 -m pytest -q
}

validate_evalmaxsat_backend() {
    smoke_dir=$(mktemp -d)
    trap 'rm -r "$smoke_dir"' EXIT
    methods=(weighted lex-cos)
    for method in "${methods[@]}"; do
        "${RUNNER_PREFIX[@]}" timeout --signal=TERM --kill-after=5 45 \
            bin/release/hcorap_multi tests/instances/lex_cos_tie.txt \
            --solver "$EVALMAXSAT_BIN" --timeout 30 --method "$method" \
            --cardinality-encoding totalizer --implied-constraints both \
            --symmetry-breaking slot-service \
            > "$smoke_dir/$method.json"
    done
    python3 - "$smoke_dir" <<'PY'
import json, sys
from pathlib import Path

root = Path(sys.argv[1])
expected_calls = {"weighted": 1, "lex-cos": 3}
for method, calls in expected_calls.items():
    path = root / f"{method}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "OPTIMUM":
        raise SystemExit(f"EvalMaxSAT smoke did not prove {method}: {payload}")
    if payload.get("solver_calls") != calls:
        raise SystemExit(f"EvalMaxSAT smoke stage mismatch for {method}: {payload}")
    if payload.get("metrics", {}).get("verified") is not True:
        raise SystemExit(f"EvalMaxSAT smoke verifier failure for {method}: {payload}")
    if method == "lex-cos":
        objectives = [stage.get("objective") for stage in payload.get("stages", [])]
        if objectives != ["continuity", "overtime", "similarity"]:
            raise SystemExit(f"EvalMaxSAT LEX-COS order mismatch: {objectives}")
PY
    rm -r "$smoke_dir"
    trap - EXIT
    echo "Pinned EvalMaxSAT passed weighted and three-stage LEX-COS smoke tests."
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
                --solver "$EVALMAXSAT_BIN" --timeout 30 --method weighted \
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
    python3 experiments/validate_publication_campaign.py
    configs=(
        gcp_original_ablation
        gcp_original_lex_primary
        gcp_corrected_primary
        gcp_maxsat_commercial_validation
        gcp_evalmaxsat_lex_calibration
    )
    for name in "${configs[@]}"; do
        python3 experiments/run_reproducible_campaign.py \
            "experiments/configs/$name.json" --dry-run
    done
    validate_evalmaxsat_backend
}

run_evalmaxsat_calibration() {
    result_dir=experiments/results/gcp_evalmaxsat_lex_calibration
    run_maxsat experiments/configs/gcp_evalmaxsat_lex_calibration.json "$result_dir"
    python3 experiments/evaluate_evalmaxsat_calibration.py \
        --results "$result_dir"
}

run_maxsat() {
    config=$1
    result_dir=$2
    "${RUNNER_PREFIX[@]}" python3 experiments/run_reproducible_campaign.py "$config" \
        --resume --workers "$WORKERS"
    python3 experiments/collect_reproducible_campaign.py "$result_dir"
    validate_result_integrity "$result_dir"
    validate_peak_rss_limit "$result_dir"
}

validate_peak_rss_limit() {
    result_dir=$1
    maximum_mb=${HCORAP_MAX_PEAK_RSS_MB:-12288}
    python3 - "$result_dir/runs.csv" "$maximum_mb" <<'PY'
import csv, sys
from pathlib import Path

path = Path(sys.argv[1])
limit = float(sys.argv[2])
with path.open(newline="", encoding="utf-8") as stream:
    rows = list(csv.DictReader(stream))
missing = [row.get("run_id", "?") for row in rows if not row.get("peak_rss_mb")]
over = [
    (row.get("run_id", "?"), float(row["peak_rss_mb"]))
    for row in rows if row.get("peak_rss_mb") and float(row["peak_rss_mb"]) > limit
]
if missing or over:
    raise SystemExit(
        f"peak-RSS gate failed for {path}: missing={len(missing)}, "
        f"over_{limit:g}MB={len(over)}"
    )
PY
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
    validate_peak_rss_limit "$result_dir"
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
    python3 experiments/evaluate_commercial_correctness_smoke.py \
        --results "$result_dir"
}

run_screen() {
    # The compact campaign spends its screening budget only on the complete
    # factorial, which is itself primary evidence.  All later phases are fixed
    # and run only after this hard correctness/memory gate passes.
    run_maxsat experiments/configs/gcp_original_ablation.json \
        experiments/results/gcp_original_ablation
    python3 experiments/evaluate_screening_gates.py \
        experiments/configs/screening_gates.json
    python3 - "$SCREEN_DECISION" <<'PY'
import json, sys
from pathlib import Path

decision = json.loads(Path(sys.argv[1]).read_text())
print(f"Screening: {decision['decision']} ({decision['publication_scope']})")
for name, branch in decision["branches"].items():
    print(f"  {name}: {branch['decision']} -- {branch['action']}")
print(f"Expected measured rows for selected scope: {decision['expected_measured_runs']}")
PY
}

branch_enabled() {
    branch=$1
    if [ ! -f "$SCREEN_DECISION" ]; then
        echo "Missing screening decision: $SCREEN_DECISION" >&2
        exit 2
    fi
    if ! enabled=$(python3 - "$SCREEN_DECISION" "$branch" <<'PY'
import json, sys
from pathlib import Path

path = Path(sys.argv[1])
decision = json.loads(path.read_text())
if decision.get("decision") != "GO":
    raise SystemExit(f"hard screening decision is {decision.get('decision')}")
print("true" if decision["branches"][sys.argv[2]]["enabled"] else "false")
PY
    ); then
        echo "Cannot read branch '$branch' from $SCREEN_DECISION." >&2
        exit 2
    fi
    [ "$enabled" = "true" ]
}

run_original_primary() {
    if branch_enabled original_lexicographic; then
        run_maxsat experiments/configs/gcp_original_lex_primary.json \
            experiments/results/gcp_original_lex_primary
    else
        echo "Compact scope cannot continue after a failed factorial hard gate." >&2
        exit 2
    fi
}

run_corrected_primary() {
    if branch_enabled corrected_v2_lexicographic; then
        run_maxsat experiments/configs/gcp_corrected_primary.json \
            experiments/results/gcp_corrected_primary
        python3 experiments/analyze_primary_campaigns.py
        python3 experiments/analyze_corrected_validation.py
    else
        echo "Compact scope cannot continue after a failed factorial hard gate." >&2
        exit 2
    fi
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

prepare_commercial_backends() {
    if [ -z "${GUROBI_HOME:-}" ] || [ -z "${CPLEX_STUDIO_DIR:-}" ]; then
        echo "GUROBI_HOME and CPLEX_STUDIO_DIR must be set." >&2
        exit 2
    fi
    # Make does not track feature macros as dependencies.  Force a rebuild so a
    # prior reference-only binary cannot masquerade as a commercial build.
    make -B -j8 YICES=0 GUROBI=1 CPLEX=1 hcorap_commercial
    python3 experiments/run_commercial_campaign.py \
        experiments/configs/gcp_commercial_original.json --preflight-only
    for name in \
        gcp_commercial_corrected_calibration \
        gcp_commercial_corrected_primary \
        gcp_commercial_corrected_audit; do
        python3 experiments/run_commercial_campaign.py \
            "experiments/configs/$name.json" --dry-run
    done
}

commercial_preflight() {
    prepare_commercial_backends
    run_commercial experiments/configs/gcp_commercial_correctness_smoke.json \
        experiments/results/gcp_commercial_correctness_smoke
    validate_commercial_smoke experiments/results/gcp_commercial_correctness_smoke
}

run_all_commercial() {
    commercial_preflight
    run_commercial_primary
}

run_commercial_primary() {
    if branch_enabled original_lexicographic; then
        run_maxsat experiments/configs/gcp_maxsat_commercial_validation.json \
            experiments/results/gcp_maxsat_commercial_validation
    else
        echo "Compact scope cannot continue after a failed factorial hard gate." >&2
        exit 2
    fi
    run_commercial experiments/configs/gcp_commercial_original.json \
        experiments/results/gcp_commercial_original
    python3 experiments/analyze_cross_paradigm_validation.py \
        --scope full
}

run_corrected_commercial_evidence() {
    if ! branch_enabled corrected_v2_lexicographic; then
        echo "Corrected-v2 evidence cannot run after a failed factorial hard gate." >&2
        exit 2
    fi
    calibration_dir=experiments/results/gcp_commercial_corrected_calibration
    run_commercial \
        experiments/configs/gcp_commercial_corrected_calibration.json \
        "$calibration_dir"
    python3 experiments/evaluate_corrected_commercial_calibration.py \
        --results "$calibration_dir"
    checkpoint_results corrected-commercial-calibration

    run_commercial experiments/configs/gcp_commercial_corrected_primary.json \
        experiments/results/gcp_commercial_corrected_primary
    checkpoint_results corrected-commercial-gurobi-primary
    run_commercial experiments/configs/gcp_commercial_corrected_audit.json \
        experiments/results/gcp_commercial_corrected_audit
    python3 experiments/analyze_corrected_exact_evidence.py
}

run_lex_transfer_pilot() {
    result_dir=experiments/results/gcp_corrected_lex_encoding_transfer_pilot
    analysis_dir=experiments/results/gcp_corrected_lex_encoding_transfer_pilot_analysis
    run_maxsat experiments/configs/gcp_corrected_lex_encoding_transfer_pilot.json \
        "$result_dir"
    python3 experiments/analyze_lex_encoding_transfer.py \
        --results "$result_dir" \
        --output-dir "$analysis_dir" \
        --expected-instances 16
}

require_lex_transfer_go() {
    decision_path=experiments/results/gcp_corrected_lex_encoding_transfer_pilot_analysis/lex_encoding_transfer_validation.json
    if [ ! -f "$decision_path" ]; then
        echo "Missing lexicographic encoding-transfer pilot decision: $decision_path" >&2
        exit 2
    fi
    decision=$(python3 - "$decision_path" <<'PY'
import json, sys
from pathlib import Path
report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if report.get("structurally_valid") is not True:
    raise SystemExit("pilot analysis is structurally invalid")
print(report.get("decision", "INVALID"))
PY
    )
    if [ "$decision" != "GO" ]; then
        echo "Pilot decision is $decision; the 96-row confirmation is intentionally skipped." >&2
        exit 3
    fi
}

run_lex_transfer_full() {
    require_lex_transfer_go
    result_dir=experiments/results/gcp_corrected_lex_encoding_transfer_full
    analysis_dir=experiments/results/gcp_corrected_lex_encoding_transfer_full_analysis
    run_maxsat experiments/configs/gcp_corrected_lex_encoding_transfer_full.json \
        "$result_dir"
    python3 experiments/analyze_lex_encoding_transfer.py \
        --results "$result_dir" \
        --output-dir "$analysis_dir" \
        --expected-instances 48
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
    solver-calibration)
        check_frozen_revision
        build_and_test
        prepare_suite
        validate_maxsat_configs
        run_evalmaxsat_calibration
        checkpoint_results solver-calibration
        ;;
    screen)
        check_frozen_revision
        build_and_test
        prepare_suite
        validate_maxsat_configs
        run_warmup
        run_screen
        checkpoint_results screen
        ;;
    original-primary)
        check_clean_for_full
        build_and_test
        prepare_suite
        run_warmup
        run_original_primary
        checkpoint_results original-primary
        ;;
    corrected-primary)
        check_clean_for_full
        build_and_test
        prepare_suite
        run_warmup
        run_corrected_primary
        checkpoint_results corrected-primary
        ;;
    lex-transfer-pilot)
        check_clean_for_full
        build_and_test
        prepare_suite
        run_warmup
        run_lex_transfer_pilot
        checkpoint_results lex-transfer-pilot
        ;;
    lex-transfer-full)
        check_clean_for_full
        build_and_test
        prepare_suite
        run_warmup
        run_lex_transfer_full
        checkpoint_results lex-transfer-full
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
        checkpoint_results commercial-preflight
        ;;
    commercial)
        check_clean_for_full
        prepare_suite
        run_all_commercial
        checkpoint_results commercial
        ;;
    corrected-commercial-evidence)
        check_clean_for_full
        build_and_test
        prepare_suite
        prepare_commercial_backends
        run_corrected_commercial_evidence
        checkpoint_results corrected-commercial-evidence
        ;;
    analyze)
        branch_enabled original_lexicographic
        python3 experiments/analyze_primary_campaigns.py
        if branch_enabled corrected_v2_lexicographic; then
            python3 experiments/analyze_corrected_validation.py
            python3 experiments/analyze_corrected_exact_evidence.py
        fi
        python3 experiments/analyze_cross_paradigm_validation.py --scope full
        ;;
    package)
        bash experiments/package_experiment_artifacts.sh
        checkpoint_results package
        ;;
    all)
        check_clean_for_full
        build_and_test
        prepare_suite
        validate_maxsat_configs
        commercial_preflight
        checkpoint_results commercial-preflight
        run_warmup
        run_evalmaxsat_calibration
        checkpoint_results solver-calibration
        run_screen
        checkpoint_results screen
        run_original_primary
        checkpoint_results original-primary
        run_corrected_primary
        checkpoint_results corrected-primary
        run_commercial_primary
        checkpoint_results commercial
        run_corrected_commercial_evidence
        checkpoint_results corrected-commercial-evidence
        bash experiments/package_experiment_artifacts.sh
        checkpoint_results package
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac
