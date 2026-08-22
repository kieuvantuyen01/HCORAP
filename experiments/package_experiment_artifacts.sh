#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$PROJECT_ROOT"

STAMP=$(date -u '+%Y%m%dT%H%M%SZ')
OUTPUT_ROOT=${ARTIFACT_OUTPUT_ROOT:-artifacts}
RESULTS_ROOT=${HCORAP_RESULTS_ROOT:-experiments/results}
PACKAGE_DIR="$OUTPUT_ROOT/hcorap_iciit2027_$STAMP"

if [ -n "$(git status --porcelain)" ] && [ "${ALLOW_DIRTY_ARTIFACT:-0}" != "1" ]; then
    echo "Refusing a final artifact from a dirty worktree." >&2
    echo "Commit/tag the campaign code, or set ALLOW_DIRTY_ARTIFACT=1 for a draft archive." >&2
    exit 2
fi
if [ "${ALLOW_DIRTY_ARTIFACT:-0}" != "1" ]; then
    if [ -z "${HCORAP_EXPECTED_COMMIT:-}" ]; then
        echo "Set HCORAP_EXPECTED_COMMIT to the frozen publication commit or tag." >&2
        exit 2
    fi
    expected_commit=$(git rev-parse --verify "${HCORAP_EXPECTED_COMMIT}^{commit}")
    observed_commit=$(git rev-parse HEAD)
    if [ "$observed_commit" != "$expected_commit" ]; then
        echo "Artifact source is $observed_commit, not expected $expected_commit." >&2
        exit 2
    fi
fi
mkdir -p "$PACKAGE_DIR"

python3 experiments/audit_publication_evidence.py \
    --results-root "$RESULTS_ROOT" \
    --output "$RESULTS_ROOT/publication_evidence_audit.json"

python3 - "$RESULTS_ROOT" <<'PY'
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
config_root = Path("experiments/configs")
manifest = json.loads((config_root / "reduced_campaign_manifest.json").read_text())


def campaign_directory(declaration):
    config = json.loads((config_root / declaration["config"]).read_text())
    return root / Path(config["result_dir"]).name


for path in sorted(root.glob("gcp_*/validation.json")):
    payload = json.loads(path.read_text())
    if not payload.get("complete"):
        raise SystemExit(f"refusing to package incomplete campaign: {path}")
for path in sorted(root.glob("gcp_*/analysis.json")):
    payload = json.loads(path.read_text())
    if payload.get("valid") is False or payload.get("complete_analysis") is False:
        raise SystemExit(f"refusing to package invalid/incomplete analysis: {path}")
primary = root / "gcp_primary_analysis/analysis_validation.json"
if not primary.exists() or not json.loads(primary.read_text()).get("valid"):
    raise SystemExit(f"refusing to package invalid primary analysis: {primary}")
cross = root / "gcp_cross_paradigm_analysis/cross_paradigm_validation.json"
if not cross.exists() or not json.loads(cross.read_text()).get("valid"):
    raise SystemExit(f"refusing to package missing/invalid cross-paradigm analysis: {cross}")
screen = root / "screening_decision.json"
if not screen.exists():
    raise SystemExit(f"refusing to package without a screening decision: {screen}")
decision = json.loads(screen.read_text())
if decision.get("decision") != "GO":
    raise SystemExit(f"refusing to package a NO-GO campaign: {screen}")

corrected = root / (
    "gcp_corrected_exact_analysis/corrected_exact_validation.json"
)
for branch in ("original_lexicographic", "corrected_v2_lexicographic"):
    if decision["branches"][branch]["enabled"] is not True:
        raise SystemExit(f"compact publication branch is not enabled: {branch}")
if not corrected.exists() or not json.loads(corrected.read_text()).get(
    "manuscript_eligible"
):
    raise SystemExit(
        f"refusing to package missing/invalid corrected-v2 analysis: {corrected}"
    )

observed_rows = 0
for declaration in manifest["measured_campaigns"]:
    expected_rows = int(declaration["expected_runs"])
    path = campaign_directory(declaration) / "runs.csv"
    if not path.exists():
        raise SystemExit(f"missing selected campaign: {path}")
    with path.open(newline="", encoding="utf-8") as stream:
        count = sum(1 for _ in csv.DictReader(stream))
    if count != expected_rows:
        raise SystemExit(
            f"unexpected row count for {declaration['name']}: {count}/{expected_rows}"
        )
    observed_rows += count
expected_total = int(decision["expected_measured_runs"])
if observed_rows != expected_total:
    raise SystemExit(
        f"selected measured rows do not match decision: {observed_rows}/{expected_total}"
    )

non_measured = {
    declaration["name"]: declaration
    for declaration in manifest["non_measured_campaigns"]
}
calibration = campaign_directory(
    non_measured["evalmaxsat_lex_calibration"]
) / "runs.csv"
if not calibration.exists():
    raise SystemExit(f"missing EvalMaxSAT scalability calibration: {calibration}")
with calibration.open(newline="", encoding="utf-8") as stream:
    calibration_rows = list(csv.DictReader(stream))
calibration_optimum = sum(row["status"] == "OPTIMUM" for row in calibration_rows)
if len(calibration_rows) != 4 or calibration_optimum < 2:
    raise SystemExit(
        "EvalMaxSAT scalability calibration did not pass: "
        f"rows={len(calibration_rows)}/4, optimum={calibration_optimum}/4"
    )

commercial_calibration = campaign_directory(
    non_measured["corrected_v2_commercial_calibration"]
) / "calibration_decision.json"
if not commercial_calibration.exists() or not json.loads(
    commercial_calibration.read_text()
).get("pass"):
    raise SystemExit(
        "corrected commercial calibration did not pass: "
        f"{commercial_calibration}"
    )
PY

git rev-parse HEAD > "$PACKAGE_DIR/git_commit.txt"
git status --porcelain > "$PACKAGE_DIR/git_status.txt"
git diff --binary HEAD > "$PACKAGE_DIR/tracked_changes.patch"
python3 -m pip freeze > "$PACKAGE_DIR/python_packages.txt"
uname -a > "$PACKAGE_DIR/uname.txt"
${CXX:-g++} --version > "$PACKAGE_DIR/compiler.txt"
if [ -f /etc/os-release ]; then
    cp /etc/os-release "$PACKAGE_DIR/os-release.txt"
fi
if command -v lscpu >/dev/null 2>&1; then
    lscpu > "$PACKAGE_DIR/lscpu.txt"
fi
if [ -r /proc/cpuinfo ]; then
    cp /proc/cpuinfo "$PACKAGE_DIR/cpuinfo.txt"
fi
if command -v curl >/dev/null 2>&1; then
    metadata_base=http://metadata.google.internal/computeMetadata/v1/instance
    for field in id machine-type zone image; do
        curl --fail --silent --show-error --max-time 2 \
            -H 'Metadata-Flavor: Google' "$metadata_base/$field" \
            > "$PACKAGE_DIR/gcp_$field.txt" 2>/dev/null || \
            rm -f "$PACKAGE_DIR/gcp_$field.txt"
    done
fi
if [ -n "${EVALMAXSAT_BIN:-}" ] && [ -x "$EVALMAXSAT_BIN" ]; then
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$EVALMAXSAT_BIN" | awk '{print $1 "  EvalMaxSAT_bin"}' \
            > "$PACKAGE_DIR/EvalMaxSAT_bin.sha256"
    else
        shasum -a 256 "$EVALMAXSAT_BIN" | awk '{print $1 "  EvalMaxSAT_bin"}' \
            > "$PACKAGE_DIR/EvalMaxSAT_bin.sha256"
    fi
    realpath "$EVALMAXSAT_BIN" > "$PACKAGE_DIR/evalmaxsat_source_path.txt"
    if [ "${HCORAP_INCLUDE_SOLVER_BINARY:-NO}" = "YES" ]; then
        cp "$EVALMAXSAT_BIN" "$PACKAGE_DIR/EvalMaxSAT_bin"
        chmod 755 "$PACKAGE_DIR/EvalMaxSAT_bin"
    fi
fi

include_paths=(
    Makefile pyproject.toml README.md
    experiments/README.md
    src/proposed src/hcorap_multi.cpp src/hcorap_commercial.cpp
    experiments/configs
    experiments/run_reproducible_campaign.py
    experiments/collect_reproducible_campaign.py
    experiments/run_commercial_campaign.py
    experiments/collect_commercial_campaign.py
    experiments/generate_uncertainty_scenarios.py
    experiments/verify_uncertainty_scenarios.py
    experiments/analyze_uncertainty_campaign.py
    experiments/analyze_primary_campaigns.py
    experiments/analyze_cross_paradigm_validation.py
    experiments/analyze_corrected_validation.py
    experiments/analyze_corrected_exact_evidence.py
    experiments/analyze_lex_encoding_transfer.py
    experiments/evaluate_corrected_commercial_calibration.py
    experiments/evaluate_commercial_correctness_smoke.py
    experiments/evaluate_evalmaxsat_calibration.py
    experiments/audit_publication_evidence.py
    experiments/publication_contract.py
    experiments/generate_manuscript_results.py
    experiments/freeze_manuscript_bundle.py
    experiments/analyze_weight_sensitivity.py
    experiments/analyze_pareto_results.py
    experiments/evaluate_screening_gates.py
    experiments/validate_campaign_manifest.py
    experiments/validate_publication_campaign.py
    experiments/gcp_prepare_and_run.sh
    experiments/run_all_remaining_publication.sh
    experiments/run_remaining_corrected_evidence.sh
    experiments/run_iciit2027_reduced_campaign.sh
    experiments/run_corrected_lex_encoding_transfer.sh
    experiments/prepare_benchmark_suite.sh
    experiments/prepare_uncertainty_screen.sh
    experiments/verify_benchmark_batch.py
    experiments/package_experiment_artifacts.sh
    "$RESULTS_ROOT"
    bin/release/hcorap_multi
    bin/release/hcorap_commercial
    instances/paperInstances
    instances/corrected_v2_reduced_suite
    instances/uncertainty_screen
    docs/EXPERIMENT_GAP_AUDIT_20260808.md
    docs/COMPACT_EXPERIMENT_MATRIX_20260820.md
    docs/EXPERIMENT_SUPPLEMENT_MATRIX_20260822.md
    docs/FAIR_EXPERIMENT_PROTOCOL.md
    docs/GCP_EXPERIMENT_RUNBOOK.md
    tests
)

existing_paths=()
for path in "${include_paths[@]}"; do
    if [ -e "$path" ]; then
        existing_paths+=("$path")
    fi
done

archive="$OUTPUT_ROOT/hcorap_iciit2027_$STAMP.tar.gz"
tar -czf "$archive" "$PACKAGE_DIR" "${existing_paths[@]}"
if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$archive" > "$archive.sha256"
else
    shasum -a 256 "$archive" > "$archive.sha256"
fi
echo "Artifact: $archive"
echo "Checksum: $archive.sha256"
