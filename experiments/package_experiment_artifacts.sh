#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$PROJECT_ROOT"

STAMP=$(date -u '+%Y%m%dT%H%M%SZ')
OUTPUT_ROOT=${ARTIFACT_OUTPUT_ROOT:-artifacts}
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

python3 - <<'PY'
import csv
import json
from pathlib import Path

for path in sorted(Path("experiments/results").glob("gcp_*/validation.json")):
    payload = json.loads(path.read_text())
    if not payload.get("complete"):
        raise SystemExit(f"refusing to package incomplete campaign: {path}")
for path in sorted(Path("experiments/results").glob("gcp_*/analysis.json")):
    payload = json.loads(path.read_text())
    if payload.get("valid") is False or payload.get("complete_analysis") is False:
        raise SystemExit(f"refusing to package invalid/incomplete analysis: {path}")
primary = Path("experiments/results/gcp_primary_analysis/analysis_validation.json")
if not primary.exists() or not json.loads(primary.read_text()).get("valid"):
    raise SystemExit(f"refusing to package invalid primary analysis: {primary}")
cross = Path(
    "experiments/results/gcp_cross_paradigm_analysis/cross_paradigm_validation.json"
)
if not cross.exists() or not json.loads(cross.read_text()).get("valid"):
    raise SystemExit(f"refusing to package missing/invalid cross-paradigm analysis: {cross}")
screen = Path("experiments/results/screening_decision.json")
if not screen.exists():
    raise SystemExit(f"refusing to package without a screening decision: {screen}")
decision = json.loads(screen.read_text())
if decision.get("decision") != "GO":
    raise SystemExit(f"refusing to package a NO-GO campaign: {screen}")

corrected = Path(
    "experiments/results/gcp_corrected_analysis/corrected_validation.json"
)
for branch in ("original_lexicographic", "corrected_v2_lexicographic"):
    if decision["branches"][branch]["enabled"] is not True:
        raise SystemExit(f"compact publication branch is not enabled: {branch}")
if not corrected.exists() or not json.loads(corrected.read_text()).get("valid"):
    raise SystemExit(
        f"refusing to package missing/invalid corrected-v2 analysis: {corrected}"
    )

selected = {
    "gcp_original_ablation": 640,
    "gcp_original_lex_primary": 280,
    "gcp_original_lex_sensitivity": 70,
    "gcp_corrected_primary": 160,
    "gcp_maxsat_commercial_validation": 40,
    "gcp_commercial_original": 80,
}

observed_rows = 0
for name, expected_rows in selected.items():
    path = Path("experiments/results") / name / "runs.csv"
    if not path.exists():
        raise SystemExit(f"missing selected campaign: {path}")
    with path.open(newline="", encoding="utf-8") as stream:
        count = sum(1 for _ in csv.DictReader(stream))
    if count != expected_rows:
        raise SystemExit(f"unexpected row count for {name}: {count}/{expected_rows}")
    observed_rows += count
expected_total = int(decision["expected_measured_runs"])
if observed_rows != expected_total:
    raise SystemExit(
        f"selected measured rows do not match decision: {observed_rows}/{expected_total}"
    )

calibration = Path(
    "experiments/results/gcp_evalmaxsat_lex_calibration/runs.csv"
)
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
    experiments/generate_manuscript_results.py
    experiments/freeze_manuscript_bundle.py
    experiments/analyze_weight_sensitivity.py
    experiments/analyze_pareto_results.py
    experiments/evaluate_screening_gates.py
    experiments/validate_campaign_manifest.py
    experiments/validate_publication_campaign.py
    experiments/gcp_prepare_and_run.sh
    experiments/run_all_remaining_publication.sh
    experiments/run_iciit2027_reduced_campaign.sh
    experiments/prepare_benchmark_suite.sh
    experiments/prepare_uncertainty_screen.sh
    experiments/verify_benchmark_batch.py
    experiments/package_experiment_artifacts.sh
    experiments/results
    bin/release/hcorap_multi
    bin/release/hcorap_commercial
    instances/paperInstances
    instances/corrected_v2_reduced_suite
    instances/uncertainty_screen
    docs/EXPERIMENT_GAP_AUDIT_20260808.md
    docs/COMPACT_EXPERIMENT_MATRIX_20260820.md
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
