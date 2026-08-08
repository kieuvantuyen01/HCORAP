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
mkdir -p "$PACKAGE_DIR"

python3 - <<'PY'
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
if primary.exists() and not json.loads(primary.read_text()).get("valid"):
    raise SystemExit(f"refusing to package invalid primary analysis: {primary}")
screen = Path("experiments/results/screening_decision.json")
if screen.exists() and json.loads(screen.read_text()).get("decision") != "GO":
    raise SystemExit(f"refusing to package a NO-GO campaign: {screen}")
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
if [ -n "${OPEN_WBO_COMMIT:-}" ]; then
    printf '%s\n' "$OPEN_WBO_COMMIT" > "$PACKAGE_DIR/open_wbo_commit.txt"
fi
if [ -n "${OPEN_WBO_SOURCE_DIR:-}" ] && [ -d "$OPEN_WBO_SOURCE_DIR/.git" ]; then
    git -C "$OPEN_WBO_SOURCE_DIR" rev-parse HEAD \
        > "$PACKAGE_DIR/open_wbo_observed_commit.txt"
    git -C "$OPEN_WBO_SOURCE_DIR" status --porcelain \
        > "$PACKAGE_DIR/open_wbo_git_status.txt"
fi
if [ -n "${OPEN_WBO_BIN:-}" ] && [ -x "$OPEN_WBO_BIN" ]; then
    cp "$OPEN_WBO_BIN" "$PACKAGE_DIR/open-wbo"
    chmod 755 "$PACKAGE_DIR/open-wbo"
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$PACKAGE_DIR/open-wbo" > "$PACKAGE_DIR/open-wbo.sha256"
    else
        shasum -a 256 "$PACKAGE_DIR/open-wbo" > "$PACKAGE_DIR/open-wbo.sha256"
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
    experiments/analyze_weight_sensitivity.py
    experiments/analyze_pareto_results.py
    experiments/evaluate_screening_gates.py
    experiments/validate_campaign_manifest.py
    experiments/gcp_prepare_and_run.sh
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
