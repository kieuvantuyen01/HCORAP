#!/usr/bin/env bash
# =============================================================================
# run_main_8cfg.sh
# Chạy 8 cấu hình so sánh (ORIGINAL + 7 đề xuất) trên toàn bộ 800 instances
#
# NOTE VỀ SOLVER:
#   Bài báo gốc (Collcaballero et al.) dùng EvalMaxSAT trên Linux x86-64.
#   EvalMaxSAT không hoạt động ổn định trên macOS arm64 với WCNF lớn
#   (thoát mà không output 's OPTIMUM FOUND'; SIGSEGV khi gọi qua hcorap_multi).
#   => Script này dùng open-wbo cho TẤT CẢ 8 cấu hình.
#   So sánh thời gian với bài báo gốc là KHÔNG HỢP LỆ (khác solver + platform).
#   So sánh objective value (SIM - CONT - OT) và ranking IC/SB giữa các cfg là HỢP LỆ.
#
# Usage:
#   bash experiments/run_main_8cfg.sh            # chạy mới
#   bash experiments/run_main_8cfg.sh --resume   # tiếp tục nếu bị ngắt
#
# Biến môi trường override:
#   SOLVER=./path/to/solver
#   HCORAP_BINARY=./path/to/hcorap_multi
#   TIMEOUT=3600
#   RESULT_DIR=experiments/results/main_8cfg
#   INSTANCE_DIRS="dir1 dir2 ..."
# =============================================================================
set -eu

BINARY="${HCORAP_BINARY:-./bin/release/hcorap_multi}"
SOLVER="${SOLVER:-./open-wbo_macos}"
TIMEOUT="${TIMEOUT:-3600}"
WC="${WC:-1}"
WO="${WO:-1}"
METHOD="weighted"
RESULT_DIR="${RESULT_DIR:-experiments/results/main_8cfg}"
RESUME="${1:-}"

# ---------------------------------------------------------------------------
# 8 cấu hình: "cardinality ic sb label"
# cfg1 = ORIGINAL (sorting-network + none + none)
# ---------------------------------------------------------------------------
CONFIGS="
sorting-network none none ORIGINAL
sorting-network none slot-service SN-none-ss
sorting-network both none SN-both-none
sorting-network both slot-service SN-both-ss
totalizer none none TOT-none-none
totalizer none slot-service TOT-none-ss
totalizer both none TOT-both-none
totalizer both slot-service TOT-both-ss
"

# Instance directories (space-separated)
INSTANCE_DIRS="${INSTANCE_DIRS:-instances/paperInstances/TXT_10-25_4-5_U30 instances/paperInstances/TXT_10-25_4-5_U40}"

# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------
if [ ! -x "$BINARY" ]; then
    echo "ERROR: Binary not found or not executable: $BINARY" >&2
    echo "       Run: make -j4  (or: make -C src -j4)" >&2
    exit 2
fi
if [ ! -x "$SOLVER" ]; then
    echo "ERROR: Solver not found or not executable: $SOLVER" >&2
    exit 2
fi

mkdir -p "$RESULT_DIR"
LOG="$RESULT_DIR/run.log"
CSV="$RESULT_DIR/results_per_instance.csv"
DONE_FILE="$RESULT_DIR/.done_runs"   # tracks completed run keys for --resume

# CSV header
if [ ! -f "$CSV" ] || [ "$RESUME" != "--resume" ]; then
    printf 'cfg_id\tlabel\tinstance_name\tcardinality\tic\tsb\tstatus\telapsed_s\tsolve_s\tvariables\thard_clauses\tsoft_clauses\tbest_value\tsimilarity\tcontinuity\tovertme\tcoverage\n' > "$CSV"
fi
touch "$DONE_FILE"

# Environment snapshot (once)
ENV_FILE="$RESULT_DIR/environment.txt"
if [ ! -f "$ENV_FILE" ]; then
    {
        printf 'created_utc='; date -u '+%Y-%m-%dT%H%%3AM%%3A%SZ'
        printf 'uname='; uname -a
        printf 'hcorap_binary=%s\n' "$BINARY"
        printf 'hcorap_sha256='; shasum -a 256 "$BINARY" | awk '{print $1}'
        printf 'solver_used=%s\n' "$SOLVER"  # open-wbo (NOT EvalMaxSAT as in original paper)
        printf 'solver_sha256='; shasum -a 256 "$SOLVER" | awk '{print $1}'
        printf 'solver_note=open-wbo built from source; paper used EvalMaxSAT on Linux x86-64\n'
        printf 'timeout=%s\nwc=%s\nwo=%s\nmethod=%s\n' "$TIMEOUT" "$WC" "$WO" "$METHOD"
        printf 'git_commit='; git rev-parse HEAD 2>/dev/null || printf 'unknown\n'
    } > "$ENV_FILE"
fi

# ---------------------------------------------------------------------------
# Count total work
# ---------------------------------------------------------------------------
total_instances=0
for idir in $INSTANCE_DIRS; do
    c=$(ls "$idir"/*.txt 2>/dev/null | wc -l | tr -d ' ')
    total_instances=$((total_instances + c))
done
total_configs=$(echo "$CONFIGS" | grep -c '\S' || true)
total_runs=$((total_instances * total_configs))

echo "=== HCORAP Main Experiment: 8 Configurations ===" | tee -a "$LOG"
echo "Instances : $total_instances" | tee -a "$LOG"
echo "Configs   : $total_configs" | tee -a "$LOG"
echo "Total runs: $total_runs" | tee -a "$LOG"
echo "Timeout   : ${TIMEOUT}s/run" | tee -a "$LOG"
echo "Results   : $RESULT_DIR" | tee -a "$LOG"
echo "Started   : $(date)" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
cfg_id=0
run_count=0
skip_count=0
fail_count=0

echo "$CONFIGS" | grep '\S' | while IFS=' ' read -r card ic sb label; do
    cfg_id=$((cfg_id + 1))
    cfg_tag=$(printf 'cfg%d_%s' "$cfg_id" "$label")
    mkdir -p "$RESULT_DIR/$cfg_tag"

    for idir in $INSTANCE_DIRS; do
        for instance in "$idir"/*.txt; do
            [ -f "$instance" ] || continue
            iname=$(basename "$instance" .txt)
            run_key="${cfg_tag}__${iname}"

            # --resume: skip already-done runs
            if [ "$RESUME" = "--resume" ] && grep -qF "$run_key" "$DONE_FILE" 2>/dev/null; then
                skip_count=$((skip_count + 1))
                continue
            fi

            run_count=$((run_count + 1))
            outfile="$RESULT_DIR/$cfg_tag/${iname}.json"

            ts_start=$(date +%s)
            printf "[%s] %s | %s ... " "$cfg_tag" "$iname" "$(date '+%H:%M:%S')" | tee -a "$LOG"

            # Run solver
            "$BINARY" "$instance" \
                --solver "$SOLVER" \
                --timeout "$TIMEOUT" \
                --method "$METHOD" \
                --wc "$WC" --wo "$WO" \
                --cardinality-encoding "$card" \
                --implied-constraints "$ic" \
                --symmetry-breaking "$sb" \
                --output "$outfile" 2>>"$LOG" || true

            ts_end=$(date +%s)
            wall_s=$((ts_end - ts_start))

            # Parse JSON → CSV row
            python3 - << PYEOF >> "$CSV"
import json, sys, os
f = "$outfile"
iname = "$iname"
cfg_id = "$cfg_id"
label = "$label"
card = "$card"
ic = "$ic"
sb = "$sb"

try:
    with open(f) as fh:
        d = json.load(fh)

    status = d.get("status", "ERROR")
    elapsed = d.get("elapsed_seconds", 0)

    stages = d.get("stages", [])
    solve_s = sum(s.get("solve_seconds", 0) for s in stages)
    variables = stages[0].get("variables", 0) if stages else 0
    hard_cl   = stages[0].get("hard_clauses", 0) if stages else 0
    soft_cl   = stages[0].get("soft_clauses", 0) if stages else 0

    m = d.get("metrics") or {}
    sim  = m.get("similarity", "")
    cont = m.get("continuity", "")
    ot   = m.get("overtime", "")
    cov  = 120 if d.get("full_coverage") else m.get("coverage", "")

    # best_value = sim - wc*cont - wo*|P|*ot
    wc = float("$WC"); wo = float("$WO")
    P = d.get("P_value") or -1  # fallback
    try:
        best = sim - wc * cont - wo * abs(P) * ot
    except:
        best = ""

    print(f"{cfg_id}\t{label}\t{iname}\t{card}\t{ic}\t{sb}\t"
          f"{status}\t{elapsed:.4f}\t{solve_s:.4f}\t"
          f"{variables}\t{hard_cl}\t{soft_cl}\t"
          f"{best}\t{sim}\t{cont}\t{ot}\t{cov}")
except Exception as e:
    print(f"{cfg_id}\t{label}\t{iname}\t{card}\t{ic}\t{sb}\t"
          f"PARSE_ERROR\t\t\t\t\t\t\t\t\t\t")
PYEOF

            # Extract status for log
            st=$(python3 -c "import json; d=json.load(open('$outfile')); print(d.get('status','?'))" 2>/dev/null || echo "ERROR")
            printf "%s  (%ds wall)\n" "$st" "$wall_s" | tee -a "$LOG"

            # Mark done
            echo "$run_key" >> "$DONE_FILE"

            if [ "$st" = "ERROR" ] || [ "$st" = "PARSE_ERROR" ]; then
                fail_count=$((fail_count + 1))
            fi
        done
    done
done

echo "" | tee -a "$LOG"
echo "=== DONE ===" | tee -a "$LOG"
echo "Finished : $(date)" | tee -a "$LOG"
echo "Completed: $run_count runs" | tee -a "$LOG"
echo "Skipped  : $skip_count (--resume)" | tee -a "$LOG"
echo "Errors   : $fail_count" | tee -a "$LOG"
echo "CSV      : $CSV" | tee -a "$LOG"

# Generate summary
python3 experiments/collect_main_results.py "$RESULT_DIR" && \
    echo "Summary  : $RESULT_DIR/summary_by_config.csv" | tee -a "$LOG"
