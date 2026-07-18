#!/usr/bin/env bash
# Pilot: chạy 50 cấu hình (2 cardinality × 5 IC × 5 SB) trên 1 instance
# Usage: bash experiments/pilot_50configs.sh
set -u

BINARY="./bin/release/hcorap_multi"
SOLVER="./open-wbo_macos"
INSTANCE="instances/paperInstances/TXT_10-25_4-5_U30/instance_30_10_4_1.txt"
RESULT_DIR="experiments/results/pilot_50"
TIMEOUT=60   # giây mỗi cấu hình (50 cfg × 60s = ~50 phút max)
METHOD="weighted"

CARDINALITIES="sorting-network totalizer"
ICS="none user-slots slot-capacity both both-plus"
SBS="none slots services slot-service all"

mkdir -p "$RESULT_DIR"
SUMMARY="$RESULT_DIR/summary.tsv"
printf 'cfg\tcardinality\tic\tsb\tstatus\tsolve_s\tvariables\thard_clauses\tsoft_clauses\tsimilarity\tcontinuity\tovertime\n' > "$SUMMARY"

cfg=0
total=50
echo "=== Pilot 50 configurations on $(basename $INSTANCE) ==="
echo "Solver: $SOLVER  Timeout: ${TIMEOUT}s  Method: $METHOD"
echo ""

for card in $CARDINALITIES; do
    for ic in $ICS; do
        for sb in $SBS; do
            cfg=$((cfg + 1))
            label=$(printf '%02d' $cfg)
            tag="${card}__${ic}__${sb}"
            outfile="$RESULT_DIR/cfg${label}_${tag}.json"

            printf "[%02d/%d] card=%-16s ic=%-14s sb=%-12s ... " \
                   "$cfg" "$total" "$card" "$ic" "$sb"

            "$BINARY" "$INSTANCE" \
                --solver "$SOLVER" \
                --timeout "$TIMEOUT" \
                --method "$METHOD" \
                --cardinality-encoding "$card" \
                --implied-constraints "$ic" \
                --symmetry-breaking "$sb" \
                --output "$outfile" 2>/dev/null
            ec=$?

            # Parse JSON result
            if [ -f "$outfile" ]; then
                status=$(python3 -c "import json,sys; d=json.load(open('$outfile')); print(d.get('status','?'))" 2>/dev/null)
                solve_s=$(python3 -c "
import json,sys
d=json.load(open('$outfile'))
stages=d.get('stages',[])
t=sum(s.get('solve_seconds',0) for s in stages)
print(f'{t:.2f}')
" 2>/dev/null)
                variables=$(python3 -c "
import json,sys
d=json.load(open('$outfile'))
stages=d.get('stages',[])
v=stages[0].get('variables',0) if stages else 0
print(v)
" 2>/dev/null)
                hclauses=$(python3 -c "
import json,sys
d=json.load(open('$outfile'))
stages=d.get('stages',[])
v=stages[0].get('hard_clauses',0) if stages else 0
print(v)
" 2>/dev/null)
                sclauses=$(python3 -c "
import json,sys
d=json.load(open('$outfile'))
stages=d.get('stages',[])
v=stages[0].get('soft_clauses',0) if stages else 0
print(v)
" 2>/dev/null)
                sim=$(python3 -c "
import json,sys
d=json.load(open('$outfile'))
m=d.get('metrics') or {}
print(m.get('similarity','?'))
" 2>/dev/null)
                cont=$(python3 -c "
import json,sys
d=json.load(open('$outfile'))
m=d.get('metrics') or {}
print(m.get('continuity','?'))
" 2>/dev/null)
                ot=$(python3 -c "
import json,sys
d=json.load(open('$outfile'))
m=d.get('metrics') or {}
print(m.get('overtime','?'))
" 2>/dev/null)
            else
                status="NO_OUTPUT"; solve_s="-"; variables="-"; hclauses="-"; sclauses="-"; sim="-"; cont="-"; ot="-"
            fi

            echo "${status}  ${solve_s}s"
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                "cfg${label}" "$card" "$ic" "$sb" \
                "$status" "$solve_s" "$variables" "$hclauses" "$sclauses" \
                "$sim" "$cont" "$ot" >> "$SUMMARY"
        done
    done
done

echo ""
echo "=== DONE. Summary: $SUMMARY ==="
echo ""
echo "--- OPTIMUM results sorted by solve time ---"
awk -F'\t' 'NR>1 && $5=="OPTIMUM" {print}' "$SUMMARY" \
    | sort -t$'\t' -k6 -n \
    | awk -F'\t' 'BEGIN{printf "%-4s %-18s %-16s %-14s %8s %8s %8s %8s\n","cfg","cardinality","ic","sb","t(s)","vars","sim","cont"}
                  {printf "%-4s %-18s %-16s %-14s %8s %8s %8s %8s\n",$1,$2,$3,$4,$6,$7,$10,$11}'

echo ""
echo "--- TIMEOUT / non-OPTIMUM ---"
awk -F'\t' 'NR>1 && $5!="OPTIMUM" {printf "%-4s %-18s %-16s %-14s %s\n",$1,$2,$3,$4,$5}' "$SUMMARY"
