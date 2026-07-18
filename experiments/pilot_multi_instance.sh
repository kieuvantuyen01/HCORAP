#!/usr/bin/env bash
# Pilot multi-instance: chạy 50 cfg × N instances để tổng hợp ranking tin cậy hơn
# Usage: bash experiments/pilot_multi_instance.sh
set -u

BINARY="./bin/release/hcorap_multi"
SOLVER="./open-wbo_macos"
TIMEOUT=60
METHOD="weighted"

# Chọn 5 instance nhỏ cùng kích thước (30U, 120S, 10A) — seed khác nhau
INSTANCES="
instances/paperInstances/TXT_10-25_4-5_U30/instance_30_10_4_1.txt
instances/paperInstances/TXT_10-25_4-5_U30/instance_30_10_4_2.txt
instances/paperInstances/TXT_10-25_4-5_U30/instance_30_10_4_3.txt
instances/paperInstances/TXT_10-25_4-5_U30/instance_30_10_4_4.txt
instances/paperInstances/TXT_10-25_4-5_U30/instance_30_10_4_5.txt
"

CARDINALITIES="sorting-network totalizer"
ICS="none user-slots slot-capacity both both-plus"
SBS="none slots services slot-service all"

RESULT_DIR="experiments/results/pilot_multi"
mkdir -p "$RESULT_DIR"

DETAIL="$RESULT_DIR/detail.tsv"
printf 'instance\tcardinality\tic\tsb\tstatus\tsolve_s\tvariables\thard_clauses\tsoft_clauses\tsimilarity\tcontinuity\tovertime\tweighted\n' > "$DETAIL"

parse_json() {
    local f="$1" field="$2"
    python3 -c "
import json, sys
try:
    d = json.load(open('$f'))
    print(d$field)
except: print('?')
" 2>/dev/null
}

echo "=== Multi-instance Pilot: 5 instances × 50 configs ==="
echo ""

inst_num=0
for instance in $INSTANCES; do
    inst_num=$((inst_num + 1))
    iname=$(basename "$instance" .txt)
    echo ">>> Instance $inst_num/5: $iname"
    cfg=0
    for card in $CARDINALITIES; do
        for ic in $ICS; do
            for sb in $SBS; do
                cfg=$((cfg + 1))
                outfile="$RESULT_DIR/${iname}__${card}__${ic}__${sb}.json"
                "$BINARY" "$instance" \
                    --solver "$SOLVER" \
                    --timeout "$TIMEOUT" \
                    --method "$METHOD" \
                    --cardinality-encoding "$card" \
                    --implied-constraints "$ic" \
                    --symmetry-breaking "$sb" \
                    --output "$outfile" 2>/dev/null

                if [ -f "$outfile" ]; then
                    status=$(python3 -c "import json; d=json.load(open('$outfile')); print(d.get('status','?'))" 2>/dev/null)
                    solve_s=$(python3 -c "
import json
d=json.load(open('$outfile'))
t=sum(s.get('solve_seconds',0) for s in d.get('stages',[]))
print(f'{t:.4f}')
" 2>/dev/null)
                    variables=$(python3 -c "
import json
d=json.load(open('$outfile'))
st=d.get('stages',[])
print(st[0].get('variables',0) if st else 0)
" 2>/dev/null)
                    hc=$(python3 -c "
import json
d=json.load(open('$outfile'))
st=d.get('stages',[])
print(st[0].get('hard_clauses',0) if st else 0)
" 2>/dev/null)
                    sc=$(python3 -c "
import json
d=json.load(open('$outfile'))
st=d.get('stages',[])
print(st[0].get('soft_clauses',0) if st else 0)
" 2>/dev/null)
                    sim=$(python3 -c "
import json
d=json.load(open('$outfile'))
m=d.get('metrics') or {}
print(m.get('similarity','?'))
" 2>/dev/null)
                    cont=$(python3 -c "
import json
d=json.load(open('$outfile'))
m=d.get('metrics') or {}
print(m.get('continuity','?'))
" 2>/dev/null)
                    ot=$(python3 -c "
import json
d=json.load(open('$outfile'))
m=d.get('metrics') or {}
print(m.get('overtime','?'))
" 2>/dev/null)
                    wt=$(python3 -c "
import json
d=json.load(open('$outfile'))
m=d.get('metrics') or {}
s=m.get('similarity',0); c=m.get('continuity',0); o=m.get('overtime',0)
if isinstance(s,int): print(s - c - o)
else: print('?')
" 2>/dev/null)
                else
                    status="NO_OUTPUT"; solve_s="-"; variables="-"; hc="-"; sc="-"; sim="-"; cont="-"; ot="-"; wt="-"
                fi
                printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                    "$iname" "$card" "$ic" "$sb" \
                    "$status" "$solve_s" "$variables" "$hc" "$sc" \
                    "$sim" "$cont" "$ot" "$wt" >> "$DETAIL"
            done
        done
    done
    echo "    Done $cfg configs"
done

echo ""
echo "=== AGGREGATED SUMMARY (mean solve time per IC, across all instances & SB) ==="
python3 - <<'PYEOF'
import csv, collections, sys

rows = []
with open("experiments/results/pilot_multi/detail.tsv") as f:
    reader = csv.DictReader(f, delimiter='\t')
    for r in reader:
        rows.append(r)

# Group by (cardinality, ic): avg solve_s, avg vars
from statistics import mean

def grp_avg(rows, key_fields, val_field):
    groups = collections.defaultdict(list)
    for r in rows:
        if r[val_field] not in ('-', '?', 'NO_OUTPUT'):
            try:
                groups[tuple(r[k] for k in key_fields)].append(float(r[val_field]))
            except: pass
    return {k: mean(v) for k, v in groups.items()}

# IC summary
print("\n--- IC ranking by avg solve time (mean over all instances & SB) ---")
ic_times = grp_avg(rows, ['cardinality','ic'], 'solve_s')
ic_vars  = grp_avg(rows, ['cardinality','ic'], 'variables')
ic_rows = sorted(ic_times.items(), key=lambda x: x[1])
print(f"{'cardinality':<18} {'ic':<14} {'avg_solve_s':>12} {'avg_vars':>10}")
print("-"*58)
for (card, ic), t in ic_rows:
    v = ic_vars.get((card,ic), 0)
    mark = " ← baseline" if ic == "none" else (" ← BEST" if t == min(v2 for (_,ic2),v2 in ic_times.items() if _ == card) else "")
    print(f"{card:<18} {ic:<14} {t:>12.4f} {v:>10.0f}{mark}")

# SB summary
print("\n--- SB ranking by avg solve time (mean over all instances & IC) ---")
sb_times = grp_avg(rows, ['cardinality','sb'], 'solve_s')
sb_rows = sorted(sb_times.items(), key=lambda x: x[1])
print(f"{'cardinality':<18} {'sb':<14} {'avg_solve_s':>12}")
print("-"*46)
for (card, sb), t in sb_rows:
    mark = " ← baseline" if sb == "none" else ""
    print(f"{card:<18} {sb:<14} {t:>12.4f}{mark}")

# Non-OPTIMUM
print("\n--- Non-OPTIMUM cases ---")
bad = [r for r in rows if r['status'] != 'OPTIMUM']
if bad:
    for r in bad:
        print(f"  {r['instance']:40s} {r['cardinality']:18s} {r['ic']:14s} {r['sb']:14s} {r['status']}")
else:
    print("  (none — all OPTIMUM)")
PYEOF

echo ""
echo "Detail log: $DETAIL"
