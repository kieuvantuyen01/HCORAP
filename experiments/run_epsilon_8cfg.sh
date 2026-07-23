#!/usr/bin/env bash
# Run B2 separately from the weighted B0 and lexicographic B1 campaigns.
#
# Every similarity budget gets a dedicated result directory. After all budgets
# finish, collect_epsilon_results.py creates all-delta and deduplicated tables.
#
# Usage:
#   bash experiments/run_epsilon_8cfg.sh
#   bash experiments/run_epsilon_8cfg.sh --resume
#
# Optional environment variables are the same as run_main_8cfg.sh, plus:
#   DELTAS="0 0.01 0.025 0.05 0.10"
#   RESULT_ROOT=experiments/results/epsilon_8cfg

set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
DELTAS="${DELTAS:-0 0.01 0.025 0.05 0.10}"
RESULT_ROOT="${RESULT_ROOT:-experiments/results/epsilon_8cfg}"
RESUME="${1:-}"

if [ -n "$RESUME" ] && [ "$RESUME" != "--resume" ]; then
    echo "ERROR: expected --resume, received: $RESUME" >&2
    exit 2
fi
if [ -z "$DELTAS" ]; then
    echo "ERROR: DELTAS must contain at least one similarity budget" >&2
    exit 2
fi

seen_tags=" "
for delta in $DELTAS; do
    if [[ ! "$delta" =~ ^(0|0\.[0-9]{1,9}|1|1\.0{1,9})$ ]]; then
        echo "ERROR: invalid delta in DELTAS: $delta" >&2
        exit 2
    fi
    delta_tag="${delta//./p}"
    case "$seen_tags" in
        *" $delta_tag "*)
            echo "ERROR: duplicate delta directory tag: $delta" >&2
            exit 2
            ;;
    esac
    seen_tags="${seen_tags}${delta_tag} "

    result_dir="$RESULT_ROOT/delta_$delta_tag"
    echo "=== B2 similarity budget: delta=$delta ==="
    if [ -n "$RESUME" ]; then
        METHOD=epsilon DELTA="$delta" RESULT_DIR="$result_dir" \
            bash "$SCRIPT_DIR/run_main_8cfg.sh" "$RESUME"
    else
        METHOD=epsilon DELTA="$delta" RESULT_DIR="$result_dir" \
            bash "$SCRIPT_DIR/run_main_8cfg.sh"
    fi
done

python3 "$SCRIPT_DIR/collect_epsilon_results.py" "$RESULT_ROOT"
echo "B2 results: $RESULT_ROOT"
