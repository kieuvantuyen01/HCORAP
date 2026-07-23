#!/usr/bin/env bash
# Run B1 separately from the weighted B0 campaign.
#
# Each policy gets its own result directory so --resume keys, JSON files and
# summaries cannot be mixed with B0 or with the other lexicographic order.
#
# Usage:
#   bash experiments/run_lex_8cfg.sh
#   bash experiments/run_lex_8cfg.sh --resume
#
# Optional environment variables are the same as run_main_8cfg.sh, plus:
#   RESULT_ROOT=experiments/results/lex_8cfg

set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
RESULT_ROOT="${RESULT_ROOT:-experiments/results/lex_8cfg}"
RESUME="${1:-}"

if [ -n "$RESUME" ] && [ "$RESUME" != "--resume" ]; then
    echo "ERROR: expected --resume, received: $RESUME" >&2
    exit 2
fi

for method in lex-continuity lex-overtime; do
    result_dir="$RESULT_ROOT/$method"
    echo "=== B1 policy: $method ==="
    if [ -n "$RESUME" ]; then
        METHOD="$method" RESULT_DIR="$result_dir" \
            bash "$SCRIPT_DIR/run_main_8cfg.sh" "$RESUME"
    else
        METHOD="$method" RESULT_DIR="$result_dir" \
            bash "$SCRIPT_DIR/run_main_8cfg.sh"
    fi
done

echo "B1 results: $RESULT_ROOT"
