#!/usr/bin/env sh

# All timed HCORAP work is performed by the C++ executable and one pinned C++
# MaxSAT backend.  This shell file only dispatches runs and writes a manifest.
set -u

if [ "$#" -lt 3 ]; then
    echo "Usage: $0 OPEN_WBO_BIN RESULT_DIR INSTANCE [INSTANCE ...]" >&2
    exit 2
fi

SOLVER=$1
RESULT_DIR=$2
shift 2

BINARY=${HCORAP_MULTI_BIN:-./bin/release/hcorap_multi}
TIMEOUT=${TIMEOUT:-300}
WC=${WC:-1}
WO=${WO:-1}
DELTAS=${DELTAS:-"0 0.01 0.025 0.05 0.10"}
# Use '-' instead of ':-' so an explicitly empty METHODS value enables an
# epsilon-only campaign while an unset value keeps the historical defaults.
METHODS=${METHODS-"weighted lex-continuity lex-overtime"}
RUN_EPSILON=${RUN_EPSILON:-1}
CARDINALITY_ENCODINGS=${CARDINALITY_ENCODINGS:-"sorting-network"}
IMPLIED_CONFIGS=${IMPLIED_CONFIGS:-"none"}
SYMMETRY_CONFIGS=${SYMMETRY_CONFIGS:-"none"}
SOFT_COVERAGE=${SOFT_COVERAGE:-0}
SOLVER_ID=${SOLVER_ID:-unknown}

if [ ! -x "$BINARY" ]; then
    echo "Missing C++ executable: $BINARY (run: make -j4 YICES=0)" >&2
    exit 2
fi
if [ ! -x "$SOLVER" ]; then
    echo "Missing C++ MaxSAT solver: $SOLVER" >&2
    exit 2
fi

mkdir -p "$RESULT_DIR"
MANIFEST="$RESULT_DIR/manifest.tsv"
manifest_header='run_id\tsha256\tinstance\tcardinality_encoding\timplied_constraints\tsymmetry_breaking\tmethod\tdelta\tresult\texit_code'
if [ ! -f "$MANIFEST" ]; then
    printf '%b\n' "$manifest_header" > "$MANIFEST"
elif [ "$(sed -n '1p' "$MANIFEST")" != "$(printf '%b' "$manifest_header")" ]; then
    echo "Manifest schema mismatch: $MANIFEST" >&2
    echo "Use a new result directory for this encoding campaign." >&2
    exit 2
fi

ENVIRONMENT="$RESULT_DIR/environment.txt"
if [ ! -f "$ENVIRONMENT" ]; then
    {
        printf 'created_utc='; date -u '+%Y-%m-%dT%H:%M:%SZ'
        printf 'uname='; uname -a
        printf 'solver_id=%s\n' "$SOLVER_ID"
        printf 'solver_path=%s\n' "$SOLVER"
        printf 'solver_sha256='; shasum -a 256 "$SOLVER" | awk '{print $1}'
        printf 'hcorap_binary=%s\n' "$BINARY"
        printf 'hcorap_sha256='; shasum -a 256 "$BINARY" | awk '{print $1}'
        printf 'timeout=%s\nwc=%s\nwo=%s\nmethods=%s\nrun_epsilon=%s\ndeltas=%s\ncardinality_encodings=%s\nimplied_configs=%s\nsymmetry_configs=%s\nsoft_coverage=%s\n' \
            "$TIMEOUT" "$WC" "$WO" "$METHODS" "$RUN_EPSILON" "$DELTAS" \
            "$CARDINALITY_ENCODINGS" "$IMPLIED_CONFIGS" \
            "$SYMMETRY_CONFIGS" "$SOFT_COVERAGE"
        printf 'compiler='; ${CXX:-g++} --version | sed -n '1p'
        printf 'git_commit='; git rev-parse HEAD 2>/dev/null || printf 'unknown\n'
        printf 'git_dirty_files='; git status --porcelain 2>/dev/null | wc -l | tr -d ' '
    } > "$ENVIRONMENT"
fi

CONFIGURATION_MATRIX="$RESULT_DIR/configuration_matrix.tsv"
if [ ! -f "$CONFIGURATION_MATRIX" ]; then
    printf 'configuration_id\tcardinality_encoding\timplied_constraints\tsymmetry_breaking\n' \
        > "$CONFIGURATION_MATRIX"
    configuration_index=0
    for matrix_cardinality in $CARDINALITY_ENCODINGS; do
        for matrix_implied in $IMPLIED_CONFIGS; do
            for matrix_symmetry in $SYMMETRY_CONFIGS; do
                configuration_index=$((configuration_index + 1))
                printf 'cfg_%03d\t%s\t%s\t%s\n' \
                    "$configuration_index" "$matrix_cardinality" \
                    "$matrix_implied" "$matrix_symmetry" \
                    >> "$CONFIGURATION_MATRIX"
            done
        done
    done
fi

coverage_option=
if [ "$SOFT_COVERAGE" = "1" ]; then
    coverage_option=--soft-coverage
fi

case "$RUN_EPSILON" in
    0|1) ;;
    *)
        echo "RUN_EPSILON must be 0 or 1: $RUN_EPSILON" >&2
        exit 2
        ;;
esac

run_index=$(awk 'END { print (NR > 0 ? NR - 1 : 0) }' "$MANIFEST")
for instance in "$@"; do
    if [ ! -f "$instance" ]; then
        echo "Skipping missing instance: $instance" >&2
        continue
    fi
    sha256=$(shasum -a 256 "$instance" | awk '{print $1}')
    base=$(basename "$instance" .txt)

    for cardinality_encoding in $CARDINALITY_ENCODINGS; do
        case "$cardinality_encoding" in
            sorting-network|totalizer) ;;
            *)
                echo "Unsupported cardinality encoding: $cardinality_encoding" >&2
                exit 2
                ;;
        esac

        for implied_config in $IMPLIED_CONFIGS; do
            case "$implied_config" in
                none|user-slots|slot-capacity|both|both-plus) ;;
                *)
                    echo "Unsupported implied-constraints config: $implied_config" >&2
                    exit 2
                    ;;
            esac

            for symmetry_config in $SYMMETRY_CONFIGS; do
                case "$symmetry_config" in
                    none|slots|services|slot-service|all) ;;
                    *)
                        echo "Unsupported symmetry-breaking config: $symmetry_config" >&2
                        exit 2
                        ;;
                esac

                for method in $METHODS; do
                    case "$method" in
                        weighted|lex-continuity|lex-overtime) ;;
                        *)
                            echo "Unsupported method: $method" >&2
                            exit 2
                            ;;
                    esac
                    run_index=$((run_index + 1))
                    run_id=$(printf '%05d_%s_%s_%s_%s_%s' \
                        "$run_index" "$base" "$cardinality_encoding" \
                        "$implied_config" "$symmetry_config" "$method")
                    result="$RESULT_DIR/$run_id.json"
                    "$BINARY" "$instance" --solver "$SOLVER" --timeout "$TIMEOUT" \
                        --method "$method" --wc "$WC" --wo "$WO" \
                        --cardinality-encoding "$cardinality_encoding" \
                        --implied-constraints "$implied_config" \
                        --symmetry-breaking "$symmetry_config" \
                        $coverage_option --output "$result"
                    exit_code=$?
                    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                        "$run_id" "$sha256" "$instance" \
                        "$cardinality_encoding" "$implied_config" \
                        "$symmetry_config" "$method" "-" "$result" \
                        "$exit_code" >> "$MANIFEST"
                done

                if [ "$RUN_EPSILON" = "1" ]; then
                    for delta in $DELTAS; do
                        run_index=$((run_index + 1))
                        run_id=$(printf '%05d_%s_%s_%s_%s_epsilon_%s' \
                            "$run_index" "$base" "$cardinality_encoding" \
                            "$implied_config" "$symmetry_config" "$delta")
                        result="$RESULT_DIR/$run_id.json"
                        "$BINARY" "$instance" --solver "$SOLVER" --timeout "$TIMEOUT" \
                            --method epsilon --delta "$delta" --wc "$WC" --wo "$WO" \
                            --cardinality-encoding "$cardinality_encoding" \
                            --implied-constraints "$implied_config" \
                            --symmetry-breaking "$symmetry_config" \
                            $coverage_option --output "$result"
                        exit_code=$?
                        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                            "$run_id" "$sha256" "$instance" \
                            "$cardinality_encoding" "$implied_config" \
                            "$symmetry_config" "epsilon" "$delta" "$result" \
                            "$exit_code" >> "$MANIFEST"
                    done
                fi
            done
        done
    done
done

if ! python3 experiments/collect_cpp_results.py "$RESULT_DIR"; then
    echo "Failed to build campaign CSV logs." >&2
    exit 2
fi

echo "Results: $RESULT_DIR"
echo "Manifest: $MANIFEST"
echo "Configuration matrix: $CONFIGURATION_MATRIX"
echo "Excel-ready log: $RESULT_DIR/runs.csv"
