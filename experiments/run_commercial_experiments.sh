#!/usr/bin/env sh

# The shell only dispatches runs. Parsing, model construction, solving,
# extraction, and independent verification remain inside the C++ executable.
set -u

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 RESULT_DIR INSTANCE [INSTANCE ...]" >&2
    exit 2
fi

RESULT_DIR=$1
shift

BINARY=${HCORAP_COMMERCIAL_BIN:-./bin/release/hcorap_commercial}
COMMERCIAL_CONFIGS=${COMMERCIAL_CONFIGS:-"gurobi-mip:mip-e cplex-mip:mip-e cplex-cp:cp-t cplex-cp:cp-i"}
METHODS=${METHODS:-"weighted lex-continuity lex-overtime"}
RUN_EPSILON=${RUN_EPSILON:-1}
DELTAS=${DELTAS:-"0 0.01 0.025 0.05 0.10"}
TIMEOUT=${TIMEOUT:-300}
THREADS=${THREADS:-1}
SEED=${SEED:-0}
WC=${WC:-1}
WO=${WO:-1}
MIP_GAP=${MIP_GAP:-0}
ABSOLUTE_MIP_GAP=${ABSOLUTE_MIP_GAP:-0}
SOFT_COVERAGE=${SOFT_COVERAGE:-0}
PRINT_ASSIGNMENTS=${PRINT_ASSIGNMENTS:-1}
NATIVE_LOGS=${NATIVE_LOGS:-0}
RESUME=${RESUME:-0}
ENUMERATION_LIMIT=${ENUMERATION_LIMIT:-5000000}
GUROBI_PARAM_FILE=${GUROBI_PARAM_FILE:-}
CPLEX_PARAM_FILE=${CPLEX_PARAM_FILE:-}

if [ ! -x "$BINARY" ]; then
    echo "Missing executable: $BINARY" >&2
    echo "Build it with one of the commands in docs/COMMERCIAL_SOLVERS.md." >&2
    exit 2
fi

for toggle in \
    "$RUN_EPSILON" "$SOFT_COVERAGE" "$PRINT_ASSIGNMENTS" \
    "$NATIVE_LOGS" "$RESUME"
do
    case "$toggle" in
        0|1) ;;
        *)
            echo "RUN_EPSILON, SOFT_COVERAGE, PRINT_ASSIGNMENTS, NATIVE_LOGS and RESUME must be 0 or 1." >&2
            exit 2
            ;;
    esac
done

sha256_file() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    else
        shasum -a 256 "$1" | awk '{print $1}'
    fi
}

mkdir -p "$RESULT_DIR"
if [ "$NATIVE_LOGS" = "1" ]; then
    mkdir -p "$RESULT_DIR/native_logs"
fi

MANIFEST="$RESULT_DIR/manifest.tsv"
manifest_header='run_id\tinstance_sha256\tinstance\tbackend\tformulation\tmethod\tdelta\tresult\texit_code'
if [ ! -f "$MANIFEST" ]; then
    printf '%b\n' "$manifest_header" > "$MANIFEST"
elif [ "$(sed -n '1p' "$MANIFEST")" != "$(printf '%b' "$manifest_header")" ]; then
    echo "Manifest schema mismatch: $MANIFEST" >&2
    echo "Use a fresh result directory." >&2
    exit 2
fi

ENVIRONMENT="$RESULT_DIR/environment.txt"
if [ ! -f "$ENVIRONMENT" ]; then
    {
        printf 'created_utc='; date -u '+%Y-%m-%dT%H:%M:%SZ'
        printf 'uname='; uname -a
        printf 'binary=%s\n' "$BINARY"
        printf 'binary_sha256='; sha256_file "$BINARY"
        printf 'commercial_configs=%s\n' "$COMMERCIAL_CONFIGS"
        printf 'methods=%s\nrun_epsilon=%s\ndeltas=%s\n' \
            "$METHODS" "$RUN_EPSILON" "$DELTAS"
        printf 'timeout=%s\nthreads=%s\nseed=%s\nwc=%s\nwo=%s\n' \
            "$TIMEOUT" "$THREADS" "$SEED" "$WC" "$WO"
        printf 'mip_gap=%s\nabsolute_mip_gap=%s\nsoft_coverage=%s\n' \
            "$MIP_GAP" "$ABSOLUTE_MIP_GAP" "$SOFT_COVERAGE"
        printf 'gurobi_param_file=%s\ncplex_param_file=%s\n' \
            "$GUROBI_PARAM_FILE" "$CPLEX_PARAM_FILE"
        if [ -n "$GUROBI_PARAM_FILE" ] && [ -f "$GUROBI_PARAM_FILE" ]; then
            printf 'gurobi_param_sha256='
            sha256_file "$GUROBI_PARAM_FILE"
        fi
        if [ -n "$CPLEX_PARAM_FILE" ] && [ -f "$CPLEX_PARAM_FILE" ]; then
            printf 'cplex_param_sha256='
            sha256_file "$CPLEX_PARAM_FILE"
        fi
        printf 'compiler='; ${CXX:-g++} --version | sed -n '1p'
        printf 'git_commit='; git rev-parse HEAD 2>/dev/null || printf 'unknown\n'
        printf 'git_dirty_files='; git status --porcelain 2>/dev/null | wc -l | tr -d ' '
        printf '\nbackend_inventory=\n'
        "$BINARY" --list-backends
    } > "$ENVIRONMENT"
fi

for instance in "$@"; do
    if [ ! -f "$instance" ]; then
        echo "Skipping missing instance: $instance" >&2
        continue
    fi
    instance_sha=$(sha256_file "$instance")
    short_sha=$(printf '%s' "$instance_sha" | cut -c1-12)
    base=$(basename "$instance" .txt)

    for commercial_config in $COMMERCIAL_CONFIGS; do
        backend=${commercial_config%%:*}
        formulation=${commercial_config#*:}
        case "$commercial_config" in
            gurobi-mip:mip-e|cplex-mip:mip-e|cplex-cp:cp-t|cplex-cp:cp-i|reference-enumerator:direct-schedule-enumeration) ;;
            *)
                echo "Unsupported commercial config: $commercial_config" >&2
                exit 2
                ;;
        esac

        parameter_file=
        case "$backend" in
            gurobi-mip) parameter_file=$GUROBI_PARAM_FILE ;;
            cplex-mip) parameter_file=$CPLEX_PARAM_FILE ;;
        esac
        if [ -n "$parameter_file" ] && [ ! -f "$parameter_file" ]; then
            echo "Missing parameter file for $backend: $parameter_file" >&2
            exit 2
        fi

        methods_to_run=$METHODS
        if [ "$RUN_EPSILON" = "1" ]; then
            methods_to_run="$methods_to_run epsilon"
        fi

        for method in $methods_to_run; do
            case "$method" in
                weighted|lex-continuity|lex-overtime) delta_values="-" ;;
                epsilon) delta_values=$DELTAS ;;
                *)
                    echo "Unsupported method: $method" >&2
                    exit 2
                    ;;
            esac

            for delta in $delta_values; do
                run_id="${short_sha}_${base}_${backend}_${formulation}_${method}"
                if [ "$delta" != "-" ]; then
                    run_id="${run_id}_delta_${delta}"
                fi
                result="$RESULT_DIR/$run_id.json"
                if [ "$RESUME" = "1" ] && [ -s "$result" ]; then
                    echo "Skipping existing result: $result"
                    continue
                fi

                set -- "$BINARY" "$instance" \
                    --backend "$backend" --formulation "$formulation" \
                    --method "$method" --timeout "$TIMEOUT" \
                    --threads "$THREADS" --seed "$SEED" \
                    --wc "$WC" --wo "$WO" \
                    --mip-gap "$MIP_GAP" \
                    --absolute-mip-gap "$ABSOLUTE_MIP_GAP" \
                    --enumeration-limit "$ENUMERATION_LIMIT" \
                    --output "$result"
                if [ "$delta" != "-" ]; then
                    set -- "$@" --delta "$delta"
                fi
                if [ "$SOFT_COVERAGE" = "1" ]; then
                    set -- "$@" --soft-coverage
                fi
                if [ "$PRINT_ASSIGNMENTS" = "1" ]; then
                    set -- "$@" --print-assignments
                fi
                if [ -n "$parameter_file" ]; then
                    set -- "$@" --parameter-file "$parameter_file"
                fi
                if [ "$NATIVE_LOGS" = "1" ]; then
                    set -- "$@" --solver-log \
                        "$RESULT_DIR/native_logs/$run_id.log"
                fi

                "$@"
                exit_code=$?
                printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                    "$run_id" "$instance_sha" "$instance" \
                    "$backend" "$formulation" "$method" "$delta" \
                    "$result" "$exit_code" >> "$MANIFEST"
            done
        done
    done
done

echo "Results: $RESULT_DIR"
echo "Manifest: $MANIFEST"
