#!/usr/bin/env bash
# Run PrivacyAwareGUIOwl on a real ADB device for a single task.
#
# Usage:
#   bash run_realman_test.sh --task "Open the Weather app and check today's forecast."
#   bash run_realman_test.sh --task "Set alarm 7:30" --max_steps 20
#   bash run_realman_test.sh --task "打开Chrome" --debug
#   bash run_realman_test.sh --task "..." --device_serial <DEVICE_SERIAL>
#   bash run_realman_test.sh --task "打开Chrome" --use_avd
#
# If only one device is connected, --device_serial is auto-detected.
# To target an AVD emulator instead, add --use_avd.
#
# Outputs:
#   default      run.log, ner_mapping_final.json, GUIOwl task subdir
#   with --debug above plus debug/step_* artifacts and privacy_hooks.log
# All artifacts are saved together under realman_test_<datetime>/.

set -euo pipefail

export NO_PROXY=localhost,127.0.0.1
export no_proxy=localhost,127.0.0.1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── Configuration ──────────────────────────────────────────────────────────
MODEL="GUI-Owl-7B"
API_KEY="EMPTY"
BASE_URL="http://localhost:8000/v1"
PRIVACY_URL="http://localhost:8001"
ADB_PATH="${HOME}/Library/Android/sdk/platform-tools/adb"
MAX_STEPS=30
ADD_INFO=""

usage() {
    cat <<EOF
Usage:
  bash run_realman_test.sh --task "<task instruction>" [--xxx ...]

Required:
  --task <text>   Task instruction for the agent

Common extra args:
  --debug                Save per-step debug artifacts under debug/
  --device_serial <id>   Use a specific ADB device
  --use_avd              Use AndroidWorld AVD instead of a real device
  --output_dir <dir>     Override the auto-generated output directory

Notes:
  - This wrapper is bash-only. Do not invoke it with 'sh'.
  - Positional task text is not supported. Always use --task.
  - Other --xxx arguments are forwarded to run_realman_test.py unchanged.
EOF
}

INSTRUCTION=""
USER_OUTPUT_DIR=""
FORWARD_ARGS=("$@")

if [ $# -eq 0 ]; then
    usage
    exit 1
fi

while [ $# -gt 0 ]; do
    case "$1" in
        --help|-h)
            usage
            exit 0
            ;;
        --task)
            if [ $# -lt 2 ] || [[ "$2" == --* ]]; then
                echo "Error: --task requires a value." >&2
                exit 1
            fi
            INSTRUCTION="$2"
            shift 2
            ;;
        --task=*)
            INSTRUCTION="${1#*=}"
            if [ -z "${INSTRUCTION}" ]; then
                echo "Error: --task requires a value." >&2
                exit 1
            fi
            shift
            ;;
        --output_dir)
            if [ $# -lt 2 ] || [[ "$2" == --* ]]; then
                echo "Error: --output_dir requires a value." >&2
                exit 1
            fi
            USER_OUTPUT_DIR="$2"
            shift 2
            ;;
        --output_dir=*)
            USER_OUTPUT_DIR="${1#*=}"
            if [ -z "${USER_OUTPUT_DIR}" ]; then
                echo "Error: --output_dir requires a value." >&2
                exit 1
            fi
            shift
            ;;
        --*)
            # Unknown --xxx args are forwarded unchanged. If the next token is a
            # non-flag, treat it as this flag's value for parsing purposes.
            if [ $# -ge 2 ] && [[ "$2" != --* ]]; then
                shift 2
            else
                shift
            fi
            ;;
        *)
            echo "Error: unexpected positional argument '$1'." >&2
            echo "Use --task \"...\" instead of positional task text." >&2
            exit 1
            ;;
    esac
done

if [ -z "${INSTRUCTION}" ]; then
    echo "Error: missing required --task." >&2
    usage >&2
    exit 1
fi

# ── Output directory (all artifacts go here, including run.log) ────────────
if [ -n "${USER_OUTPUT_DIR}" ]; then
    OUT_DIR="${USER_OUTPUT_DIR}"
else
    current_time=$(date +"%Y-%m-%d_%H-%M-%S")
    OUT_DIR="${SCRIPT_DIR}/realman_test_${current_time}"
fi
mkdir -p "${OUT_DIR}"

echo "Task      : ${INSTRUCTION}"
echo "Output    : ${OUT_DIR}"
echo ""

# ── Run ────────────────────────────────────────────────────────────────────
cd "${SCRIPT_DIR}"
PY_ARGS=(
    --model "${MODEL}"
    --api_key "${API_KEY}"
    --base_url "${BASE_URL}"
    --privacy_url "${PRIVACY_URL}"
    --adb_path "${ADB_PATH}"
    --max_steps "${MAX_STEPS}"
    "${FORWARD_ARGS[@]}"
)

if [ -n "${ADD_INFO}" ]; then
    PY_ARGS+=(--add_info "${ADD_INFO}")
fi

if [ -z "${USER_OUTPUT_DIR}" ]; then
    PY_ARGS+=(--output_dir "${OUT_DIR}")
fi

python run_realman_test.py \
    "${PY_ARGS[@]}" \
    2>&1 | grep -v "fork_posix\|tcp_posix\|error:\ Message\ too\ long"
