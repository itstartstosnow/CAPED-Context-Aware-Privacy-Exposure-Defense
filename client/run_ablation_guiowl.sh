#!/usr/bin/env bash
# =============================================================================
# run_ablation_guiowl.sh — Run ablation agent evaluation on AndroidWorld
# =============================================================================
#
# Usage:
#   sh run_ablation_guiowl.sh <agent> [--start N] [--end N] [--debug]
#
# Agent (required, first argument):
#   text_only          OCR/NER-only text redaction, no visual masking
#   no_elem_verify     CAPED without VQA/embedding element rescue
#   no_task_extract    CAPED without task requirement extraction
#
# Options:
#   --start N   First task index (1-based, inclusive). Default: 1
#   --end   N   Last  task index (1-based, inclusive). Default: last task
#   --debug     Save per-step debug artifacts
#   --llm_model Model for task extraction (0.8b or 2b). Default: 0.8b
#
# Examples:
#   sh run_ablation_guiowl.sh text_only
#   sh run_ablation_guiowl.sh no_elem_verify --start 1 --end 20
#   sh run_ablation_guiowl.sh no_task_extract --debug
#
# Outputs: eval_ablation_<agent>_<timestamp>/
#   traj_test/          — per-task trajectory JSON
#   checkpoints/        — incremental checkpoints for resuming
#   log_ma3_<ts>.log    — system decision log
#
# =============================================================================
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: sh run_ablation_guiowl.sh <text_only|no_elem_verify|no_task_extract> [options]" >&2
    exit 1
fi

AGENT="$1"
shift

case "$AGENT" in
    text_only|no_elem_verify|no_task_extract) ;;
    *) echo "Unknown agent: $AGENT (must be text_only, no_elem_verify, or no_task_extract)" >&2; exit 1 ;;
esac

current_time=$(date +"%Y-%m-%d_%H-%M-%S")

export NO_PROXY=localhost,127.0.0.1
export no_proxy=localhost,127.0.0.1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

EVAL_DIR="${SCRIPT_DIR}/eval_ablation_${AGENT}_${current_time}"
mkdir -p "${EVAL_DIR}"

TRAJ_OUTPUT_PATH="${EVAL_DIR}/traj_test"
CHECKPOINT_DIR="${EVAL_DIR}/checkpoints"
LOG="${EVAL_DIR}/log_ma3_${current_time}.log"

mkdir -p "${TRAJ_OUTPUT_PATH}" "${CHECKPOINT_DIR}"

MODEL="GUI-Owl-7B"
API_KEY="EMPTY"
BASE_URL="http://localhost:8000/v1"
PRIVACY_SERVER_URL="http://localhost:8001"

# --- Parse remaining CLI arguments ---
_START=""
_END=""
_DEBUG=""
_LLM_MODEL="0.8b"
while [ "$#" -gt 0 ]; do
    case "$1" in
        --start)     _START="$2";     shift 2 ;;
        --end)       _END="$2";       shift 2 ;;
        --debug)     _DEBUG="--debug"; shift ;;
        --llm_model) _LLM_MODEL="$2"; shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done
TASKS=""

python run_ma3.py \
  --suite_family=android_world \
  --agent_name=$AGENT \
  --model=$MODEL \
  --api_key=$API_KEY \
  --base_url=$BASE_URL \
  --privacy_server_url=$PRIVACY_SERVER_URL \
  --llm_model=$_LLM_MODEL \
  --traj_output_path=$TRAJ_OUTPUT_PATH \
  --checkpoint_dir=$CHECKPOINT_DIR \
  --grpc_port=8554 \
  --console_port=5554 \
  ${TASKS:+--tasks=$TASKS} \
  ${_START:+--start=$_START} \
  ${_END:+--end=$_END} \
  ${_DEBUG} 2>&1 \
  | grep -vE 'fork_posix|tcp_posix|error: Message too long' \
  | grep -vE 'data:image/|;base64,|[A-Za-z0-9+/]{200,}={0,2}' \
  | tee "$LOG"
