#!/usr/bin/env bash
# =============================================================================
# run_privacy_guiowl.sh — Run PrivacyAwareGUIOwl evaluation on AndroidWorld
# =============================================================================
#
# Usage:
#   sh run_privacy_guiowl.sh [--start N] [--end N] [--debug]
#
# Options:
#   --start N   First task index (1-based, inclusive). Default: 1 (first task)
#   --end   N   Last  task index (1-based, inclusive). Default: last task
#   --debug     Save per-step debug artifacts (raw PNG, detection JSON,
#               labeled PNG, privacy_hooks.log) to checkpoints/debug/.
#               Without this flag only log_ma3 and traj_test are written.
#   (no args)   Run all tasks without debug artifacts
#
# Examples:
#   sh run_privacy_guiowl.sh --start 60 --end 90
#   sh run_privacy_guiowl.sh --start 60 --debug
#   sh run_privacy_guiowl.sh --debug
#   sh run_privacy_guiowl.sh --llm_model 2b
#
# Prerequisites:
#   1. AVD running:
#        ~/Library/Android/sdk/emulator/emulator \
#          -avd AndroidWorldAvd -no-snapshot -grpc 8554 -skip-adb-auth
#   2. GUI-Owl-7B vLLM server on port 8000:
#        (on GPU server) python -m vllm.entrypoints.openai.api_server \
#          --model GUI-Owl-7B --port 8000
#   3. agentPrivacy server on port 8001:
#        (on GPU server) python -m uvicorn api.main:app --host 0.0.0.0 --port 8001
#   4. Python venv active: workon android_world
#      (or use the interpreter directly: python)
#
# Task selection (set at most one; if both empty, all 116 tasks are run):
#   TASKS      — comma-separated task names, e.g. "CameraTakePhoto,ContactsAddContact"
#                Edit the TASKS= line below to set this.
#
#
# Outputs (all under eval_privacy_<timestamp>/):
#   traj_test/          — per-task trajectory JSON
#   checkpoints/        — incremental checkpoints for resuming
#   checkpoints/debug/  — per-step raw PNG, detection JSON, context JSON,
#                         labeled PNG, privacy_hooks.log  (only with --debug)
#   log_ma3_<ts>.log    — privacy system decision log (no image b64)
#
# =============================================================================
current_time=$(date +"%Y-%m-%d_%H-%M-%S")

# 绕过代理访问 localhost
export NO_PROXY=localhost,127.0.0.1
export no_proxy=localhost,127.0.0.1

# 告知 agent 可用 open_app 动作及 app 列表（DETAILED_TIPS）
# export ADD_GENERAL_ADD_INFO=1

# 获取脚本所在目录，使用绝对路径
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# All outputs (traj, checkpoints, log, eval_comparison) go under one eval dir
EVAL_DIR="${SCRIPT_DIR}/eval_privacy_${current_time}"
mkdir -p "${EVAL_DIR}"

TRAJ_OUTPUT_PATH="${EVAL_DIR}/traj_test"
CHECKPOINT_DIR="${EVAL_DIR}/checkpoints"
LOG="${EVAL_DIR}/log_ma3_${current_time}.log"

# Ensure downstream code can write trajectories/checkpoints immediately.
mkdir -p "${TRAJ_OUTPUT_PATH}" "${CHECKPOINT_DIR}"

MODEL_NAME="privacy_guiowl"
MODEL="GUI-Owl-7B"
API_KEY="EMPTY"
BASE_URL="http://localhost:8000/v1"
PRIVACY_SERVER_URL="http://localhost:8001"

# --- Parse CLI arguments ---
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
# 如需指定具体任务名（逗号分隔），在此设置：
TASKS=""

python run_ma3.py \
  --suite_family=android_world \
  --agent_name=$MODEL_NAME \
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
