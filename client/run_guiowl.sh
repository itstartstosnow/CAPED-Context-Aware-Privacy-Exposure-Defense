current_time=$(date +"%Y-%m-%d_%H-%M-%S")

# 绕过代理访问 localhost
export NO_PROXY=localhost,127.0.0.1
export no_proxy=localhost,127.0.0.1

# 获取脚本所在目录，使用绝对路径
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# All outputs (traj, checkpoints, log, eval_comparison) go under one eval dir
EVAL_DIR="${SCRIPT_DIR}/eval_${current_time}"
mkdir -p "${EVAL_DIR}"

TRAJ_OUTPUT_PATH="${EVAL_DIR}/traj_test"
CHECKPOINT_DIR="${EVAL_DIR}/checkpoints"
LOG="${EVAL_DIR}/log_ma3_${current_time}.log"

MODEL_NAME="gui_owl"
MODEL="GUI-Owl-7B"
API_KEY="EMPTY"
BASE_URL="http://localhost:8000/v1"

python run_ma3.py \
  --suite_family=android_world \
  --agent_name=$MODEL_NAME \
  --model=$MODEL \
  --api_key=$API_KEY \
  --base_url=$BASE_URL \
  --traj_output_path=$TRAJ_OUTPUT_PATH \
  --checkpoint_dir=$CHECKPOINT_DIR \
  --grpc_port=8554 \
  --console_port=5554 2>&1 | grep -v "fork_posix\|tcp_posix\|error:\ Message\ too\ long" | tee "$LOG"