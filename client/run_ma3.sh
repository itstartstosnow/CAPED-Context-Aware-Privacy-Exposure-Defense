current_time=$(date +"%Y-%m-%d_%H-%M-%S")

# 获取脚本所在目录，使用绝对路径
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# All outputs (traj, log) go under one eval dir
EVAL_DIR="${SCRIPT_DIR}/eval_${current_time}"
mkdir -p "${EVAL_DIR}"

TRAJ_OUTPUT_PATH="${EVAL_DIR}/traj"
LOG="${EVAL_DIR}/log_ma3_${current_time}.log"

MODEL_NAME="mobile_agent_v3"
MODEL="your model name"
API_KEY="your api key"
BASE_URL="your base url"

python run_ma3.py \
  --suite_family=android_world \
  --agent_name=$MODEL_NAME \
  --model=$MODEL \
  --api_key=$API_KEY \
  --base_url=$BASE_URL \
  --traj_output_path=$TRAJ_OUTPUT_PATH \
  --grpc_port=8554 \
  --console_port=5554 2>&1 | tee "$LOG"