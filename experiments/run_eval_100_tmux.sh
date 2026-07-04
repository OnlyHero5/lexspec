#!/usr/bin/env bash
# 在 tmux 中启动 LexSpec-100 单模型评测（需先切换远程模型）

set -o pipefail

RED='\033[0;31m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

usage() {
  cat <<'EOF'
用法: run_eval_100_tmux.sh <model> [run_eval_100.sh 的额外参数...]

  model: gemma4 | qwen36 | qwen35

示例:
  ./experiments/run_eval_100_tmux.sh qwen36 --skip-eval
  ./experiments/run_eval_100_tmux.sh gemma4 --skip-reflexion

查看: tmux attach -t <model>-eval100
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

MODEL="$1"
shift
SESSION="${MODEL}-eval100"
LOG_DIR="${PROJECT_ROOT}/outputs/eval_100/${MODEL}/logs"
mkdir -p "$LOG_DIR"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo -e "${RED}tmux session 已存在: ${SESSION}${NC}" >&2
  echo "  tmux attach -t ${SESSION}"
  exit 1
fi

chmod +x "${SCRIPT_DIR}/run_eval_100.sh" "${SCRIPT_DIR}/eval_100_status.sh"

tmux new-session -d -s "$SESSION" \
  "cd '${PROJECT_ROOT}' && ./experiments/run_eval_100.sh '${MODEL}' $* 2>&1 | tee -a '${LOG_DIR}/run.log'; echo '--- exited with code' \$? '---'; read -p 'Press enter to close...' _"

echo "已启动 tmux session: ${SESSION}"
echo "  tmux attach -t ${SESSION}"
echo "  tail -f ${LOG_DIR}/run.log"
