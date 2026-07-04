#!/usr/bin/env bash
# =============================================================================
# LexSpec 500 条金标评测 — 多模型顺序测试
# =============================================================================
#
# 在 gold_testset_500.jsonl / gold_triplets_500.jsonl 上，对指定模型依次运行
# 步骤 03–07（baseline → dep → reflexion → evaluate → error analysis）。
#
# 推荐顺序（需远程 llama-server 切换模型后执行）:
#   1. gemma4   — Gemma 4 31B（当前远程已加载时可立即跑）
#   2. qwen36   — Qwen3.6 27B
#   3. qwen35   — Qwen3.5 9B（论文主实验模型）
#
# 用法:
#   chmod +x experiments/run_eval_500.sh
#
#   # 仅跑 Gemma4（远程已加载 gemma-4-31B-it-Q8_0.gguf）
#   ./experiments/run_eval_500.sh gemma4
#
#   # 切换远程模型后跑 Qwen3.6 27B
#   ./experiments/run_eval_500.sh qwen36
#
#   # 切换远程模型后跑 Qwen3.5 9B
#   ./experiments/run_eval_500.sh qwen35
#
#   # 检查三台模型是否就绪后依次全跑（中间需人工切换服务）
#   ./experiments/run_eval_500.sh all
#
#   # 只跑抽取，跳过评估
#   ./experiments/run_eval_500.sh gemma4 --skip-eval
#
# 输出目录:
#   outputs/eval_500/<model_key>/
#     predictions/{baseline,ours_dep,ours_reflexion}.jsonl
#     metrics/{task_metrics,linguistic_metrics,significance}.json
#     error_cases/
#
# 监控进度:
#   tail -f outputs/eval_500/gemma4/predictions/baseline.progress
# =============================================================================

set -o pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

TESTSET="${PROJECT_ROOT}/data/processed/gold_testset_500.jsonl"
GOLD="${PROJECT_ROOT}/data/processed/gold_triplets_500.jsonl"
CONSTRAINTS="${PROJECT_ROOT}/configs/constraints.yaml"
BASE_URL="http://10.0.16.254:8080/v1"

SKIP_EVAL=false
MODEL_KEYS=()

declare -A MODEL_CONFIG=(
  [gemma4]="configs/model_gemma4.yaml"
  [qwen36]="configs/model_qwen36.yaml"
  [qwen35]="configs/model.yaml"
)
declare -A MODEL_SERVER_NAME=(
  [gemma4]="gemma-4-31B-it-Q8_0.gguf"
  [qwen36]="qwen3.6-27b"
  [qwen35]="qwen3.5-9b"
)
declare -A MODEL_DISPLAY=(
  [gemma4]="Gemma 4 31B"
  [qwen36]="Qwen3.6 27B"
  [qwen35]="Qwen3.5 9B"
)

usage() {
  cat <<'EOF'
用法: run_eval_500.sh <model...> [--skip-eval]

  model:  gemma4 | qwen36 | qwen35 | all

  --skip-eval   只跑步骤 03–05，跳过 06–07
  --help        显示本帮助
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-eval) SKIP_EVAL=true; shift ;;
    --help|-h) usage; exit 0 ;;
    all) MODEL_KEYS+=(gemma4 qwen36 qwen35); shift ;;
    gemma4|qwen36|qwen35) MODEL_KEYS+=("$1"); shift ;;
    *) echo "未知参数: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ ${#MODEL_KEYS[@]} -eq 0 ]]; then
  echo -e "${RED}请指定模型: gemma4 | qwen36 | qwen35 | all${NC}" >&2
  usage
  exit 1
fi

for f in "$TESTSET" "$GOLD" "$CONSTRAINTS"; do
  if [[ ! -f "$f" ]]; then
    echo -e "${RED}[失败]${NC} 缺少文件: $f"
    exit 1
  fi
done

if ! python3 -c "
import json, sys
from pathlib import Path
def ids(p):
    return [json.loads(l)['clause_id'] for l in Path(p).read_text().splitlines() if l.strip()]
gold_ids = ids('${GOLD}')
test_ids = ids('${TESTSET}')
if gold_ids != test_ids:
    print('金标与测试集 clause_id 顺序/数量不一致', file=sys.stderr)
    sys.exit(1)
print(f'数据对齐: {len(gold_ids)} 条 clause_id 一致')
"; then
  echo -e "${RED}[失败]${NC} 金标 ${GOLD} 与测试集 ${TESTSET} 未对齐"
  exit 1
fi

check_server_model() {
  local expected="$1"
  local loaded
  loaded="$(curl -sf --connect-timeout 8 "${BASE_URL}/models" \
    | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['data'][0]['id'] if d.get('data') else '')" 2>/dev/null || true)"
  if [[ -z "$loaded" ]]; then
    echo -e "${RED}[失败]${NC} 无法连接 ${BASE_URL} 或无法读取模型列表"
    return 1
  fi
  if [[ "$loaded" != "$expected" ]]; then
    echo -e "${YELLOW}[等待]${NC} 远程当前模型: ${loaded}"
    echo -e "${YELLOW}       ${NC} 需要模型:     ${expected}"
    echo -e "${YELLOW}       ${NC} 请在远程切换 llama-server 后重新运行本脚本。"
    return 1
  fi
  echo -e "${GREEN}[就绪]${NC} 远程模型: ${loaded}"
  return 0
}

warn_gold_annotation_bias() {
  local key="$1"
  case "$key" in
    qwen36)
      echo -e "${YELLOW}[注意]${NC} qwen36 与金标 primary 标注模型相同，分数含自评偏差，勿与 qwen35 直接比模型能力。"
      ;;
    gemma4)
      echo -e "${YELLOW}[注意]${NC} gemma4 与金标 secondary 标注模型相同，分数含自评偏差，勿与 qwen35 直接比模型能力。"
      ;;
  esac
}

run_one_model() {
  local key="$1"
  local config="${PROJECT_ROOT}/${MODEL_CONFIG[$key]}"
  local server_name="${MODEL_SERVER_NAME[$key]}"
  local display="${MODEL_DISPLAY[$key]}"
  local out_root="${PROJECT_ROOT}/outputs/eval_500/${key}"
  local pred_dir="${out_root}/predictions"
  local log_dir="${out_root}/logs"

  echo ""
  echo -e "${BOLD}${BLUE}============================================================${NC}"
  echo -e "${BOLD}${BLUE}  500 条评测 — ${display} (${key})${NC}"
  echo -e "${BOLD}${BLUE}  配置: ${config}${NC}"
  echo -e "${BOLD}${BLUE}  输出: ${out_root}${NC}"
  echo -e "${BOLD}${BLUE}============================================================${NC}"

  if ! check_server_model "$server_name"; then
    return 1
  fi
  warn_gold_annotation_bias "$key"

  mkdir -p "$pred_dir" "$log_dir" "${out_root}/metrics" "${out_root}/error_cases"
  cd "$PROJECT_ROOT" || return 1

  local step_failed=false

  echo -e "${BLUE}[步骤 03]${NC} baseline ..."
  if ! python3 experiments/step_03_extract_baseline.py \
      --config "$config" \
      --testset "$TESTSET" \
      --output-dir "$out_root" \
      2>&1 | tee -a "${log_dir}/step_03_baseline.log"; then
    step_failed=true
  fi

  echo -e "${BLUE}[步骤 04]${NC} ours_dep ..."
  if ! python3 experiments/step_04_extract_dep.py \
      --config "$config" \
      --testset "$TESTSET" \
      --output-dir "$out_root" \
      2>&1 | tee -a "${log_dir}/step_04_dep.log"; then
    step_failed=true
  fi

  echo -e "${BLUE}[步骤 05]${NC} ours_reflexion ..."
  if ! python3 experiments/step_05_extract_reflexion.py \
      --config "$config" \
      --testset "$TESTSET" \
      --output-dir "$out_root" \
      2>&1 | tee -a "${log_dir}/step_05_reflexion.log"; then
    step_failed=true
  fi

  if [[ "$SKIP_EVAL" == "true" ]]; then
    echo -e "${YELLOW}[跳过]${NC} 步骤 06–07（--skip-eval）"
    [[ "$step_failed" == "true" ]] && return 1 || return 0
  fi

  echo -e "${BLUE}[步骤 06]${NC} evaluate ..."
  if ! python3 experiments/step_06_evaluate.py \
      --config "$config" \
      --predictions-dir "$pred_dir" \
      --gold "$GOLD" \
      --testset "$TESTSET" \
      --constraints "$CONSTRAINTS" \
      --output-dir "$out_root" \
      2>&1 | tee -a "${log_dir}/step_06_evaluate.log"; then
    step_failed=true
  fi

  echo -e "${BLUE}[步骤 07]${NC} analyze_errors ..."
  if ! python3 experiments/step_07_analyze_errors.py \
      --config "$config" \
      --predictions-dir "$pred_dir" \
      --gold "$GOLD" \
      --testset "$TESTSET" \
      --constraints "$CONSTRAINTS" \
      --output-dir "$out_root" \
      2>&1 | tee -a "${log_dir}/step_07_errors.log"; then
    step_failed=true
  fi

  if [[ "$step_failed" == "true" ]]; then
    echo -e "${RED}[失败]${NC} ${display} 部分步骤出错，见 ${log_dir}/"
    return 1
  fi

  echo -e "${GREEN}[完成]${NC} ${display} — 结果: ${out_root}/metrics/"
  return 0
}

echo -e "${BOLD}LexSpec 500 条金标评测${NC}"
echo "  测试集: ${TESTSET}"
echo "  金标:   ${GOLD}"
echo "  模型:   ${MODEL_KEYS[*]}"

overall_ok=true
for key in "${MODEL_KEYS[@]}"; do
  if [[ -z "${MODEL_CONFIG[$key]:-}" ]]; then
    echo -e "${RED}未知模型键: ${key}${NC}"
    overall_ok=false
    continue
  fi
  if ! run_one_model "$key"; then
    overall_ok=false
    if [[ ${#MODEL_KEYS[@]} -gt 1 ]]; then
      echo -e "${YELLOW}继续下一个模型...${NC}"
    fi
  fi
done

if [[ "$overall_ok" == "true" ]]; then
  echo -e "\n${GREEN}${BOLD}全部请求模型评测完成。${NC}"
  exit 0
fi
echo -e "\n${RED}${BOLD}部分模型未完成。切换远程模型后重跑对应项。${NC}"
exit 1
