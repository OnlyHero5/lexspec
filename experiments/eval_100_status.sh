#!/usr/bin/env bash
# 查看 LexSpec-100 九组抽取进度（3 模型 × baseline/dep/reflexion）

set -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TESTSET="${ROOT}/data/processed/lexspec_100.jsonl"
TOTAL="$(wc -l < "$TESTSET" | tr -d ' ')"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

count_jsonl() {
  local f="$1"
  if [[ -f "$f" ]]; then
    wc -l < "$f" | tr -d ' '
  else
    echo "0"
  fi
}

read_progress() {
  local f="$1"
  if [[ -f "$f" ]]; then
    head -1 "$f"
  fi
}

status_cell() {
  local model="$1"
  local step="$2"
  local file="$3"
  local prog="$4"
  local n
  n="$(count_jsonl "$file")"
  if [[ "$n" -ge "$TOTAL" && -f "$file" ]]; then
    echo -e "${GREEN}完成 ${n}/${TOTAL}${NC}"
  elif [[ -f "$prog" ]]; then
    echo -e "${YELLOW}进行中${NC}  $(cat "$prog")"
  elif [[ "$n" -gt 0 ]]; then
    echo -e "${YELLOW}部分 ${n}/${TOTAL}${NC}"
  else
    echo -e "${RED}未开始${NC}"
  fi
}

echo "LexSpec-100 评测进度（测试集 ${TOTAL} 条: gold_500 前 100，lexspec_100.jsonl）"
echo ""
printf "%-8s %-12s %-12s %-12s\n" "模型" "baseline" "ours_dep" "ours_reflexion"
printf "%-8s %-12s %-12s %-12s\n" "----" "--------" "--------" "--------------"

for key in gemma4 qwen36 qwen35; do
  base="${ROOT}/outputs/eval_100/${key}/predictions"
  b="$(status_cell "$key" baseline "${base}/baseline.jsonl" "${base}/baseline.progress")"
  d="$(status_cell "$key" dep "${base}/ours_dep.jsonl" "${base}/ours_dep.progress")"
  r="$(status_cell "$key" reflexion "${base}/ours_reflexion.jsonl" "${base}/ours_reflexion.progress")"
  printf "%-8s %-12b %-12b %-12b\n" "$key" "$b" "$d" "$r"
done

echo ""
echo "金标来源: gold_triplets_500 前 100 条（见 data/processed/gold_100_manifest.json）"
