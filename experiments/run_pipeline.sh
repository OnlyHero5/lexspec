#!/usr/bin/env bash
# =============================================================================
# LexSpec: 完整实验流水线编排器
# =============================================================================
#
# 按顺序执行 7 步实验流水线，是复现论文全部结果的唯一入口。
#
# 流水线步骤:
#   01 — 构建语料: 从 CUAD v1 数据构建 LexSpec 评估语料库
#   02 — 标注金标: 大语言模型独立标注合同条款
#   03 — 基线抽取: 纯大语言模型抽取（无约束）
#   04 — Dep 抽取:  大语言模型 + UD 约束校验
#   05 — Reflexion: 大语言模型 + UD 约束 + Reflexion 修正
#   06 — 评估:      双轨评估 + 显著性检验
#   07 — 错误分析:  语言学错误分类
#
# 用法:
#   chmod +x experiments/run_pipeline.sh
#   ./experiments/run_pipeline.sh
#
#   # 自定义配置
#   ./experiments/run_pipeline.sh --config /path/to/model.yaml
#
#   # 跳过已完成的步骤
#   ./experiments/run_pipeline.sh --skip-01 --skip-02
#
# 环境变量:
#   VENV_PATH    — Python 虚拟环境路径（可选）。默认: .venv/
#   EXTRA_ARGS   — 传递给所有 Python 脚本的额外参数。
# =============================================================================

set -o pipefail

# ---------------------------------------------------------------------------
# 终端颜色
# ---------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

# ---------------------------------------------------------------------------
# 项目根目录
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# 默认评测数据（curated gold_500 前 100 条，见 scripts/build_gold_100_from_500.py）
CONFIG="${PROJECT_ROOT}/configs/model.yaml"
CONSTRAINTS="${PROJECT_ROOT}/configs/constraints.yaml"
TESTSET="${PROJECT_ROOT}/data/processed/lexspec_100.jsonl"
GOLD="${PROJECT_ROOT}/data/processed/gold_triplets_100.jsonl"
OUTPUT_DIR="${PROJECT_ROOT}/outputs"
VENV_PATH="${VENV_PATH:-${PROJECT_ROOT}/.venv}"
LOG_DIR="${OUTPUT_DIR}/logs"
TIMING_FILE="${LOG_DIR}/pipeline_timing.log"

# 步骤跳过标志
SKIP_01=false; SKIP_02=false; SKIP_03=false
SKIP_04=false; SKIP_05=false; SKIP_06=false; SKIP_07=false

# 各步骤结果追踪
declare -A STEP_RESULTS
declare -A STEP_TIMES
PIPELINE_START=""
PIPELINE_END=""

# ---------------------------------------------------------------------------
# 命令行参数解析
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)       CONFIG="$2"; shift 2 ;;
        --constraints)  CONSTRAINTS="$2"; shift 2 ;;
        --output-dir)   OUTPUT_DIR="$2"; LOG_DIR="${OUTPUT_DIR}/logs"
                        TIMING_FILE="${LOG_DIR}/pipeline_timing.log"; shift 2 ;;
        --venv)         VENV_PATH="$2"; shift 2 ;;
        --skip-01) SKIP_01=true; shift ;;
        --skip-02) SKIP_02=true; shift ;;
        --skip-03) SKIP_03=true; shift ;;
        --skip-04) SKIP_04=true; shift ;;
        --skip-05) SKIP_05=true; shift ;;
        --skip-06) SKIP_06=true; shift ;;
        --skip-07) SKIP_07=true; shift ;;
        --help|-h)
            echo "用法: $0 [--config PATH] [--output-dir PATH] [--venv PATH]"
            echo "          [--skip-01] ... [--skip-07]"
            echo ""
            echo "按顺序执行 LexSpec 全部实验步骤。"
            exit 0 ;;
        *)  echo "未知选项: $1" >&2; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# 激活虚拟环境
# ---------------------------------------------------------------------------
activate_venv() {
    if [[ -f "${VENV_PATH}/bin/activate" ]]; then
        echo -e "${BLUE}[虚拟环境]${NC} 激活: ${VENV_PATH}"
        source "${VENV_PATH}/bin/activate"
    elif [[ -n "${VIRTUAL_ENV:-}" ]]; then
        echo -e "${GREEN}[虚拟环境]${NC} 已激活: ${VIRTUAL_ENV}"
    else
        echo -e "${YELLOW}[虚拟环境]${NC} 未找到 ${VENV_PATH} — 使用系统 Python"
    fi
}

# ---------------------------------------------------------------------------
# 前置检查
# ---------------------------------------------------------------------------
check_prerequisites() {
    echo ""
    echo -e "${BOLD}=============================================${NC}"
    echo -e "${BOLD}  LexSpec 流水线 — 前置检查${NC}"
    echo -e "${BOLD}=============================================${NC}"
    echo ""

    local PYTHON_BIN
    PYTHON_BIN="$(command -v python3 || command -v python || echo "")"
    if [[ -z "${PYTHON_BIN}" ]]; then
        echo -e "${RED}[失败]${NC} 未找到 Python 3。"
        return 1
    fi
    local PY_VERSION
    PY_VERSION="$("${PYTHON_BIN}" --version 2>&1 | awk '{print $2}')"
    echo -e "  Python:  ${PY_VERSION}  (${PYTHON_BIN})"
    local MAJOR MINOR
    MAJOR="$(echo "${PY_VERSION}" | cut -d. -f1)"
    MINOR="$(echo "${PY_VERSION}" | cut -d. -f2)"
    if [[ "${MAJOR}" -lt 3 ]] || { [[ "${MAJOR}" -eq 3 && "${MINOR}" -lt 10 ]]; }; then
        echo -e "${RED}[失败]${NC} Python >= 3.10 需要。当前版本 ${PY_VERSION}。"
        return 1
    fi
    echo -e "           ${GREEN}通过${NC} (>= 3.10)"

    echo ""
    echo "  检查必需依赖包..."
    local REQUIRED_PKGS=("stanza" "openai" "pydantic" "yaml" "numpy" "scipy" "tqdm")
    local ALL_OK=true
    for pkg in "${REQUIRED_PKGS[@]}"; do
        if "${PYTHON_BIN}" -c "import ${pkg}" 2>/dev/null; then
            echo -e "    ${pkg}: ${GREEN}通过${NC}"
        else
            echo -e "    ${pkg}: ${RED}缺失${NC}"
            ALL_OK=false
        fi
    done
    if [[ "${ALL_OK}" == "false" ]]; then
        echo ""
        echo -e "${YELLOW}[警告]${NC} 部分依赖包缺失。安装命令: pip install -r requirements.txt"
    fi

    echo ""
    echo "  检查数据文件..."
    local CUAD_FILE="${PROJECT_ROOT}/data/raw/CUAD_v1/CUAD_v1.json"
    if [[ -f "${CUAD_FILE}" ]]; then
        echo -e "    CUAD_v1.json: ${GREEN}存在${NC}"
    else
        echo -e "    CUAD_v1.json: ${YELLOW}缺失${NC} (步骤 01 需要)"
    fi

    for cfg in "model.yaml" "constraints.yaml" "prompts.yaml"; do
        local cfg_path="${PROJECT_ROOT}/configs/${cfg}"
        if [[ -f "${cfg_path}" ]]; then
            echo -e "    ${cfg}: ${GREEN}存在${NC}"
        else
            echo -e "    ${cfg}: ${YELLOW}缺失${NC}"
        fi
    done

    echo ""
    return 0
}

# ---------------------------------------------------------------------------
# 时间戳
# ---------------------------------------------------------------------------
timestamp() { date '+%Y-%m-%d %H:%M:%S'; }

# ---------------------------------------------------------------------------
# 执行单个步骤并记录结果
# ---------------------------------------------------------------------------
run_step() {
    local STEP_NUM="$1"
    local STEP_NAME="$2"
    shift 2
    local -a CMD_ARGS=("$@")
    local SCRIPT_PATH="${SCRIPT_DIR}/step_${STEP_NUM}_${STEP_NAME}.py"

    echo ""
    echo -e "${BOLD}============================================================${NC}"
    echo -e "${BOLD}  步骤 ${STEP_NUM}: ${STEP_NAME}${NC}"
    echo -e "${BOLD}  开始: $(timestamp)${NC}"
    echo -e "${BOLD}============================================================${NC}"
    echo ""

    local STEP_START
    STEP_START="$(date +%s)"
    local EXIT_CODE=0

    cd "${PROJECT_ROOT}" || {
        echo -e "${RED}[错误]${NC} 无法进入 ${PROJECT_ROOT}" >&2
        STEP_RESULTS["${STEP_NUM}"]="失败"
        STEP_TIMES["${STEP_NUM}"]="0"
        return 1
    }

    set +e
    if ((${#CMD_ARGS[@]})); then
        python3 "${SCRIPT_PATH}" "${CMD_ARGS[@]}"
    else
        python3 "${SCRIPT_PATH}"
    fi
    EXIT_CODE=$?
    set -e

    local STEP_END
    STEP_END="$(date +%s)"
    local ELAPSED=$((STEP_END - STEP_START))
    local H=$((ELAPSED / 3600))
    local M=$(((ELAPSED % 3600) / 60))
    local S=$((ELAPSED % 60))
    local TIME_STR
    TIME_STR="$(printf '%d:%02d:%02d' ${H} ${M} ${S})"

    echo ""
    if [[ ${EXIT_CODE} -eq 0 ]]; then
        echo -e "${GREEN}[通过]${NC} 步骤 ${STEP_NUM} ${STEP_NAME} 完成，耗时 ${TIME_STR}"
        STEP_RESULTS["${STEP_NUM}"]="通过"
    else
        echo -e "${RED}[失败]${NC} 步骤 ${STEP_NUM} ${STEP_NAME} 失败 (退出码 ${EXIT_CODE})，耗时 ${TIME_STR}"
        STEP_RESULTS["${STEP_NUM}"]="失败"
    fi

    STEP_TIMES["${STEP_NUM}"]="${TIME_STR}"
    echo "  ${STEP_NUM} ${STEP_NAME} ${STEP_RESULTS[${STEP_NUM}]} ${TIME_STR}" >> "${TIMING_FILE}"
    return ${EXIT_CODE}
}

run_step_02_annotation() {
    local STEP_NUM="02"
    local STEP_NAME="annotate_gold"
    local SCRIPT_PATH="${SCRIPT_DIR}/step_${STEP_NUM}_${STEP_NAME}.py"
    local -a SUBCOMMANDS=(
        "annotate --model gemma"
        "annotate --model qwen"
        "review --reviewer qwen --source gemma"
        "review --reviewer gemma --source qwen"
        "merge"
    )
    local sub
    local failed=false

    for sub in "${SUBCOMMANDS[@]}"; do
        # shellcheck disable=SC2206
        local -a SUB_ARGS=(${sub})
        echo ""
        echo -e "${BOLD}  步骤 02 子命令: ${sub}${NC}"
        set +e
        python3 "${SCRIPT_PATH}" \
            --config "${CONFIG}" \
            --prompts "${PROJECT_ROOT}/configs/prompts.yaml" \
            "${SUB_ARGS[@]}"
        local sub_exit=$?
        set -e
        if [[ ${sub_exit} -ne 0 ]]; then
            echo -e "${RED}[失败]${NC} 步骤 02 子命令失败: ${sub} (退出码 ${sub_exit})"
            failed=true
        fi
    done

    if [[ "${failed}" == "true" ]]; then
        STEP_RESULTS["02"]="失败"
        return 1
    fi
    STEP_RESULTS["02"]="通过"
    return 0
}

# ---------------------------------------------------------------------------
# 主流水线
# ---------------------------------------------------------------------------
main() {
    mkdir -p "${OUTPUT_DIR}/predictions"
    mkdir -p "${OUTPUT_DIR}/metrics"
    mkdir -p "${OUTPUT_DIR}/error_cases"
    mkdir -p "${LOG_DIR}"

    echo "# LexSpec 流水线计时日志 — $(timestamp)" > "${TIMING_FILE}"
    echo "# 项目根目录: ${PROJECT_ROOT}" >> "${TIMING_FILE}"
    echo "" >> "${TIMING_FILE}"

    activate_venv

    if ! check_prerequisites; then
        echo -e "${RED}[中止]${NC} 前置检查失败。" >&2
        exit 1
    fi

    # 若已有 curated 100 条金标/测试集，默认跳过 01/02，避免覆盖评测数据
    if [[ -f "${GOLD}" && -f "${TESTSET}" ]]; then
        if [[ "${SKIP_01}" != "true" ]]; then
            echo -e "${BLUE}[信息]${NC} 检测到 curated 评测集，自动跳过步骤 01 (build_corpus)"
            SKIP_01=true
        fi
        if [[ "${SKIP_02}" != "true" ]]; then
            echo -e "${BLUE}[信息]${NC} 使用 ${GOLD}，自动跳过步骤 02 (annotate_gold)"
            SKIP_02=true
        fi
    else
        echo -e "${YELLOW}[警告]${NC} 未找到 ${GOLD} 或 ${TESTSET}。"
        echo -e "         可运行: python3 scripts/build_gold_100_from_500.py"
    fi

    PIPELINE_START="$(date +%s)"

    echo ""
    echo -e "${BOLD}${BLUE}============================================================${NC}"
    echo -e "${BOLD}${BLUE}  LexSpec 实验流水线${NC}"
    echo -e "${BOLD}${BLUE}  开始: $(timestamp)${NC}"
    echo -e "${BOLD}${BLUE}============================================================${NC}"

    # ---- 步骤 01: 构建语料 ----
    if [[ "${SKIP_01}" == "true" ]]; then
        echo -e "${YELLOW}[跳过]${NC} 步骤 01: build_corpus"
        STEP_RESULTS["01"]="跳过"; STEP_TIMES["01"]="0:00:00"
    else
        run_step "01" "build_corpus" \
            --config "${CONFIG}" \
            --constraints "${CONSTRAINTS}" \
            --cuad-path "${PROJECT_ROOT}/data/raw/CUAD_v1/CUAD_v1.json" \
            --output "${PROJECT_ROOT}/data/processed/lexspec_100.jsonl" || true
    fi

    # ---- 步骤 02: 标注金标 (四阶段) ----
    if [[ "${SKIP_02}" == "true" ]]; then
        echo -e "${YELLOW}[跳过]${NC} 步骤 02: annotate_gold"
        STEP_RESULTS["02"]="跳过"; STEP_TIMES["02"]="0:00:00"
    else
        run_step_02_annotation || true
    fi

    # ---- 步骤 03: 基线抽取 ----
    if [[ "${SKIP_03}" == "true" ]]; then
        echo -e "${YELLOW}[跳过]${NC} 步骤 03: extract_baseline"
        STEP_RESULTS["03"]="跳过"; STEP_TIMES["03"]="0:00:00"
    else
        run_step "03" "extract_baseline" \
            --config "${CONFIG}" \
            --output-dir "${OUTPUT_DIR}" \
            --testset "${TESTSET}" || true
    fi

    # ---- 步骤 04: Dep 抽取 ----
    if [[ "${SKIP_04}" == "true" ]]; then
        echo -e "${YELLOW}[跳过]${NC} 步骤 04: extract_dep"
        STEP_RESULTS["04"]="跳过"; STEP_TIMES["04"]="0:00:00"
    else
        run_step "04" "extract_dep" \
            --config "${CONFIG}" \
            --output-dir "${OUTPUT_DIR}" \
            --testset "${TESTSET}" || true
    fi

    # ---- 步骤 05: Reflexion 抽取 ----
    if [[ "${SKIP_05}" == "true" ]]; then
        echo -e "${YELLOW}[跳过]${NC} 步骤 05: extract_reflexion"
        STEP_RESULTS["05"]="跳过"; STEP_TIMES["05"]="0:00:00"
    else
        run_step "05" "extract_reflexion" \
            --config "${CONFIG}" \
            --output-dir "${OUTPUT_DIR}" \
            --testset "${TESTSET}" || true
    fi

    # ---- 步骤 06: 评估 ----
    if [[ "${SKIP_06}" == "true" ]]; then
        echo -e "${YELLOW}[跳过]${NC} 步骤 06: evaluate"
        STEP_RESULTS["06"]="跳过"; STEP_TIMES["06"]="0:00:00"
    else
        run_step "06" "evaluate" \
            --config "${CONFIG}" \
            --constraints "${CONSTRAINTS}" \
            --output-dir "${OUTPUT_DIR}" \
            --predictions-dir "${OUTPUT_DIR}/predictions" \
            --gold "${GOLD}" \
            --testset "${TESTSET}" || true
    fi

    # ---- 步骤 07: 错误分析 ----
    if [[ "${SKIP_07}" == "true" ]]; then
        echo -e "${YELLOW}[跳过]${NC} 步骤 07: analyze_errors"
        STEP_RESULTS["07"]="跳过"; STEP_TIMES["07"]="0:00:00"
    else
        run_step "07" "analyze_errors" \
            --config "${CONFIG}" \
            --constraints "${CONSTRAINTS}" \
            --output-dir "${OUTPUT_DIR}" \
            --predictions-dir "${OUTPUT_DIR}/predictions" \
            --gold "${GOLD}" \
            --testset "${TESTSET}" || true
    fi

    PIPELINE_END="$(date +%s)"
    local TOTAL_ELAPSED=$((PIPELINE_END - PIPELINE_START))
    local TH=$((TOTAL_ELAPSED / 3600))
    local TM=$(((TOTAL_ELAPSED % 3600) / 60))
    local TS=$((TOTAL_ELAPSED % 60))
    local TOTAL_STR
    TOTAL_STR="$(printf '%d:%02d:%02d' ${TH} ${TM} ${TS})"

    # -----------------------------------------------------------------------
    # 流水线汇总
    # -----------------------------------------------------------------------
    echo ""
    echo -e "${BOLD}============================================================${NC}"
    echo -e "${BOLD}  LexSpec 流水线汇总${NC}"
    echo -e "${BOLD}  完成时间: $(timestamp)${NC}"
    echo -e "${BOLD}============================================================${NC}"
    echo ""
    echo -e "${BOLD}步骤  描述                    结果      耗时${NC}"
    echo -e "${BOLD}----  ----------------------  --------  ---------${NC}"

    local STEP_NAMES=(
        "01" "build_corpus"
        "02" "annotate_gold"
        "03" "extract_baseline"
        "04" "extract_dep"
        "05" "extract_reflexion"
        "06" "evaluate"
        "07" "analyze_errors"
    )

    local ALL_PASS=true
    local HAS_FAILURES=false

    for ((i=0; i<${#STEP_NAMES[@]}; i+=2)); do
        local NUM="${STEP_NAMES[$i]}"
        local DESC="${STEP_NAMES[$((i+1))]}"
        local RESULT="${STEP_RESULTS[${NUM}]:-未知}"
        local TIME="${STEP_TIMES[${NUM}]:-0:00:00}"

        local STATUS_COLOUR=""
        case "${RESULT}" in
            通过) STATUS_COLOUR="${GREEN}" ;;
            失败) STATUS_COLOUR="${RED}"; HAS_FAILURES=true; ALL_PASS=false ;;
            跳过) STATUS_COLOUR="${YELLOW}" ;;
            *)    STATUS_COLOUR="${RED}" ;;
        esac

        printf "%-5s %-25s ${STATUS_COLOUR}%-8s${NC}  %s\n" \
            "${NUM}" "${DESC}" "${RESULT}" "${TIME}"
    done

    echo ""
    echo "流水线总耗时: ${TOTAL_STR}"
    echo "输出目录:     ${OUTPUT_DIR}"
    echo ""

    if [[ "${ALL_PASS}" == "true" ]]; then
        echo -e "${GREEN}${BOLD}全部步骤通过。${NC}"
        exit 0
    elif [[ "${HAS_FAILURES}" == "true" ]]; then
        echo -e "${RED}${BOLD}部分步骤失败。检查日志: ${LOG_DIR}${NC}"
        exit 1
    else
        echo -e "${YELLOW}${BOLD}流水线完成（部分步骤跳过）。${NC}"
        exit 0
    fi
}

main "$@"
