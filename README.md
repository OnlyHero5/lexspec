# LexSpec

> **基于依存句法与论元结构约束的法律合同要素抽取智能体研究**
>
> *Dependency-Parse-Constrained Legal Contract Element Extraction with Reflexion Agents*

---

## 目录

1. [项目概述](#1-项目概述)
2. [系统架构](#2-系统架构)
3. [目录结构](#3-目录结构)
4. [环境配置](#4-环境配置)
5. [使用方法](#5-使用方法)
6. [配置说明](#6-配置说明)
7. [核心模块详解](#7-核心模块详解)
8. [评估体系](#8-评估体系)
9. [实验流水线](#9-实验流水线)
10. [依赖项](#10-依赖项)
11. [论文引用](#11-论文引用)
12. [许可证](#12-许可证)

---

## 1. 项目概述

### 1.1 研究问题

法律合同文本中存在大量复杂的谓词-论元结构，其主语、宾语和条件从句的边界识别是自然语言处理中的难点。法律英语特有的被动语态、长距离依存、情态动词（shall/may/must）与否定词的交互，给大语言模型的信息抽取带来了系统性挑战。

**LexSpec** 提出了一种将**依存句法约束**与**大语言模型自我修正（Reflexion）**相结合的方法，在法律合同要素抽取任务上提升大语言模型的结构化预测能力。

### 1.2 核心思路

```
                        法律合同条款
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              v              v              v
        ┌──────────┐  ┌──────────┐  ┌──────────┐
        │ 大语言模型 │  │ Stanza   │  │ 约束规则  │
        │ 零样本抽取 │  │ UD 解析  │  │ 知识库    │
        └─────┬────┘  └─────┬────┘  └─────┬────┘
              │              │              │
              └──────────────┼──────────────┘
                             │
                             v
                  ┌─────────────────────┐
                  │   UD 约束校验器      │
                  │  (7 步校验算法)      │
                  └─────────┬───────────┘
                            │
              ┌─────────────┼─────────────┐
              │             │             │
           VALID       CORRECTED    REFLEXION_REQUIRED
              │             │             │
              v             v             v
           直接使用     自动修正        ┌──────────┐
           抽取结果     后使用         │ Reflexion │
                                       │ 自我修正  │
                                       └─────┬────┘
                                             │
                                             v
                                       修正后重新校验
```

### 1.3 三大技术支柱

| 技术组件 | 作用 | 实现 |
|----------|------|------|
| **依存句法解析** | 提供句法基础：识别主语(nsubj)、宾语(obj)、被动(nsubj:pass)、否定(neg)、从句边界(advcl+mark) | Stanza (StanfordNLP) |
| **大语言模型抽取** | 零样本结构化信息抽取：从合同条款中抽取 (主体, 动作, 条件) 三元组 | Qwen3.5 9B (llama.cpp) |
| **Reflexion 自我修正** | 将句法校验发现的错误以结构化提示形式反馈给大语言模型，触发重新推理和修正 | ReflexionGenerator |

### 1.4 模型隔离策略

```
┌─────────────────────────────────────────────────────────┐
│                    模型隔离边界                           │
│                                                         │
│  ┌──────────────────────────┐  ┌──────────────────────┐ │
│  │ 标注模型（仅用于金标生成） │  │ 实验模型（抽取与修正） │ │
│  │                          │  │                      │ │
│  │  Qwen3.6 27B  (主标注)   │  │  Qwen3.5 9B          │ │
│  │  Gemma4 31B   (副标注)   │  │                      │ │
│  │                          │  │  - 基线抽取           │ │
│  │  - 独立标注              │  │  - Reflexion 修正     │ │
│  │  - 交叉审查              │  │  - 绝不见到标注数据    │ │
│  │  - 共识合并              │  │                      │ │
│  │                          │  │                      │ │
│  │  绝不参与实验阶段         │  │                      │ │
│  └──────────────────────────┘  └──────────────────────┘ │
│                                                         │
│  标注模型的输出仅用于构造金标准测试集，绝不会泄露到        │
│  实验模型的训练、提示词或评估中                            │
└─────────────────────────────────────────────────────────┘
```

---

## 2. 系统架构

### 2.1 总体架构图

```
┌──────────────────────────────────────────────────────────────────────┐
│                          LexSpec 系统架构                              │
│                                                                      │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐   │
│  │   configs/       │  │   experiments/   │  │   src/            │   │
│  │                  │  │                  │  │                  │   │
│  │  model.yaml      │  │  run_pipeline.sh │  │  extraction/     │   │
│  │  prompts.yaml    │  │  step_01 ... 07  │  │  linguistic/     │   │
│  │  constraints.yaml│  │                  │  │  correction/     │   │
│  └────────┬─────────┘  └────────┬─────────┘  │  annotation/     │   │
│           │                     │             │  evaluation/     │   │
│           │                     │             │  utils/          │   │
│           │                     │             └────────┬─────────┘   │
│           │                     │                      │             │
│           v                     v                      v             │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                      数据流向                                  │   │
│  │                                                              │   │
│  │  data/raw/CUAD_v1 ──► step_01 ──► data/processed/lexspec_100 │   │
│  │                                        │                     │   │
│  │                                        v                     │   │
│  │                              step_02 (双模型标注)              │   │
│  │                                        │                     │   │
│  │                                        v                     │   │
│  │                              data/processed/gold_triplets.jsonl │
│  │                                        │                     │   │
│  │                         ┌──────────────┼──────────────┐      │   │
│  │                         v              v              v      │   │
│  │                     step_03        step_04        step_05    │   │
│  │                    (Baseline)    (Ours-Dep)  (Ours-Reflex)   │   │
│  │                         │              │              │      │   │
│  │                         └──────────────┼──────────────┘      │   │
│  │                                        v                     │   │
│  │                              step_06 (双轨评估)               │   │
│  │                              step_07 (错误分析)               │   │
│  │                                        │                     │   │
│  │                                        v                     │   │
│  │                              outputs/ (结果报告)              │   │
│  └──────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────┘
```

### 2.2 核心模块依赖关系

```
┌─────────────────────────────────────────────────────────────────┐
│                         模块依赖图                                │
│                                                                 │
│   experiments/   ───────── 调用 ────────►  src/                  │
│       │                                       │                 │
│       │                              ┌────────┼────────┐        │
│       │                              │        │        │        │
│       │                         extraction  linguistic  utils   │
│       │                              │        │        │        │
│       │                              │   ┌────┼────┐   │        │
│       │                              │   │    │    │   │        │
│       │                              │ stanza passive cond  │   │
│       │                              │ parser detect  extr  │   │
│       │                              │   │    │    │        │   │
│       │                              │   └────┼────┘        │   │
│       │                              │        │             │   │
│       │                              │   validator         │   │
│       │                              │        │             │   │
│       │                              │   ┌────┼────┐        │   │
│       │                              │   │    │    │        │   │
│       │                         correction annotation eval  │   │
│       │                         (Reflexion) (双模型)  (F1)  │   │
│       │                              │        │        │    │   │
│       └──────────────────────────────┴────────┴────────┘    │   │
│                                                                 │
│   configs/ ──── 被所有模块读取 ──────────────────────────────    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.3 数据模型层次

```
┌────────────────────────────────────────────────────────────┐
│                     Pydantic 数据模型层次                     │
│                                                            │
│  ┌─────────────────────┐                                   │
│  │    LegalTriplet     │  ◄── 核心抽取输出                   │
│  │  ┌───────────────┐  │                                   │
│  │  │ Subject       │  │  text: str                        │
│  │  │               │  │  role: LegalRole (枚举)            │
│  │  ├───────────────┤  │       ├── obligor                 │
│  │  │ Action        │  │       ├── right_holder            │
│  │  │               │  │       ├── prohibited_party        │
│  │  │               │  │       ├── indemnifying_party      │
│  │  │               │  │       └── other                   │
│  │  ├───────────────┤  │                                   │
│  │  │ Condition     │  │  text: str                        │
│  │  │               │  │  type: ConditionType (枚举)        │
│  │  └───────────────┘  │       ├── trigger                 │
│  └─────────────────────┘       ├── temporal                │
│                                ├── exception               │
│  ┌─────────────────────┐       └── none                    │
│  │  DependencyTree     │  ◄── UD 句法树封装                  │
│  │  ┌───────────────┐  │                                   │
│  │  │ Token[]       │  │  index, text, lemma               │
│  │  │               │  │  upos, xpos, deprel               │
│  │  │               │  │  head, feats                      │
│  │  └───────────────┘  │                                   │
│  └─────────────────────┘                                   │
│                                                            │
│  ┌─────────────────────┐                                   │
│  │  ValidationResult   │  ◄── 校验器输出                     │
│  │                     │  status: VALID|CORRECTED|REFLEXION │
│  │                     │  original_prediction: LegalTriplet │
│  │                     │  corrected_prediction: LegalTriplet│
│  │                     │  linguistic_evidence               │
│  │                     │  corrections: FieldCorrection[]    │
│  │                     │  feedback: str                     │
│  └─────────────────────┘                                   │
│                                                            │
│  ┌─────────────────────┐                                   │
│  │  ErrorCase          │  ◄── 错误分析输出                   │
│  │                     │  primary_category: ErrorCategory   │
│  │                     │  secondary_category: FieldErrorType│
│  │                     │  linguistic_explanation (双语)     │
│  └─────────────────────┘                                   │
└────────────────────────────────────────────────────────────┘
```

---

## 3. 目录结构

```
计算语言学/                              # 项目根目录
│
├── README.md                            # 项目说明文档（本文件）
├── pyproject.toml                        # Python 项目元数据 (setuptools)
├── requirements.txt                      # Python 依赖清单（带详细注释）
│
├── configs/                              # 配置文件（版本控制，确保可复现）
│   ├── model.yaml                        #   服务器连接、模型标识、生成参数、
│   │                                     #   Stanza 流水线、Reflexion 参数
│   ├── prompts.yaml                      #   全部提示词模板：标注 (annotation)、
│   │                                     #   抽取 (extraction.baseline)、
│   │                                     #   Reflexion 修正 (reflexion)
│   └── constraints.yaml                  #   UD 依存关系清单、条件从句标记词、
│                                         #   情态规则、校验阈值、F1 权重、
│                                         #   文本规范化、现象抽样阈值
│
├── src/                                  # 源代码
│   ├── __init__.py                       #   包文档与版本信息 (v1.0.0)
│   │
│   ├── extraction/                       # 抽取模块 — 大语言模型调用与结果解析
│   │   ├── __init__.py                   #     公开 API 导出
│   │   ├── schema/                       #     Pydantic v2 数据模型包
│   │   │                                 #     LegalTriplet, Subject, Action,
│   │   │                                 #     Condition, DependencyTree, Token,
│   │   │                                 #     ValidationResult, ErrorCase 等
│   │   ├── client.py                     #     LLM 客户端 (OpenAI SDK 封装)
│   │   │                                 #     指数退避重试、超时控制、
│   │   │                                 #     json_object / json_schema 模式
│   │   └── extractor.py                  #     LegalTripletExtractor
│   │                                     #     提示词加载→格式化→大语言模型调用→
│   │                                     #     多策略 JSON 解析→Pydantic 校验
│   │
│   ├── linguistic/                       # 语言学模块 — UD 依存句法分析与校验
│   │   ├── __init__.py                   #     公开 API 导出
│   │   ├── stanza_parser.py              #     Stanza 流水线封装（单例模式）
│   │   │                                 #     parse() → DependencyTree
│   │   ├── ud_features.py                #     UD 特征提取 (1,000+ 行)
│   │   │                                 #     find_root_predicate, find_nsubj,
│   │   │                                 #     find_obj, find_nsubj_pass,
│   │   │                                 #     find_obl_agent, get_dependency_path,
│   │   │                                 #     compute_mean_dependency_distance 等
│   │   ├── passive_detector.py           #     被动语态检测与论元恢复
│   │   │                                 #     is_passive(), restore_passive_args()
│   │   ├── condition_extractor.py        #     条件从句边界提取与分类
│   │   │                                 #     extract(), extract_all(),
│   │   │                                 #     compute_condition_overlap()
│   │   ├── polarity_detector.py          #     情态/否定检测 → 法律角色分类
│   │   │                                 #     detect(), detect_modality(),
│   │   │                                 #     detect_role_with_voice()
│   │   └── validator.py                  #     ★ 核心算法 — 7 步约束校验器
│   │                                     #     ConstraintValidator.validate()
│   │
│   ├── correction/                       # 修正模块 — Reflexion 自我修正
│   │   ├── __init__.py                   #     公开 API 导出
│   │   ├── reflexion.py                  #     ReflexionGenerator
│   │   ├── reflexion_error_mapper.py     #     校验结果 → Reflexion 错误提示键
│   │   ├── prompt_loader.py              #     提示词加载 (re-export)
│   │   └── response_parser.py            #     Reflexion 响应解析
│   │
│   ├── annotation/                       # 标注模块 — 双模型金标构建
│   │   ├── __init__.py                   #     公开 API 导出
│   │   ├── llm_annotator.py              #     LLMAnnotator — 单模型标注器
│   │   │                                 #     annotate(), annotate_batch()
│   │   ├── reviewer.py                   #     CrossModelReviewer — 交叉审查
│   │   ├── consensus.py                  #     字段级投票共识
│   │   │                                 #     field_level_consensus(),
│   │   │                                 #     build_gold_from_consensus(),
│   │   │                                 #     generate_annotation_stats()
│   │   └── disagreement_logger.py        #     标注分歧记录与持久化
│   │
│   ├── evaluation/                       # 评估模块 — 双轨评估与统计分析
│   │   ├── __init__.py                   #     公开 API 导出
│   │   ├── triplet_f1.py                 #     加权 F1 计算（主要指标）
│   │   │                                 #     compute_triplet_f1(),
│   │   │                                 #     compute_per_sample_f1()
│   │   ├── normalization.py              #     文本规范化
│   │   │                                 #     冠词去除、数字规范化、方别名映射
│   │   ├── linguistic_metrics.py         #     语言学辅助指标 (4 项)
│   │   │                                 #     依存路径合法性、被动恢复准确率、
│   │   │                                 #     条件边界 IoU、修正成功率
│   │   ├── significance.py               #     统计显著性检验
│   │   │                                 #     配对自举 (bootstrap)、Wilcoxon、
│   │   │                                 #     分层显著性检验
│   │   └── error_analyzer.py             #     错误分类与双语解释生成
│   │                                     #     classify_errors(),
│   │                                     #     error_distribution_report(),
│   │                                     #     save_error_cases()
│   │
│   └── utils/                            # 工具模块
│       ├── __init__.py                   #     公开 API 导出
│       ├── config.py                      #     模型配置加载 (model.yaml)
│       ├── constraints.py                 #     约束配置加载 (constraints.yaml)
│       ├── io.py                         #     JSONL 读写、Pydantic 序列化
│       ├── logging.py                    #     集中式日志配置
│       └── prompt_loader.py             #     统一提示词加载 (prompts.yaml)
│
├── experiments/                          # 实验脚本 (7 步流水线)
│   ├── run_pipeline.sh                   #   ★ 一键运行完整流水线
│   ├── step_01_build_corpus.py           #   步骤 01: 从 CUAD 构建测试语料
│   ├── step_02_annotate_gold.py          #   步骤 02: 双模型分阶段金标标注
│   ├── step_03_extract_baseline.py       #   步骤 03: 基线 — 纯大语言模型抽取
│   ├── step_04_extract_dep.py            #   步骤 04: Ours-Dep — +UD 约束校验
│   ├── step_05_extract_reflexion.py      #   步骤 05: Ours-Reflexion — +自我修正
│   ├── step_06_evaluate.py               #   步骤 06: 双轨评估 + 显著性检验
│   └── step_07_analyze_errors.py         #   步骤 07: 语言学错误分类分析
│
├── data/                                 # 数据目录
│   ├── raw/                              #   原始语料
│   │   └── CUAD_v1/                      #     CUAD v1 合同数据集
│   ├── processed/                        #   预处理后的数据
│   │   └── lexspec_100.jsonl             #     LexSpec-100 测试集
│   └── annotations/                      #   标注中间文件 (运行时生成)
│       └── gemma_annotations.jsonl       #     Gemma 模型标注结果
│
├── data/processed/                       #   预处理与金标输出
│   ├── lexspec_100.jsonl                 #     LexSpec-100 测试集
│   └── gold_triplets.jsonl               #     共识合并后的金标三元组
│
├── outputs/                              # 实验输出（运行时生成）
│   ├── predictions/                      #   三元组预测结果
│   ├── metrics/                          #   评估指标
│   ├── error_cases/                      #   错误案例分析
│   └── logs/                             #   运行日志
│
└── papers/                               # 参考文献 (10 篇)
    ├── CUAD_2021_Hendrycks.pdf           #   CUAD 合同理解数据集
    ├── ContractNLI_2021_Koreeda.pdf       #   ContractNLI 自然语言推理
    ├── LegalBench_2023_Guha.pdf           #   LegalBench 法律推理基准
    ├── LexGLUE_2022_Chalkidis.pdf         #   LexGLUE 法律 NLP 基准
    ├── LLM_in_Law_Survey_2023_Lai.pdf     #   大语言模型在法律中的应用综述
    ├── Legal_AI_Survey_2025_Hou.pdf       #   法律人工智能综述 (2025)
    ├── Universal_Dependencies_2021_deMarneffe.pdf  # UD 依存语法
    ├── UD_v2_2020_Nivre.pdf              #   UD v2 指南
    ├── PropBank_2005_Palmer.pdf           #   PropBank 语义角色标注
    ├── Reflexion_2023_Shinn.pdf           #   Reflexion 自我修正
    └── Self-Refine_2023_Madaan.pdf        #   Self-Refine 迭代修正
```

---

## 4. 环境配置

### 4.1 前置条件

| 条件 | 说明 |
|------|------|
| Python | ≥ 3.10 |
| 内存 | ≥ 16 GB（Stanza 模型 ~500MB + 数据处理） |
| GPU | 推荐（llama.cpp 支持 CPU 推理） |
| llama.cpp 服务器 | 运行在 `http://10.0.16.254:8080/v1` |

### 4.2 安装步骤

```bash
# 1. 进入项目目录
cd /home/psx/homework/计算语言学

# 2. 创建并激活虚拟环境
python -m venv .venv
source .venv/bin/activate

# 3. 安装 Python 依赖
pip install -r requirements.txt

# 4. 预下载 Stanza 英文模型（首次运行时也会自动下载）
python -c "import stanza; stanza.download('en')"
```

### 4.3 启动 llama.cpp 服务器

```bash
llama-server \
  -m /path/to/qwen3.5-9b.gguf \
  -m /path/to/qwen3.6-27b.gguf \
  -m /path/to/gemma-4-31B-it-Q8_0.gguf \
  --port 8080 \
  --n-gpu-layers 99
```

### 4.4 验证安装

```bash
python -c "
from src.extraction.client import ClientConfig, LLMClient
from src.linguistic.stanza_parser import StanzaParser

# 测试 Stanza
parser = StanzaParser()
tree = parser.parse('Seller shall deliver the Goods within 30 days.')
print(f'Stanza OK — root: {tree.get_token(tree.root_index).lemma}')

# 测试 LLM 客户端配置
cfg = ClientConfig(base_url='http://10.0.16.254:8080/v1', model='qwen3.5-9b')
print(f'LLM client configured: {cfg.model} @ {cfg.base_url}')
print('Setup OK')
"
```

---

## 5. 使用方法

### 5.1 一键运行完整流水线

```bash
chmod +x experiments/run_pipeline.sh
./experiments/run_pipeline.sh
```

流水线按顺序执行全部 7 个步骤。支持选择性跳过和自定义参数：

```bash
# 跳过已完成的步骤
./experiments/run_pipeline.sh --skip-01 --skip-02

# 自定义配置
./experiments/run_pipeline.sh \
  --config configs/model.yaml \
  --constraints configs/constraints.yaml \
  --output-dir outputs/
```

步骤 02 在流水线中自动执行完整的四阶段标注（Gemma 标注 → Qwen 标注 → 双向交叉审查 → merge）。

# 使用自定义虚拟环境
VENV_PATH=/path/to/venv ./experiments/run_pipeline.sh
```

### 5.2 分步运行

#### 步骤 01 — 构建测试语料

```bash
# 分层抽样模式（默认 100 条，保证各语言现象均衡覆盖）
python experiments/step_01_build_corpus.py \
  --cuad-path data/raw/CUAD_v1/CUAD_v1.json \
  --output data/processed/lexspec_100.jsonl

# 全量模式（使用全部 510 份合同的所有有效条款）
python experiments/step_01_build_corpus.py \
  --all \
  --output data/processed/lexspec_corpus.jsonl
```

#### 步骤 02 — 双模型金标标注

```bash
# 阶段 1: Gemma 独立标注
python experiments/step_02_annotate_gold.py annotate --model gemma

# 阶段 2: 切换到 Qwen，Qwen 标注 + Qwen 审查 Gemma
python experiments/step_02_annotate_gold.py annotate --model qwen
python experiments/step_02_annotate_gold.py review --reviewer qwen --source gemma

# 阶段 3: 切换回 Gemma，Gemma 审查 Qwen
python experiments/step_02_annotate_gold.py review --reviewer gemma --source qwen

# 阶段 4: 共识合并（无需大语言模型）
python experiments/step_02_annotate_gold.py merge
```

#### 步骤 03-05 — 三种抽取实验

```bash
# 基线: 纯大语言模型零样本抽取
python experiments/step_03_extract_baseline.py \
  --testset data/processed/lexspec_100.jsonl

# Ours-Dep: 大语言模型 + UD 约束校验自动修正
python experiments/step_04_extract_dep.py \
  --testset data/processed/lexspec_100.jsonl

# Ours-Reflexion: 大语言模型 + UD 约束 + Reflexion 自我修正
python experiments/step_05_extract_reflexion.py \
  --testset data/processed/lexspec_100.jsonl
```

#### 步骤 06-07 — 评估与分析

```bash
# 双轨评估（任务指标 + 语言学指标）+ 显著性检验
python experiments/step_06_evaluate.py

# 语言学错误分类与双语解释生成
python experiments/step_07_analyze_errors.py
```

### 5.3 编程接口示例

```python
# 端到端使用示例
from src.extraction.client import LLMClient, ClientConfig
from src.extraction.extractor import LegalTripletExtractor
from src.linguistic.stanza_parser import StanzaParser
from src.linguistic.validator import ConstraintValidator
from src.correction.reflexion import ReflexionGenerator

# 1. 初始化组件
client = LLMClient(ClientConfig(model="qwen3.5-9b"))
extractor = LegalTripletExtractor(client)
parser = StanzaParser()
validator = ConstraintValidator(parser=parser)

# 2. 抽取 + 校验
clause = "The Goods shall be delivered by Seller within 30 days."
triplet = extractor.extract(clause)
tree = parser.parse(clause)
result = validator.validate(triplet, clause, tree)

# 3. 根据校验结果处理
if result.status.value == "VALID":
    final = triplet
elif result.status.value == "CORRECTED":
    final = result.corrected_prediction
else:
    # REFLEXION_REQUIRED — 触发自我修正
    reflexion = ReflexionGenerator(client)
    final = reflexion.correct(clause, result) or triplet
```

---

## 6. 配置说明

所有实验参数均存储在版本控制的 YAML 配置文件中。**运行时参数必须从 YAML 加载；缺失或无效配置将立即抛出异常，禁止静默回退或硬编码默认值。**

- `configs/model.yaml` — 服务器、模型、Stanza、Reflexion 迭代参数
- `configs/constraints.yaml` — UD 约束、F1 权重、文本规范化、语料抽样、距离阈值
- `configs/prompts.yaml` — 标注、抽取、Reflexion 提示词（含 `error_hints.default`）

### 6.1 `configs/model.yaml` — 模型与服务器配置

```
model.yaml
├── server                         # llama.cpp 连接参数
│   ├── base_url                   #   http://10.0.16.254:8080/v1
│   ├── api_key                    #   "not-needed" (llama.cpp 无需认证)
│   ├── timeout                    #   1200 秒（大模型标注）
│   └── max_retries                #   3 次重试
│
├── models                         # 模型标识（按角色分组）
│   ├── experiment                 #   实验模型: Qwen3.5 9B (仅用于抽取)
│   └── annotation                 #   标注模型: Qwen3.6 27B + Gemma4 31B
│       ├── primary                #     主标注模型
│       └── secondary              #     副标注模型
│
├── generation                     # 生成参数
│   ├── temperature                #   0.0（贪心解码，确保可复现）
│   ├── max_tokens                 #   8192
│   ├── annotation_max_tokens      #   8192（标注专用 token 预算）
│   └── seed                       #   42
│
├── stanza                         # Stanza NLP 流水线
│   ├── lang                       #   "en" (英语)
│   ├── processors                 #   tokenize,mwt,pos,lemma,depparse
│   └── download_method            #   REUSE_RESOURCES
│
└── reflexion                      # Reflexion 自我修正参数
    ├── max_iterations             #   1（最多一轮修正，防过度修正）
    └── temperature                #   0.0
```

### 6.2 `configs/constraints.yaml` — 约束规则与校验配置

```
constraints.yaml
├── ud_relations                   # UD 依存关系清单 (10 种)
│   ├── active_subject             #   nsubj — 主动态主语
│   ├── active_object              #   obj — 主动态宾语
│   ├── passive_subject            #   nsubj:pass — 被动态受事主语
│   ├── passive_agent              #   obl:agent — 被动态施事
│   ├── advcl                      #   advcl — 状语从句
│   ├── mark                       #   mark — 从句引导词
│   ├── neg                        #   neg — 否定
│   ├── aux                        #   aux — 助动词
│   ├── aux_pass                   #   aux:pass — 被动助动词
│   └── acl_relcl                  #   acl:relcl — 关系从句
│
├── condition_markers              # 条件从句标记词分类
│   ├── trigger                    #   触发型: if, provided that, ...
│   ├── temporal                   #   时间型: when, upon, after, ...
│   └── exception                  #   例外型: unless, except, ...
│
├── modality_rules                 # 情态角色分类规则
│   ├── obligor                    #   shall/must/will + 肯定 → 义务方
│   ├── right_holder               #   may/can + 肯定 → 权利方
│   └── prohibited_party           #   shall/may/must + 否定 → 被禁止方
│
├── validation                     # 校验阈值
│   ├── condition_overlap          #   0.5 (条件从句 IoU 最低阈值)
│   ├── subject_match              #   0.8 (主语模糊匹配阈值)
│   ├── object_match               #   0.8 (宾语模糊匹配阈值)
│   ├── long_distance_tokens       #   3 (论元依存距离 — Reflexion/错误分析)
│   └── long_distance_mdd          #   6.0 (语料现象检测 MDD 阈值)
│
├── f1_weights                     # 加权 F1 各分量权重
│   ├── subject_text               #   0.35
│   ├── subject_role               #   0.10
│   ├── predicate                  #   0.20
│   ├── object                     #   0.20
│   └── condition                  #   0.15
│
├── normalization                  # 文本规范化规则
│   ├── remove_articles            #   去除冠词 (a/an/the)
│   ├── lemmatize                  #   词形还原 (评估管线)
│   ├── number_normalization       #   数字规范化 (thirty ↔ 30)
│   └── use_party_aliases          #   是否启用 party_alias_mappings
│
├── party_alias_mappings           # 方别称映射 (canonical → aliases[])
│
├── corpus_sampling                # 语料构建默认参数
│   ├── target_count_default       #   100
│   └── random_seed                #   42
│
└── phenomenon_thresholds          # 测试集分层抽样最低比例
    ├── passive                    #   0.20 → min 20/100
    ├── conditional                #   0.25
    ├── relative                   #   0.15
    ├── long_distance              #   0.15 (MDD > long_distance_mdd)
    └── negation                   #   0.15
```

### 6.3 `configs/prompts.yaml` — 提示词模板

```
prompts.yaml
├── annotation                     # 金标标注提示词
│   ├── system                    #   标注系统提示词（含详细规则和示例）
│   ├── user                      #   标注用户模板 {sentence}
│   └── review                    #   交叉审查提示词
│       ├── system                #     审查系统提示词
│       └── user                  #     审查用户模板
│
├── extraction                     # 抽取实验提示词
│   └── baseline                  #   基线零样本抽取
│       ├── system                #     抽取系统提示词
│       └── user                  #     抽取用户模板 {sentence}
│
└── reflexion                      # Reflexion 修正提示词
    ├──     feedback_template         #   反馈模板
    │                              #   {error_type}, {text}, {prediction},
    │                              #   {linguistic_evidence}, {specific_hint}
    └── error_hints                #   错误修正提示 (含必需的 default)
        ├── passive_subject       #     被动语态主语错误
        ├── passive_object        #     被动语态宾语错误
        ├── condition_boundary    #     条件从句边界错误
        ├── long_distance_object  #     长距离依存错误
        ├── negation_role         #     否定角色错误
        └── role_mismatch         #     角色-情态不匹配
```

---

## 7. 核心模块详解

### 7.1 约束校验器 — 7 步校验算法

```
┌─────────────────────────────────────────────────────────────────────┐
│              ConstraintValidator.validate() — 7 步算法               │
│                                                                     │
│  输入: LegalTriplet (LLM) + 原始文本 + DependencyTree (Stanza)      │
│  输出: ValidationResult {VALID | CORRECTED | REFLEXION_REQUIRED}    │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ Step 1: 定位根谓词                                           │   │
│  │   通过 head=0 在 UD 树中找到主句动词                           │   │
│  │   "Seller shall DELIVER the Goods" → root = "deliver"        │   │
│  └────────────────────────────┬────────────────────────────────┘   │
│                               v                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ Step 2: 检测语态，恢复语义论元                                 │   │
│  │   主动: nsubj→agent(主语), obj→patient(宾语)                  │   │
│  │   被动: obl:agent→agent(主语), nsubj:pass→patient(宾语)       │   │
│  └────────────────────────────┬────────────────────────────────┘   │
│                               v                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ Step 3: 校验主语                                             │   │
│  │   对照 LLM subject.text 与 UD 语义施事 (agent)                │   │
│  │   匹配策略: 规范化后精确匹配 → 子串匹配 → 内容词重叠          │   │
│  └────────────────────────────┬────────────────────────────────┘   │
│                               v                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ Step 4: 校验宾语                                             │   │
│  │   对照 LLM action.object 与 UD 语义受事 (patient)             │   │
│  │   处理不及物动词、介词补足语等边界情况                         │   │
│  └────────────────────────────┬────────────────────────────────┘   │
│                               v                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ Step 5: 校验条件从句                                          │   │
│  │   计算 LLM 条件文本与 UD advcl 子树的 token 级 IoU            │   │
│  │   IoU ≥ 0.5 → 接受; IoU < 0.5 → 边界修正                     │   │
│  │   检测条件遗漏 (omission) 和过度扩展 (over-extraction)         │   │
│  └────────────────────────────┬────────────────────────────────┘   │
│                               v                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ Step 6: 校验情态/角色                                         │   │
│  │   对照 LLM subject.role 与 UD 推导的法律角色                  │   │
│  │   规则: shall+肯定→obligor, may+肯定→right_holder,            │   │
│  │         shall/may+否定→prohibited_party                      │   │
│  └────────────────────────────┬────────────────────────────────┘   │
│                               v                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ Step 7: 确定输出状态                                          │   │
│  │   无修正 → VALID                                              │   │
│  │   有修正且 UD 有候选 → CORRECTED                              │   │
│  │   有修正但 UD 缺证据 → REFLEXION_REQUIRED                     │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### 7.2 Reflexion 自我修正机制

```
┌─────────────────────────────────────────────────────────────────┐
│                    Reflexion 自我修正流程                         │
│                                                                 │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐       │
│  │ 原始大语言模型 │────►│ UD 约束校验 │────►│ Validation  │       │
│  │   抽取结果    │     │    器       │     │   Result    │       │
│  └─────────────┘     └─────────────┘     └──────┬──────┘       │
│                                                  │              │
│                                   status=REFLEXION_REQUIRED     │
│                                                  │              │
│                                                  v              │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                  ReflexionGenerator                       │  │
│  │                                                          │  │
│  │  1. 分析 FieldCorrection 列表                             │  │
│  │     │                                                    │  │
│  │     v                                                    │  │
│  │  2. 确定错误类型 (优先级排序)                              │  │
│  │     passive_subject > condition_boundary                  │  │
│  │     > negation_role > role_mismatch                       │  │
│  │     > long_distance_object > default                      │  │
│  │     │                                                    │  │
│  │     v                                                    │  │
│  │  3. 选择对应的 ERROR_HINT                                 │  │
│  │     │                                                    │  │
│  │     v                                                    │  │
│  │  4. 组装反馈提示词                                         │  │
│  │     ┌─────────────────────────────────────┐              │  │
│  │     │ ## Error Type                       │              │  │
│  │     │ passive_subject                     │              │  │
│  │     │                                     │              │  │
│  │     │ ## Original Clause Text             │              │  │
│  │     │ {原始合同条款}                        │              │  │
│  │     │                                     │              │  │
│  │     │ ## Your Previous Extraction         │              │  │
│  │     │ {错误的三元组 JSON}                   │              │  │
│  │     │                                     │              │  │
│  │     │ ## Linguistic Evidence (UD Parse)   │              │  │
│  │     │ {UD 句法证据 JSON}                   │              │  │
│  │     │                                     │              │  │
│  │     │ ## Correction Guidance              │              │  │
│  │     │ {针对性修正提示}                      │              │  │
│  │     └─────────────────────────────────────┘              │  │
│  │     │                                                    │  │
│  │     v                                                    │  │
│  │  5. 调用大语言模型重新抽取                                  │  │
│  │  6. 解析修正后的 LegalTriplet                             │  │
│  │  7. 最多迭代 1 次 (防止过度修正)                           │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                  │              │
│                                                  v              │
│                                         修正后的 LegalTriplet   │
└─────────────────────────────────────────────────────────────────┘
```

### 7.3 双模型标注共识机制

```
┌───────────────────────────────────────────────────────────────────┐
│                 双模型标注 → 金标共识流程                           │
│                                                                   │
│  合同条款                                                          │
│  "Seller shall deliver the Goods within 30 days."                 │
│       │                                                           │
│       ├──────────────────┬──────────────────┐                     │
│       v                  v                  v                     │
│  ┌──────────┐      ┌──────────┐      ┌──────────┐                │
│  │  Qwen    │      │  Gemma   │      │ 交叉审查  │                │
│  │ 3.6 27B  │      │ 4 31B    │      │          │                │
│  │          │      │          │      │ Qwen 审查 │                │
│  │ 独立标注  │      │ 独立标注  │      │ Gemma输出│                │
│  └────┬─────┘      └────┬─────┘      │          │                │
│       │                 │             │ Gemma审查 │                │
│       │                 │             │ Qwen输出 │                │
│       │                 │             └────┬─────┘                │
│       │                 │                  │                      │
│       └────────┬────────┘                  │                      │
│                │                           │                      │
│                v                           v                      │
│  ┌─────────────────────────────────────────────────────┐         │
│  │           field_level_consensus()                    │         │
│  │                                                     │         │
│  │  逐字段比较 (6 字段):                                  │         │
│  │    1. subject.text    规范化文本比较 (去冠词/小写)      │         │
│  │    2. subject.role    枚举值精确匹配                   │         │
│  │    3. action.predicate 规范化文本比较                   │         │
│  │    4. action.object   规范化文本比较                   │         │
│  │    5. condition.text  规范化文本比较                   │         │
│  │    6. condition.type  枚举值精确匹配                   │         │
│  │                                                     │         │
│  │  一致 → 采纳                          │              │         │
│  │  不一致 → 标记需人工审核, Qwen 值暂用   │              │         │
│  └────────────────────────────────────┬────────────────┘         │
│                                       │                          │
│                                       v                          │
│  ┌─────────────────────────────────────────────────────┐         │
│  │           build_gold_from_consensus()                 │         │
│  │                                                     │         │
│  │  对已人工裁决的分歧 → 采用裁决结果                     │         │
│  │  对未裁决的分歧     → 保留 Qwen 值 (暂定)              │         │
│  └────────────────────────────────────┬────────────────┘         │
│                                       │                          │
│                                       v                          │
│                              金标 LegalTriplet                   │
│                           (保存为 data/processed/gold_triplets.jsonl)
└───────────────────────────────────────────────────────────────────┘
```

---

## 8. 评估体系

### 8.1 双轨评估框架

LexSpec 从两个维度评估抽取质量：

```
┌─────────────────────────────────────────────────────────────────┐
│                      双轨评估框架                                 │
│                                                                 │
│  ┌─────────────────────────┐  ┌─────────────────────────────┐  │
│  │  轨道 1: 任务指标         │  │  轨道 2: 语言学指标           │  │
│  │  (Weighted Triplet F1)  │  │  (Linguistic Metrics)       │  │
│  │                         │  │                             │  │
│  │  加权 F1 = Σ wᵢ × F1ᵢ   │  │  1. 依存路径合法性率          │  │
│  │                         │  │     抽取的论元对是否在 UD     │  │
│  │  分量 (5 项):            │  │     树中有合法依存路径        │  │
│  │  ┌─────────────┬──────┐ │  │                             │  │
│  │  │ subject_text│ 0.35 │ │  │  2. 被动语态恢复准确率        │  │
│  │  │ subject_role│ 0.10 │ │  │     是否正确识别 obl:agent   │  │
│  │  │ predicate   │ 0.20 │ │  │     为法律主体                │  │
│  │  │ object      │ 0.20 │ │  │                             │  │
│  │  │ condition   │ 0.15 │ │  │  3. 条件边界 IoU             │  │
│  │  └─────────────┴──────┘ │  │     预测条件从句与 UD 条件    │  │
│  │                         │  │     从句的 token 级重叠度     │  │
│  │  匹配方式:               │  │                             │  │
│  │  - 文本字段: token 级 F1 │  │  4. 修正成功率               │  │
│  │    (部分匹配给分)         │  │     CORRECTED / (CORRECTED  │  │
│  │  - 角色/类型: 精确匹配   │  │     + REFLEXION)            │  │
│  └─────────────────────────┘  └─────────────────────────────┘  │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  显著性检验 (3 种方法)                                    │    │
│  │                                                         │    │
│  │  1. 配对自举 (Bootstrap) 10,000 次重采样 — 主要方法       │    │
│  │     → 95% 置信区间 + 单侧 p 值                           │    │
│  │  2. Wilcoxon 符号秩检验 — 补充方法                        │    │
│  │     → 双侧 p 值                                          │    │
│  │  3. 分层显著性 — 按语言现象子集分别检验                    │    │
│  │     → "被动语态上改进是否显著?"                            │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

### 8.2 错误分类体系（二级分类）

```
┌──────────────────────────────────────────────────────────────────┐
│                   错误分类: 一级 × 二级 交叉                        │
│                                                                  │
│  一级分类 (语言学现象)                 二级分类 (错误字段)          │
│  ──────────────────────              ──────────────────────       │
│                                                                  │
│  ┌──────────────────────┐           ┌──────────────────────┐     │
│  │ passive_voice        │           │ subject              │     │
│  │ 被动语态错误           │           │ 主语文本错误           │     │
│  │ nsubj:pass混淆        │           │                      │     │
│  ├──────────────────────┤           ├──────────────────────┤     │
│  │ conditional_boundary │           │ role                 │     │
│  │ 条件边界错误           │           │ 角色分类错误           │     │
│  │ advcl+mark范围错误    │           │                      │     │
│  ├──────────────────────┤           ├──────────────────────┤     │
│  │ relative_clause      │           │ predicate            │     │
│  │ 关系从句混淆           │           │ 谓词识别错误           │     │
│  │ acl:relcl内嵌谓词     │           │                      │     │
│  ├──────────────────────┤           ├──────────────────────┤     │
│  │ long_distance        │           │ object               │     │
│  │ 长距离依存错误         │           │ 宾语识别错误           │     │
│  │ 依存距离 > 3          │           │                      │     │
│  ├──────────────────────┤           ├──────────────────────┤     │
│  │ negation_exception   │           │ condition_omission   │     │
│  │ 否定/例外错误          │           │ 条件遗漏              │     │
│  │ neg关系+角色反转      │           │                      │     │
│  ├──────────────────────┤           ├──────────────────────┤     │
│  │ other                │           │ condition_overext    │     │
│  │ 其他错误              │           │ 条件过度扩展           │     │
│  └──────────────────────┘           └──────────────────────┘     │
│                                                                  │
│  每个错误案例包含:                                                 │
│    - 双语解释 (中文 + English)                                     │
│    - 具体 UD 依存关系引用 (token 编号、deprel 标签)                  │
│    - 预测 vs 金标对比                                              │
└──────────────────────────────────────────────────────────────────┘
```

---

## 9. 实验流水线

### 9.1 7 步完整流水线

```
┌─────────────────────────────────────────────────────────────────────┐
│                      LexSpec 实验流水线                              │
│                                                                     │
│  Step 01                 Step 02                 Step 03            │
│  ┌──────────────┐       ┌──────────────┐       ┌──────────────┐   │
│  │ 构建测试语料  │       │ 金标标注      │       │ 基线抽取      │   │
│  │              │       │              │       │              │   │
│  │ CUAD v1      │       │ 双模型独立标注 │       │ Qwen3.5 9B   │   │
│  │ 510份合同     │──►    │ + 交叉审查    │──►    │ 零样本        │   │
│  │              │       │ + 共识合并    │       │ 纯大语言模型   │   │
│  │ 分层抽样      │       │              │       │              │   │
│  │ 100条测试集   │       │ 金标三元组    │       │ baseline     │   │
│  └──────────────┘       └──────────────┘       └──────┬───────┘   │
│                                                       │            │
│                                                       │            │
│  Step 04                 Step 05                             │     │
│  ┌──────────────┐       ┌──────────────┐                    │     │
│  │ +UD约束校验   │       │ +Reflexion   │                    │     │
│  │              │       │              │                    │     │
│  │ 大语言模型抽取 │       │ 大语言模型抽取 │                    │     │
│  │ + Stanza解析 │       │ + Stanza解析 │                    │     │
│  │ + 自动修正   │       │ + UD校验     │◄───────────────────┘     │
│  │              │       │ + 错误提示    │                          │
│  │ ours_dep     │       │ + 大语言模型重抽│                          │
│  └──────┬───────┘       └──────┬───────┘                          │
│         │                      │                                   │
│         └──────────┬───────────┘                                   │
│                    │                                               │
│                    v                                               │
│  Step 06                 Step 07                                   │
│  ┌──────────────┐       ┌──────────────┐                          │
│  │ 双轨评估      │       │ 错误分析      │                          │
│  │              │       │              │                          │
│  │ 加权 F1      │──►    │ 二级分类      │                          │
│  │ 语言学指标    │       │ 双语解释      │                          │
│  │ 显著性检验    │       │ 交叉分布      │                          │
│  │              │       │              │                          │
│  │ metrics/     │       │ error_cases/ │                          │
│  └──────────────┘       └──────────────┘                          │
└─────────────────────────────────────────────────────────────────────┘
```

### 9.2 实验对照设计

| 实验条件 | 大语言模型 | UD 约束校验 | Reflexion 修正 | 变量 |
|----------|-----------|------------|---------------|------|
| **Baseline** | Qwen3.5 9B | ✗ | ✗ | 纯大语言模型下界 |
| **Ours-Dep** | Qwen3.5 9B | ✓ (自动修正) | ✗ | 测试句法约束的单独贡献 |
| **Ours-Reflexion** | Qwen3.5 9B | ✓ | ✓ (1轮) | 完整 LexSpec 系统 |

对比分析：
- **Baseline vs Ours-Dep**: 测试 UD 约束校验是否能在不增加大语言模型调用的情况下提升抽取质量
- **Ours-Dep vs Ours-Reflexion**: 测试 Reflexion 自我修正的增量收益
- **Baseline vs Ours-Reflexion**: 测试完整 LexSpec 系统的总提升

---

## 10. 依赖项

### 10.1 Python 包

| 包名 | 最低版本 | 用途 |
|------|---------|------|
| `stanza` | ≥ 1.6.0 | UD 依存句法解析（分词、词性标注、词形还原、依存分析） |
| `openai` | ≥ 1.0.0 | llama.cpp OpenAI 兼容 API 客户端 |
| `pandas` | ≥ 2.0.0 | 结构化数据处理 |
| `numpy` | ≥ 1.24.0 | 数值计算 |
| `scipy` | ≥ 1.10.0 | Wilcoxon 符号秩检验 |
| `scikit-learn` | ≥ 1.3.0 | 分类指标（精确率、召回率、F1） |
| `matplotlib` | ≥ 3.7.0 | 可视化图表生成 |
| `pyyaml` | ≥ 6.0 | YAML 配置文件解析 |
| `pydantic` | ≥ 2.0.0 | 运行时类型校验 (Pydantic v2) |
| `tqdm` | ≥ 4.65.0 | 批量处理进度条 |

### 10.2 开发依赖

| 包名 | 最低版本 | 用途 |
|------|---------|------|
| `pytest` | ≥ 7.0 | 单元测试框架 |
| `pytest-cov` | ≥ 4.0 | 测试覆盖率报告 |
| `mypy` | ≥ 1.0 | 静态类型检查 |
| `ruff` | ≥ 0.1 | 代码规范检查与格式化 |

### 10.3 外部服务

- **llama.cpp 服务器**: 本地 HTTP 服务器，暴露 OpenAI 兼容的 `/v1/chat/completions` 端点
  - 默认地址: `http://10.0.16.254:8080/v1`
  - 模型文件: Qwen3.5 9B (~5GB GGUF)、Qwen3.6 27B (~15GB GGUF)、Gemma4 31B (~18GB GGUF)

### 10.4 Stanza 模型

首次运行时自动下载英文模型（~500MB）。包含:
- `tokenize` — 句子分割与词语分词
- `mwt` — 多词 token 展开 (e.g., "don't" → "do not")
- `pos` — 词性标注 (UPOS + XPOS + UFeats)
- `lemma` — 词形还原
- `depparse` — UD v2 依存句法解析

---

## 11. 论文引用

本项目参考以下学术文献：

| 论文 | 引用 | 在本项目中的作用 |
|------|------|-----------------|
| CUAD (Hendrycks et al., 2021) | Contract Understanding Atticus Dataset | 测试集构建的数据来源 |
| ContractNLI (Koreeda et al., 2021) | Document-level Natural Language Inference for Contracts | 标注模式参考 |
| LegalBench (Guha et al., 2023) | A Benchmark for Legal Reasoning | 评估基准设计参考 |
| LexGLUE (Chalkidis et al., 2022) | Multi-Task Benchmark for Legal NLP | 法律 NLP 任务设计 |
| Universal Dependencies (de Marneffe et al., 2021) | UD v2 Dependency Guidelines | 依存语法理论基础 |
| UD v2 (Nivre et al., 2020) | An Ever-growing Multilingual Treebank Collection | 依存解析实践 |
| PropBank (Palmer et al., 2005) | Proposition Bank — Semantic Role Labeling | 语义角色标注理论 |
| Reflexion (Shinn et al., 2023) | Reflexion: Language Agents with Verbal Reinforcement | Reflexion 自我修正方法 |
| Self-Refine (Madaan et al., 2023) | Self-Refine: Iterative Refinement | 迭代自我修正机制 |
| LLM in Law (Lai et al., 2023) | 大语言模型在法律中的应用综述 | 法律 AI 领域综述 |
| Legal AI Survey (Hou et al., 2025) | Legal Artificial Intelligence Survey | 法律 AI 最新进展 |

---

## 12. 许可证

本项目仅供学术研究使用。源代码和配置文件按 MIT 许可证提供，用于计算语言学实验的可复现性。

模型权重（Qwen、Gemma）受其各自许可证条款约束。使用前请参阅模型提供方的使用条款。

---

*LexSpec — 基于依存句法与论元结构约束的法律合同要素抽取智能体研究*

*最后更新: 2026-07-01*
