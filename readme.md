# test
# Clasp 项目说明

这是一个用于社交行为数据处理与训练数据构造的项目。

## 核心能力

- `process_dataset/sft_data_generator.py`：基于用户行为序列生成 SFT 样本。
- `process_dataset/community_data_splitter.py`：按社区切分用户并导出数据文件。
- `src/`：DPO 对构造与画像生成模型训练的完整 pipeline。

## 目录说明

- `src/`：DPO pipeline 核心代码
- `process_dataset/`：数据处理脚本
- `data/`：导出后的训练/测试/评估数据
- `comparison/`：对比实验相关代码
- `scripts/`：辅助脚本
- `saves/`：模型或中间结果保存目录

## 社区切分与导出

脚本：`process_dataset/community_data_splitter.py`

功能：
1. 从社区 `0,1,3,4,5` 中按每个社区 70%/30% 划分训练和测试用户；
2. 从社区 `6,7` 中抽取 500 个未见用户用于评估；
3. 将用户及其序列动作导出到 `data` 目录下，按社区分文件。

运行命令：

```bash
python process_dataset/community_data_splitter.py \
  --train-communities 0,1,3,4,5 \
  --eval-communities 6,7 \
  --train-ratio 0.7 \
  --eval-users 500 \
  --output-dir data
```

## DPO Pipeline 概览

关键模块：
- `src/config.py`：全局配置（模型路径、窗口参数、评分权重、DPO 阈值）
- `src/window_splitter.py`：窗口切分器（T=10, W0~W4）
- `src/scorer.py`：评分（F(S)、L(S)、Q(S)）
- `src/action_predictor.py`：动作预测（决策类 + 内容生成类）
- `src/profile_generator.py`：画像生成与精炼
- `src/dpo_pipeline.py`：DPO 全流程编排（主入口）

流程摘要：
1. 窗口切分（不足 50 条动作的用户跳过）
2. 生成初始画像 `S0`
3. 在 `W0/W1/W2` 上做 baseline 评分
4. 构造行为偏差信号
5. 生成候选精炼画像（默认 N=15）
6. 对候选画像评分并构造 DPO 正负样本对

## 常用运行命令

```bash
# Step 1: 窗口切分（示例）
python -m src.window_splitter \
  --input data/test/community_5.jsonl \
  --output output/windowed/test/community_5.jsonl

# Step 2: 运行 DPO Pipeline（示例）
python -m src.dpo_pipeline \
  --input output/windowed/test/community_5.jsonl \
  --output-dir output/dpo \
  --max-users 2
```

调试模式：

```bash
python -m src.dpo_pipeline \
  --input output/windowed/test/community_5.jsonl \
  --output-dir output/dpo \
  --max-users 1 \
  --debug
```

## 模型部署

支持两种方式：
- vLLM API（推荐）：两个模型分别在不同端口启动服务
- 本地 transformers：无需单独服务，但显存要求更高

在 `src/config.py` 中通过 `USE_VLLM_API` 切换模式。

## 完整流程

1. 安装依赖：

```bash
pip install torch transformers sentence-transformers psycopg2-binary openai
# 若使用 vLLM
pip install vllm
```

2. 数据切分：运行 `process_dataset/community_data_splitter.py`
3. 窗口切分：运行 `src.window_splitter`
4. 启动模型服务（仅 vLLM）
5. 运行 `src.dpo_pipeline` 生成 DPO 数据
# Clasp 项目说明

这是一个用于社交行为数据处理与训练数据构造的项目。

当前已包含的核心能力：
- `process_dataset/sft_data_generator.py`：基于用户行为序列生成 SFT 样本。
- `process_dataset/community_data_splitter.py`：按社区切分用户，导出数据文件。
- `src/`：DPO 对构造与画像生成模型训练的完整 pipeline。

## 目录说明

- `src/`：DPO pipeline 核心代码
- `process_dataset/`：数据处理脚本
- `data/`：导出后的训练/测试/评估数据
- `comparison/`：对比实验相关代码
- `scripts/`：辅助脚本
- `saves/`：模型或中间结果保存目录

## 社区切分与导出

脚本：`process_dataset/community_data_splitter.py`

功能：
1. 从社区 `0,1,3,4,5` 中按每个社区 70%/30% 划分训练和测试用户；
2. 从社区 `6,7` 中抽取 500 个未见用户用于评估；
3. 将用户及其序列动作导出到 `data` 目录下，按社区分文件。

### 运行命令

```bash
python process_dataset/community_data_splitter.py \
  --train-communities 0,1,3,4,5 \
  --eval-communities 6,7 \
  --train-ratio 0.7 \
  --eval-users 500 \
  --output-dir data
```

### 输出结构

```text
data/
├── train/community_0.jsonl ... community_5.jsonl
├── test/community_0.jsonl ... community_5.jsonl
├── eval_unseen/community_6.jsonl, community_7.jsonl
└── split_summary.json
```

### 字段说明

- `community_id`：用户所属社区
- `user_id`：用户 ID
- `actions`：动作序列（时间顺序），`post` 动作的 `target` 为 `null`
- `action_count`：序列长度

---

## DPO 对构造 - 画像生成模型训练

### 代码结构

```text
src/
├── __init__.py
├── config.py               # 全局配置（模型路径、窗口参数、评分权重、DPO 阈值）
├── prompts.py              # 所有提示词模板（画像生成/精炼、动作预测）
├── window_splitter.py      # 窗口切分器：T=10, W0~W4
├── scorer.py               # 评分：F(S) 加权F1 + L(S) 语义对齐 + Q(S) 综合
├── action_predictor.py     # 动作预测：决策类 + 内容生成类
├── profile_generator.py    # 画像生成 & 精炼（N=15 候选）
└── dpo_pipeline.py         # DPO 全流程编排（主入口）
```

## DPO 联合损失函数训练
对于DPO微调，温度系数β设置为0.2，注意dpo微调对 除了好画像y1，坏画像y2，还需要记录输入上下文x（旧人格+真实/预测差异）
为了避免 DPO微调 只管偏好，不管正确性，所以增加 SFT损失函数，强制模型保持高质量的persona画像
L = L（DPO）+ a·L（SFT）
a = 0.1 


### Pipeline 流程

1. **窗口切分**：将用户动作按 T=10 切为 W0~W4（共 50 条），不足 50 条的用户跳过
2. **初始画像 S0**：用 W0 动作 + `profile_generation_model_raw` 生成
3. **Baseline 评分**：用 S0 在 W0/W1/W2 上预测动作，计算 Q(S0)（W0 为**整窗**共 T 条：空历史起逐步预测，每步用真实动作推进历史；W1=用 W0 作历史预测 W1；W2=用 W1 作历史预测 W2）
4. **偏差信号**：对比 W1 预测与真实，构造 behavior_discrepancies
5. **候选画像**：基于偏差信号用 `profile_generation_model_raw` 生成 N=15 个精炼画像
6. **候选评分**：每个候选画像在 W0/W1/W2 上评分，计算 r(all) = r(pre)+r(cur)+r(fut)
7. **DPO 对**：r > τ⁺=0.05 为正，r < τ⁻=-0.05 为负，且正-负 > δ=0.2

### 评分公式

- **F(S)**：交互决策加权 F1（post=0.35, reply=0.30, repost=0.20, like=0.15）
- **L(S)**：内容语义对齐度（sentence-transformers 余弦相似度均值）
- **Q(S)** = α·F(S) + (1-α)·L(S)，α = 0.4

### 运行命令

```bash
# Step 1: 窗口切分（以 test/community_5.jsonl 为例）
python -m src.window_splitter \
  --input data/test/community_5.jsonl \
  --output output/windowed/test/community_5.jsonl

# Step 2: 运行 DPO Pipeline
python -m src.dpo_pipeline \
  --input output/windowed/test/community_5.jsonl \
  --output-dir output/dpo \
  --max-users 2   # 调试时限制用户数

# 调试：1）Step3 在终端打印「行为偏差全文」（含 reply 的 Replied-to original）2）各候选「画像精炼」的 LLM 块
# 不打印「每次动作预测」的 API 级调试（避免刷屏）。若也要看 action 的 prompt：再加 --debug-actions
python -m src.dpo_pipeline \
  --input output/windowed/test/community_5.jsonl \
  --output-dir output/dpo \
  --max-users 1 \
  --debug
# python -m src.dpo_pipeline ... --debug --debug-actions   # 另打印每次决策/内容预测的 [LLM-DEBUG]

# 换一批测试用户：先按种子打乱 jsonl 中用户顺序，再只跑前 1 个（同一文件多试几个 seed）
python -m src.dpo_pipeline \
  --input output/windowed/test/community_5.jsonl \
  --output-dir output/dpo \
  --max-users 1 \
  --seed 42
```

也可在 `src/config.py` 中设置 `DEBUG_LLM = True`（或调整 `DEBUG_LLM_MAX_*` 截断长度）。摘要文件 `dpo_summary_*.json` 的 `config.debug_llm` 会记录是否开启调试。

### 输出文件

```text
output/
├── windowed/                              # 窗口切分结果
│   ├── train/community_*.jsonl
│   └── test/community_*.jsonl
└── dpo/                                   # DPO 对构造结果
    ├── dpo_pairs_community_5.jsonl        # DPO 正负对（每行一个 pair）
    ├── dpo_detail_community_5.json        # 每用户的完整评分细节
    └── dpo_summary_community_5.json       # 统计摘要
```

### 配置说明（src/config.py）

| 参数 | 默认值 | 说明 |
|---|---|---|
| `WINDOW_SIZE` | 10 | 每窗口动作数 |
| `NUM_WINDOWS` | 5 | W0~W4 |
| `NUM_CANDIDATE_PROFILES` | 15 | 每轮精炼候选画像数 |
| `ALPHA` | 0.4 | Q(S) 中 F(S) 的权重 |
| `TAU_PLUS` | 0.05 | 正向 DPO 阈值 |
| `TAU_MINUS` | -0.05 | 负向 DPO 阈值 |
| `DELTA` | 0.20 | 正负对最小差距 |
| `USE_VLLM_API` | True | True 时用 vLLM；False 时仅当传入非 None 的 model/tokenizer 才走本地 transformers |
| `DEBUG_LLM` | False | True 或 `--debug`：打印 Step3 **偏差全文** + **画像精炼** LLM 块；不打印动作 API 块 |
| `DEBUG_LLM_INCLUDE_ACTIONS` | False | True 或 `--debug-actions`（需同时 `--debug`）：再打印每次动作预测的 `[LLM-DEBUG]` |
| `DEBUG_LLM_*` | 见 config | 精炼调试时重点展示行为误差与节选；长输出头尾由 `PROFILE_OUTPUT_*` / `DISCREPANCY_MAX` 等控制 |

---

## 模型部署 & 启动命令

项目涉及两个 LLM 模型和一个 Sentence Transformer：

| 模型 | 路径 | 用途 |
|---|---|---|
| Meta-Llama-3-8B-Instruct | `/data/LLM_models/Meta-Llama-3-8B-Instruct` | 画像生成 & 精炼 |
| Meta-Llama-3-8B-Instruct-bluesky-sft-289 | `/data/LLM_models/Meta-Llama-3-8B-Instruct-bluesky-sft-289` | 动作预测（决策 + 内容） |
| all-mpnet-base-v2 | `/data/LLM_models/sentence-transformers/all-mpnet-base-v2` | 语义相似度评分（自动加载，无需部署） |

### 方式一：vLLM 部署（推荐）

vLLM 单实例只支持一个模型，所以需要在**不同端口**启动两个实例。
两个 8B 模型各需约 16GB 显存（bfloat16），总计约 32GB，一张 A100-80GB 可同时运行。

```bash
# 终端 1：启动画像生成模型（端口 8000，GPU 0）
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
  --model /data/LLM_models/Meta-Llama-3-8B-Instruct \
  --served-model-name Meta-Llama-3-8B-Instruct \
  --port 8000 \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.45

# 终端 2：启动动作预测模型（端口 8001，GPU 0）
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
  --model /data/LLM_models/Meta-Llama-3-8B-Instruct-bluesky-sft-289 \
  --served-model-name Meta-Llama-3-8B-Instruct-bluesky-sft-289 \
  --port 8001 \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.45

--远程服务器上启动了
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server   --model ./Meta-Llama-3-8B-Instruct-bluesky-sft-289   --served-model-name Meta-Llama-3-8B-Instruct-bluesky-sft-289   --port 8001   --dtype bfloat16   --max-model-len 2048   --gpu-memory-utilization 0.35

```

> **多 GPU 环境**：如果有多张 GPU，可以分别指定 `CUDA_VISIBLE_DEVICES=0` 和 `CUDA_VISIBLE_DEVICES=1`，
> 并将 `--gpu-memory-utilization` 调高到 `0.9`。

验证服务是否正常：

```bash
# 检查画像生成模型
curl http://localhost:8000/v1/models

# 检查动作预测模型
curl http://localhost:8001/v1/models
```

### 方式二：本地 transformers 加载

不需要额外启动服务，Pipeline 会自动用 `transformers` 库加载模型到 GPU。
需要 GPU 显存至少 40GB（同时加载两个 8B 模型）。

### 切换模式

在 `src/config.py` 中修改：

```python
# 使用 vLLM API 模式
USE_VLLM_API = True

# 使用本地 transformers 加载（默认）
USE_VLLM_API = False
```


```bash
python -m src.dpo_pipeline --input output/windowed/test/community_5.jsonl  --output-dir output/dpo   --max-users 1 --debug


```

> 测试模式的 API 配置在 `src/config.py` 的 `TEST_*` 系列变量中，可按需修改。

---

## 完整运行流程

### Step 0：安装依赖

```bash
pip install torch transformers sentence-transformers psycopg2-binary openai
# 如果使用 vLLM 部署
pip install vllm
```

### Step 1：数据切分

```bash
DB_HOST=127.0.0.1 python process_dataset/community_data_splitter.py \
  --train-communities 0,1,3,4,5 \
  --eval-communities 6,7 \
  --train-ratio 0.7 \
  --eval-users 500 \
  --output-dir data
```

### Step 2：窗口切分

```bash
# 对所有训练数据做窗口切分
for f in data/train/community_*.jsonl; do
  out="output/windowed/train/$(basename $f)"
  python -m src.window_splitter --input "$f" --output "$out"
done

# 对测试数据做窗口切分
for f in data/test/community_*.jsonl; do
  out="output/windowed/test/$(basename $f)"
  python -m src.window_splitter --input "$f" --output "$out"
done
```

### Step 3：启动模型服务（仅 vLLM 模式）

```bash
# 按照上面「方式一」在两个终端分别启动两个 vLLM 实例
# 然后修改 src/config.py 中 USE_VLLM_API = True
```

### Step 4：运行 DPO Pipeline

```bash
# 单个社区（调试）
python -m src.dpo_pipeline \
  --input output/windowed/test/community_5.jsonl \
  --output-dir output/dpo \
  --max-users 2

# 批量处理所有训练社区
for f in output/windowed/train/community_*.jsonl; do
  python -m src.dpo_pipeline --input "$f" --output-dir output/dpo
done
```
# Clasp 项目说明

这是一个用于社交行为数据处理与训练数据构造的项目。

当前已包含的核心能力：
- `process_dataset/sft_data_generator.py`：基于用户行为序列生成 SFT 样本。
- `process_dataset/community_data_splitter.py`：按社区切分用户，导出数据文件。
- `src/`：DPO 对构造与画像生成模型训练的完整 pipeline。

## 目录说明

- `src/`：DPO pipeline 核心代码
- `process_dataset/`：数据处理脚本
- `data/`：导出后的训练/测试/评估数据
- `comparison/`：对比实验相关代码
- `scripts/`：辅助脚本
- `saves/`：模型或中间结果保存目录

## 社区切分与导出

脚本：`process_dataset/community_data_splitter.py`

功能：
1. 从社区 `0,1,3,4,5` 中按每个社区 70%/30% 划分训练和测试用户；
2. 从社区 `6,7` 中抽取 500 个未见用户用于评估；
3. 将用户及其序列动作导出到 `data` 目录下，按社区分文件。

### 运行命令

```bash
python process_dataset/community_data_splitter.py \
  --train-communities 0,1,3,4,5 \
  --eval-communities 6,7 \
  --train-ratio 0.7 \
  --eval-users 500 \
  --output-dir data
```

### 输出结构

```text
data/
├── train/community_0.jsonl ... community_5.jsonl
├── test/community_0.jsonl ... community_5.jsonl
├── eval_unseen/community_6.jsonl, community_7.jsonl
└── split_summary.json
```

### 字段说明

- `community_id`：用户所属社区
- `user_id`：用户 ID
- `actions`：动作序列（时间顺序），`post` 动作的 `target` 为 `null`
- `action_count`：序列长度

---

## DPO 对构造 - 画像生成模型训练

### 代码结构

```text
src/
├── __init__.py
├── config.py               # 全局配置（模型路径、窗口参数、评分权重、DPO 阈值）
├── prompts.py              # 所有提示词模板（画像生成/精炼、动作预测）
├── window_splitter.py      # 窗口切分器：T=10, W0~W4
├── scorer.py               # 评分：F(S) 加权F1 + L(S) 语义对齐 + Q(S) 综合
├── action_predictor.py     # 动作预测：决策类 + 内容生成类
├── profile_generator.py    # 画像生成 & 精炼（N=15 候选）
└── dpo_pipeline.py         # DPO 全流程编排（主入口）
```

## DPO 联合损失函数训练
对于DPO微调，温度系数β设置为0.2，注意dpo微调对 除了好画像y1，坏画像y2，还需要记录输入上下文x（旧人格+真实/预测差异）
为了避免 DPO微调 只管偏好，不管正确性，所以增加 SFT损失函数，强制模型保持高质量的persona画像
L = L（DPO）+ a·L（SFT）
a = 0.1 


### Pipeline 流程

1. **窗口切分**：将用户动作按 T=10 切为 W0~W4（共 50 条），不足 50 条的用户跳过
2. **初始画像 S0**：用 W0 动作 + `profile_generation_model_raw` 生成
3. **Baseline 评分**：用 S0 在 W0/W1/W2 上预测动作，计算 Q(S0)（W0 为**整窗**共 T 条：空历史起逐步预测，每步用真实动作推进历史；W1=用 W0 作历史预测 W1；W2=用 W1 作历史预测 W2）
4. **偏差信号**：对比 W1 预测与真实，构造 behavior_discrepancies
5. **候选画像**：基于偏差信号用 `profile_generation_model_raw` 生成 N=15 个精炼画像
6. **候选评分**：每个候选画像在 W0/W1/W2 上评分，计算 r(all) = r(pre)+r(cur)+r(fut)
7. **DPO 对**：r > τ⁺=0.05 为正，r < τ⁻=-0.05 为负，且正-负 > δ=0.2

### 评分公式

- **F(S)**：交互决策加权 F1（post=0.35, reply=0.30, repost=0.20, like=0.15）
- **L(S)**：内容语义对齐度（sentence-transformers 余弦相似度均值）
- **Q(S)** = α·F(S) + (1-α)·L(S)，α = 0.4

### 运行命令

```bash
# Step 1: 窗口切分（以 test/community_5.jsonl 为例）
python -m src.window_splitter \
  --input data/test/community_5.jsonl \
  --output output/windowed/test/community_5.jsonl

# Step 2: 运行 DPO Pipeline
python -m src.dpo_pipeline \
  --input output/windowed/test/community_5.jsonl \
  --output-dir output/dpo \
  --max-users 2   # 调试时限制用户数

# 调试：1）Step3 在终端打印「行为偏差全文」（含 reply 的 Replied-to original）2）各候选「画像精炼」的 LLM 块
# 不打印「每次动作预测」的 API 级调试（避免刷屏）。若也要看 action 的 prompt：再加 --debug-actions
python -m src.dpo_pipeline \
  --input output/windowed/test/community_5.jsonl \
  --output-dir output/dpo \
  --max-users 1 \
  --debug
# python -m src.dpo_pipeline ... --debug --debug-actions   # 另打印每次决策/内容预测的 [LLM-DEBUG]

# 换一批测试用户：先按种子打乱 jsonl 中用户顺序，再只跑前 1 个（同一文件多试几个 seed）
python -m src.dpo_pipeline \
  --input output/windowed/test/community_5.jsonl \
  --output-dir output/dpo \
  --max-users 1 \
  --seed 42
```

也可在 `src/config.py` 中设置 `DEBUG_LLM = True`（或调整 `DEBUG_LLM_MAX_*` 截断长度）。摘要文件 `dpo_summary_*.json` 的 `config.debug_llm` 会记录是否开启调试。

### 输出文件

```text
output/
├── windowed/                              # 窗口切分结果
│   ├── train/community_*.jsonl
│   └── test/community_*.jsonl
└── dpo/                                   # DPO 对构造结果
    ├── dpo_pairs_community_5.jsonl        # DPO 正负对（每行一个 pair）
    ├── dpo_detail_community_5.json        # 每用户的完整评分细节
    └── dpo_summary_community_5.json       # 统计摘要
```

### 配置说明（src/config.py）

| 参数 | 默认值 | 说明 |
|---|---|---|
| `WINDOW_SIZE` | 10 | 每窗口动作数 |
| `NUM_WINDOWS` | 5 | W0~W4 |
| `NUM_CANDIDATE_PROFILES` | 15 | 每轮精炼候选画像数 |
| `ALPHA` | 0.4 | Q(S) 中 F(S) 的权重 |
| `TAU_PLUS` | 0.05 | 正向 DPO 阈值 |
| `TAU_MINUS` | -0.05 | 负向 DPO 阈值 |
| `DELTA` | 0.20 | 正负对最小差距 |
| `USE_VLLM_API` | True | True 时用 vLLM；False 时仅当传入非 None 的 model/tokenizer 才走本地 transformers |
| `DEBUG_LLM` | False | True 或 `--debug`：打印 Step3 **偏差全文** + **画像精炼** LLM 块；不打印动作 API 块 |
| `DEBUG_LLM_INCLUDE_ACTIONS` | False | True 或 `--debug-actions`（需同时 `--debug`）：再打印每次动作预测的 `[LLM-DEBUG]` |
| `DEBUG_LLM_*` | 见 config | 精炼调试时重点展示行为误差与节选；长输出头尾由 `PROFILE_OUTPUT_*` / `DISCREPANCY_MAX` 等控制 |

---

## 模型部署 & 启动命令

项目涉及两个 LLM 模型和一个 Sentence Transformer：

| 模型 | 路径 | 用途 |
|---|---|---|
| Meta-Llama-3-8B-Instruct | `/data/LLM_models/Meta-Llama-3-8B-Instruct` | 画像生成 & 精炼 |
| Meta-Llama-3-8B-Instruct-bluesky-sft-289 | `/data/LLM_models/Meta-Llama-3-8B-Instruct-bluesky-sft-289` | 动作预测（决策 + 内容） |
| all-mpnet-base-v2 | `/data/LLM_models/sentence-transformers/all-mpnet-base-v2` | 语义相似度评分（自动加载，无需部署） |

### 方式一：vLLM 部署（推荐）

vLLM 单实例只支持一个模型，所以需要在**不同端口**启动两个实例。
两个 8B 模型各需约 16GB 显存（bfloat16），总计约 32GB，一张 A100-80GB 可同时运行。

```bash
# 终端 1：启动画像生成模型（端口 8000，GPU 0）
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
  --model /data/LLM_models/Meta-Llama-3-8B-Instruct \
  --served-model-name Meta-Llama-3-8B-Instruct \
  --port 8000 \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.45

# 终端 2：启动动作预测模型（端口 8001，GPU 0）
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
  --model /data/LLM_models/Meta-Llama-3-8B-Instruct-bluesky-sft-289 \
  --served-model-name Meta-Llama-3-8B-Instruct-bluesky-sft-289 \
  --port 8001 \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.45

--远程服务器上启动了
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server   --model ./Meta-Llama-3-8B-Instruct-bluesky-sft-289   --served-model-name Meta-Llama-3-8B-Instruct-bluesky-sft-289   --port 8001   --dtype bfloat16   --max-model-len 2048   --gpu-memory-utilization 0.35

```

> **多 GPU 环境**：如果有多张 GPU，可以分别指定 `CUDA_VISIBLE_DEVICES=0` 和 `CUDA_VISIBLE_DEVICES=1`，
> 并将 `--gpu-memory-utilization` 调高到 `0.9`。

验证服务是否正常：

```bash
# 检查画像生成模型
curl http://localhost:8000/v1/models

# 检查动作预测模型
curl http://localhost:8001/v1/models
```

### 方式二：本地 transformers 加载

不需要额外启动服务，Pipeline 会自动用 `transformers` 库加载模型到 GPU。
需要 GPU 显存至少 40GB（同时加载两个 8B 模型）。

### 切换模式

在 `src/config.py` 中修改：

```python
# 使用 vLLM API 模式
USE_VLLM_API = True

# 使用本地 transformers 加载（默认）
USE_VLLM_API = False
```


```bash
python -m src.dpo_pipeline --input output/windowed/test/community_5.jsonl  --output-dir output/dpo   --max-users 1 --debug


```

> 测试模式的 API 配置在 `src/config.py` 的 `TEST_*` 系列变量中，可按需修改。

---

## 完整运行流程

### Step 0：安装依赖

```bash
pip install torch transformers sentence-transformers psycopg2-binary openai
# 如果使用 vLLM 部署
pip install vllm
```

### Step 1：数据切分

```bash
DB_HOST=127.0.0.1 python process_dataset/community_data_splitter.py \
  --train-communities 0,1,3,4,5 \
  --eval-communities 6,7 \
  --train-ratio 0.7 \
  --eval-users 500 \
  --output-dir data
```

### Step 2：窗口切分

```bash
# 对所有训练数据做窗口切分
for f in data/train/community_*.jsonl; do
  out="output/windowed/train/$(basename $f)"
  python -m src.window_splitter --input "$f" --output "$out"
done

# 对测试数据做窗口切分
for f in data/test/community_*.jsonl; do
  out="output/windowed/test/$(basename $f)"
  python -m src.window_splitter --input "$f" --output "$out"
done
```

### Step 3：启动模型服务（仅 vLLM 模式）

```bash
# 按照上面「方式一」在两个终端分别启动两个 vLLM 实例
# 然后修改 src/config.py 中 USE_VLLM_API = True
```

### Step 4：运行 DPO Pipeline

```bash
# 单个社区（调试）
python -m src.dpo_pipeline \
  --input output/windowed/test/community_5.jsonl \
  --output-dir output/dpo \
  --max-users 2

# 批量处理所有训练社区
for f in output/windowed/train/community_*.jsonl; do
  python -m src.dpo_pipeline --input "$f" --output-dir output/dpo
done
```
# Clasp 项目说明

这是一个用于社交行为数据处理与训练数据构造的项目。

当前已包含的核心能力：
- `process_dataset/sft_data_generator.py`：基于用户行为序列生成 SFT 样本。
- `process_dataset/community_data_splitter.py`：按社区切分用户，导出数据文件。
- `src/`：DPO 对构造与画像生成模型训练的完整 pipeline。

## 目录说明

- `src/`：DPO pipeline 核心代码
- `process_dataset/`：数据处理脚本
- `data/`：导出后的训练/测试/评估数据
- `comparison/`：对比实验相关代码
- `scripts/`：辅助脚本
- `saves/`：模型或中间结果保存目录

## 社区切分与导出

脚本：`process_dataset/community_data_splitter.py`

功能：
1. 从社区 `0,1,3,4,5` 中按每个社区 70%/30% 划分训练和测试用户；
2. 从社区 `6,7` 中抽取 500 个未见用户用于评估；
3. 将用户及其序列动作导出到 `data` 目录下，按社区分文件。

### 运行命令

```bash
python process_dataset/community_data_splitter.py \
  --train-communities 0,1,3,4,5 \
  --eval-communities 6,7 \
  --train-ratio 0.7 \
  --eval-users 500 \
  --output-dir data
```

### 输出结构

```text
data/
├── train/community_0.jsonl ... community_5.jsonl
├── test/community_0.jsonl ... community_5.jsonl
├── eval_unseen/community_6.jsonl, community_7.jsonl
└── split_summary.json
```

### 字段说明

- `community_id`：用户所属社区
- `user_id`：用户 ID
- `actions`：动作序列（时间顺序），`post` 动作的 `target` 为 `null`
- `action_count`：序列长度

---

## DPO 对构造 - 画像生成模型训练

### 代码结构

```text
src/
├── __init__.py
├── config.py               # 全局配置（模型路径、窗口参数、评分权重、DPO 阈值）
├── prompts.py              # 所有提示词模板（画像生成/精炼、动作预测）
├── window_splitter.py      # 窗口切分器：T=10, W0~W4
├── scorer.py               # 评分：F(S) 加权F1 + L(S) 语义对齐 + Q(S) 综合
├── action_predictor.py     # 动作预测：决策类 + 内容生成类
├── profile_generator.py    # 画像生成 & 精炼（N=15 候选）
└── dpo_pipeline.py         # DPO 全流程编排（主入口）
```

## DPO 联合损失函数训练
对于DPO微调，温度系数β设置为0.2，注意dpo微调对 除了好画像y1，坏画像y2，还需要记录输入上下文x（旧人格+真实/预测差异）
为了避免 DPO微调 只管偏好，不管正确性，所以增加 SFT损失函数，强制模型保持高质量的persona画像
L = L（DPO）+ a·L（SFT）
a = 0.1 


### Pipeline 流程

1. **窗口切分**：将用户动作按 T=10 切为 W0~W4（共 50 条），不足 50 条的用户跳过
2. **初始画像 S0**：用 W0 动作 + `profile_generation_model_raw` 生成
3. **Baseline 评分**：用 S0 在 W0/W1/W2 上预测动作，计算 Q(S0)（W0 为**整窗**共 T 条：空历史起逐步预测，每步用真实动作推进历史；W1=用 W0 作历史预测 W1；W2=用 W1 作历史预测 W2）
4. **偏差信号**：对比 W1 预测与真实，构造 behavior_discrepancies
5. **候选画像**：基于偏差信号用 `profile_generation_model_raw` 生成 N=15 个精炼画像
6. **候选评分**：每个候选画像在 W0/W1/W2 上评分，计算 r(all) = r(pre)+r(cur)+r(fut)
7. **DPO 对**：r > τ⁺=0.05 为正，r < τ⁻=-0.05 为负，且正-负 > δ=0.2

### 评分公式

- **F(S)**：交互决策加权 F1（post=0.35, reply=0.30, repost=0.20, like=0.15）
- **L(S)**：内容语义对齐度（sentence-transformers 余弦相似度均值）
- **Q(S)** = α·F(S) + (1-α)·L(S)，α = 0.4

### 运行命令

```bash
# Step 1: 窗口切分（以 test/community_5.jsonl 为例）
python -m src.window_splitter \
  --input data/test/community_5.jsonl \
  --output output/windowed/test/community_5.jsonl

# Step 2: 运行 DPO Pipeline
python -m src.dpo_pipeline \
  --input output/windowed/test/community_5.jsonl \
  --output-dir output/dpo \
  --max-users 2   # 调试时限制用户数

# 调试：1）Step3 在终端打印「行为偏差全文」（含 reply 的 Replied-to original）2）各候选「画像精炼」的 LLM 块
# 不打印「每次动作预测」的 API 级调试（避免刷屏）。若也要看 action 的 prompt：再加 --debug-actions
python -m src.dpo_pipeline \
  --input output/windowed/test/community_5.jsonl \
  --output-dir output/dpo \
  --max-users 1 \
  --debug
# python -m src.dpo_pipeline ... --debug --debug-actions   # 另打印每次决策/内容预测的 [LLM-DEBUG]

# 换一批测试用户：先按种子打乱 jsonl 中用户顺序，再只跑前 1 个（同一文件多试几个 seed）
python -m src.dpo_pipeline \
  --input output/windowed/test/community_5.jsonl \
  --output-dir output/dpo \
  --max-users 1 \
  --seed 42
```

也可在 `src/config.py` 中设置 `DEBUG_LLM = True`（或调整 `DEBUG_LLM_MAX_*` 截断长度）。摘要文件 `dpo_summary_*.json` 的 `config.debug_llm` 会记录是否开启调试。

### 输出文件

```text
output/
├── windowed/                              # 窗口切分结果
│   ├── train/community_*.jsonl
│   └── test/community_*.jsonl
└── dpo/                                   # DPO 对构造结果
    ├── dpo_pairs_community_5.jsonl        # DPO 正负对（每行一个 pair）
    ├── dpo_detail_community_5.json        # 每用户的完整评分细节
    └── dpo_summary_community_5.json       # 统计摘要
```

### 配置说明（src/config.py）

| 参数 | 默认值 | 说明 |
|---|---|---|
| `WINDOW_SIZE` | 10 | 每窗口动作数 |
| `NUM_WINDOWS` | 5 | W0~W4 |
| `NUM_CANDIDATE_PROFILES` | 15 | 每轮精炼候选画像数 |
| `ALPHA` | 0.4 | Q(S) 中 F(S) 的权重 |
| `TAU_PLUS` | 0.05 | 正向 DPO 阈值 |
| `TAU_MINUS` | -0.05 | 负向 DPO 阈值 |
| `DELTA` | 0.20 | 正负对最小差距 |
| `USE_VLLM_API` | True | True 时用 vLLM；False 时仅当传入非 None 的 model/tokenizer 才走本地 transformers |
| `DEBUG_LLM` | False | True 或 `--debug`：打印 Step3 **偏差全文** + **画像精炼** LLM 块；不打印动作 API 块 |
| `DEBUG_LLM_INCLUDE_ACTIONS` | False | True 或 `--debug-actions`（需同时 `--debug`）：再打印每次动作预测的 `[LLM-DEBUG]` |
| `DEBUG_LLM_*` | 见 config | 精炼调试时重点展示行为误差与节选；长输出头尾由 `PROFILE_OUTPUT_*` / `DISCREPANCY_MAX` 等控制 |

---

## 模型部署 & 启动命令

项目涉及两个 LLM 模型和一个 Sentence Transformer：

| 模型 | 路径 | 用途 |
|---|---|---|
| Meta-Llama-3-8B-Instruct | `/data/LLM_models/Meta-Llama-3-8B-Instruct` | 画像生成 & 精炼 |
| Meta-Llama-3-8B-Instruct-bluesky-sft-289 | `/data/LLM_models/Meta-Llama-3-8B-Instruct-bluesky-sft-289` | 动作预测（决策 + 内容） |
| all-mpnet-base-v2 | `/data/LLM_models/sentence-transformers/all-mpnet-base-v2` | 语义相似度评分（自动加载，无需部署） |

### 方式一：vLLM 部署（推荐）

vLLM 单实例只支持一个模型，所以需要在**不同端口**启动两个实例。
两个 8B 模型各需约 16GB 显存（bfloat16），总计约 32GB，一张 A100-80GB 可同时运行。

```bash
# 终端 1：启动画像生成模型（端口 8000，GPU 0）
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
  --model /data/LLM_models/Meta-Llama-3-8B-Instruct \
  --served-model-name Meta-Llama-3-8B-Instruct \
  --port 8000 \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.45

# 终端 2：启动动作预测模型（端口 8001，GPU 0）
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
  --model /data/LLM_models/Meta-Llama-3-8B-Instruct-bluesky-sft-289 \
  --served-model-name Meta-Llama-3-8B-Instruct-bluesky-sft-289 \
  --port 8001 \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.45

--远程服务器上启动了
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server   --model ./Meta-Llama-3-8B-Instruct-bluesky-sft-289   --served-model-name Meta-Llama-3-8B-Instruct-bluesky-sft-289   --port 8001   --dtype bfloat16   --max-model-len 2048   --gpu-memory-utilization 0.35

```

> **多 GPU 环境**：如果有多张 GPU，可以分别指定 `CUDA_VISIBLE_DEVICES=0` 和 `CUDA_VISIBLE_DEVICES=1`，
> 并将 `--gpu-memory-utilization` 调高到 `0.9`。

验证服务是否正常：

```bash
# 检查画像生成模型
curl http://localhost:8000/v1/models

# 检查动作预测模型
curl http://localhost:8001/v1/models
```

### 方式二：本地 transformers 加载

不需要额外启动服务，Pipeline 会自动用 `transformers` 库加载模型到 GPU。
需要 GPU 显存至少 40GB（同时加载两个 8B 模型）。

### 切换模式

在 `src/config.py` 中修改：

```python
# 使用 vLLM API 模式
USE_VLLM_API = True

# 使用本地 transformers 加载（默认）
USE_VLLM_API = False
```


```bash
python -m src.dpo_pipeline --input output/windowed/test/community_5.jsonl  --output-dir output/dpo   --max-users 1 --debug


```

> 测试模式的 API 配置在 `src/config.py` 的 `TEST_*` 系列变量中，可按需修改。

---

## 完整运行流程

### Step 0：安装依赖

```bash
pip install torch transformers sentence-transformers psycopg2-binary openai
# 如果使用 vLLM 部署
pip install vllm
```

### Step 1：数据切分

```bash
DB_HOST=127.0.0.1 python process_dataset/community_data_splitter.py \
  --train-communities 0,1,3,4,5 \
  --eval-communities 6,7 \
  --train-ratio 0.7 \
  --eval-users 500 \
  --output-dir data
```

### Step 2：窗口切分

```bash
# 对所有训练数据做窗口切分
for f in data/train/community_*.jsonl; do
  out="output/windowed/train/$(basename $f)"
  python -m src.window_splitter --input "$f" --output "$out"
done

# 对测试数据做窗口切分
for f in data/test/community_*.jsonl; do
  out="output/windowed/test/$(basename $f)"
  python -m src.window_splitter --input "$f" --output "$out"
done
```

### Step 3：启动模型服务（仅 vLLM 模式）

```bash
# 按照上面「方式一」在两个终端分别启动两个 vLLM 实例
# 然后修改 src/config.py 中 USE_VLLM_API = True
```

### Step 4：运行 DPO Pipeline

```bash
# 单个社区（调试）
python -m src.dpo_pipeline \
  --input output/windowed/test/community_5.jsonl \
  --output-dir output/dpo \
  --max-users 2

# 批量处理所有训练社区
for f in output/windowed/train/community_*.jsonl; do
  python -m src.dpo_pipeline --input "$f" --output-dir output/dpo
done
```
