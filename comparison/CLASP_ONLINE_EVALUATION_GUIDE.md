# Clasp_online 评估逻辑与大规模测试指南

## 一、Clasp_online 评估逻辑详解

### 1.1 核心思想

**Clasp_online** 是一种基于**预测误差驱动**的在线画像精炼方法，通过持续观察用户行为与预测的差异来动态更新用户画像。

### 1.2 评估流程（窗口链协议）

#### 时间窗口设置
- **窗口大小 T**: 10 条动作/窗口
- **评估窗口数**: 6 个窗口（W0, W1, W2, W3, W4, W5）
- **总动作数**: 60 条（6 × 10）

#### 评估步骤（5步前向预测链）

```
初始化：W0 → 生成初始画像 S0

Step 0: 用 S0 + 历史W0 → 预测 W1 → 计算 F/L/Q
        ├─ 预测误差 → 精炼画像 → 候选画像集
        ├─ 用候选画像 + 历史W1 → 预测 W2（验证集）
        └─ 选择 Q 值最高的候选 → 更新为 S1

Step 1: 用 S1 + 历史W1 → 预测 W2 → 计算 F/L/Q
        ├─ 预测误差 → 精炼画像 → 候选画像集
        ├─ 用候选画像 + 历史W2 → 预测 W3（验证集）
        └─ 选择 Q 值最高的候选 → 更新为 S2

Step 2: 用 S2 + 历史W2 → 预测 W3 → 计算 F/L/Q
        ├─ 预测误差 → 精炼画像 → 候选画像集
        ├─ 用候选画像 + 历史W3 → 预测 W4（验证集）
        └─ 选择 Q 值最高的候选 → 更新为 S3

Step 3: 用 S3 + 历史W3 → 预测 W4 → 计算 F/L/Q
        ├─ 预测误差 → 精炼画像 → 候选画像集
        ├─ 用候选画像 + 历史W4 → 预测 W5（验证集）
        └─ 选择 Q 值最高的候选 → 更新为 S4

Step 4: 用 S4 + 历史W4 → 预测 W5 → 计算 F/L/Q
        （最后一步，不再更新画像）
```

#### 关键修复（2026-05-06）

**问题**: 原代码用当前窗口（W_t → W_{t+1}）评估候选画像，导致过拟合
**修复**: 现在用下一步窗口（W_{t+1} → W_{t+2}）评估候选画像，真实反映泛化能力

### 1.3 评分指标

#### F(S) - 交互决策加权 F1
```
权重分配:
- post:   0.35
- reply:  0.30
- repost: 0.20
- like:   0.15
```

#### L(S) - 内容语义对齐度
- 使用 Sentence-Transformer (all-mpnet-base-v2)
- 计算预测文本与真实文本的余弦相似度
- 仅对生成类动作（post/reply）计算

#### Q(S) - 综合得分
```
Q(S) = α × F(S) + (1-α) × L(S)
α = 0.6  (当前配置)
```

### 1.4 画像精炼机制

1. **构建误差信号**: 对比预测动作与真实动作，生成 behavior_discrepancies
2. **生成候选画像**: 基于误差信号生成 N 个候选画像（默认 N=10）
3. **候选评估**: 在下一步窗口上评估每个候选的 Q 值
4. **选择最优**: 选择 Q 值最高的候选作为新画像

## 二、测试集合说明

### 2.1 数据集划分

项目包含 **3 个测试集合**，来自 8 个社区（0,1,3,4,5,6,7）：

#### 1. 训练集 (train/)
- **来源社区**: 0, 1, 3, 4, 5
- **划分比例**: 每个社区的 70% 用户
- **总用户数**: 3,252 个用户
- **总动作数**: 673,713 条动作
- **用途**: DPO 对构造、模型训练

**各社区详情**:
```
community_0: 1,192 用户, 244,095 动作
community_1:   408 用户,  81,124 动作
community_3:   615 用户, 131,899 动作
community_4:   908 用户, 190,407 动作
community_5:   129 用户,  26,188 动作
```

#### 2. 测试集 (test/)
- **来源社区**: 0, 1, 3, 4, 5（与训练集同社区）
- **划分比例**: 每个社区的 30% 用户
- **总用户数**: 1,397 个用户
- **总动作数**: 286,003 条动作
- **用途**: 同社区泛化能力评估、基线对比

**各社区详情**:
```
community_0: 511 用户, 103,063 动作
community_1: 176 用户,  34,293 动作
community_3: 264 用户,  57,474 动作
community_4: 390 用户,  79,850 动作
community_5:  56 用户,  11,323 动作
```

#### 3. 未见社区评估集 (eval_unseen/)
- **来源社区**: 6, 7（训练时完全未见）
- **用户数**: 500 个用户（从两个社区随机抽取）
- **总动作数**: 104,735 条动作
- **用途**: 跨社区泛化能力评估

**各社区详情**:
```
community_6: 279 用户, 56,320 动作
community_7: 221 用户, 48,415 动作
```

### 2.2 数据格式

每个 JSONL 文件，每行一个用户：

```json
{
  "community_id": 0,
  "user_id": 539166,
  "actions": [
    {
      "timestamp": "2023-09-01 03:47",
      "action_type": "repost",
      "target": "原文内容...",
      "action_text": null,
      "date": "202309010347"
    },
    ...
  ]
}
```

**字段说明**:
- `action_type`: post, reply, repost, like
- `target`: 被回复/转发/点赞的原文（post 时为 null）
- `action_text`: 用户生成的文本（post/reply 时有内容）

### 2.3 测试集选择建议

#### 快速验证（调试）
```bash
# 使用最小的社区 5
--input-jsonl data/test/community_5.jsonl
--max-users 10
```

#### 同社区泛化测试（标准）
```bash
# 使用完整测试集
--split test
--windowed-root output/windowed_eval_chain
```

#### 跨社区泛化测试（挑战）
```bash
# 使用未见社区
--split eval_unseen
--windowed-root output/windowed_eval_chain
```

## 三、模型配置说明

### 3.1 涉及的模型

项目需要 **3 个模型**：

| 模型 | 用途 | 部署方式 | 显存需求 |
|------|------|---------|---------|
| **Meta-Llama-3-8B-Instruct** | 画像生成与精炼 | vLLM API (端口 8000) | ~16GB |
| **Meta-Llama-3-8B-Instruct-bluesky-sft** | 动作预测 | vLLM API (端口 8001) | ~16GB |
| **all-mpnet-base-v2** | 语义相似度评分 | 自动加载 | ~1GB |

**总显存需求**: 约 33GB（推荐 A100-40GB 或 A100-80GB）

### 3.2 当前配置（src/config.py）

#### 模型路径
```python
# 本地模型路径（transformers 模式）
ACTION_GENERATION_MODEL = "/data/LLM_models/Meta-Llama-3-8B-Instruct-bluesky"
PROFILE_GENERATION_MODEL_RAW = "/data/LLM_models/Meta-Llama-3-8B-Instruct"
SENTENCE_TRANSFORMER_MODEL = "/data/LLM_models/sentence-transformers/all-mpnet-base-v2"
```

#### vLLM API 配置
```python
USE_VLLM_API = True  # 使用 vLLM 模式

# 画像生成 API
PROFILE_API_BASE = "http://175.6.27.230:8001/v1"
PROFILE_API_MODEL = "Meta-Llama-3-8B-Instruct"

# 动作预测 API
ACTION_API_BASE = "http://localhost:8002/v1"
ACTION_API_MODEL = "Meta-Llama-3-8B-Instruct-bluesky-sft"
```

#### 商用画像模型（可选）
```python
ENABLE_COMMERCIAL_PROFILE = True
COMMERCIAL_PROFILE_RATIO = 0.4  # 40% 候选由商用模型生成
PROFILE_MODEL = "gpt-4o-mini"
OPENAI_API_KEY = "sk-..."
OPENAI_BASE_URL = "https://api.huiyan-ai.cn/v1"
```

### 3.3 关键参数配置

#### 窗口参数
```python
WINDOW_SIZE = 10                # 每窗口动作数
NUM_WINDOWS = 5                 # 训练/DPO: W0~W4
NUM_WINDOWS_EVAL_CHAIN = 6      # 评估链: W0~W5
MIN_ACTIONS = 50                # 用户最少动作数
```

#### 评分权重
```python
ACTION_WEIGHTS = {
    "post":   0.35,
    "reply":  0.30,
    "repost": 0.20,
    "like":   0.15,
}
ALPHA = 0.6  # Q(S) = 0.6×F + 0.4×L
```

#### 候选画像生成
```python
NUM_CANDIDATE_PROFILES = 10     # 每轮生成候选数
TEMPERATURE_PROFILE = 0.8       # 画像生成温度（多样性）
TEMPERATURE_ACTION = 0          # 动作预测温度（确定性）
```

#### 上下文长度限制（避免 4k 上下文溢出）
```python
ACTION_API_MAX_CONTEXT_TOKENS = 4096
PROFILE_BEHAVIOR_TEXT_MAX_CHARS = 6000
ACTION_PROMPT_HISTORY_MAX_CHARS = 6000
ACTION_PROMPT_PROFILE_MAX_CHARS = 3500
PROFILE_REFINEMENT_OLD_PERSONA_MAX_CHARS = 3500
PROFILE_REFINEMENT_DISCREPANCY_MAX_CHARS = 3500
```

#### 并发参数
```python
DPO_WORKERS = 10                # 候选画像评估线程数
DPO_USER_PROCESSES = 5          # 多用户并行进程数
DPO_SCORER_DEVICE = None        # 语义分设备（None=CPU，避免显存竞争）
```

## 四、大规模测试运行指南

### 4.1 环境准备

#### Step 1: 启动 vLLM 服务

**画像生成模型（端口 8000）**:
```bash
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
  --model /data/LLM_models/Meta-Llama-3-8B-Instruct \
  --served-model-name Meta-Llama-3-8B-Instruct \
  --port 8000 \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.45
```

**动作预测模型（端口 8001）**:
```bash
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
  --model /data/LLM_models/Meta-Llama-3-8B-Instruct-bluesky-sft-289 \
  --served-model-name Meta-Llama-3-8B-Instruct-bluesky-sft-289 \
  --port 8001 \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.45
```

**验证服务**:
```bash
curl http://localhost:8000/v1/models
curl http://localhost:8001/v1/models
```

#### Step 2: 检查配置

确认 `src/config.py` 中：
```python
USE_VLLM_API = True
PROFILE_API_BASE = "http://localhost:8000/v1"  # 或远程地址
ACTION_API_BASE = "http://localhost:8001/v1"
```

### 4.2 测试场景与命令

#### 场景 1: 快速验证（单社区，少量用户）

**目的**: 验证代码和配置正确性

```bash
# 测试 community_5（最小社区）的前 10 个用户
python -m comparison.run_baseline_comparison \
  --split test \
  --skip-window-split \
  --windowed-root output/windowed_eval_chain \
  --methods clasp_online \
  --max-users 10 \
  --input-jsonl data/test/community_5.jsonl \
  --comparison-root output/comparison \
  --scorer-device cpu
```

**预期时间**: 约 10-20 分钟（取决于 API 速度）

#### 场景 2: 单社区完整测试

**目的**: 评估单个社区的完整性能

```bash
# 测试 community_0 的所有用户
python -m comparison.run_baseline_comparison \
  --split test \
  --windowed-root output/windowed_eval_chain \
  --methods clasp_online \
  --input-jsonl output/windowed_eval_chain/test/community_0.jsonl \
  --comparison-root output/comparison \
  --scorer-device cpu \
  --plot community_0_clasp.png
```

**预期时间**: 约 2-4 小时（511 个用户）

#### 场景 3: 测试集完整评估（推荐）

**目的**: 在所有测试社区上评估 clasp_online

```bash
# 自动处理 test/ 下所有社区
python -m comparison.run_baseline_comparison \
  --split test \
  --windowed-root output/windowed_eval_chain \
  --methods clasp_online \
  --comparison-root output/comparison \
  --scorer-device cpu \
  --skip-window-split
```

**注意**: 需要先运行窗口切分（见下方）

**预期时间**: 约 8-12 小时（1,397 个用户）

#### 场景 4: 多基线对比测试

**目的**: 对比 clasp_online 与其他基线方法

```bash
# 同时运行 6 种方法
python -m comparison.run_baseline_comparison \
  --split test \
  --windowed-root output/windowed_eval_chain \
  --methods static_s0,prefix_refresh,clasp_online,incremental_persona,s0_sliding_history,user_full_history \
  --comparison-root output/comparison \
  --scorer-device cpu \
  --skip-window-split \
  --plot baseline_comparison.png
```

**预期时间**: 约 48-72 小时（6 种方法 × 1,397 个用户）

#### 场景 5: 未见社区泛化测试

**目的**: 评估跨社区泛化能力

```bash
# 在未见社区 6, 7 上测试
python -m comparison.run_baseline_comparison \
  --split eval_unseen \
  --windowed-root output/windowed_eval_chain \
  --methods clasp_online \
  --comparison-root output/comparison \
  --scorer-device cpu \
  --skip-window-split
```

**预期时间**: 约 3-5 小时（500 个用户）

### 4.3 窗口切分（预处理）

在运行评估前，需要先对数据进行窗口切分：

```bash
# 切分测试集（6 个窗口）
python -m src.window_splitter \
  --input data \
  --output output/windowed_eval_chain \
  --split test \
  --num-windows 6

# 切分未见社区评估集
python -m src.window_splitter \
  --input data \
  --output output/windowed_eval_chain \
  --split eval_unseen \
  --num-windows 6
```

**输出结构**:
```
output/windowed_eval_chain/
├── test/
│   ├── community_0.jsonl
│   ├── community_1.jsonl
│   ├── community_3.jsonl
│   ├── community_4.jsonl
│   └── community_5.jsonl
└── eval_unseen/
    ├── community_6.jsonl
    └── community_7.jsonl
```

### 4.4 输出文件说明

#### 评估结果文件

```
output/comparison/
├── clasp_online/
│   ├── baseline_chain_test.jsonl          # 每行一个用户的评估结果
│   ├── baseline_chain_test_F.png          # F 指标折线图
│   ├── baseline_chain_test_L.png          # L 指标折线图
│   └── baseline_chain_test_Q.png          # Q 指标折线图
├── static_s0/
│   └── ...
└── prefix_refresh/
    └── ...
```

#### JSONL 结果格式

```json
{
  "user_id": 539166,
  "community_id": 0,
  "method": "clasp_online",
  "window_keys": ["W0", "W1", "W2", "W3", "W4", "W5"],
  "refinement_variants": 1,
  "always_accept_refinement": false,
  "steps": [
    {
      "step_index": 0,
      "history_window": "W0",
      "target_window": "W1",
      "F": 0.6234,
      "L": 0.4521,
      "Q": 0.5547
    },
    ...
  ],
  "mean_Q": 0.5623,
  "mean_F": 0.6145,
  "mean_Q_chain": 0.5623,
  "mean_F_chain": 0.6145
}
```

### 4.5 性能优化建议

#### 1. 并行处理
```python
# 在 src/config.py 中调整
DPO_WORKERS = 10              # 候选画像评估线程数（推荐 5-15）
DPO_USER_PROCESSES = 5        # 多用户并行进程数（推荐 3-8）
```

#### 2. 语义分设备选择
```bash
# CPU 模式（避免显存竞争，推荐）
--scorer-device cpu

# GPU 模式（显存充足时可加速）
--scorer-device cuda
```

#### 3. 减少候选数（快速测试）
```python
# 在 src/config.py 中
NUM_CANDIDATE_PROFILES = 5    # 默认 10，可降至 5 加速
```

## 五、常见问题与解决方案

### 5.1 模型相关

**Q: vLLM 服务启动失败，提示显存不足**
```bash
# 降低显存占用
--gpu-memory-utilization 0.35  # 默认 0.45

# 或减少上下文长度
--max-model-len 2048  # 默认 4096
```

**Q: API 调用超时或 4xx 错误**
- 检查 API 地址和端口是否正确
- 检查模型名称是否匹配
- 查看 vLLM 服务日志排查错误

### 5.2 评估相关

**Q: 用户被跳过（动作数不足）**
- 窗口链需要至少 60 条动作（6 × 10）
- 检查原始数据中用户的动作数量

**Q: 评估速度太慢**
- 增加 `DPO_WORKERS` 和 `DPO_USER_PROCESSES`
- 使用 `--max-users` 限制用户数进行测试
- 考虑使用更快的 GPU 或多 GPU 部署

**Q: 内存溢出（OOM）**
- 降低 `DPO_USER_PROCESSES`（减少并行进程）
- 使用 `DPO_SCORER_DEVICE = None`（语义分用 CPU）
- 减少 `NUM_CANDIDATE_PROFILES`

### 5.3 结果分析

**Q: 如何查看汇总统计？**
```bash
# 运行结束后会自动打印汇总
# 或使用绘图工具重新分析
python -m comparison.plot_chain_from_jsonl \
  output/comparison/clasp_online/baseline_chain_test.jsonl \
  --plot output/comparison/clasp_online/replot.png
```

**Q: 如何对比多个方法？**
- 每个方法的结果在独立目录下
- 可以编写脚本读取各方法的 JSONL 文件进行对比
- 或使用 `--combined-jsonl` 模式合并输出

## 六、推荐测试流程

### 阶段 1: 验证（1-2 小时）
```bash
# 1. 小规模测试
python -m comparison.run_baseline_comparison \
  --input-jsonl data/test/community_5.jsonl \
  --methods clasp_online \
  --max-users 5 \
  --comparison-root output/comparison_debug

# 2. 检查输出和日志
# 3. 确认配置正确
```

### 阶段 2: 单方法完整测试（8-12 小时）
```bash
# 在测试集上运行 clasp_online
python -m comparison.run_baseline_comparison \
  --split test \
  --windowed-root output/windowed_eval_chain \
  --methods clasp_online \
  --comparison-root output/comparison \
  --scorer-device cpu \
  --skip-window-split
```

### 阶段 3: 多基线对比（2-3 天）
```bash
# 运行所有 6 种方法
python -m comparison.run_baseline_comparison \
  --split test \
  --windowed-root output/windowed_eval_chain \
  --methods static_s0,prefix_refresh,clasp_online,incremental_persona,s0_sliding_history,user_full_history \
  --comparison-root output/comparison \
  --scorer-device cpu \
  --skip-window-split
```

### 阶段 4: 泛化能力测试（3-5 小时）
```bash
# 在未见社区上测试
python -m comparison.run_baseline_comparison \
  --split eval_unseen \
  --windowed-root output/windowed_eval_chain \
  --methods clasp_online \
  --comparison-root output/comparison \
  --scorer-device cpu \
  --skip-window-split
```

## 七、参考资料

- **代码实现**: `comparison/window_chain_eval.py`
- **运行入口**: `comparison/run_baseline_comparison.py`
- **配置文件**: `src/config.py`
- **审查报告**: `comparison/code_review_report.md`
- **项目说明**: `readme.md`











