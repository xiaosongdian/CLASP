# PolicySim（Agent 训练精简版）

这个仓库已按你的要求重构为：**只保留 Agent 与训练相关代码**。  
不再包含干预策略、推荐优化、仿真环境等模块说明。

## 当前目标

- 仅做 `Agent` 行为建模
- 仅做 `SFT + DPO` 训练流水线
- 提供最小可运行示例（单 Agent 调用）

## 目录（只看这个）

```text
policysim/
├── __init__.py
├── agent.py                    # Agent 主类（动作生成、记忆更新、JSON解析）
├── config.py                   # 模型与训练配置
├── memory.py                   # Agent 记忆结构
├── types.py                    # 动作与消息类型定义
├── llm/
│   ├── base.py                 # LLM 抽象接口
│   └── openai_chat.py          # OpenAI 兼容模型调用
├── prompt/
│   └── agent_prompt.py         # Agent 决策 Prompt
├── train/
│   ├── build_sft_data.py       # 构建 SFT 数据
│   ├── train_sft.py            # SFT 训练
│   ├── build_dpo_data.py       # 构建 DPO 偏好数据
│   └── train_dpo.py            # DPO 训练
├── examples/
│   └── run_agent_demo.py       # Agent 推理示例
└── requirements.txt            # 精简依赖
```

## 快速开始

### 1) 安装

```bash
pip install -r policysim/requirements.txt
```

### 2) 配置 API（如使用在线模型）

```bash
export POLICYSIM_API_KEY="你的Key"
export POLICYSIM_BASE_URL="https://open.bigmodel.cn/api/paas/v4/"
export POLICYSIM_MODEL_NAME="glm-4-flash"
```

### 3) 跑 Agent 示例

```bash
python -m policysim.examples.run_agent_demo
```

## 训练流程

### SFT 数据格式（原始）

```json
[
  {
    "event": "某个事件",
    "profile": {"likely_identity":"student","posting_style":"direct"},
    "action": "reply",
    "content": "我不同意这个观点"
  }
]
```

### 1) 构建 SFT 数据

```bash
python -m policysim.train.build_sft_data --source data/raw_sft.json --output data/sft.jsonl
```

### 2) 训练 SFT

```bash
python -m policysim.train.train_sft \
  --model Qwen/Qwen2.5-3B-Instruct \
  --data data/sft.jsonl \
  --output outputs/sft_model
```

### DPO 数据格式（原始）

```json
[
  {
    "event": "某个事件",
    "profile": {"likely_identity":"student"},
    "chosen": {"action":"reply","content":"更合理的输出"},
    "rejected": {"action":"reply","content":"较差的输出"}
  }
]
```

### 3) 构建 DPO 数据

```bash
python -m policysim.train.build_dpo_data --source data/raw_dpo.json --output data/dpo.jsonl
```

### 4) 训练 DPO

```bash
python -m policysim.train.train_dpo \
  --model outputs/sft_model \
  --data data/dpo.jsonl \
  --output outputs/dpo_model
```

## 说明

- 当前版本是 **Agent + 训练专用结构**。
- 你后续如果要，我可以继续帮你做两件事：
  1. 把旧的非 Agent 目录彻底清理掉（物理删除）  
  2. 把训练脚本升级为 LoRA/QLoRA 版本（更省显存）

