#!/usr/bin/env python3
"""
全局配置：模型路径、窗口参数、评分权重、DPO 阈值
"""
import os

# ============================================================================
# 模型路径（本地 transformers 加载时使用）
# ============================================================================
ACTION_GENERATION_MODEL = "/data/LLM_models/Meta-Llama-3-8B-Instruct-bluesky"
PROFILE_GENERATION_MODEL_RAW = "/data/LLM_models/Meta-Llama-3-8B-Instruct"
SENTENCE_TRANSFORMER_MODEL = "/data/LLM_models/sentence-transformers/all-mpnet-base-v2"

# ============================================================================
# vLLM API 配置
# 使用预启动的 vLLM 服务时保持 True；与 dpo_pipeline 仅载 Sentence-Transformer 一致
# ============================================================================
USE_VLLM_API = True

PROFILE_API_BASE = "http://175.6.27.230:8001/v1"
PROFILE_API_MODEL = "Meta-Llama-3-8B-Instruct"

# 商用画像模型（候选画像混合生成）
# 例如 n=10, ratio=0.4 => 4 个候选由商用模型生成
ENABLE_COMMERCIAL_PROFILE = True
COMMERCIAL_PROFILE_RATIO = 0.4
OPENAI_API_KEY = "sk-1JY7edl1HTvrqYHQM8wfFSL72eNuhPJqM6WLJNNbqIciTUAB"
OPENAI_BASE_URL = "https://api.huiyan-ai.cn/v1"
PROFILE_MODEL = "gpt-4o-mini"

ACTION_API_BASE = "http://localhost:8002/v1"
ACTION_API_MODEL = "Meta-Llama-3-8B-Instruct-bluesky-sft"

# ============================================================================
# 测试模式（用单个远程 API 替代所有 LLM，跑通 pipeline 用）
# ============================================================================
TEST_MODE = False

TEST_API_BASE = "https://api.scnet.cn/api/llm/v1"
TEST_API_KEY = "sk-MTA4LTExMTExNjQwOTE0LTE3NzQ1NzkxMDQxNzM="
TEST_API_MODEL = "DeepSeek-R1-Distill-Qwen-7B"
TEST_NUM_CANDIDATES = 3       # 测试时只生成 3 个候选画像（正式 15 个）

# ============================================================================
# 窗口参数
# ============================================================================
WINDOW_SIZE = 10       # 每个窗口的动作数
NUM_WINDOWS = 5        # W0 ~ W4
MIN_ACTIONS = WINDOW_SIZE * NUM_WINDOWS  # 用户至少需要 50 条动作

# ============================================================================
# 动作预测 / SFT 样本构造
# ============================================================================
TEXT_LONG = 500        # 文本截断长度

# ============================================================================
# 评分权重（交互决策 F1）
# ============================================================================
ACTION_WEIGHTS = {
    "post":   0.35,
    "reply":  0.30,
    "repost": 0.20,
    "like":   0.15,
}

# 综合得分 Q(S) = ALPHA * F(S) + (1 - ALPHA) * L(S)
ALPHA = 0.4

# ============================================================================
# DPO 对构造阈值
# ============================================================================
NUM_CANDIDATE_PROFILES = 10   # 每轮精炼候选画像数
TAU_PLUS = 0.05               # 正向阈值
TAU_MINUS = 0             # 负向阈值
DELTA = 0.1                  # 正负对之间最小差距

# ============================================================================
# 模型推理参数
# ============================================================================
MAX_NEW_TOKENS_PROFILE = 2048     # 画像生成最大 token
MAX_NEW_TOKENS_ACTION = 512       # 动作预测最大 token
TEMPERATURE_PROFILE = 0.8         # 画像候选多样性
TEMPERATURE_ACTION = 0.3          # 动作预测倾向确定性

# ============================================================================
# Debug：打印每次 LLM 请求/响应（由 --debug 或 DEBUG_LLM=True 开启）
# ============================================================================
DEBUG_LLM = False
DEBUG_LLM_MAX_INSTRUCTION_CHARS = 1200
# 未使用结构化 focus 时，user 与 output 的通用截断
DEBUG_LLM_MAX_USER_CHARS = 2000
DEBUG_LLM_PRINT_FULL_OUTPUT = True   # 对「动作预测」等：True 为尽量完整；画像类见下方 head/tail
# 画像类：行为历史、误差、原画像、模型输出的头尾长度（重点打印误差，其它节选）
DEBUG_LLM_BEHAVIOR_HEAD = 2500
DEBUG_LLM_BEHAVIOR_TAIL = 1500
DEBUG_LLM_DISCREPANCY_MAX = 50000   # 行为误差可接近全量打印的上限
DEBUG_LLM_OLD_PERSONA_HEAD = 1000
DEBUG_LLM_OLD_PERSONA_TAIL = 800
DEBUG_LLM_PROFILE_OUTPUT_HEAD = 2000
DEBUG_LLM_PROFILE_OUTPUT_TAIL = 1200
# 动作预测：user 里 profile 极长时仅打印摘要
DEBUG_LLM_ACTION_USER_MAX = 4000
# True 时额外打印「每次动作预测」的 LLM-DEBUG（量大）；默认 False，建议用 Step3 偏差全文 + 画像精炼 debug
DEBUG_LLM_INCLUDE_ACTIONS = False

# ============================================================================
# 并发参数（DPO 评估与候选画像生成）
# ============================================================================
DPO_WORKERS = 4
