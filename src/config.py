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
NUM_WINDOWS = 5        # 训练 / DPO：W0 ~ W4（共 5 窗）
# 窗口链评估：W0 建 S0，再 S0→W1 … S4→W5，共 6 窗、6*T 条动作
NUM_WINDOWS_EVAL_CHAIN = 6
MIN_ACTIONS = WINDOW_SIZE * NUM_WINDOWS  # 用户至少需要 50 条动作（训练默认）

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
# 经验（train_copy community_4）：F 多在 0～0.8、std≈0.16；L 多在 0～0.35、std≈0.09。α=0.7 时 Q 几乎由 F 主导；
# α≈0.55～0.62 可在不明显牺牲动作项的前提下让文本相似度 L 对 margin 更有存在感。
ALPHA = 0.58
# 若为 True：先将 L 从 [-1,1] 线性映射到 [0,1] 再算 Q（Q 可能高于原始 F/L）；False 则直接用原始 L
# 当前数据里 L 已多为非负余弦相似度，再做 (L+1)/2 会压窄有效动态范围，一般保持 False。
NORMALIZE_L_TO_UNIT = False

# 作图（baseline 链 F/L/Q）：按用户 mean_Q 去掉最低/最高各该比例后再聚合折线；0=不去极值
PLOT_TRIM_EACH_TAIL = 0.05

# ============================================================================
# DPO 对构造阈值
# ============================================================================
NUM_CANDIDATE_PROFILES = 10   # 每轮精炼候选画像数
# r_all 为三窗 Q 差之和；略抬高 TAU+、负侧略低于 0，可减少「几乎打平」的弱偏好对。
TAU_PLUS = 0.06               # r_all > TAU+ 视为正例候选
TAU_MINUS = -0.02             # r_all < TAU- 视为负例候选（原为 0 时任意负即负例，对更严）
DELTA = 0.05                  # tau_delta：正例与负例 r_all 至少相差 DELTA
ABS_DELTA = DELTA * 2         # abs_delta 规则下 Hi/Lo 最小 |Δr|
# ============================================================================
# 模型推理参数
# ============================================================================
MAX_NEW_TOKENS_PROFILE = 2048     # 画像生成最大 token
MAX_NEW_TOKENS_ACTION = 512       # 动作预测最大 token

# 画像 API 常见 max_context=4096：行为拼接过长会 400。限制送入画像模型的行为正文长度（头尾保留）。
PROFILE_BEHAVIOR_TEXT_MAX_CHARS = 8000
# 窗口链里 s0_sliding_history / user_full_history 拼到动作预测「画像」后的额外块上限（为 scenario 等留 token）。
ACTION_PROMPT_HISTORY_MAX_CHARS = 6000
# 画像精炼 prompt 两段的字符上限（与 PROFILE_BEHAVIOR_TEXT_MAX_CHARS 分开，避免 old+误差一起爆上下文）
PROFILE_REFINEMENT_OLD_PERSONA_MAX_CHARS = 3500
PROFILE_REFINEMENT_DISCREPANCY_MAX_CHARS = 3500
TEMPERATURE_PROFILE = 0.8         # 画像候选多样性
TEMPERATURE_ACTION = 0         # 动作预测倾向确定性

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
DPO_WORKERS = 10
# 并行处理的用户进程数：1=串行；多用户时设 2~5 可显著压缩墙钟时间（子进程内仍用 DPO_WORKERS 线程评候选）
DPO_USER_PROCESSES = 5
# 多用户多进程时，第 i 个用户任务在子进程内会额外等待 i*秒（0 不等待），把各进程真正开始打 API 的时间错开
DPO_USER_PROCESS_STAGGER_SEC = 0.3
# 多进程时 Sentence-Transformer 设备：None 表示多进程时由管道默认用 cpu，单进程为自动 cuda/cpu。
# 若显存足够且需加速语义分，可设为 "cuda"（多进程会各占一份显存，易 OOM）
DPO_SCORER_DEVICE = None
# DPO 滚动轮次（第1轮 S0->候选，选最优作为 S1；第2轮窗口前移继续）
DPO_ROUNDS = 2

# ============================================================================
# DPO 微调（train_profile_dpo_joint.py，TRL sigmoid + SFT）
# β 与 readme 中「DPO 温度系数」描述一致
# ============================================================================
DPO_BETA = 0.2
DPO_SFT_LOSS_WEIGHT = 0.1
