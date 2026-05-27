#!/usr/bin/env python3
"""
Global configuration: model paths, window parameters, scoring weights, DPO thresholds
"""
import os

# Load environment variables from .env file if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, use system environment variables

# ============================================================================
# Model paths (for local transformers loading)
# ============================================================================
ACTION_GENERATION_MODEL = "path/to/Meta-Llama-3.1-8B-Instruct-bluesky"
PROFILE_GENERATION_MODEL_RAW = "path/to/Meta-Llama-3.1-8B-Instruct"
SENTENCE_TRANSFORMER_MODEL = "sentence-transformers/all-mpnet-base-v2"

# ============================================================================
# vLLM API Configuration
# Set to True when using pre-started vLLM services
# ============================================================================
USE_VLLM_API = True

PROFILE_API_BASE = "http://localhost:8001/v1"
PROFILE_API_MODEL = "Meta-Llama-3.1-8B-Instruct"

# Commercial persona model (for hybrid candidate generation)
# Example: n=10, ratio=0.4 => 4 candidates from commercial model
ENABLE_COMMERCIAL_PROFILE = True
COMMERCIAL_PROFILE_RATIO = 0.4
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-openai-api-key")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
PROFILE_MODEL = "gpt-4o-mini"

ACTION_API_BASE = "http://localhost:8002/v1"
ACTION_API_MODEL = "Meta-Llama-3.1-8B-Instruct"

ACTION_API_EXTRA_BASES: tuple[str, ...] = ()  # Example: ("http://server2:8080/v1",)


def _normalize_openai_v1_base(url: str) -> str:
    """Normalize action API base URL (must end with /v1 for OpenAI client compatibility)."""
    u = (url or "").strip().rstrip("/")
    if not u:
        return ""
    if not u.startswith("http://") and not u.startswith("https://"):
        u = "http://" + u
    if not u.endswith("/v1"):
        u = u + "/v1"
    return u


def effective_action_api_bases() -> tuple[str, ...]:
    """
    List of action API endpoints for round-robin (deduplicated, primary endpoint first).
    """
    seen: set[str] = set()
    out: list[str] = []
    for raw in (ACTION_API_BASE, *ACTION_API_EXTRA_BASES):
        u = _normalize_openai_v1_base(raw)
        if u and u not in seen:
            seen.add(u)
            out.append(u)
    return tuple(out) if out else ("http://127.0.0.1:8002/v1",)

COMPARISON_BASELINE_VLLM_MODEL = "Meta-Llama-3.1-8B-Instruct"
COMPARISON_CLASP_PROFILE_VLLM_MODEL = (
    "Meta-Llama-3.1-8B-Instruct-clasp-dpo-stage2"
)
COMPARISON_CLASP_ACTION_VLLM_MODEL = (
    "Meta-Llama-3.1-8B-Instruct-bluesky-sft"
)

COMPARISON_CLASP_PROFILE_STAGE1_VLLM_MODEL = (
    "Meta-Llama-3.1-8B-Instruct-clasp-dpo-stage1"
)

# ============================================================================
# Test mode (use single remote API for all LLMs to test pipeline)
# ============================================================================
TEST_MODE = False

TEST_API_BASE = os.getenv("TEST_API_BASE", "https://api.openai.com/v1")
TEST_API_KEY = os.getenv("TEST_API_KEY", "your-test-api-key")
TEST_API_MODEL = os.getenv("TEST_API_MODEL", "gpt-3.5-turbo")
TEST_NUM_CANDIDATES = 3      

# ============================================================================
# Window parameters
# ============================================================================
WINDOW_SIZE = 10       # Actions per window
NUM_WINDOWS = 5        # Training/DPO: W0 ~ W4 (5 windows total)
# Chain evaluation: W0 builds S0, then S0→W1 ... S4→W5, total 6 windows, 6*T actions
NUM_WINDOWS_EVAL_CHAIN = 6

MONTHLY_CHAIN_NUM_MONTHS = 6
MONTHLY_CHAIN_WINDOWS_PER_MONTH = 1
MIN_ACTIONS = WINDOW_SIZE * NUM_WINDOWS  # Minimum 50 actions per user (training default)

# ============================================================================
# Action prediction / SFT sample construction
# ============================================================================
TEXT_LONG = 500        # Text truncation length

# Sliding history window for action prediction prompts: use N most recent real actions before current timestep
ACTION_PREDICTION_HISTORY_WINDOW = 5


ACTION_PROMPT_INCLUDE_OBSERVED_HISTORY = True

ACTION_WEIGHTS = {
    "post":   0.35,
    "reply":  0.30,
    "repost": 0.20,
    "like":   0.15,
}

ALPHA = 0.6

NORMALIZE_L_TO_UNIT = False


PLOT_TRIM_EACH_TAIL = 0.05

# ============================================================================
# DPO pair construction thresholds
# ============================================================================
NUM_CANDIDATE_PROFILES = 10   # Number of candidate personas per refinement round

TAU_PLUS = 0.06               
TAU_MINUS = -0.02             
DELTA = 0.05                  
ABS_DELTA = DELTA * 2         
# ============================================================================
# Model inference parameters
# ============================================================================
MAX_NEW_TOKENS_PROFILE = 2048     # Max tokens for persona generation
MAX_NEW_TOKENS_ACTION = 512       # Max tokens for action prediction


ACTION_API_MAX_CONTEXT_TOKENS = 4096
ACTION_API_COMPLETION_SAFETY_MARGIN = 64

ACTION_API_CHARS_PER_TOKEN_ESTIMATE = 3.0


PROFILE_BEHAVIOR_TEXT_MAX_CHARS = 6000

ACTION_PROMPT_HISTORY_MAX_CHARS = 6000

ACTION_PROMPT_PROFILE_MAX_CHARS = 3500


HISTORY_ONLY_ACTION_LINE_MAX_CHARS = 0  # 0 means use TEXT_LONG
HISTORY_ONLY_HISTORY_BUDGET_CHARS = 0
HISTORY_ONLY_PROMPT_NON_HISTORY_RESERVE_CHARS = 4200

PROFILE_REFINEMENT_OLD_PERSONA_MAX_CHARS = 3500
PROFILE_REFINEMENT_DISCREPANCY_MAX_CHARS = 3500
TEMPERATURE_PROFILE = 0.8         # Persona candidate diversity
TEMPERATURE_ACTION = 0         # Action prediction: increase temperature for persona sensitivity

# ============================================================================
# Debug: print each LLM request/response (enabled by --debug or DEBUG_LLM=True)
# ============================================================================
DEBUG_LLM = False
DEBUG_LLM_MAX_INSTRUCTION_CHARS = 1200

DEBUG_LLM_MAX_USER_CHARS = 2000
DEBUG_LLM_PRINT_FULL_OUTPUT = True   

DEBUG_LLM_BEHAVIOR_HEAD = 2500
DEBUG_LLM_BEHAVIOR_TAIL = 1500
DEBUG_LLM_DISCREPANCY_MAX = 50000  
DEBUG_LLM_OLD_PERSONA_HEAD = 1000
DEBUG_LLM_OLD_PERSONA_TAIL = 800
DEBUG_LLM_PROFILE_OUTPUT_HEAD = 2000
DEBUG_LLM_PROFILE_OUTPUT_TAIL = 1200

DEBUG_LLM_ACTION_USER_MAX = 4000

DEBUG_LLM_INCLUDE_ACTIONS = False

# ============================================================================
# Concurrency parameters (DPO evaluation and candidate persona generation)
# ============================================================================
DPO_WORKERS = 10

DPO_USER_PROCESSES = 5

DPO_USER_PROCESS_STAGGER_SEC = 0.3

DPO_SCORER_DEVICE = None
# DPO rolling rounds (round 1: S0->candidates, select best as S1; round 2: shift window and continue)
DPO_ROUNDS = 2

# Action prediction parallelization
ACTION_PREDICTION_PARALLEL = True  
ACTION_PREDICTION_WORKERS = 10     

# ============================================================================
# DPO fine-tuning (train_profile_dpo_joint.py, TRL sigmoid + SFT)
# β consistent with "DPO temperature coefficient" description in readme
# ============================================================================
DPO_BETA = 0.2
DPO_SFT_LOSS_WEIGHT = 0.1
