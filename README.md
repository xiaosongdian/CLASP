# CLASP: A Closed-Loop Behavior Alignment Framework for Social Media Agent Simulation

A closed-loop behavior alignment framework that leverages explicit, dynamically refined personas to bridge users' historical behaviors and subsequent agent actions, ensuring behavioral consistency and temporal adaptability in social media simulation.

## Overview

Large language models (LLMs) provide a powerful foundation for social media simulation, but their utility is fundamentally limited by the accuracy of user behavior modeling. Existing LLM agents exhibit behavioral homogenization and fail to capture users' inherent identities and evolving preferences.

CLASP addresses these challenges through:
- **Explicit Persona Modeling**: Creating dynamically refined user personas from behavioral history
- **Decoupled Dual-Model Design**: Separating persona refinement from action generation
- **Closed-Loop Alignment**: Continual behavior alignment driven by action prediction discrepancies
- **Temporal Adaptability**: Capturing evolving user preferences across time windows

## Key Features

- **Closed-Loop Behavior Alignment**: Continually refines personas based on action prediction discrepancies
- **Decoupled Dual-Model Architecture**: 
  - **Persona Model**: Generates and refines explicit user personas
  - **Action Model**: Predicts user actions conditioned on personas
- **DPO-based Persona Refinement**: Generates multiple candidate personas and selects the best one using preference optimization
- **Comprehensive Evaluation**: Measures both action accuracy (F1) and semantic similarity (L) of generated content
- **Long-Horizon Simulation**: Supports multi-window evaluation to assess temporal consistency
- **BlueTrack Dataset**: Large-scale Bluesky dataset with complete action elements and heterogeneous behaviors

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              CLASP: Closed-Loop Alignment                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Window Splitting (W0, W1, ..., W5)                    │
│     └─> Temporal segmentation of user actions              │
│                                                             │
│  2. Initial Persona Generation (S0)                        │
│     └─> Persona Model: Generate explicit user persona      │
│         from W0 historical behaviors                        │
│                                                             │
│  3. Action Prediction (Decoupled)                          │
│     └─> Action Model: Conditioned on persona S0            │
│         ├─> Decision: Predict action type                  │
│         └─> Content: Generate post/reply text              │
│                                                             │
│  4. Discrepancy Detection                                  │
│     └─> Compare predicted vs. actual actions               │
│         ├─> Action type mismatches                         │
│         └─> Content semantic differences                   │
│                                                             │
│  5. Persona Refinement (Closed-Loop)                       │
│     └─> Persona Model: Refine based on discrepancies       │
│         ├─> Generate N candidate personas (DPO)            │
│         ├─> Evaluate on multiple windows                   │
│         └─> Select best persona (highest Q score)          │
│                                                             │
│  6. Iterative Alignment (Long-Horizon)                     │
│     └─> S0 → W1 → S1 → W2 → S2 → ... → W5                 │
│         (Continual behavior alignment)                      │
│                                                             │
│  Evaluation Metrics:                                        │
│     ├─> F: Action alignment (weighted F1)                  │
│     ├─> L: Content similarity (cosine)                     │
│     └─> Q: Overall quality (α*F + (1-α)*L)                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Project Structure

```
Clasp_git/
├── src/                          # Core modules
│   ├── config.py                 # Global configuration
│   ├── action_predictor.py      # Action Model: action prediction
│   ├── action_predictor_parallel.py  # Parallel action prediction
│   ├── profile_generator.py     # Persona Model: generation & refinement
│   ├── scorer.py                # Evaluation metrics (F, L, Q)
│   ├── prompts.py               # LLM prompt templates
│   └── window_splitter.py       # Temporal window splitting
│
├── train/                        # Training pipelines
│   ├── dpo/
│   │   └── dpo_pipeline.py      # DPO training pipeline
│   └── sft/
│       ├── sft_data_generator.py    # SFT data generation
│       └── sft_model_trainer.py     # SFT model training
│
├── comparison/                   # Baseline comparison
│   ├── run_baseline_comparison.py   # Multi-baseline evaluation
│   ├── run_baseline_parallel.py     # Parallel evaluation
│   ├── run_dpo_profile_slice_eval.py  # DPO slice evaluation
│   ├── action_predictor_batch.py    # Batch prediction
│   ├── Clasp/
│   │   └── profile_client.py    # Profile service client
│   └── plot/                     # Visualization scripts
│       ├── visualize_baseline_chain.py
│       ├── visualize_f_l_comprehensive.py
│       ├── visualize_q_stability_final.py
│       └── ...
│
├── scripts/                      # Utility scripts
│   ├── start_vllm_baseline_tmux.sh   # Start baseline vLLM service
│   ├── start_vllm_clasp_tmux.sh      # Start CLASP vLLM service
│   └── start_vllm_dpo_slice_tmux.sh  # Start DPO slice service
│
└── output/                       # Experiment outputs (gitignored)
    ├── windowed/                 # Windowed data
    ├── dpo/                      # DPO training outputs
    └── comparison/               # Evaluation results
```

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- vLLM (for efficient inference)
- Sentence-Transformers
- PostgreSQL (for data storage)

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd Clasp_git

# Install dependencies
pip install torch transformers vllm sentence-transformers


# Configure model paths in src/config.py
# Set up vLLM services (see scripts/)
```

## Configuration

Key parameters in `src/config.py`:

```python
# Window parameters
WINDOW_SIZE = 10              # Actions per window
NUM_WINDOWS = 5               # Training windows (W0-W4)
NUM_WINDOWS_EVAL_CHAIN = 6    # Evaluation windows (W0-W5)

# Scoring weights
ACTION_WEIGHTS = {
    "post": 0.35,
    "reply": 0.30,
    "repost": 0.20,
    "like": 0.15,
}
ALPHA = 0.6                   # Q = α*F + (1-α)*L

# DPO thresholds
NUM_CANDIDATE_PROFILES = 10   # Candidate profiles per round
TAU_PLUS = 0.06              # Positive pair threshold
TAU_MINUS = -0.02            # Negative pair threshold
DELTA = 0.05                 # Minimum margin

# Model configuration
USE_VLLM_API = True          # Use vLLM for inference
ENABLE_COMMERCIAL_PROFILE = True  # Use GPT-4o-mini for candidates
COMMERCIAL_PROFILE_RATIO = 0.4    # 40% candidates from GPT
```

## Usage

### 1. Data Preparation

We use **BlueTrack**, a large-scale Bluesky dataset covering long-horizon and heterogeneous social behaviors with complete action elements.

```bash
# Split user actions into temporal windows
python -m src.window_splitter \
    --input data/raw/bluetrack_actions.jsonl \
    --output output/windowed/ \
    --window-size 10 \
    --num-windows 5
```

### 2. DPO Training (Persona Model)

```bash
# Run DPO pipeline to generate preference pairs for persona refinement
python -m train.dpo.dpo_pipeline \
    --windowed-root output/windowed/ \
    --split train \
    --output output/dpo/dpo_pairs.jsonl \
    --rounds 2 \
    --workers 10
```

### 3. Model Training (Decoupled Dual-Model)

```bash
# Step 1: Train Action Model (SFT)
python -m train.sft.sft_model_trainer \
    --data output/sft/train_data.jsonl \
    --model-name Meta-Llama-3.1-8B-Instruct \
    --output saves/action_model/

# Step 2: Train Persona Model (DPO)
# Use standard DPO training with generated preference pairs
# This enables closed-loop behavior alignment
```

### 4. Baseline Comparison

```bash
# Start vLLM services
bash scripts/start_vllm_baseline_tmux.sh
bash scripts/start_vllm_clasp_tmux.sh

# Run multi-baseline evaluation
python -m comparison.run_baseline_comparison \
    --split test \
    --windowed-root output/windowed/ \
    --methods static_s0,prefix_refresh,clasp_online,history_only \
    --plot
```

### 5. Visualization

```bash
# Visualize baseline comparison results
python -m comparison.plot.visualize_baseline_chain \
    --input output/comparison/clasp_online/baseline_chain_test.jsonl \
    --output output/comparison/clasp_online/chain_results.png

# Comprehensive F/L analysis
python -m comparison.plot.visualize_f_l_comprehensive \
    --method clasp_online \
    --split test
```

## Evaluation Metrics

### Action Accuracy (F)
Weighted F1 score across action types:
- **Post**: 35%
- **Reply**: 30%
- **Repost**: 20%
- **Like**: 15%

### Semantic Similarity (L)
Cosine similarity between predicted and actual content (for post/reply actions).

### Combined Score (Q)
```
Q = α * F + (1 - α) * L
```
where α = 0.6 (configurable)

## Baseline Methods

1. **static_s0**: Fixed initial persona (no refinement)
2. **prefix_refresh**: Regenerate persona from all observed actions
3. **clasp_online**: Closed-loop alignment with discrepancy-driven refinement (**our method**)
4. **history_only**: No persona, use raw action history
5. **incremental_persona**: Refine with new actions (no discrepancy signals)

**Performance**: CLASP improves overall alignment quality by up to **31.8%** and reduces variance by **32.1%** during long-sequence simulation compared to baselines.

## Advanced Features

### Parallel Processing

```python
# Enable parallel action prediction
ACTION_PREDICTION_PARALLEL = True
ACTION_PREDICTION_WORKERS = 10

# Multi-process user evaluation
DPO_USER_PROCESSES = 5
```

### Debug Mode

```python
# Enable detailed LLM call logging
DEBUG_LLM = True
DEBUG_LLM_INCLUDE_ACTIONS = True  # Log action predictions
```

### Hybrid Persona Generation

```python
# Mix local and commercial models for diverse candidate personas
ENABLE_COMMERCIAL_PROFILE = True
COMMERCIAL_PROFILE_RATIO = 0.4  # 40% candidates from GPT-4o-mini
OPENAI_API_KEY = "your-api-key"
OPENAI_BASE_URL = "https://api.openai.com/v1"
```

## Performance Tips

1. **Use vLLM**: Set `USE_VLLM_API = True` for 10-20x faster inference
2. **Parallel Prediction**: Enable `ACTION_PREDICTION_PARALLEL` for multi-threaded action prediction
3. **Multi-Process Evaluation**: Set `DPO_USER_PROCESSES > 1` for parallel user processing
4. **GPU Acceleration**: Use CUDA for Sentence-Transformer scoring (set `DPO_SCORER_DEVICE = "cuda"`)

## Dataset: BlueTrack

BlueTrack is a large-scale Bluesky dataset that covers:
- **Long-horizon behaviors**: Multi-window temporal sequences
- **Heterogeneous actions**: Post, reply, repost, like with complete elements
- **Complete action elements**: Timestamps, content, targets, and context

This dataset facilitates realistic social media simulation and evaluation.

## Citation

If you use this code or dataset in your research, please cite:

```bibtex
@article{clasp2024,
  title={CLASP: A Closed-Loop Behavior Alignment Framework for Social Media Agent Simulation},
  author={Your Name},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2024}
}
```

## License

[Specify your license here]

## Contact

For questions or issues, please open an issue on GitHub or contact [your-email].

## Key Results

- **+31.8%** improvement in overall alignment quality
- **-32.1%** reduction in alignment variance during long-sequence simulation
- Demonstrates behavioral consistency and temporal adaptability
- Outperforms static, prefix-based, and history-only baselines

## Acknowledgments

- Built on Meta's Llama-3.1-8B-Instruct
- Uses vLLM for efficient inference
- Sentence-Transformers for semantic similarity
- DPO training framework (Rafailov et al., 2023)
- BlueTrack dataset from Bluesky social network
