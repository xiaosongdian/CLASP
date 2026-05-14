# Clasp 画像 DPO 训练超参数汇总

本文档归纳 **LLaMA-Factory** 下 Clasp 画像 **DPO 两阶段** LoRA 微调的超参数，主来源为：

- `DPO_Train/config/clasp_profile_dpo_stage1_llama3_lora.yaml`
- `DPO_Train/config/clasp_profile_dpo_stage2_llama3_lora.yaml`
- 阶段 1 续训示例：`DPO_Train/config/clasp_profile_dpo_stage1_resume_checkpoint500.yaml`（与阶段 1 主 yaml 训练超参一致，仅 `overwrite_cache` / `resume_from_checkpoint` / `overwrite_output_dir` 不同）

与 **`src/config.py`** 中 **`DPO_BETA`**、**`DPO_SFT_LOSS_WEIGHT`** 对齐关系：`pref_beta` ↔ `DPO_BETA`，`pref_ftx` ↔ `DPO_SFT_LOSS_WEIGHT`（当前均为 **0.2** / **0.1**）。

启动入口：`DPO_Train/run_llamafactory_two_stage.sh`、`run_llamafactory_stage1.sh`、`run_llamafactory_stage2.sh`（在 `DEEPER/LLaMA-Factory` 目录执行 `llamafactory-cli train <yaml>`）。

---

## 1. 阶段 1 vs 阶段 2（主配置对照表）

| 超参数 | 阶段 1 `clasp_profile_dpo_stage1_llama3_lora.yaml` | 阶段 2 `clasp_profile_dpo_stage2_llama3_lora.yaml` |
|--------|-----------------------------------------------------|-----------------------------------------------------|
| **基座** `model_name_or_path` | `/data/LLM_models/Meta-Llama-3-8B-Instruct` | 同左 |
| **挂载已有 LoRA** `adapter_name_or_path` | 无（从零 LoRA） | `.../saves/clasp_profile_dpo_stage1_base`（脚本可改为阶段 1 的 `checkpoint-*`） |
| **`create_new_adapter`** | （未写，默认） | `false`（在阶段 1 LoRA 上继续训） |
| **`trust_remote_code`** | `true` | `true` |
| **`stage`** | `dpo` | `dpo` |
| **`do_train`** | `true` | `true` |
| **`finetuning_type`** | `lora` | `lora` |
| **`dataset_dir`** | `/home/xiaosong/personality/DEEPER/data` | 同左 |
| **`dataset`** | `clasp_profile_dpo_stage1` | `clasp_profile_dpo_stage2` |
| **`template`** | `llama3` | `llama3` |
| **`cutoff_len`** | 8192 | 8192 |
| **`overwrite_cache`** | `true` | `true` |
| **`preprocessing_num_workers`** | 4 | 4 |
| **`dataloader_num_workers`** | 2 | 2 |
| **`lora_target`** | `all` | `all` |
| **`lora_rank`** | 16 | 16 |
| **`lora_alpha`** | 32 | 32 |
| **`lora_dropout`** | 0.1 | 0.1 |
| **DPO** `pref_beta`（≈ β） | 0.2 | 0.2 |
| **DPO + chosen SFT** `pref_ftx` | 0.1 | 0.1 |
| **`pref_loss`** | `sigmoid` | `sigmoid` |
| **`output_dir`** | `.../saves/clasp_profile_dpo_stage1_base` | `.../saves/clasp_profile_dpo_stage2_commercial` |
| **`overwrite_output_dir`** | `true` | **`false`**（便于阶段 2 续训） |
| **`resume_from_checkpoint`** | 未写 | `null`（由 shell 动态覆盖） |
| **`per_device_train_batch_size`** | 2 | 2 |
| **`gradient_accumulation_steps`** | 8 | 8 |
| **有效 batch（单卡）** | 2×8 = **16** | 同左 |
| **`learning_rate`** | 5.0e-6 | **3.0e-6** |
| **`num_train_epochs`** | **4** | **3** |
| **`lr_scheduler_type`** | `cosine` | `cosine` |
| **`warmup_ratio`** | **0.05** | **0.03** |
| **`max_grad_norm`** | 1.0 | 1.0 |
| **`bf16`** | `true` | `true` |
| **`ddp_timeout`** | 180000000 | 180000000 |
| **`optim`** | `adamw_torch` | `adamw_torch` |
| **`do_eval`** | `true` | `true` |
| **`eval_strategy`** | `steps` | `steps` |
| **`eval_steps`** | 100 | 100 |
| **`val_size`** | 0.05 | 0.05 |
| **`per_device_eval_batch_size`** | 2 | 2 |
| **`logging_steps`** | 5 | 5 |
| **`save_steps`** | 100 | 100 |
| **`save_total_limit`** | 4 | 4 |
| **`plot_loss`** | `true` | `true` |
| **`report_to`** | `none` | `none` |

**yaml 中未显式写出**（一般由 LLaMA-Factory / HF Trainer 默认）：如 `weight_decay`、`seed` 等；若需与论文完全一致，可在 yaml 中补写。

---

## 2. 训练数据与导出路径（注释约定）

| 阶段 | 导出命令入参（Clasp 根目录） | 写入 jsonl（DEEPER） |
|------|------------------------------|----------------------|
| 1 | `output/dpo/train/dpo_pairs_stage1_base_only.jsonl` | `DEEPER/data/DEEPER_train_data/clasp/profile_dpo_stage1.jsonl` |
| 2 | `output/dpo/train/dpo_pairs_stage2_commercial_involved.jsonl` | `DEEPER/data/DEEPER_train_data/clasp/profile_dpo_stage2.jsonl` |

数据集名在 **`DEEPER/data/dataset_info.json`** 注册为 `clasp_profile_dpo_stage1` / `clasp_profile_dpo_stage2`（`ranking: true`，列含 `system` / `prompt` / `chosen` / `rejected`）。

---

## 3. 续训专用：阶段 1 `resume_checkpoint500`

与阶段 1 **主 yaml 相同的训练超参**；差异仅：

- `overwrite_cache: false`
- `overwrite_output_dir: false`
- `resume_from_checkpoint: .../checkpoint-500`（需按实际 checkpoint 修改）

其余 LoRA / DPO / batch / LR / epoch 等与阶段 1 主表一致。

---

## 4. 生成 DPO 对时相关常量（`src/config.py`，非 LLaMA-Factory yaml）

用于 **构造偏好对** 与流水线，不等同于上表「微调步长」，但常与 DPO 实验一起报告：

| 常量 | 值 | 含义（简述） |
|------|-----|----------------|
| `NUM_CANDIDATE_PROFILES` | 10 | 每轮精炼候选画像数 |
| `TAU_PLUS` | 0.06 | `r_all` 正侧阈值 |
| `TAU_MINUS` | -0.02 | `r_all` 负侧阈值 |
| `DELTA` / `ABS_DELTA` | 0.05 / 0.10 | 正负例 `r_all` 间隔等 |
| `DPO_BETA` | 0.2 | 与 yaml `pref_beta` 对齐 |
| `DPO_SFT_LOSS_WEIGHT` | 0.1 | 与 yaml `pref_ftx` 对齐 |
| `DPO_ROUNDS` | 2 | 滚动 DPO 轮次 |
| `DPO_WORKERS` | 10 | 候选评估等线程数 |
| `DPO_USER_PROCESSES` | 5 | 多用户并行进程数 |

---

## 5. 副本说明（`DEEPER/scripts/`）

`~/personality/DEEPER/scripts/clasp_profile_dpo_stage{1,2}_llama3_lora.yaml` 为另一份拷贝；其中 **LoRA `lora_dropout`** 曾为 **0.05**，**`eval_steps` / `save_steps`** 等与 `DPO_Train/config/` 版本可能不一致。以 **`Clasp/DPO_Train/config/`** 为维护主源。

---

*若修改 yaml 或 `src/config.py` 中 DPO 相关项，请同步更新本文件。*
