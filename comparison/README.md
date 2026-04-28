# Clasp 对比评估

在仓库根目录执行。动作预测与语义分与 `src.config` / `src.scorer` 一致；画像与纠偏调用与主项目 DPO pipeline 相同（`PROFILE_API_*`、精炼 prompt 等）。**行为文本过长**时由 `PROFILE_BEHAVIOR_TEXT_MAX_CHARS`、`ACTION_PROMPT_HISTORY_MAX_CHARS` 及精炼段上限截断（头尾保留），避免 4k 上下文 400。

## 统一测试协议：窗口链

推荐在统一测试集上使用 **窗口链**：

1. 用 **W0** 上的行为生成**基础画像 S0**。
2. 测试集窗口化推荐 **6 个窗口**（W0…W5，每窗 T 条动作，共 6T）：`python3 -m src.window_splitter --input data --output output/windowed --split test --num-windows 6`。
3. **前向链**（无「空历史预测 W0」步）：`step_index=0…4` 依次为 **S0→W1**（历史 W0）、**S1→W2**（历史 W1）…直至 **S4→W5**；每步在「历史 = `W_t`、目标 = `W_{t+1}`」上算 F/L/Q。
4. **画像 / 上下文更新**：`clasp_online` 误差精炼；`prefix_refresh` 前缀全量重算画像；`static_s0` 始终 S0；`incremental_persona` 用上一画像 + 当前窗行为精炼；`s0_sliding_history` 固定 S0 但在动作 prompt 中附加当前窗行为；`user_full_history` 不显式画像、仅用 W0…W_t 拼接行为块。可选 `--always-accept-refinement`、`--refinement-variants N`（主要对 `clasp_online`）。

## 入口

### 窗口链多基线

`python3 -m comparison.run_baseline_comparison --split test`

- **clasp_online**：上述「误差驱动更新画像」的 Clasp 测试流程。
- **static_s0**：仅用 W0 的 S0，逐步以 W_t 为历史预测 W_{t+1}（画像不变）。
- **prefix_refresh**：每步后按已观测 W0…W_{t+1} 整段重算「初始画像」。
- **incremental_persona**：每步后用 S_{t-1} 与**当前窗**真实行为（无预测误差文本）走精炼 prompt，单次更新画像。
- **s0_sliding_history**：画像固定为 W0 的 S0；每步在 prompt 中附加的是**已观测历史窗 W_t** 的行为全文，**不包含**待预测的 W_{t+1}（无标签泄漏）。
- **user_full_history**：不设独立画像字符串；每步在 prompt 中附加 **W0 至当前窗** 拼接后的全量行为。

可选：`--methods clasp_online,static_s0,incremental_persona` 等（完整列表见 CLI `--help`）、`--max-users 20`、`--skip-window-split`、`--windowed-root output/windowed`。**自动切分**时默认 `--num-windows` 与 `config.NUM_WINDOWS_EVAL_CHAIN`（6）一致；训练/DPO 仍为 5 窗时可单独跑 splitter。

- **语义分（SentenceTransformer）**：默认 **`--scorer-device cpu`**，避免与 GPU 上 vLLM 等争显存；显存充裕时可 `--scorer-device cuda` 加速。

- **单份窗口化 jsonl**：`--input-jsonl output/windowed/test/community_3.jsonl`（不要求 `data/test` 目录）。
- **折线图**（需 matplotlib）：`--plot clasp_c3.png` 时，默认写入 **`--comparison-root` 下对应 method 子目录**。  
  **去极值比例**：`--plot-trim-each-tail P`（按用户 `mean_Q` 去掉最低/最高各 **P**；`P=0` 关闭）；省略则用 **`config.PLOT_TRIM_EACH_TAIL`**（默认 `0.05`）。不写回 jsonl。

**输出目录（默认按方法分离）**

- 根目录：`--comparison-root`（默认 `output/comparison`）。
- 每种 method 一个子文件夹：`<comparison-root>/<method>/baseline_chain_<split或输入文件名>.jsonl`。
- 若需要**旧版单文件**（所有 method 混在一个 jsonl）：加 **`--combined-jsonl`**，并用 **`--output`** 指定完整路径。

评估已跑完、只需作图时（指向对应 method 下的 jsonl）：

`python3 -m comparison.plot_chain_from_jsonl output/comparison/clasp_online/baseline_chain_community_3.jsonl --plot output/comparison/clasp_online/clasp_c3.png`

### PolicySim（baseline 比较方法之一）

`comparison/policysim/` 为策略智能体与相关训练脚本，与其它基线并列用于对比；说明见 `comparison/policysim/README.md` 与 `Policysim.pdf`。

## 指标（与 `src/scorer.py` 一致）

| 符号 | 含义 |
|------|------|
| **F** | 决策加权 F1（交互动作类型对齐） |
| **L** | post/reply 预测文本与真实的语义余弦均值 |
| **Q** | α·F + (1-α)·L；若 `NORMALIZE_L_TO_UNIT` 为 True 则 L 先映射为 (L+1)/2（见 `src/config.py`） |

## 目录说明

| 路径 | 作用 |
|------|------|
| `comparison/window_chain_eval.py` | 窗口链策略核心（S0→W1 … S4→W5、可选纠偏） |
| `comparison/run_baseline_comparison.py` | 窗口链多基线 CLI（支持 `--input-jsonl`、`--plot`） |
| `comparison/window_chain_plot.py` | 按步聚合 F/L/Q、绑图 |
| `comparison/plot_chain_from_jsonl.py` | 仅从 JSONL 重画折线（不调模型） |
| `comparison/policysim/` | Baseline 之一：策略仿真子包 |
| `comparison/Clasp/profile_client.py` | 微调画像 API 封装（供需单独指定画像端点时使用） |
