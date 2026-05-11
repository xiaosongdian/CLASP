# Clasp 对比评估

在仓库根目录执行。动作预测与语义分与 `src.config` / `src.scorer` 一致；画像与纠偏调用与主项目 DPO pipeline 相同（`PROFILE_API_*`、精炼 prompt 等）。**行为文本过长**时由 `PROFILE_BEHAVIOR_TEXT_MAX_CHARS`、`ACTION_PROMPT_HISTORY_MAX_CHARS` 及精炼段上限截断（头尾保留），避免 4k 上下文 400。

## 统一测试协议：窗口链

推荐在统一测试集上使用 **窗口链**：

1. 用 **W0** 上的行为生成**基础画像 S0**。
2. 测试集窗口化推荐 **6 个窗口**（W0…W5）：`python3 -m src.window_splitter --input data --output output/windowed --split test --num-windows 6`。若需**缓解短时动作扎堆**，可用 **`--split-mode monthly_chain`**（默认连续 **6** 个自然月，每月 **1** 窗、在全月时间跨度内均匀抽 N 条，共 6N 条）；合并 test+eval_unseen 并统一命名输出可运行 `python3 scripts/build_monthly_chain_windowed.py` → `output/windowed/test/monthly_chain_community_{id}.jsonl`。
3. **前向链**（无「空历史预测 W0」步）：`step_index=0…4` 依次为 **S0→W1**（历史 W0）、**S1→W2**（历史 W1）…直至 **S4→W5**；每步在「历史 = `W_t`、目标 = `W_{t+1}`」上算 F/L/Q。
4. **统一历史机制**：所有方法都使用相同的历史输入（profile_suffix），确保公平对比画像更新策略。
5. **画像更新策略**：`clasp_online` 误差精炼；**`clasp_online_no_hist`** 与前者相同但动作侧不载入观测历史（可与 `clasp_online` 同跑对照）；`prefix_refresh` 前缀全量重算画像；`static_s0` 始终 S0；`incremental_persona` 用上一画像 + 当前窗行为精炼。可选 `--always-accept-refinement`、`--refinement-variants N`（主要对 Clasp 系）。

### vLLM 模型切换（基线对比）

`comparison/window_chain_eval.evaluate_user_window_chain` 在调用画像/动作 API 前会按方法切换 `cfg` 中的模型名：**`clasp_online`** 与 **`clasp_online_no_hist`** 使用画像 checkpoint `COMPARISON_CLASP_PROFILE_VLLM_MODEL` 与动作 checkpoint `COMPARISON_CLASP_ACTION_VLLM_MODEL`；**其余方法**画像与动作均使用 `COMPARISON_BASELINE_VLLM_MODEL`。常量定义见 [`src/config.py`](../src/config.py)。两端 vLLM 服务注册的 `model` 字段须与这些字符串一致。

## 入口

### 窗口链多基线

`python3 -m comparison.run_baseline_comparison --split test`

**核心方法**（已统一历史输入机制；`--methods` 任选子集）：

- **clasp_online**：误差驱动更新画像（主方法）。
- **clasp_online_no_hist**：与 `clasp_online` 相同画像与模型路由，但动作 prompt **不**含观测历史（用于与 `clasp_online` 同批对照；输出在 `comparison-root/clasp_online_no_hist/`）。
- **static_s0**：仅用 W0 的 S0，画像不变（基线）。
- **prefix_refresh**：每步后按已观测 W0…W_{t+1} 整段重算「初始画像」。
- **incremental_persona**：每步后用 S_{t-1} 与**当前窗**真实行为（无预测误差文本）走精炼 prompt，单次更新画像。

可选：`--methods clasp_online,static_s0,prefix_refresh,incremental_persona` 等（完整列表见 CLI `--help`）、`--max-users 20`、`--skip-window-split`、`--windowed-root output/windowed`。**已窗口化数据类型**：`--windowed-dataset contiguous`（默认，匹配 `community_*.jsonl`）或 **`--windowed-dataset monthly_chain`**（匹配 `monthly_chain_community_*.jsonl`；未写 `--file-glob` 时按类型自动选 glob，也可显式 `--file-glob` 覆盖）。**自动切分**时默认 `--num-windows` 与 `config.NUM_WINDOWS_EVAL_CHAIN`（6）一致；训练/DPO 仍为 5 窗时可单独跑 splitter。

- **语义分（SentenceTransformer）**：默认 **`--scorer-device cpu`**，避免与 GPU 上 vLLM 等争显存；显存充裕时可 `--scorer-device cuda` 加速。

- **单份窗口化 jsonl**：`--input-jsonl output/windowed/test/community_3.jsonl`（不要求 `data/test` 目录）。
- **monthly_chain 多文件目录模式**：`--split test --skip-window-split --windowed-root output/windowed --windowed-dataset monthly_chain`。
- **折线图**（需 matplotlib）：`--plot clasp_c3.png` 时，默认写入 **`--comparison-root` 下对应 method 子目录**。  
  **去极值**：`--plot-trim-each-tail P`；**`--plot-trim-scope user`**（默认）按用户 `mean_Q` 整行去最低/最高各 **P**；**`--plot-trim-scope step`** 则在**每个链上窗口内**去尾后再聚合（`--plot-step-trim-basis deviation` 为 Q−当步均值，**`value`** 为当步 Q 分位）；**仅去低档**加 `--plot-trim-sides lower`；`P=0` 关闭；若带 **`--plot`** 且未指定该项，则用 **`config.PLOT_TRIM_EACH_TAIL`**（默认 `0.05`）。不写回 jsonl。

**输出目录（默认按方法分离）**

- 根目录：`--comparison-root`（默认 `output/comparison`）。
- 每种 method 一个子文件夹：`<comparison-root>/<method>/baseline_chain_<split或标签>_<数据集类型>.jsonl`（数据集类型为 `--windowed-dataset`：`contiguous` 或 `monthly_chain`，避免与另一类窗口化结果混写覆盖）。
- 若需要**旧版单文件**（所有 method 混在一个 jsonl）：加 **`--combined-jsonl`**，并用 **`--output`** 指定路径（若文件名主干未含 `_contiguous` / `_monthly_chain`，会自动追加后缀以区分数据集类型）。

- **clasp_online 画像快照**：当 `--methods` 含 **`clasp_online`** 或 **`clasp_online_no_hist`** 时，默认将各用户、各阶段画像写入**对应 method 子目录下**的单个文件  
  `<comparison-root>/<method>/profile_snapshots/<output_stem>/profiles.jsonl`  
  （每行 JSON：`user_id`、`phase`、`step_index`、`profile`、`profile_length` 等）。加 **`--no-profile-snapshots`** 可关闭。全新跑会清空该文件；**`--resume`** 时在末尾追加未跑过的用户。

- **不带观测历史的动作 prompt**：加 **`--no-action-prompt-observed-history`** 时，动作模型输入不再包含 (1) 拼在画像后的本窗行为块；(2) 「Recent user actions」滑窗；仍保留 **Current scenario**（目标窗口内当前条的上下文）。可与 `src/config.py` 中 **`ACTION_PROMPT_INCLUDE_OBSERVED_HISTORY`** 配合（CLI 会写入每条结果里的 `action_prompt_include_observed_history`）。

- **汇总图（按社区链上 F/L/Q）**：`python3 -m comparison.plot.visualize_baseline_chain <baseline_chain_*.jsonl> --out viz.png`。按 **`community_id`** 分组；**无论几个社区均写入同一个 `--out`**，多个社区时在一张 PNG 内 **纵向多子图**。**默认**以 **W1** 步聚合值为基线画 **Q** 的水平虚线；`--baseline-metric all` 为 F/L/Q 三条；`--no-baseline` 关闭。**降噪**：`--plot-trim-scope user` 时 `--plot-trim-each-tail 0.05` 双侧去极端用户行；**按步去极值**用 `--plot-trim-scope step --plot-trim-each-tail 0.05 --plot-trim-sides lower`（默认 `--step-trim-basis deviation`）；`--aggregate median` 逐步取中位数。加 **`--watch 10`** 每 10 秒重读 jsonl。

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
| `comparison/run_baseline_comparison.py` | 窗口链多基线 CLI（`--input-jsonl`、`--plot`、**`--windowed-dataset`** 选 contiguous / monthly_chain） |
| `comparison/window_chain_plot.py` | 按步聚合 F/L/Q、绑图 |
| `comparison/plot_chain_from_jsonl.py` | 仅从 JSONL 重画折线（不调模型） |
| `comparison/plot/visualize_baseline_chain.py` | 按社区绘制链上 F/L/Q（单 PNG、多社区为纵向子图），`--watch` 刷新 |
| `comparison/policysim/` | Baseline 之一：策略仿真子包 |
| `comparison/Clasp/profile_client.py` | 微调画像 API 封装（供需单独指定画像端点时使用） |
