# Clasp

面向 Bluesky/社区用户行为与画像的 DPO 数据与训练相关代码，核心流程在 `src/dpo_pipeline.py`：对窗口化后的用户数据生成画像、对候选画像做动作预测与语义评分，并构造 DPO 正负对。

## 依赖与运行

- Python 3；需已配置 `src/config.py` 中的 vLLM（画像 / 动作）API、Sentence-Transformer 本地路径等。
- 主流程示例：

```bash
python3 -m src.dpo_pipeline --input path/to/windowed.jsonl --output-dir output/dpo
```



## 速度：多进程 + 多线程

- **用户间**：`--user-processes N`（或 `config.DPO_USER_PROCESSES`）使用最多 `min(N, 待处理用户数)` 个进程，**并行处理不同用户**（适合 IO 多路访问远程 LLM API）。
- **用户内（候选间）**：`--workers`（`config.DPO_WORKERS`）在单进程内用 **ThreadPool** 对候选画像做 quick/全量评分等；与多进程是「进程 × 线程」关系。
- **语义模型（Sentence-Transformer）**：多进程时每个子进程会各加载一份；默认在子进程用 **CPU**（`DPO_SCORER_DEVICE` 未设置时由管道选 `cpu`），避免多份模型同时占满同一块 **GPU 显存**。若显存充足且要加速，可在 `config` 中设 `DPO_SCORER_DEVICE = "cuda"` 或传 `--scorer-device cuda`（有 OOM 风险，请自行调小 `--user-processes`）。

## 启动命令
python3 -m src.dpo_pipeline \
  --input-dir /home/xiaosong/personality/Clasp/output/windowed/train \
  --input-glob 'community_*.jsonl' \
  --output-dir /home/xiaosong/personality/Clasp/output/dpo/train \
  --user-processes 5 \
  --workers 10 \
  --scorer-device cpu \
  --resume

## 用户数据窗口拆分
cd /home/xiaosong/personality/Clasp
python3 -m src.window_splitter \
  --input /home/xiaosong/personality/Clasp/data \
  --output /home/xiaosong/personality/Clasp/output/windowed \
  --split train


## 常用参数

| 参数 | 说明 |
|------|------|
| `--input` / `--input-dir` | 窗口化 `jsonl` 输入 |
| `--output-dir` | DPO 对与明细输出目录 |
| `--max-users` | 本 run 最多处理用户数 |
| `--workers` | 单用户内候选级线程数 |
| `--user-processes` | 多用户并行进程数（默认 1=用户串行） |
| `--scorer-device` | 语义分设备：`cpu` / `cuda` 等 |
| `--rounds` | 滚动 DPO 轮次 |
| `--resume` | 从 `dpo_detail_<stem>.jsonl` 续跑已完成的 `user_id` |

## DPO 画像切片评估（comparison）

- `python3 -m comparison.run_dpo_profile_slice_eval`：`--slice-eval-mode` 含 `w1_w2`（默认）、`w2_w3`、`w0_w1_w2_p0p1`。
  - `w0_w1_w2_p0p1`：P0 与精炼后的 P1 各对 **W0/W1/W2** 做动作预测与 F/L/Q；行内写 `P0_W0_F`…`P1_W2_Q`，并写 `W1_*`/`W2_*` 为 P0/P1 **各自三窗**算术均值；`plot_dpo_profile_slice_radar` 对 p0p1 成功行作图时聚合 **P0@W1**（`P0_W1_*`）与 **P1@W2**（`P1_W2_*`）单窗。**W0 从不**向动作模型注入近期行为；**W1/W2** 是否注入由 `--no-action-prompt-observed-history` 控制（不加该参数时 W1/W2 与旧切片一致带历史）。
  - 输出文件：`dpo_profile_slice_<split>_w0w1w2_p0p1.jsonl`（勿与其它 mode 混在同一文件上 `--resume`）。
- `python3 -m comparison.plot_dpo_profile_slice_radar`：**单次**运行默认在终端打印 **切片统计**（行数、成功行、各 variant、`slice_eval_mode`、全局 mean(W1_Q)/mean(W2_Q)，p0p1 与作图对齐）；加 `--no-slice-stats` 可关。`--watch N` 每轮打印统计及与上一轮 **Δ**；可加 `--watch-skip-unchanged` 在文件 mtime+size 未变时跳过重读与重画。可加 `--aggregate-top-fraction 0.3`：按社区×方法×**每个指标**单独取该指标最高的约 30% 用户再求柱上均值（假定 F/L/Q 越大越好）。可加 `--asymmetric-quantile-demo`：**GPT/Base 柱**每桶取该指标**偏低侧**约 95%（⌈0.95·n⌉ 人）再均，**Clasp** 取**偏高侧**约 95%（演示、非公平对比）；等价于 `--demo-plot-asymmetric-median-split --demo-asymmetric-split-fraction 0.95`。可加 `--export-stats-csv [PATH]`：**仅**导出与柱图柱顶数值一一对应的窄表（`metric`×`community`×图例柱；UTF-8 BOM）；不写 `PATH` 时默认为 `<out 主名>_plot_stats.csv` 与 `--out` 同目录。

## 模块说明（简要）

- `src/config.py`：API、阈值、窗口、`DPO_WORKERS` / `DPO_USER_PROCESSES` 等。
- `src/dpo_pipeline.py`：DPO 全流程与 CLI。
- `src/scorer.py`：`SemanticScorer`（可指定 `device`）。
- `src/action_predictor.py`、`src/profile_generator.py`：动作预测与画像生成。

## 当前改进与注意

- 多用户多进程会提高**墙钟时间**上的吞吐，但整体 CPU/内存会上升；`--user-processes` 与 vLLM 最大并发/队列能力需一起观察，避免 429 或打满服务端。
- 多进程时 `dpo_detail_*.jsonl` 的写入顺序**按完成先后**，与串行时不同，不影响 `resume`（按 `user_id` 去重）。
