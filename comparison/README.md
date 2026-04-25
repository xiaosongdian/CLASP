# Clasp 对比评估

在仓库根目录执行。动作预测与语义分与 `src.config` / `src.scorer` 一致；画像与纠偏调用与主项目 DPO pipeline 相同（`PROFILE_API_*`、精炼 prompt 等）。

## 统一测试协议：窗口链

推荐在统一测试集上使用 **窗口链**：

1. 用 **W0** 上的行为生成**基础画像**（初始人设）。
2. 依次对 **W1 → W2 → W3 → W4** 评估：每一步在「历史 = 上一窗口 `W_{k-1}`、目标 = `W_k`」上计算 F、L、Q（与 `evaluate_profile_on_window` 一致）。
3. **画像不是固定不变**：在预测完当前窗口后，根据**该窗口上的真实 vs 预测误差**更新画像（纠偏），再用更新后的画像去预测**下一个**窗口。Clasp 主方法对应 `clasp_online`（候选精炼 + 按 Q 选优，**不写入 DPO 对**）。

## 入口

### 窗口链多基线

`python3 -m comparison.run_baseline_comparison --split test`

- **clasp_online**：上述「误差驱动更新画像」的 Clasp 测试流程。
- **static_s0**：仅用 W0 初始画像预测 W1…W4（画像始终不变），作对照。
- **prefix_refresh**：每步用已观测前缀 W0…W_{k-1} 整段重算「初始画像」，作对照。

可选：`--methods clasp_online,static_s0`、`--max-users 20`、`--skip-window-split`、`--windowed-root output/windowed`。

输出默认：`output/comparison/baseline_chain_<split>.jsonl`（每用户、每个 method 一行，含各步 F/L/Q 与 mean_Q）。

### PolicySim（baseline 比较方法之一）

`comparison/policysim/` 为策略智能体与相关训练脚本，与其它基线并列用于对比；说明见 `comparison/policysim/README.md` 与 `Policysim.pdf`。

## 指标（与 `src/scorer.py` 一致）

| 符号 | 含义 |
|------|------|
| **F** | 决策加权 F1（交互动作类型对齐） |
| **L** | post/reply 预测文本与真实的语义余弦均值 |
| **Q** | α·F + (1-α)·L_norm（见 config） |

## 目录说明

| 路径 | 作用 |
|------|------|
| `comparison/window_chain_eval.py` | 窗口链策略核心（W1…W4、可选纠偏） |
| `comparison/run_baseline_comparison.py` | 窗口链多基线 CLI |
| `comparison/policysim/` | Baseline 之一：策略仿真子包 |
| `comparison/Clasp/profile_client.py` | 微调画像 API 封装（供需单独指定画像端点时使用） |
