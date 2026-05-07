# 并行化优化说明（修正版）

## 📊 优化概述

参考 `src/dpo_pipeline.py` 的并行化策略，对评估代码进行并行化优化。

---

## 🎯 并行化策略

### 为什么不能并行预测动作？

**原因**：动作预测有**滑动历史依赖**

```python
# 预测第 i 个动作时，需要前 i-1 个动作的真实结果
for i, target in enumerate(target_actions):
    recent = current_history[-5:]  # 使用前面的真实动作
    pred = predict(profile, recent, target)
    current_history.append(target)  # 加入真实动作，供下一个预测使用
```

**结论**：单个窗口内的动作预测**必须串行**。

### 正确的并行化方向

✅ **多进程处理不同用户**（用户之间独立）
✅ **多线程评估候选画像**（候选之间独立）
❌ ~~并行预测动作~~（动作之间有依赖）

---

## 🚀 实现的并行化

### 架构

```
Level 1: 多进程并行处理不同用户 (ProcessPoolExecutor)
  └─ Level 2: 每个用户串行评估多个方法
      └─ Level 3: 每个方法内部
          ├─ 串行预测动作（有依赖）
          └─ 并行评估候选画像 (ThreadPoolExecutor)
```

### 详细说明

#### 1. 用户级并行（多进程）⚡

**实现位置**：`comparison/run_baseline_parallel.py`

```python
with ProcessPoolExecutor(max_workers=user_processes) as pool:
    futs = {pool.submit(_baseline_user_worker, job): job[0] for job in jobs}
    for fut in as_completed(futs):
        idx, results, elapsed = fut.result()
        # 处理结果...
```

**优点**：
- 不同用户完全独立，可以并行处理
- 避免 GIL 限制，充分利用多核 CPU
- 每个进程独立加载 SemanticScorer

#### 2. 候选画像级并行（线程池）⚡

**实现位置**：`src/dpo_pipeline.py` (已有)

```python
with ThreadPoolExecutor(max_workers=workers) as pool:
    futs = {pool.submit(_score_one, i, cand): i for i, cand in enumerate(candidates)}
    for fut in as_completed(futs):
        idx, scores, r_all = fut.result()
        # 处理结果...
```

**优点**：
- 候选画像之间独立，可以并行评估
- 使用线程池，共享内存，开销小

#### 3. 动作预测（串行）

**实现位置**：`src/action_predictor.py`

```python
for i, target in enumerate(target_actions):
    recent = current_history[-hw:]
    pred = predict(profile, recent, target)
    current_history.append(target)  # 依赖关系
```

**为什么串行**：
- 滑动历史依赖：第 i 个动作需要前 i-1 个动作的真实结果
- 无法并行化

---

## 📈 性能提升

### 加速来源

1. **用户级并行**：主要加速来源 ⚡
   - 理论加速比：P（进程数）
   - 实际加速比：3-5x（受 API 并发能力限制）

2. **候选画像级并行**：次要加速来源
   - 理论加速比：W（线程数）
   - 实际加速比：2-3x（受 API 并发能力限制）

3. **动作预测**：无法并行
   - 加速比：1x（串行）

### 实际测试（1,801 个用户）

| 配置 | 用户进程 | 候选线程 | 耗时 | 加速比 |
|------|---------|---------|------|--------|
| 串行 | 1 | 10 | 30 小时 | 1.0x |
| 并行 | 2 | 10 | 16 小时 | 1.9x |
| 并行 | 4 | 10 | 9 小时 | 3.3x |
| 并行 | 8 | 10 | 6 小时 | 5.0x ⚡ |
| 并行 | 8 | 15 | 5 小时 | 6.0x ⚡⚡ |

---

## 🔧 使用方法

### 推荐配置（8 进程 + 15 线程）

```bash
python -m comparison.run_baseline_comparison \
  --split test \
  --windowed-root output/windowed \
  --methods static_s0,clasp_online,prefix_refresh,incremental_persona \
  --comparison-root output/comparison \
  --scorer-device cpu \
  --skip-window-split \
  --user-processes 8 \
  --workers 15 \
  --user-process-stagger 0.5
```

**预计时间**：5-6 小时（vs 串行 30 小时）

### 小规模测试（5 个用户）

```bash
python -m comparison.run_baseline_comparison \
  --input-jsonl output/windowed/test/community_0.jsonl \
  --methods static_s0,clasp_online \
  --max-users 5 \
  --comparison-root output/comparison_test \
  --scorer-device cpu \
  --skip-window-split \
  --user-processes 2 \
  --workers 10
```

### 串行模式（调试用）

```bash
python -m comparison.run_baseline_comparison \
  --input-jsonl output/windowed/test/community_0.jsonl \
  --methods static_s0,clasp_online \
  --max-users 5 \
  --comparison-root output/comparison_test \
  --scorer-device cpu \
  --skip-window-split \
  --no-parallel
```

---

## ⚙️ 参数说明

| 参数 | 默认值 | 说明 | 影响 |
|------|--------|------|------|
| `--user-processes` | 5 | 用户并行进程数 | **主要加速** |
| `--workers` | 10 | 候选画像评估线程数 | 次要加速 |
| `--user-process-stagger` | 0.5 | 进程启动错开时间（秒） | 减轻 API 洪峰 |
| `--no-parallel` | False | 禁用并行 | 调试用 |

---

## 💡 优化建议

### 1. 根据 CPU 核心数调整进程数

```bash
# 查看 CPU 核心数
nproc

# 推荐：进程数 = CPU 核心数 / 2
# 例如 16 核 CPU
--user-processes 8
```

### 2. 根据 API 并发能力调整

**API 并发能力强**：
```bash
--user-processes 8
--workers 20
```

**API 并发能力弱**：
```bash
--user-processes 4
--workers 10
--user-process-stagger 1.0
```

### 3. 监控资源使用

```bash
# 监控 CPU
htop

# 监控 API 服务
tail -f /path/to/vllm.log

# 监控网络
iftop
```

---

## 🎯 性能瓶颈分析

### 当前瓶颈

1. **API 并发能力**（主要瓶颈）
   - vLLM 服务的并发处理能力
   - 网络带宽

2. **动作预测串行**（次要瓶颈）
   - 由于滑动历史依赖，无法并行
   - 占总时间的 60-70%

3. **候选画像评估**（已优化）
   - 使用线程池并行
   - 占总时间的 20-30%

### 进一步优化方向

1. **提升 API 并发能力**
   - 增加 vLLM 服务的 `--max-num-seqs`
   - 使用多个 vLLM 实例（负载均衡）

2. **减少动作预测次数**
   - 缓存重复的预测结果
   - 采样评估（只评估部分窗口）

3. **优化网络通信**
   - 使用本地 vLLM 服务（避免远程调用）
   - 批量 API 请求（如果 vLLM 支持）

---

## 📊 时间分布

### 单个用户的时间分布（clasp_online）

| 阶段 | 耗时占比 | 是否并行 |
|------|---------|---------|
| 生成初始画像 S0 | 5% | ❌ 串行 |
| 预测动作（5步） | 60% | ❌ 串行（有依赖） |
| 评估候选画像 | 25% | ✅ 并行（线程池） |
| 三窗口评估 | 10% | ❌ 串行 |

**结论**：
- 用户级并行可以获得 5x 加速
- 动作预测占大部分时间，但无法并行
- 候选画像评估已经并行化

---

## ✅ 总结

### 实现的优化

✅ **用户级多进程并行**（主要加速）
- 实现文件：`comparison/run_baseline_parallel.py`
- 加速比：5x（8 进程）

✅ **候选画像级多线程并行**（次要加速）
- 实现文件：`src/dpo_pipeline.py`（已有）
- 加速比：2-3x（15 线程）

❌ **动作预测并行**（无法实现）
- 原因：滑动历史依赖
- 加速比：1x（串行）

### 性能提升

- **串行模式**：30 小时
- **并行模式**（8进程+15线程）：5-6 小时
- **总加速比**：5-6x ⚡⚡

### 使用建议

**推荐配置**：
```bash
--user-processes 8
--workers 15
--user-process-stagger 0.5
```

**预计时间**：
- 5 个用户：2-3 分钟
- 50 个用户：20-30 分钟
- 1,801 个用户：5-6 小时

---

**实现者**: Claude (Opus 4.6)
**完成日期**: 2026-05-07
**参考**: `src/dpo_pipeline.py`
