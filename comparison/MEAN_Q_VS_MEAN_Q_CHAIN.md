# mean_Q vs mean_Q_chain 说明

## 📊 指标说明

### 当前状态

在当前实现中，这两组指标**完全相同**：

```python
mean_Q == mean_Q_chain
mean_F == mean_F_chain
```

### 原因

所有步骤都是**前向跨窗步**（forward chain steps），因此：

```python
steps_out = [step0, step1, step2, step3, step4]
chain_steps = steps_out  # 完全相同

mean_Q = average([step0.Q, step1.Q, step2.Q, step3.Q, step4.Q])
mean_Q_chain = average([step0.Q, step1.Q, step2.Q, step3.Q, step4.Q])  # 相同
```

---

## 🔍 历史背景

### 可能的旧版设计

在某个历史版本中，可能存在两种评估方式：

#### 1. 标准评估（mean_Q / mean_F）

可能只包含某些特定步骤，例如：
- 只包含画像更新后的步骤
- 只包含主要评估步骤

#### 2. 窗口链评估（mean_Q_chain / mean_F_chain）

包含所有前向跨窗步骤：
- W0 → W1
- W1 → W2
- W2 → W3
- W3 → W4
- W4 → W5

### 当前设计

现在**所有步骤都是窗口链步骤**，所以两者相同。

---

## 🔧 代码优化

### 优化前

```python
qs = [float(s["Q"]) for s in steps_out]
fs = [float(s["F"]) for s in steps_out]
mean_q = sum(qs) / len(qs) if qs else None
mean_f = sum(fs) / len(fs) if fs else None

# 重复计算
chain_steps = list(steps_out)
qc = [float(s["Q"]) for s in chain_steps]
fc = [float(s["F"]) for s in chain_steps]
mean_q_chain = sum(qc) / len(qc) if qc else None
mean_f_chain = sum(fc) / len(fc) if fc else None
```

### 优化后

```python
qs = [float(s["Q"]) for s in steps_out]
fs = [float(s["F"]) for s in steps_out]
mean_q = sum(qs) / len(qs) if qs else None
mean_f = sum(fs) / len(fs) if fs else None

# 直接复用，避免重复计算
return {
    "mean_Q": mean_q,
    "mean_F": mean_f,
    "mean_Q_chain": mean_q,  # 相同
    "mean_F_chain": mean_f,  # 相同
}
```

---

## 📈 使用建议

### 分析时使用哪个？

**推荐使用 `mean_Q` 和 `mean_F`**：

```python
import json

with open('output/comparison/clasp_online/baseline_chain_test.jsonl', 'r') as f:
    results = [json.loads(line) for line in f]

# 使用 mean_Q（推荐）
mean_qs = [r['mean_Q'] for r in results]
print(f"平均 Q 值: {np.mean(mean_qs):.4f}")

# 或者使用 mean_Q_chain（结果相同）
mean_q_chains = [r['mean_Q_chain'] for r in results]
print(f"平均 Q 值（chain）: {np.mean(mean_q_chains):.4f}")

# 两者相同
assert mean_qs == mean_q_chains
```

### 为什么保留 mean_Q_chain？

**向后兼容**：
- 下游脚本可能依赖 `mean_Q_chain` 字段
- 保留该字段避免破坏现有代码
- 未来可以考虑删除

---

## 🎯 总结

### 当前状态

✅ `mean_Q` == `mean_Q_chain`（完全相同）
✅ `mean_F` == `mean_F_chain`（完全相同）
✅ 保留 `mean_Q_chain` 字段供向后兼容

### 使用建议

✅ **分析时使用 `mean_Q` 和 `mean_F`**
✅ 忽略 `mean_Q_chain` 和 `mean_F_chain`（除非需要兼容旧代码）

### 代码优化

✅ 避免重复计算，直接复用 `mean_q` 和 `mean_f`
✅ 代码更简洁，性能更好

---

**说明者**: Claude (Opus 4.6)
**日期**: 2026-05-07
