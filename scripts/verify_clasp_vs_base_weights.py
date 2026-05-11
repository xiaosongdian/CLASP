#!/usr/bin/env python3
"""
对比两个 HuggingFace 格式目录下的 safetensors 权重，用于验证「微调/合并后」与「基座」
在数值上是否不同，从而辅助判断「不是误指向了同一份底模」。

说明（重要）：

1) **能回答什么**  
   - 对应张量名、形状一致时，逐元素 |Δw| 的均值/最大值/非零比例。  
   - 若整体 diff 为 0，且 key 完全对齐，则两目录极可能为同一份权重（或完全未改动的拷贝）。

2) **不能单独证明什么**  
   - 权值有变化 **≠** 下游任务一定更好；任务有效性需用 **F/L/Q、人工阅读画像** 等。  
   - 若你对比的是 **LoRA 适配器目录 vs 基座**，形状可能对不齐——应用 `--adapter-path` 只做适配器内部统计，或与合并后的全量目录对比。

3) **与 Clasp 的关系**  
   - `COMPARISON_CLASP_PROFILE_VLLM_MODEL` / `COMPARISON_CLASP_ACTION_VLLM_MODEL` 在 config 里常为 **vLLM 注册的模型名字符串**。  
   - 本脚本需要 **本机磁盘上的模型目录**（如 `/data/LLM_models/Meta-Llama-3-8B-Instruct`），不能直接用 API 上的别名。

依赖：pip install safetensors numpy torch
（Llama-3 权重含 bfloat16，需用 PyTorch 读入再转 numpy 做差分；仅用 numpy 会报 dtype 错误。）

用法示例：

  python scripts/verify_clasp_vs_base_weights.py \\
    --base /data/LLM_models/Meta-Llama-3-8B-Instruct \\
    --finetuned /data/LLM_models/Meta-Llama-3-8B-Instruct-clasp-dpo-stage2

  python scripts/verify_clasp_vs_base_weights.py \\
    --base ... \\
    --finetuned ... \\
    --max-keys 50

"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


def _index_tensor_locations(root: Path) -> Dict[str, Path]:
    """ tensor_name -> safetensors 文件路径 """
    idx: Dict[str, Path] = {}
    files = sorted(root.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"未找到 *.safetensors: {root}")
    try:
        from safetensors import safe_open
    except ImportError as e:
        raise SystemExit("请先安装: pip install safetensors") from e
    try:
        import torch  # noqa: F401
    except ImportError as e:
        raise SystemExit("列出张量需 PyTorch（与 bfloat16 权重一致）: pip install torch") from e

    for fp in files:
        with safe_open(fp, framework="pt", device="cpu") as f:
            for k in f.keys():
                idx[k] = fp
    return idx


def _load_tensor_as_float64_np(root_idx: Dict[str, Path], key: str):
    """加载张量并转为 float64 numpy（支持 bfloat16，避免 safetensors+numpy 不识别 bf16）。"""
    from safetensors import safe_open

    try:
        import torch
    except ImportError as e:
        raise SystemExit("加载 bfloat16 权重需: pip install torch") from e

    fp = root_idx[key]
    with safe_open(fp, framework="pt", device="cpu") as f:
        t = f.get_tensor(key)
    return t.detach().cpu().to(dtype=torch.float64).numpy()


def compare_trees(
    base: Path,
    finetuned: Path,
    *,
    max_keys: int | None,
    fp16_tol: float,
) -> Dict[str, Any]:
    import numpy as np

    base = Path(base).resolve()
    ft = Path(finetuned).resolve()
    ib = _index_tensor_locations(base)
    ift = _index_tensor_locations(ft)

    keys_b = set(ib.keys())
    keys_f = set(ift.keys())
    common = sorted(keys_b & keys_f)
    only_b = sorted(keys_b - keys_f)
    only_f = sorted(keys_f - keys_b)

    report: Dict[str, Any] = {
        "base": str(base),
        "finetuned": str(ft),
        "n_keys_base": len(keys_b),
        "n_keys_finetuned": len(keys_f),
        "n_keys_common": len(common),
        "keys_only_in_base": only_b[:200],
        "keys_only_in_finetuned": only_f[:200],
    }

    if not common:
        report["error"] = "无共同张量名，无法逐层对比（可能格式或模型不同）"
        return report

    per_key: List[Dict[str, Any]] = []
    total_el = 0
    wsum_abs = 0.0
    wmax_glob = 0.0
    n_nonzero = 0
    n_total_params = 0

    use_keys = common
    if max_keys is not None and max_keys > 0:
        use_keys = common[: int(max_keys)]

    for k in use_keys:
        tb = _load_tensor_as_float64_np(ib, k)
        tf = _load_tensor_as_float64_np(ift, k)
        if tb.shape != tf.shape:
            per_key.append(
                {
                    "key": k,
                    "error": f"shape {tb.shape} vs {tf.shape}",
                }
            )
            continue
        d = np.abs(tf - tb)
        n = d.size
        m = float(d.mean()) if n else 0.0
        mx = float(d.max()) if n else 0.0
        nz = int(np.count_nonzero(d > fp16_tol))
        total_el += n
        wsum_abs += float(d.sum())
        wmax_glob = max(wmax_glob, mx)
        n_nonzero += nz
        n_total_params += n
        per_key.append(
            {
                "key": k,
                "numel": n,
                "mean_abs_diff": m,
                "max_abs_diff": mx,
                "frac_nonzero_gt_tol": (nz / n) if n else 0.0,
            }
        )

    per_key.sort(key=lambda x: x.get("mean_abs_diff", 0.0), reverse=True)
    report["per_key_top"] = per_key[:30]
    report["aggregate_on_sampled_keys"] = {
        "keys_compared": len(use_keys),
        "weighted_mean_abs_diff": (wsum_abs / total_el) if total_el else 0.0,
        "max_abs_diff": wmax_glob,
        "total_elements": total_el,
        "fraction_nonzero_gt_tol": (n_nonzero / n_total_params) if n_total_params else 0.0,
        "fp16_tol": fp16_tol,
    }

    agg = report["aggregate_on_sampled_keys"]
    mam = agg["weighted_mean_abs_diff"]
    if mam == 0.0 and len(common) == len(use_keys) and not only_b and not only_f:
        report["interpretation_hint"] = (
            "采样范围内权重完全一致：很可能两路径指向同一份文件或均为未改动拷贝。"
        )
    elif mam > 0.0:
        report["interpretation_hint"] = (
            "权重存在数值差异：微调/合并确实改变了参数（仍需任务指标验证有效性）。"
        )

    return report


def main() -> None:
    p = argparse.ArgumentParser(description="对比基座与微调目录下 safetensors 权重差异")
    p.add_argument("--base", type=Path, required=True, help="基座模型目录（含 *.safetensors）")
    p.add_argument(
        "--finetuned",
        type=Path,
        required=True,
        help="微调/合并后模型目录（含 *.safetensors）",
    )
    p.add_argument(
        "--max-keys",
        type=int,
        default=None,
        help="仅对比前 N 个共同 key（按字母序；调试/省时）",
    )
    p.add_argument(
        "--fp16-tol",
        type=float,
        default=1e-6,
        help="视为「非零差异」的最小 |Δ|（默认 1e-6，兼容 fp16 噪声）",
    )
    p.add_argument("--json", type=Path, default=None, help="可选：写入完整 JSON 报告")
    args = p.parse_args()

    rep = compare_trees(
        args.base,
        args.finetuned,
        max_keys=args.max_keys,
        fp16_tol=float(args.fp16_tol),
    )

    agg = rep.get("aggregate_on_sampled_keys")
    print("[verify_weights] base:", rep.get("base"), flush=True)
    print("[verify_weights] finetuned:", rep.get("finetuned"), flush=True)
    print(
        f"[verify_weights] keys: base={rep.get('n_keys_base')} "
        f"finetuned={rep.get('n_keys_finetuned')} common={rep.get('n_keys_common')}",
        flush=True,
    )
    if rep.get("keys_only_in_base"):
        print(f"[verify_weights] only_in_base (trunc): {rep['keys_only_in_base'][:5]}...", flush=True)
    if rep.get("keys_only_in_finetuned"):
        print(
            f"[verify_weights] only_in_finetuned (trunc): {rep['keys_only_in_finetuned'][:5]}...",
            flush=True,
        )
    if agg:
        print(
            "[verify_weights] aggregate:",
            json.dumps(agg, indent=2, ensure_ascii=False),
            flush=True,
        )
    if rep.get("interpretation_hint"):
        print("[verify_weights] =>", rep["interpretation_hint"], flush=True)
    if rep.get("per_key_top"):
        print("[verify_weights] top mean_abs_diff keys:", flush=True)
        for row in rep["per_key_top"][:10]:
            if "mean_abs_diff" in row:
                print(
                    f"    {row['key'][:80]}... mean_abs={row['mean_abs_diff']:.6g}",
                    flush=True,
                )

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[verify_weights] JSON -> {args.json.resolve()}", flush=True)


if __name__ == "__main__":
    main()
