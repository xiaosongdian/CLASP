"""
从 grid_contiguous_plot_stats.csv 绘制各社区 Q 随窗口（W1–W5）的变化；
每个社区一个子图（2 行 × 3 列），每种方法一条折线。风格对齐 example_code.py。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 与 example_code.py 一致：优先 Times New Roman；当前环境无该字体时由 matplotlib 依次回退
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = [
    "Times New Roman",
    "Liberation Serif",
    "DejaVu Serif",
    "Nimbus Roman",
]
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"

# CSV 中的 method 键 → 论文展示名
METHOD_DISPLAY = {
    "static_s0": "Static_Persona",
    "incremental_persona": "Incremental_Persona",
    "prefix_refresh": "Regeneration_Persona",
    "history_only": "Full_History",
    "clasp_online": "Clasp",
}

METHOD_ORDER = list(METHOD_DISPLAY.keys())

# 顶会常用定性色（色相拉开）
METHOD_COLORS = {
    "static_s0": "#4E79A7",
    "incremental_persona": "#59A14F",
    "prefix_refresh": "#F28E2B",
    "history_only": "#B07AA1",
    "clasp_online": "#E15759",
}

METHOD_LINEWIDTH = {"clasp_online": 2.4}
METHOD_ZORDER = {"clasp_online": 4}
_DEFAULT_LW = 1.8
_DEFAULT_Z = 2


def _style_axes(ax: plt.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_edgecolor("black")
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.grid(axis="y", ls="--", alpha=0.3)


def plot_q_by_community(
    csv_path: Path,
    out_png: Path,
    out_pdf: Path | None = None,
    dpi: int = 300,
) -> None:
    df = pd.read_csv(csv_path)
    communities = sorted(df["community_id"].unique())
    if len(communities) != 6:
        raise ValueError(f"期望 6 个社区，实际为 {len(communities)}: {communities}")

    fig, axes = plt.subplots(2, 3, figsize=(12.6, 7.8), constrained_layout=False)
    # 子图上沿与图例底边：间距略收，使图例与 Community 标题更近（仍留一点防叠字）
    _axes_top = 0.883
    _legend_bottom = 0.918
    fig.subplots_adjust(
        hspace=0.28, wspace=0.22, bottom=0.14, top=_axes_top, left=0.08, right=0.98
    )

    q_all = df["Q"].astype(float)
    y_min = float(q_all.min())
    y_max = float(q_all.max())
    pad = max((y_max - y_min) * 0.12, 0.01)
    y_axis_min = max(0.0, y_min - pad)
    y_axis_max = y_max + pad

    window_ticks = None
    for ax, cid in zip(np.ravel(axes), communities):
        sub = df[df["community_id"] == cid]
        for meth in METHOD_ORDER:
            msub = sub[sub["method"] == meth].sort_values("step_order")
            if msub.empty:
                continue
            x = msub["step_order"].to_numpy()
            y = msub["Q"].to_numpy(dtype=float)
            if window_ticks is None:
                window_ticks = msub["target_window_label"].tolist()
            ax.plot(
                x,
                y,
                lw=METHOD_LINEWIDTH.get(meth, _DEFAULT_LW),
                marker="o",
                ms=5,
                color=METHOD_COLORS.get(meth, "#333333"),
                label=METHOD_DISPLAY.get(meth, meth),
                zorder=METHOD_ZORDER.get(meth, _DEFAULT_Z),
            )

        _style_axes(ax)
        # pad 略小：标题更靠坐标轴上沿，减少侵入图例区域
        ax.set_title(f"Community {cid}", fontsize=13, pad=3)
        ax.set_xlim(-0.15, 4.35)
        ax.set_xticks(np.arange(5))
        if window_ticks and len(window_ticks) == 5:
            ax.set_xticklabels(window_ticks, fontsize=13)
        else:
            ax.set_xticklabels([f"W{i}" for i in range(1, 6)], fontsize=13)
        ax.tick_params(axis="y", labelsize=13)
        ax.set_ylim(y_axis_min, y_axis_max)

    axes_flat = np.ravel(axes)
    axes_flat[0].set_ylabel("Q", fontsize=15, rotation=90)
    axes_flat[3].set_ylabel("Q", fontsize=15, rotation=90)
    for ax in (axes_flat[1], axes_flat[2], axes_flat[4], axes_flat[5]):
        ax.set_yticklabels([])

    for ax in axes_flat[3:]:
        ax.set_xlabel("Window", fontsize=14)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, _legend_bottom),
        ncol=5,
        fontsize=14,
        frameon=True,
        edgecolor="black",
        fancybox=False,
        columnspacing=1.35,
        handlelength=2.8,
        labelspacing=0.6,
    )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=dpi, bbox_inches="tight")
    if out_pdf is not None:
        plt.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--csv",
        type=Path,
        default=root / "data" / "grid_contiguous_plot_stats.csv",
        help="plot_stats CSV 路径",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=root / "grid_contiguous_q_by_community.png",
        help="输出 PNG",
    )
    p.add_argument(
        "--pdf",
        type=Path,
        default=None,
        help="可选输出 PDF（默认不写）",
    )
    p.add_argument("--dpi", type=int, default=300)
    args = p.parse_args()
    plot_q_by_community(args.csv, args.out, args.pdf, dpi=args.dpi)
    print(f"已写入: {args.out.resolve()}")
    if args.pdf is not None:
        print(f"已写入: {args.pdf.resolve()}")


if __name__ == "__main__":
    main()
