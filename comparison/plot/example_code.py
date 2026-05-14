import matplotlib.pyplot as plt
import numpy as np

# -------------------------------------------------
# 1）全局字体加粗
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'

# 数据
labels = ['Low-Temperature(= 0.1)\nwithout rollback',
          'High-Temperature(= 0.7)\nwithout rollback',
          'High-Temperature(= 0.7)\nwith rollback']

eval_lines = [
    [2.0, 2.1, 1.8, 1.5, 1.4, 1.7, 1.6],
    [2.0, 4.0, 3.8, 3.2, 3.1, 2.9, 3.0],
    [2.0, 1.5, 3.5, 1.5, 1.0, 0.05]
]
rollback_idx = [[1, 2], [], [2]]
success_idx = [3, 6, 7]

success_rates = [67, 72, 90]
hallucination = [0.6, 1.8, 1.9]

color_brick = '#C14958'
color_gray = '#5C5C5C'
color_black = '#000000'

# -------------------------------------------------
fig = plt.figure(figsize=(8.4, 5.2))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], hspace=0.22, wspace=0.25)

y_axis_min = 0
y_axis_max = 4.2
major_loc = 1

for col, (curve, rolls, succ, sr, hi, lab) in enumerate(
        zip(eval_lines, rollback_idx, success_idx, success_rates, hallucination, labels)):
    # ---- 上方折线 ----
    ax_line = fig.add_subplot(gs[0, col])
    steps = np.arange(len(curve))
    ax_line.plot(steps, curve, lw=1.8, marker='o', ms=5, color=color_brick)
    
    # 移动坐标轴到0点
    ax_line.spines['left'].set_position(('data', 0))
    ax_line.spines['bottom'].set_position(('data', 0))
    
    # 1. 精确设置 x 轴范围，确保上边框与数据对齐
    x_max = len(curve) - 1
    ax_line.set_xlim(0, x_max+0.3)
    ax_line.spines['top'].set_bounds(0, x_max+0.3)  # 限制上边框从 x=0 到 x=x_max
    
    # 2. 设置边框端点样式为 projecting，确保角落闭合
    # ax_line.spines['top'].set_capstyle('projecting')
    # ax_line.spines['right'].set_capstyle('projecting')
    # ax_line.spines['left'].set_capstyle('projecting')
    # ax_line.spines['bottom'].set_capstyle('projecting')
    
    # 3. 确保所有边框可见且样式一致
    ax_line.spines['top'].set_visible(True)
    ax_line.spines['right'].set_visible(True)
    ax_line.spines['left'].set_visible(True)
    ax_line.spines['bottom'].set_visible(True)
    
    for spine in ax_line.spines.values():
        spine.set_linewidth(1.0)
        spine.set_edgecolor('black')
    
    if col == 2:
        ax_line.scatter(len(curve) - 1, curve[-1], s=100, marker='*', color='lime', edgecolors='k', zorder=6)
        max_idx = int(np.argmax(curve))
        ax_line.text(max_idx, curve[max_idx] - 0.3, '😢',
                     fontsize=21, ha='center', va='bottom', zorder=7,
                     fontproperties={'family': 'DejaVu Sans', 'weight': 'normal'})
        ax_line.text(max_idx, curve[max_idx] - 0.23, 'rollback', fontsize=12, color=color_black,
                     ha='center', va='top', zorder=7, weight='bold')

    ax_line.text(1.21, 0.03, 'steps', fontsize=14, color=color_black,
                 ha='right', va='bottom', transform=ax_line.transAxes, zorder=8,
                 weight='bold')
    
    ax_line.set_ylim(y_axis_min, y_axis_max)
    ax_line.yaxis.set_major_locator(plt.MultipleLocator(major_loc))
    
    # 隐藏 y 轴的 0 标签（只保留 x 轴的 0）
    yticks = ax_line.get_yticks()
    yticklabels = ['' if t == 0 else str(int(t)) for t in yticks]
    ax_line.set_yticklabels(yticklabels)
    
    ax_line.set_title(lab, fontsize=14)
    ax_line.grid(axis='y', ls='--', alpha=0.3)
    ax_line.set_xticks(steps[::max(1, len(steps) // 4)])
    ax_line.tick_params(axis='x', labelsize=13)
    ax_line.tick_params(axis='y', labelsize=13)
    
    if col == 0:
        ax_line.text(-0.25, 0.5, 'Incorrectness Score', fontsize=15, rotation=90,
                     ha='center', va='center', transform=ax_line.transAxes, weight='bold')
    else:
        ax_line.set_yticklabels([])

    # ---- 下方柱状图 ----
    ax_bar = fig.add_subplot(gs[1, col])
    ax_bar_right = ax_bar.twinx()
    ax_bar.set_xlim(-0.5, 0.5)
    bar_w = 0.25
    dist = 0.45
    x0 = -dist / 2
    x1 = dist / 2
    ax_bar.bar(x0, sr, bar_w, color=color_brick)
    ax_bar_right.bar(x1, hi, bar_w, color=color_gray)

    ax_bar.set_xticks([x0, x1])
    ax_bar.set_xticklabels(['Success\nRate', 'Hallucination\nScore'], fontsize=13, weight='bold')
    ax_bar.plot([x0 - bar_w / 2, x0 + bar_w / 2], [sr, sr], ls='--', color=color_brick, lw=1.2)
    ax_bar.text(x0, sr + 1, f'{sr}%', ha='center', va='bottom', color=color_brick, fontsize=13,
                weight='bold')
    ax_bar_right.plot([x1 - bar_w / 2, x1 + bar_w / 2], [hi, hi], ls='--', color=color_gray, lw=1.2)
    ax_bar_right.text(x1, hi + 0.1, f'{hi}', ha='center', va='bottom', color=color_gray, fontsize=13,
                      weight='bold')

    ax_bar.set_ylim(0, 100)
    ax_bar_right.set_ylim(0, 2.5)
    ax_bar_right.yaxis.set_major_locator(plt.MultipleLocator(0.5))

    if col == 0:
        ax_bar.text(-0.25, 0.5, 'Semantic Accuracy Rate %', fontsize=15, rotation=90,
                    ha='center', va='center', transform=ax_bar.transAxes, color=color_brick,
                    weight='bold')
        ax_bar.tick_params(axis='y', labelsize=13)
        ax_bar_right.set_yticklabels([])
    elif col == 2:
        ax_bar_right.text(1.25, 0.5, 'Hallucination Score', fontsize=16, rotation=90,
                          ha='center', va='center', transform=ax_bar.transAxes, color=color_gray,
                          weight='bold')
        ax_bar_right.tick_params(axis='y', labelsize=13)
        ax_bar.set_yticklabels([])
    else:
        ax_bar.set_yticklabels([])
        ax_bar_right.set_yticklabels([])
    ax_bar.grid(axis='y', ls='--', alpha=0.3)

    x_center = (x0 + x1) / 2
    ax_bar.text(x_center, -20, f'({chr(97 + col)})', ha='center', va='top', fontsize=18, weight='bold',
                transform=ax_bar.transData)

# 保存
plt.savefig('figure.png', dpi=800, bbox_inches='tight')
plt.savefig('figure.pdf', bbox_inches='tight')
plt.show()