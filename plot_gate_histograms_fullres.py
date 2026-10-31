# -*- coding: utf-8 -*-
"""Figure 2（全分辨率版）：各模块、各阶段的通道门控 sigmoid 分布。

数据来自 gate_stats_fullres.py 保存的 figs/gates_fullres_{module}_seed{seed}.npz
（每个文件是 all 配置下、该 seed 在完整验证集上的门控激活）。
三个 seed 的激活合并绘制，标题中的 mean/std/%>0.9 也在合并样本上计算。

输出：figs/gate_histograms_fullres.png（300 dpi，满足 Elsevier 投稿要求）
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paths import FIGURES as FIG_DIR
OUT = os.path.join(FIG_DIR, "gate_histograms_fullres.png")
MODULES = ["se", "cbam", "eca"]
SEEDS = [0, 1, 2]
STAGE_NAMES = ["att$_1$ (layer1)", "att$_2$ (layer2)",
               "att$_3$ (layer3)", "att$_4$ (layer4)"]

FILL = "#4C78A8"      # 单一色相：每格只有一条序列，颜色不承担信息
REF = "#8C8C8C"       # 0.5 参考线（浅层塌缩的视觉基准）

plt.rcParams.update({
    "font.size": 9,
    "axes.edgecolor": "#B0B0B0",
    "axes.linewidth": 0.8,
    "xtick.color": "#404040",
    "ytick.color": "#404040",
    "axes.labelcolor": "#404040",
    "axes.titlesize": 9,
})


def load(module, stage):
    parts = []
    for s in SEEDS:
        p = os.path.join(FIG_DIR, f"gates_fullres_{module}_seed{s}.npz")
        if os.path.exists(p):
            with np.load(p) as d:
                if f"att{stage}" in d:
                    parts.append(d[f"att{stage}"])
    return np.concatenate(parts) if parts else None


def main():
    # 期刊版：不画 suptitle（图注已说明），把纵向空间留给子图，便于缩小排版时阅读
    fig, axes = plt.subplots(3, 4, figsize=(15, 8.6), sharex=True)

    for r, module in enumerate(MODULES):
        for c in range(4):
            ax = axes[r, c]
            w = load(module, c + 1)
            if w is None:
                ax.axis("off")
                continue

            # density=True：各 stage 的样本量差 8 倍（att1 12 万 vs att4 96 万），
            # 归一化后子图之间才可直接比较形状
            ax.hist(w, bins=50, range=(0, 1), density=True, color=FILL,
                    alpha=0.9, edgecolor="white", linewidth=0.2)
            ax.axvline(0.5, color=REF, linestyle="--", linewidth=0.9, zorder=0)

            ax.set_title(
                f"{module.upper()}  {STAGE_NAMES[c]}\n"
                f"mean={w.mean():.3f}  std={w.std():.3f}  "
                f"$>$0.9: {100 * (w > 0.9).mean():.1f}%",
                fontsize=8.5, color="#202020")
            ax.set_xlim(0, 1)
            ax.grid(axis="y", color="#E8E8E8", linewidth=0.6, zorder=0)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)

            if c == 0:
                ax.set_ylabel("density")
            if r == 2:
                ax.set_xlabel("channel gate value $s$")

    fig.tight_layout()
    fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"saved {OUT}", flush=True)


if __name__ == "__main__":
    main()
