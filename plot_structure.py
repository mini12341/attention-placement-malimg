# -*- coding: utf-8 -*-
"""ResNet18 放置结构图：backbone 流程 + 4 个可选注意力插入点 + 5 种放置组合标注。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

import os

from paths import FIGURES

OUT = os.path.join(FIGURES, "resnet18_placement.png")


def box(ax, x, y, w, h, text, fc, fs=9, rounded=False):
    style = "round,pad=0.02" if rounded else "square,pad=0.02"
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style,
                                fc=fc, ec="black", lw=1.1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def main():
    fig, ax = plt.subplots(figsize=(15, 5.6))
    ax.set_xlim(0, 15); ax.set_ylim(0, 6); ax.axis("off")

    # main flow (top row)
    y, h = 3.9, 0.85
    blocks = [
        (0.15, "Input\n224\u00d7224", "#eef2f7"),
        (1.55, "conv1\n7\u00d77, 64", "#dbe7f5"),
        (2.95, "layer1\n64 \u00d7 2", "#c3d9f0"),
        (4.55, "layer2\n128 \u00d7 2", "#abcbec"),
        (6.15, "layer3\n256 \u00d7 2", "#93bde7"),
        (7.75, "layer4\n512 \u00d7 2", "#7bafe3"),
        (9.35, "GAP\n+ FC(25)", "#eef2f7"),
    ]
    for x, t, c in blocks:
        box(ax, x, y, 1.2, h, t, c)
    for i in range(len(blocks) - 1):
        x1 = blocks[i][0] + 1.2
        x2 = blocks[i + 1][0]
        ax.annotate("", xy=(x2, y + h / 2), xytext=(x1, y + h / 2),
                    arrowprops=dict(arrowstyle="->", lw=1.3, color="black"))

    # attention modules under layer1..layer4
    att_y, att_h = 1.5, 0.7
    for x, lab in zip([2.95, 4.55, 6.15, 7.75], ["att1", "att2", "att3", "att4"]):
        box(ax, x, att_y, 1.2, att_h, f"{lab}\nSE / CBAM / ECA", "#ffe0b2", fs=8, rounded=True)
        ax.annotate("", xy=(x + 0.6, y), xytext=(x + 0.6, att_y + att_h),
                    arrowprops=dict(arrowstyle="<->", lw=1.0, ls="--", color="gray"))

    ax.text(0.15, 2.7, "attention module\ninserted after layer_i\n(optional)", fontsize=8,
            color="gray", va="top")

    # 5 placement combos legend
    ax.text(10.75, 5.6, "Placement configurations", fontsize=10, weight="bold")
    combos = [
        ("all", "att1 + att2 + att3 + att4"),
        ("att4", "att4 only"),
        ("att3+4", "att3 + att4"),
        ("att2+3+4", "att2 + att3 + att4"),
        ("att1+4", "att1 + att4"),
    ]
    ax.text(10.75, 5.15, "\n".join(f"\u2022 {a}: {b}" for a, b in combos),
            fontsize=9, va="top", linespacing=1.6)

    plt.savefig(OUT, dpi=150, bbox_inches="tight")
    print("saved", OUT)


if __name__ == "__main__":
    main()
