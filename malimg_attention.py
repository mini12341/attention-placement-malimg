# -*- coding: utf-8 -*-
"""
Malimg 注意力对比实验：none / SE / FCA(DCT 频率注意力)

三种配置共用同样的 seed、数据划分、lr、epoch、batch、输入尺寸，
保证唯一的变量就是「注意力模块」，公平对比。

SE  : 全局平均池化(GAP) -> MLP -> 通道权重  （GAP 本质只用了 DC 频率分量）
FCA : 多个 2D-DCT 频率分量 -> MLP -> 通道权重 （FcaNet 风格，描述频率分布）

用法:
    # 单独跑某个配置
    python malimg_attention.py --attention fca --epochs 12 --workers 0
    # 依次跑 none/se/fca 并输出对比表
    python malimg_attention.py --run-all --epochs 12 --workers 0
"""
import argparse
import json
import os
import math
import statistics
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models

from paths import DATA_32, RESULTS


# ---------- 注意力模块 ----------
class SE(nn.Module):
    """Squeeze-and-Excitation：GAP -> MLP -> 通道权重（只用 DC 分量）。"""

    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y = x.mean(dim=(2, 3))           # GAP
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class CBAM(nn.Module):
    """CBAM：通道注意力(avg+max 共享 MLP) + 空间注意力。"""

    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False),
            nn.Sigmoid(),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 通道注意力（amax 替代 AdaptiveMaxPool2d：数值等价，但 backward 有确定性实现）
        ch = self.sigmoid(self.mlp(self.avg_pool(x)) + self.mlp(x.amax(dim=(2, 3), keepdim=True)))
        x = x * ch
        # 空间注意力
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        sp = self.spatial(torch.cat([avg, mx], dim=1))
        return x * sp


class ECA(nn.Module):
    """ECA：1D 卷积建模局部跨通道交互（不降维）。"""

    def __init__(self, channels, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.shape
        y = self.avg_pool(x).squeeze(-1).transpose(-1, -2)   # (b, 1, c)
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)      # (b, c, 1, 1)
        return x * self.sigmoid(y)


class FCA(nn.Module):
    """频率通道注意力（FcaNet 风格）：用多个 2D-DCT 频率分量代替 GAP。"""

    FREQS = [(0, 0), (0, 1), (1, 0), (1, 1), (0, 2), (2, 0), (2, 2), (1, 2)]

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.freqs = self.FREQS
        hidden = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.Linear(channels * len(self.freqs), hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, h, w = x.shape
        coeffs = []
        hi = torch.arange(h, device=x.device).float()
        wi = torch.arange(w, device=x.device).float()
        for (u, v) in self.freqs:
            # 2D-DCT-II 基函数，带 FcaNet 的正交归一化（build_filter 同款）：
            #   单轴 u==0 -> 1/sqrt(H)，u>0 -> sqrt(2/H)；v 同理，二维两轴相乘。
            #   目的：让 DC 与 AC 各频率分量的量级可比，不被 DC 主导。
            cu = (1.0 / math.sqrt(h)) if u == 0 else math.sqrt(2.0 / h)
            cv = (1.0 / math.sqrt(w)) if v == 0 else math.sqrt(2.0 / w)
            bh = (cu * torch.cos(math.pi * u * (2 * hi + 1) / (2 * h))).view(1, 1, h, 1)
            bw = (cv * torch.cos(math.pi * v * (2 * wi + 1) / (2 * w))).view(1, 1, 1, w)
            basis = bh * bw                          # (1,1,h,w)
            coeffs.append((x * basis).sum(dim=(2, 3)))   # (b,c)
        y = torch.stack(coeffs, dim=2).reshape(b, c * len(self.freqs))
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


# ---------- 模型 ----------
class _ResNet18Att(nn.Module):
    """把注意力模块插到 ResNet18 四个 stage 之后。"""

    def __init__(self, base, atts, num_classes):
        super().__init__()
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool = base.avgpool
        self.fc = nn.Linear(base.fc.in_features, num_classes)
        self.att1, self.att2, self.att3, self.att4 = atts

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.att1(self.layer1(x))
        x = self.att2(self.layer2(x))
        x = self.att3(self.layer3(x))
        x = self.att4(self.layer4(x))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def build_model(att_type, num_classes=25, att_stages="all"):
    base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    channels = [64, 128, 256, 512]
    if att_type in ("se", "cbam", "eca"):
        if att_stages == "all":
            active = {1, 2, 3, 4}
        else:
            active = {int(s) for s in att_stages.replace(",", " ").split() if s.strip()}
        mod = {"se": SE, "cbam": CBAM, "eca": ECA}[att_type]
        atts = [mod(c) if i in active else nn.Identity()
                for i, c in enumerate(channels, start=1)]
    elif att_type == "fca":
        atts = [FCA(c) for c in channels]
    else:
        atts = [nn.Identity() for _ in channels]
    return _ResNet18Att(base, atts, num_classes)


# ---------- 训练 ----------
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / total


def run(att_type, args, device):
    # 固定本 run 内所有随机性（fc/注意力层初始化、数据 shuffle），
    # 让同一 seed 下 none/se/fca 三者的划分、shuffle、初值完全一致。
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    full = datasets.ImageFolder(
        root=args.data_dir, transform=transform,
        is_valid_file=lambda p: p.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")),
    )
    n_val = int(len(full) * args.val_ratio)
    n_train = len(full) - n_val
    train_ds, val_ds = random_split(
        full, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed)
    )
    g = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.workers, generator=g)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.workers)

    model = build_model(att_type, num_classes=len(full.classes),
                        att_stages=args.att_stages).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    val_accs = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
        va = evaluate(model, val_loader, device)
        val_accs.append(va)
        print(f"  [{att_type}] epoch {epoch}/{args.epochs} val_acc={va:.4f} "
              f"time={time.time() - t0:.1f}s", flush=True)
    return val_accs


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, default=DATA_32)
    p.add_argument("--attention", type=str, default="none",
                   choices=["none", "se", "fca", "cbam", "eca"])
    p.add_argument("--att-stages", type=str, default="all",
                   help="注意力插入的 stage（1-4，逗号分隔）；all=全部。如 4=只插最后一层")
    p.add_argument("--run-all", action="store_true",
                   help="依次跑 none/se/fca 并输出对比表")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seeds", type=int, nargs="+", default=None,
                   help="多种子列表，如 --seeds 0 1 2（需配合 --run-all）")
    p.add_argument("--out", type=str, default=os.path.join(RESULTS, "multiseed_results.json"),
                   help="多种子曲线保存路径（JSON）")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def _summarize(results, seeds, atts):
    """跨 seed 汇总 best 与末轮 val_acc 的 mean±std。"""
    print("\n===== mean±std (best val_acc per seed) =====")
    for att in atts:
        bests = [max(results[(s, att)]) for s in seeds]
        finals = [results[(s, att)][-1] for s in seeds]
        print(f"  {att:6s}: best = {statistics.mean(bests)*100:.2f}% "
              f"± {statistics.stdev(bests)*100:.2f}%   "
              f"末轮 = {statistics.mean(finals)*100:.2f}% "
              f"± {statistics.stdev(finals)*100:.2f}%")


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[info] device: {device}", flush=True)
    atts = ["none", "se", "fca"]
    seeds = args.seeds if args.seeds else [args.seed]
    if args.run_all:
        att_list = atts
    elif args.seeds:
        att_list = [args.attention]      # 只重跑指定注意力，多种子
    else:
        run(args.attention, args, device)
        return
    results = {}
    for seed in seeds:
        args.seed = seed
        print(f"\n########## seed={seed} ##########", flush=True)
        for att in att_list:
            print(f"\n===== seed {seed} / {att} =====", flush=True)
            results[(seed, att)] = run(att, args, device)
    out = {f"{att}__seed{seed}": results[(seed, att)]
           for seed in seeds for att in att_list}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[info] 曲线已保存到 {args.out}", flush=True)
    _summarize(results, seeds, att_list)


if __name__ == "__main__":
    main()
