# -*- coding: utf-8 -*-
"""全分辨率 Malimg：CBAM 空间门控（spatial gate）统计，seed 0/1/2。

只做前向推理，不训练：直接加载 checkpoints/fullres/cbam_all_seed{s}.pt
（由 gate_stats_fullres.py 训练并保存，与主扫描逐位一致）。

空间门控 = CBAM.forward 里的 `self.spatial` 输出，形状 (b,1,h,w)，取值 (0,1)。
它与通道门控 (b,c,1,1) 的"塌缩"判据对应关系：
    通道门控塌缩 → 所有通道都取 ≈0.5（无通道选择性）
    空间门控塌缩 → 所有像素都取 ≈0.5（空间均匀，等价于整体乘 0.5）
因此统计量取：
    mean                全验证集所有像素均值
    within_std          每张图内像素的标准差（空间对比度），再对图取平均
    frac_gt_0.9 / <0.1  极端像素占比
    contrast            每张图 max-min，再取平均
"""

import ctypes
import json
import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from malimg_attention import build_model

from paths import DATA_FULL as DATA
IMG, BATCH, VAL_RATIO = 224, 64, 0.2
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

from paths import CKPT_FULLRES as CKPT_DIR
from paths import RESULTS, FIGURES
OUT = os.path.join(RESULTS, "cbam_spatial_stats_fullres.json")
SEEDS = [0, 1, 2]


def keep_awake():
    if os.name == "nt":
        try:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        except Exception:
            pass


def get_val_loader(seed):
    """与 gate_stats_fullres.get_loaders 完全相同的构造顺序。"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    transform = transforms.Compose([
        transforms.Grayscale(3), transforms.Resize((IMG, IMG)),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    full = datasets.ImageFolder(root=DATA, transform=transform,
        is_valid_file=lambda p: p.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")))
    n_val = int(len(full) * VAL_RATIO)
    _, val_ds = random_split(full, [len(full) - n_val, n_val],
        generator=torch.Generator().manual_seed(seed))
    return DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=0), full


def collect_spatial(model, vl, device):
    model.eval()
    bufs = {i: [] for i in range(1, 5)}
    hooks = []
    for i in range(1, 5):
        mod = getattr(model, f"att{i}")
        if isinstance(mod, nn.Identity):
            continue
        h = mod.spatial.register_forward_hook(
            lambda m, ii, o, i=i: bufs[i].append(o.detach().cpu().squeeze(1)))
        hooks.append(h)
    with torch.no_grad():
        for x, _ in vl:
            model(x.to(device))
    for h in hooks:
        h.remove()
    return {i: torch.cat(v).numpy() for i, v in bufs.items() if v}


def main():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    keep_awake()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device: {device}", flush=True)

    out = {}
    for seed in SEEDS:
        vl, full = get_val_loader(seed)
        model = build_model("cbam", num_classes=len(full.classes),
                            att_stages="all").to(device)
        ckpt = os.path.join(CKPT_DIR, f"cbam_all_seed{seed}.pt")
        model.load_state_dict(torch.load(ckpt, map_location=device))
        print(f"\n===== cbam/all/seed{seed}  ({ckpt}) =====", flush=True)

        maps = collect_spatial(model, vl, device)
        stats = {}
        for i, m in sorted(maps.items()):   # m: (N, h, w)
            flat = m.reshape(m.shape[0], -1)
            within = flat.std(axis=1)
            contrast = flat.max(axis=1) - flat.min(axis=1)
            stats[str(i)] = {
                "shape": list(m.shape[1:]),
                "mean": float(m.mean()),
                "within_std": float(within.mean()),
                "frac_gt_0.9": float((m > 0.9).mean()),
                "frac_lt_0.1": float((m < 0.1).mean()),
                "contrast": float(contrast.mean()),
                "n": int(m.size),
            }
            print(f"  cbam att{i} {tuple(m.shape[1:])}: mean={m.mean():.3f} "
                  f"within_std={within.mean():.4f} "
                  f">0.9={100 * (m > 0.9).mean():.2f}% "
                  f"<0.1={100 * (m < 0.1).mean():.2f}% "
                  f"contrast={contrast.mean():.4f}", flush=True)

        out[f"cbam__seed{seed}"] = stats
        np.savez_compressed(
            os.path.join(FIGURES, "cbam_spatial_fullres_seed%d.npz" % seed),
            **{f"att{k}": v[:200] for k, v in maps.items()})

        with open(OUT, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"\n[done] {OUT}", flush=True)


if __name__ == "__main__":
    main()
