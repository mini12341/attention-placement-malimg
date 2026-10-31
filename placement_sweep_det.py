# -*- coding: utf-8 -*-
"""注意力放置消融（确定性版，80 次统一重跑）。

统一在 deterministic 标准下重跑 baseline + 5 组合 × se/cbam/eca × 5 seed，
消除跨进程 cuDNN 噪声。输出到新文件 results/placement_sweep_det.json
（不复用旧 stage_sweep.json / cbam_all.json 等，避免断点续跑误跳过）。

- baseline(none)                        × 5 seed = 5 次
- 5 组合(all/4/3,4/2,3,4/1,4) × 3 模块 × 5 seed = 75 次
合计 80 次。

确定性设置（必须在任何 CUDA 算子之前设置）：
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"   # cuBLAS 固定 workspace
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    torch.use_deterministic_algorithms(True)
断点续跑：每完成一个任务立刻写 json，重跑时跳过已完成 key。
防休眠：SetThreadExecutionState(ES_CONTINUOUS|ES_SYSTEM_REQUIRED)。
"""
import ctypes
import json
import os
import time

# use_deterministic_algorithms(True) 要求 cuBLAS 固定 workspace，必须在首次 CUDA 前设置
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from malimg_attention import build_model

# ---- 确定性三件套 ----
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

from paths import DATA_32 as DATA
IMG, BATCH, LR, EPOCHS, VAL_RATIO = 224, 64, 1e-4, 12, 0.2
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
from paths import RESULTS
OUT = os.path.join(RESULTS, "placement_sweep_det.json")

MODULES = ["se", "cbam", "eca"]
# 32x32 control uses five seeds: its placement spread is of the order of the
# seed noise, so three seeds are not enough to separate the two.
SEEDS = [0, 1, 2, 3, 4]
# (label, att_stages 参数)；label 用于 key，att_stages 传给 build_model
COMBOS = [("all", "all"), ("4", "4"), ("3,4", "3,4"),
          ("2,3,4", "2,3,4"), ("1,4", "1,4")]


def keep_awake():
    if os.name == "nt":
        try:
            # ES_CONTINUOUS(0x80000000) | ES_SYSTEM_REQUIRED(0x00000001)
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        except Exception:
            pass


def load_done():
    if os.path.exists(OUT):
        with open(OUT) as f:
            return json.load(f)
    return {}


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / total


def run(module, stages, seed, device):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    transform = transforms.Compose([
        transforms.Grayscale(3), transforms.Resize((IMG, IMG)),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    full = datasets.ImageFolder(root=DATA, transform=transform,
        is_valid_file=lambda p: p.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")))
    n_val = int(len(full) * VAL_RATIO)
    train_ds, val_ds = random_split(full, [len(full) - n_val, n_val],
        generator=torch.Generator().manual_seed(seed))
    g = torch.Generator().manual_seed(seed)
    tl = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0, generator=g)
    vl = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=0)
    model = build_model(module, num_classes=len(full.classes), att_stages=stages).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    accs = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        t0 = time.time()
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss_fn(model(x), y).backward()
            opt.step()
        va = evaluate(model, vl, device)
        accs.append(va)
        keep_awake()
        print(f"  [{module}/{stages}/seed{seed}] epoch {epoch}/{EPOCHS} "
              f"val_acc={va:.4f} ({time.time() - t0:.0f}s)", flush=True)
    return accs


def task_list():
    ts = []
    for seed in SEEDS:
        ts.append(("none", "all", seed, f"baseline__seed{seed}"))
    for mod in MODULES:
        for label, stages in COMBOS:
            for seed in SEEDS:
                ts.append((mod, stages, seed, f"{mod}__{label}__seed{seed}"))
    return ts


def main():
    keep_awake()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device: {device}", flush=True)
    done = load_done()
    ts = task_list()
    total = len(ts)
    completed = sum(1 for _, _, _, k in ts if k in done)
    print(f"[resume] done {completed}/{total}", flush=True)
    t_start = time.time()
    for mod, stages, seed, key in ts:
        if key in done:
            continue
        print(f"\n===== {key}  [{completed}/{total} done] =====", flush=True)
        accs = run(mod, stages, seed, device)
        done[key] = accs
        with open(OUT, "w") as f:
            json.dump(done, f, indent=2)
        completed += 1
        el = (time.time() - t_start) / 3600
        print(f"[saved] {key} -> {OUT}  ({completed}/{total}, {el:.1f}h)", flush=True)
    print(f"\n[all done] {total}/{total}, total {(time.time() - t_start) / 3600:.1f}h",
          flush=True)


if __name__ == "__main__":
    main()
