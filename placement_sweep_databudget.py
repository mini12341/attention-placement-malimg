# -*- coding: utf-8 -*-
"""放置消融（确定性版，全分辨率原生图，受训练数据量限制的探针）。

动机：论文主结论的前提是 Malimg 在全分辨率下已饱和（99.5%+），有人会问
"天花板太低所以看不到放置效应"。§5 把"更低的精度天花板"列为可能改变结论
的条件之一。降低训练预算（只训 3 epoch）几乎压不低天花板（仍 ≈99.3%），
所以这里换一个真正能压低天花板的轴：只保留一部分训练样本，验证集不变。

协议与 placement_sweep_det_fullres.py 完全一致（同划分、同优化器、同 12
epoch、同 5 种子、同 5 种放置 × 3 模块 + 基线），唯一区别是每个种子下
训练集被确定性地抽到 FRAC 比例；同一 seed 的所有配置共用同一子集，因此
放置仍是唯一变量。

用法：
    python placement_sweep_databudget.py [frac] [smoke]

输出：results/placement_databudget.json，key 形如
    frac0.1__se__3,4__seed2
    frac0.1__baseline__seed2

确定性设置（必须在任何 CUDA 算子之前设置）：
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    torch.use_deterministic_algorithms(True)
断点续跑：每完成一个任务立刻写 json，重跑时跳过已完成 key。
"""
import ctypes
import json
import os
import sys
import time

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms

from malimg_attention import build_model

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

from paths import DATA_FULL as DATA, RESULTS

IMG, BATCH, LR, EPOCHS, VAL_RATIO = 224, 64, 1e-4, 12, 0.2
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

FRAC = float(sys.argv[1]) if len(sys.argv) > 1 else 0.1
SMOKE = len(sys.argv) > 2 and sys.argv[2] == "smoke"
TAG = f"frac{FRAC:g}"
OUT = os.path.join(RESULTS, "placement_databudget_smoke.json" if SMOKE
                   else "placement_databudget.json")

MODULES = ["se", "cbam", "eca"]
SEEDS = [0, 1, 2, 3, 4]
COMBOS = [("all", "all"), ("4", "4"), ("3,4", "3,4"),
          ("2,3,4", "2,3,4"), ("1,4", "1,4")]


def keep_awake():
    if os.name == "nt":
        try:
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
    # 训练集子采样：同一 seed 下所有配置共用，且与模型初始化互不影响
    k = max(1, int(len(train_ds) * FRAC))
    perm = torch.randperm(len(train_ds),
                          generator=torch.Generator().manual_seed(10_000 + seed))
    train_ds = Subset(train_ds, perm[:k].tolist())
    g = torch.Generator().manual_seed(seed)
    tl = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0, generator=g)
    vl = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=0)
    model = build_model(module, num_classes=len(full.classes), att_stages=stages).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    accs = []
    n_epochs = 1 if SMOKE else EPOCHS
    for epoch in range(1, n_epochs + 1):
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
        print(f"  [{TAG}/{module}/{stages}/seed{seed}] epoch {epoch}/{n_epochs} "
              f"val_acc={va:.4f} ({time.time() - t0:.0f}s)", flush=True)
    return accs


def task_list():
    ts = [("none", "all", 0, f"{TAG}__baseline__seed0")] if SMOKE else []
    if SMOKE:
        return ts
    for seed in SEEDS:
        ts.append(("none", "all", seed, f"{TAG}__baseline__seed{seed}"))
    for mod in MODULES:
        for label, stages in COMBOS:
            for seed in SEEDS:
                ts.append((mod, stages, seed, f"{TAG}__{mod}__{label}__seed{seed}"))
    return ts


def main():
    keep_awake()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device: {device}; data fraction {FRAC}; smoke={SMOKE}", flush=True)
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
    print(f"\n[all done] {completed}/{total}, total "
          f"{(time.time() - t_start) / 3600:.1f}h", flush=True)


if __name__ == "__main__":
    main()
