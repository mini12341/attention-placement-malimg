# -*- coding: utf-8 -*-
"""32x32 降采样 Malimg 下的门控选择性统计（SE / CBAM / ECA，all 配置，seed 0/1/2）。

目的：与全分辨率的结果同口径对比，检验"浅层门控塌缩"是否与分辨率无关。

流程（每个 module × seed）：
    1. 用与 placement_sweep_det_fullres.py 完全一致的协议训练 12 epoch
       （同 DATA / 同 transform / 同 split / 同 seed 序列 → 与该次扫描结果逐位一致）
    2. 保存 checkpoint 到 checkpoints/fullres/{module}_all_seed{seed}.pt
    3. 全验证集挂 forward hook 抓 sigmoid 输出，算 mean / std / >0.9 占比
    4. 存原始门控到 figs/gates_fullres_{module}_seed{seed}.npz

钩子位置与 att234_stats_fullval.py 保持一致：
    SE   -> att.fc[3]      (Sigmoid，输出 (b,c))
    CBAM -> att.sigmoid    (通道注意力 Sigmoid，输出 (b,c,1,1))
    ECA  -> att.sigmoid    (输出 (b,c,1,1))

确定性设置同主扫描脚本。断点续跑：每完成一个任务写 json。
"""

import ctypes
import json
import os
import time

# 必须在任何 CUDA 算子之前设置 cuBLAS 固定 workspace（与主扫描脚本同位置）
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from malimg_attention import build_model

from paths import DATA_32 as DATA
IMG, BATCH, LR, EPOCHS, VAL_RATIO = 224, 64, 1e-4, 12, 0.2
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
STAGES = "all"

from paths import RESULTS
OUT = os.path.join(RESULTS, "gate_stats_32.json")
from paths import CHECKPOINTS as CKPT_DIR
from paths import FIGURES as NPZ_DIR
SWEEP = os.path.join(RESULTS, "placement_sweep_det.json")

MODULES = ["se", "cbam", "eca"]
SEEDS = [0, 1, 2]


def keep_awake():
    if os.name == "nt":
        try:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        except Exception:
            pass


def set_deterministic():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def get_loaders(seed):
    """与主扫描脚本 run() 完全相同的构造顺序，保证 split 一致。"""
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
    return tl, vl, full


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / total


def train(model, tl, vl, device, tag):
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
        print(f"  [{tag}] epoch {epoch}/{EPOCHS} val_acc={va:.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)
    return accs


def gate_module(mod, att_type):
    """与 att234_stats_fullval.py 完全一致。"""
    if att_type == "se":
        return mod.fc[3]
    return mod.sigmoid


def collect_gates(model, att_type, vl, device):
    model.eval()
    bufs = {i: [] for i in range(1, 5)}
    hooks = []
    for i in range(1, 5):
        mod = getattr(model, f"att{i}")
        if isinstance(mod, nn.Identity):
            continue
        h = gate_module(mod, att_type).register_forward_hook(
            lambda m, ii, o, i=i: bufs[i].append(o.detach().cpu().flatten()))
        hooks.append(h)
    with torch.no_grad():
        for x, _ in vl:
            model(x.to(device))
    for h in hooks:
        h.remove()
    return {i: torch.cat(v).numpy() for i, v in bufs.items() if v}


def main():
    set_deterministic()
    keep_awake()
    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(NPZ_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device: {device}", flush=True)

    done = json.load(open(OUT)) if os.path.exists(OUT) else {}
    sweep = json.load(open(SWEEP)) if os.path.exists(SWEEP) else {}

    tasks = [(m, s) for m in MODULES for s in SEEDS]
    total = len(tasks)
    completed = sum(1 for m, s in tasks if f"{m}__seed{s}" in done)
    print(f"[resume] done {completed}/{total}", flush=True)
    t_start = time.time()

    for module, seed in tasks:
        key = f"{module}__seed{seed}"
        if key in done:
            continue
        print(f"\n===== {key}  [{completed}/{total} done] =====", flush=True)

        tl, vl, full = get_loaders(seed)
        model = build_model(module, num_classes=len(full.classes),
                            att_stages=STAGES).to(device)
        accs = train(model, tl, vl, device, f"{module}/all/seed{seed}")

        # 保存 checkpoint
        ckpt = os.path.join(CKPT_DIR, f"{module}_all_seed{seed}.pt")
        torch.save(model.state_dict(), ckpt)
        print(f"  [ckpt] {ckpt}", flush=True)

        # 门控统计
        gates = collect_gates(model, module, vl, device)
        stats = {}
        for i, w in sorted(gates.items()):
            stats[str(i)] = {
                "mean": float(w.mean()),
                "std": float(w.std()),
                "frac_gt_0.9": float((w > 0.9).mean()),
                "n": int(w.size),
            }
            print(f"  {module} att{i}: mean={w.mean():.3f} std={w.std():.3f} "
                  f">0.9={100 * (w > 0.9).mean():.1f}%", flush=True)
            np.savez_compressed(os.path.join(NPZ_DIR, f"gates_32_{module}_seed{seed}.npz"),
                                **{f"att{k}": v for k, v in gates.items()})

        # 一致性校验：本次训练结果应与主扫描该 key 的曲线一致
        best = max(accs)
        skey = key.replace("__seed", "__all__seed")
        expect = max(sweep[skey]) if skey in sweep else None
        match = (expect is not None and abs(best - expect) < 1e-9)
        print(f"  [check] best={best:.4f} sweep={expect} match={match}", flush=True)

        done[key] = {
            "accs": accs,
            "best": best,
            "sweep_match": bool(match),
            "gates": stats,
            "ckpt": ckpt,
        }
        with open(OUT, "w") as f:
            json.dump(done, f, indent=2)
        completed += 1
        print(f"[saved] {key} ({completed}/{total}, "
              f"{(time.time() - t_start) / 3600:.1f}h)", flush=True)

    print(f"\n[all done] {total}/{total}, total {(time.time() - t_start) / 3600:.1f}h",
          flush=True)


if __name__ == "__main__":
    main()
