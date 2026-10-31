# -*- coding: utf-8 -*-
"""Full-resolution shadowing check: are att2/att3 dead even without att1 in front?

The low-resolution control already answers this (Table 5 of the paper). The paper
notes that it was not repeated at full resolution, on the grounds that every
configuration there is saturated and the shallow gates are already locked at 0.5,
so the ablation cannot separate competing explanations. This script runs it
anyway, over three seeds, so the limitation can be stated as something checked
rather than argued.

Writes results/att234_stats_fullres.json, keyed ``{module}/att{stage}/seed{seed}``.
Resumable: finished (module, seed) pairs are skipped on restart.

    python att234_stats_fullres.py
"""
import ctypes
import json
import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from malimg_attention import build_model

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

from paths import DATA_FULL as DATA, RESULTS

IMG, BATCH, LR, EPOCHS, VAL_RATIO = 224, 64, 1e-4, 12, 0.2
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
STAGES = "2,3,4"
MODULES = ["se", "cbam", "eca"]
SEEDS = [0, 1, 2]
OUT = os.path.join(RESULTS, "att234_stats_fullres.json")


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


def get_loaders(seed):
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


def train(model, tl, vl, device):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss_fn(model(x), y).backward()
            opt.step()
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in vl:
                x, y = x.to(device), y.to(device)
                correct += (model(x).argmax(1) == y).sum().item()
                total += y.size(0)
        print(f"  epoch {epoch}/{EPOCHS} val_acc={correct / total:.4f}", flush=True)


def gate_module(mod, att_type):
    return mod.fc[3] if att_type == "se" else mod.sigmoid


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
    keep_awake()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    done = load_done()
    total = len(MODULES) * len(SEEDS)
    completed = sum(1 for m in MODULES for s in SEEDS
                    if f"{m}/att2/seed{s}" in done)
    print(f"[resume] done {completed}/{total}", flush=True)
    for att in MODULES:
        for seed in SEEDS:
            if f"{att}/att2/seed{seed}" in done:
                continue
            print(f"\n===== {att} att2+3+4, seed {seed} "
                  f"[{completed}/{total} done] =====", flush=True)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            tl, vl, full = get_loaders(seed)
            model = build_model(att, num_classes=len(full.classes),
                                att_stages=STAGES).to(device)
            train(model, tl, vl, device)
            for i, w in sorted(collect_gates(model, att, vl, device).items()):
                done[f"{att}/att{i}/seed{seed}"] = {
                    "mean": float(w.mean()), "std": float(w.std()),
                    "frac_gt_0.9": float((w > 0.9).mean()), "n": int(w.size)}
                print(f"  {att} att{i}: mean={w.mean():.3f} std={w.std():.3f} "
                      f">0.9={100 * (w > 0.9).mean():.1f}%", flush=True)
            with open(OUT, "w") as f:
                json.dump(done, f, indent=1)
            completed += 1
            print(f"[saved] {att}/seed{seed} ({completed}/{total})", flush=True)
    print("[all done]", flush=True)


if __name__ == "__main__":
    main()
