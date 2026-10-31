# -*- coding: utf-8 -*-
"""Placement sweep under a reduced training budget (fewer epochs).

Exploratory follow-up, not part of the paper's tables. It holds the input fixed
at full resolution -- so the information content is unchanged -- and lowers only
the training budget, which lowers the accuracy ceiling. If the within-module
placement spread grows as the ceiling drops, the "an effect needs headroom"
condition in the Discussion is supported by direct measurement; if the spread
stays at the full-resolution value, the driver is input information specifically
and the claim narrows.

    python placement_sweep_budget.py 3
    python placement_sweep_budget.py 3 6      # several budgets

Writes results/placement_sweep_det_ep{N}.json. Deterministic and resumable, in
the same way as placement_sweep_det_fullres.py.
"""
import ctypes
import json
import os
import sys
import time

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

IMG, BATCH, LR, VAL_RATIO = 224, 64, 1e-4, 0.2
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

MODULES = ["se", "cbam", "eca"]
# Five seeds, matching the accuracy tables. A three-seed spread is exactly the
# statistic that Section 4.4 shows to be uninterpretable on this dataset, so a
# three-seed budget probe could not support or refute the saturation condition.
SEEDS = [0, 1, 2, 3, 4]
COMBOS = [("all", "all"), ("4", "4"), ("3,4", "3,4"),
          ("2,3,4", "2,3,4"), ("1,4", "1,4")]

BUDGETS = [int(a) for a in sys.argv[1:]] or [3]


def keep_awake():
    if os.name == "nt":
        try:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        except Exception:
            pass


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / total


def run(module, stages, seed, epochs, device):
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
    for epoch in range(1, epochs + 1):
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
        print(f"  [ep{epochs}/{module}/{stages}/seed{seed}] epoch {epoch}/{epochs} "
              f"val_acc={va:.4f} ({time.time() - t0:.0f}s)", flush=True)
    return accs


def main():
    keep_awake()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for epochs in BUDGETS:
        out = os.path.join(RESULTS, f"placement_sweep_det_ep{epochs}.json")
        done = {}
        if os.path.exists(out):
            with open(out) as f:
                done = json.load(f)
        tasks = [(mod, stages, seed, f"{mod}__{label}__seed{seed}")
                 for mod in MODULES for label, stages in COMBOS for seed in SEEDS]
        tasks += [("none", "all", seed, f"baseline__seed{seed}") for seed in SEEDS]
        total = len(tasks)
        completed = sum(1 for *_, k in tasks if k in done)
        print(f"\n##### budget {epochs} epochs: {completed}/{total} done #####",
              flush=True)
        t_start = time.time()
        for mod, stages, seed, key in tasks:
            if key in done:
                continue
            print(f"\n===== ep{epochs} {key}  [{completed}/{total} done] =====",
                  flush=True)
            done[key] = run(mod, stages, seed, epochs, device)
            with open(out, "w") as f:
                json.dump(done, f, indent=2)
            completed += 1
            print(f"[saved] {key} -> {out}  ({completed}/{total}, "
                  f"{(time.time() - t_start) / 3600:.1f}h)", flush=True)
    print("\n[all done]", flush=True)


if __name__ == "__main__":
    main()
