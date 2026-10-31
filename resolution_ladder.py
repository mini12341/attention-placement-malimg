# -*- coding: utf-8 -*-
"""Resolution ladder: does the placement effect reappear as input detail shrinks?

Exploratory follow-up to the paper's "information scarcity" reading. The
32x32 control in the paper uses a different image file (the widely distributed
downsampled npz), so resolution and source image vary together. Here we hold the
source fixed -- the native byteplots -- and change only the resize target, so the
ladder isolates input detail.

Not part of the paper's tables. Results go to results/resolution_ladder.json and
are reported separately, if at all.

Everything else is bit-identical to placement_sweep_det_fullres.py: same model,
same optimizer, same schedule, same deterministic settings, same 80/20 split.

    python resolution_ladder.py                 # default: 96 and 160
    python resolution_ladder.py 64 96 160 224   # arbitrary ladder

Resumable: each finished run is written to JSON immediately and skipped on
restart, so the ladder can be stopped at any rung.
"""
import ctypes
import json
import os
import sys
import time

# use_deterministic_algorithms(True) requires a fixed cuBLAS workspace; set
# before the first CUDA call.
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

BATCH, LR, EPOCHS, VAL_RATIO = 64, 1e-4, 12, 0.2
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
OUT = os.path.join(RESULTS, "resolution_ladder.json")

MODULES = ["se", "cbam", "eca"]
SEEDS = [0, 1, 2]
COMBOS = [("all", "all"), ("4", "4"), ("3,4", "3,4"),
          ("2,3,4", "2,3,4"), ("1,4", "1,4")]

RESOLUTIONS = [int(a) for a in sys.argv[1:]] or [96, 160]


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


def run(module, stages, seed, res, device):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    transform = transforms.Compose([
        transforms.Grayscale(3), transforms.Resize((res, res)),
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
        print(f"  [{res}/{module}/{stages}/seed{seed}] epoch {epoch}/{EPOCHS} "
              f"val_acc={va:.4f} ({time.time() - t0:.0f}s)", flush=True)
    return accs


def task_list():
    ts = []
    for res in RESOLUTIONS:
        for seed in SEEDS:
            ts.append(("none", "all", seed, res, f"res{res}__baseline__seed{seed}"))
        for mod in MODULES:
            for label, stages in COMBOS:
                for seed in SEEDS:
                    ts.append((mod, stages, seed, res,
                               f"res{res}__{mod}__{label}__seed{seed}"))
    return ts


def main():
    keep_awake()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device: {device}; ladder {RESOLUTIONS}", flush=True)
    done = load_done()
    ts = task_list()
    total = len(ts)
    completed = sum(1 for *_, k in ts if k in done)
    print(f"[resume] done {completed}/{total}", flush=True)
    t_start = time.time()
    for mod, stages, seed, res, key in ts:
        if key in done:
            continue
        print(f"\n===== {key}  [{completed}/{total} done] =====", flush=True)
        accs = run(mod, stages, seed, res, device)
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
