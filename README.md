# Where Does Attention Matter?

Code and result data for the paper

> **Where Does Attention Matter? A Gate-Level Audit of SE/CBAM/ECA Placement
> in Image-Based Malware Classification**
> Zhenghao Xiao, Jianhua Mo, Shuo Yuan, Lin Li
> *Journal of Information Security and Applications* (under review)

The study is a controlled ablation, not a new architecture. We insert SE, CBAM
and ECA in five placements over the four residual stages of ResNet-18 (5
placements x 3 modules x 5 seeds = 80 deterministic runs), repeat the whole
protocol at two input resolutions, and read the learned gates out of the trained
networks with forward hooks. Both resolutions use five seeds, because at 32x32
the placement spread is of the same order as the seed noise.

## What we find

1. **At full resolution, placement does not matter.** Every run, including those
   of the no-attention baseline, lands within 99.4-99.8%; the spread within a
   module (0.06-0.12 points) is no larger than the seed-to-seed variation (up to
   0.12).
2. **Shallow gates collapse at either resolution.** `att1`-`att3` sit at
   ~0.5 with essentially no channel switched on, and independent seeds converge
   to the same constant (SD as low as 0.001). Only the deepest gate is
   selective. CBAM's spatial branch behaves the same way: at the shallow stages
   no pixel of the spatial map exceeds 0.9. Removing `att1` and retraining does
   not revive `att2`/`att3` at either resolution, so the collapse is not a
   shadowing effect.
3. **The 32x32 control does not resolve placement either.** Its run-to-run noise
   is three to four times larger (seed SD up to 0.49 against 0.12), the
   within-module spread (0.19-0.35 points) stays inside that noise, and neither a
   one-way nor a seed-blocked ANOVA finds a placement effect in any module.
   Downsampling degrades the measurement faster than it degrades the signal, so
   the cheaper control is the place where the same effect would be hardest to
   see - not the place where it surfaces.

The mechanism is in the Discussion: a gate that is constant across the dimension
it modulates is a global rescaling, and the next block's BatchNorm removes it, so
such a gate is inert at *any* value - not only at 0.5.

Two probes test whether a lower accuracy ceiling would let a placement effect
surface. Cutting the training budget to a quarter leaves accuracy at 99.36%, so
the ceiling barely moves; cutting the training set to a tenth lowers it to
98.20% but raises the seed standard deviation to 0.57-0.93, so the placement
spread still sits inside the noise and no effect appears on either axis
(Section 5 of the paper).

## Requirements

```
Python 3.10, torch 2.6.0+cu124, torchvision, numpy, matplotlib, Pillow
```

Tested on Windows 11 with an RTX 3060 (6 GB). Batch size 64 and 224x224 input
fit comfortably in 6 GB at ResNet-18.

```
pip install -r requirements.txt
```

## Data

**Full-resolution Malimg (primary).** 9,339 byteplot PNGs in 25 family folders,
original variable size (widths 64-1024, median 256; heights 208-5334,
median 424):

```bash
# https://huggingface.co/datasets/ikrambenabd/malimg-original
# expected layout: <root>/<Family.Name>/<image>.png
export MALIMG_FULLRES=/path/to/malimg_paper_dataset_imgs
```

**32x32 Malimg (control).** The widely used downsampled variant, distributed as
`malimg.npz` (9,339 images of 32x32 + labels). Put it at `data/malimg.npz` (or
point `MALIMG_NPZ` at it) and convert it to the folder layout the loaders expect:

```bash
python convert_malimg.py          # writes to $MALIMG_32 (default ./data/malimg_32)
```

## Paths

Nothing is hard-coded; `paths.py` resolves everything relative to the repository
and accepts overrides:

| variable | meaning | default |
|---|---|---|
| `MALIMG_FULLRES` | full-resolution dataset root | `./data/malimg_fullres` |
| `MALIMG_32` | 32x32 dataset root | `./data/malimg_32` |
| `RESULTS_DIR` | result JSONs | `./results` |
| `FIGURES_DIR` | figures, gate dumps | `./figures` |
| `CKPT_DIR` | checkpoints | `./checkpoints` |

## Reproducing the paper

| Artifact | Command | Cost |
|---|---|---|
| Table 1 (80 runs, full res, 5 seeds) | `python placement_sweep_det_fullres.py` | ~14 h |
| Table 2 + Figure 2 data | `python gate_stats_fullres.py` | ~2.6 h |
| Figure 2 | `python plot_gate_histograms_fullres.py` | seconds |
| Sec. 4.3 spatial-gate numbers | `python cbam_spatial_stats_fullres.py` | ~3 min |
| Table 3 (80 runs, 32x32, 5 seeds) | `python placement_sweep_det.py` | ~7.3 h |
| Table 4 (32x32 gates) | `python gate_stats_32.py` | ~50 min |
| Table 5 (32x32 shadowing, 3 seeds) | `python att234_stats_32.py` | ~40 min |
| Table 6 (full-res shadowing, 3 seeds) | `python att234_stats_fullres.py` | ~1.6 h |
| Reduced-budget probe (3 epochs, 80 runs) | `python placement_sweep_budget.py 3` | ~3.2 h |
| Data-limited probe (10% of train set, 80 runs) | `python placement_sweep_databudget.py 0.1` | ~4.5 h |
| Check every table against `results/` | `python verify_tables.py` | seconds |
| Check gate models reproduce the sweep | `python check_reproduction.py` | seconds |
| Equivalence tests for "no placement effect" | `python stats_equivalence.py` | seconds |
| ANOVA on the placement spread (Sec. 4.4) | `python stats_anova.py` | seconds |
| Figure 1 | `python plot_structure.py` | seconds |

The scripts are resumable: each finished run is written to JSON immediately and
skipped on restart.

**You do not need to retrain to check the tables.** The JSON files produced by
the runs used in the paper are checked in under `results/`, so every number in
Tables 1-6 and the spatial-gate paragraph can be recomputed in seconds. One
command does the comparison for you:

```bash
python verify_tables.py path/to/main.tex
```

It recomputes each table cell from the JSONs, compares it against the value
printed in the manuscript, and exits non-zero on any mismatch. The manuscript
source is not part of this repository; point the script at your copy. The
analysis scripts (`stats_anova.py`, `stats_equivalence.py`) and every JSON in
`results/` run without it. A single run of
the original code:

```bash
python - <<'PY'
import json
d = json.load(open("results/placement_sweep_det_fullres.json"))
print(max(d["se__all__seed0"]))   # best val accuracy of one run
PY
```

## Determinism

The comparisons rest on differences of a few tenths of a point, so all training
is bit-reproducible:

```python
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"   # before importing torch
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
```

Two consequences are worth knowing if you adapt this code:

- The environment variable must be set **before** `torch` is imported; every
  script here does that at the top of the file.
- `nn.AdaptiveMaxPool2d`, used by the original CBAM for the max-pooled
  descriptor, has no deterministic CUDA backward. `malimg_attention.CBAM`
  replaces it with `x.amax(dim=(2, 3), keepdim=True)`, which is numerically
  identical and has no parameters, so the model is unchanged.

With these settings the nine gate-analysis models (three modules x three seeds)
reproduce the corresponding sweep curves exactly, epoch by epoch;
`python check_reproduction.py` redoes the comparison. The `sweep_match` field
stored in `results/gate_stats_fullres.json` reads false because it was written
when the gate runs finished, before the final five-seed sweep file existed;
recomputing it from the distributed files gives 9/9 identical.

## Layout

```
paths.py                          filesystem layout / environment overrides
malimg_attention.py               SE, CBAM, ECA and the ResNet-18 wrapper
placement_sweep_det_fullres.py    Table 1
gate_stats_fullres.py             Table 2, gate dumps for Figure 2
cbam_spatial_stats_fullres.py     CBAM spatial-gate statistics
plot_gate_histograms_fullres.py   Figure 2
placement_sweep_det.py            Table 3
gate_stats_32.py                  Table 4
att234_stats_32.py                Table 5 (32x32 shadowing, three seeds)
verify_tables.py                  recompute every table cell from results/
stats_equivalence.py              TOST equivalence tests on paired seed differences
stats_anova.py                    one-way and seed-blocked ANOVA on placement
att234_stats_fullres.py           Table 6 (full-res shadowing, three seeds)
resolution_ladder.py              exploratory: 64/96/160 px rungs (not in the paper)
placement_sweep_budget.py         reduced-epoch (3) probe: does a lower training
                                  budget create a placement effect?
placement_sweep_databudget.py     data-limited (10% of train set) probe: same question,
                                  with an accuracy ceiling that actually drops
plot_structure.py                 Figure 1
convert_malimg.py                 malimg.npz -> per-family folders
results/                          result JSONs from the paper's runs
```

Checkpoints and the raw per-image gate activations (`85 MB`) are not stored in
the repository; they are regenerated by the scripts above and are available from
the corresponding author on request.

## Citation

```bibtex
@article{xiao2026attention,
  title   = {Where Does Attention Matter? A Gate-Level Audit of SE/CBAM/ECA
             Placement in Image-Based Malware Classification},
  author  = {Xiao, Zhenghao and Mo, Jianhua and Yuan, Shuo and Li, Lin},
  journal = {Journal of Information Security and Applications},
  year    = {2026},
  note    = {Under review}
}
```

## License

Code: MIT (see `LICENSE`). The Malimg dataset is distributed by its original
authors under its own terms; it is not included here.

## Contact

Lin Li (corresponding author) - lilin@wust.edu.cn
