# -*- coding: utf-8 -*-
"""Is the placement spread larger than the seed noise?

Two analyses per module and resolution.

One-way ANOVA treats the seed runs of each placement as replicates and asks
whether placement means differ. Its within-group variance includes the seed main
effect, which is large here (each seed is a different train/validation split),
so it is the conservative test.

The seed-blocked (two-way, additive) ANOVA enters seed as a block and removes
that main effect, asking whether placement differences exceed the placement x
seed interaction. It is the more powerful test and the one a reviewer is likely
to run.

Both are reported because they answer different questions: the blocked test asks
"is there any placement effect at all", the one-way test asks "would a
practitioner see it across splits".

    python stats_anova.py

Writes results/anova.json.
"""
import json
import os
import statistics as st

from paths import RESULTS

MODULES = ["se", "cbam", "eca"]
PLACEMENTS = ["all", "4", "3,4", "2,3,4", "1,4"]
FCRIT = {(4, 20): 2.87, (4, 16): 3.01, (4, 8): 3.84}   # 5% upper tail


def load(name):
    with open(os.path.join(RESULTS, name)) as f:
        return json.load(f)


def cell(data, module, placement, seed, tag=""):
    return 100 * max(data[f"{tag}{module}__{placement}__seed{seed}"])


def one_way(data, module, seeds, tag=""):
    groups = [[cell(data, module, p, s, tag) for s in seeds] for p in PLACEMENTS]
    k, n = len(groups), len(seeds)
    grand = st.mean([v for g in groups for v in g])
    ssb = n * sum((st.mean(g) - grand) ** 2 for g in groups)
    sse = sum((v - st.mean(g)) ** 2 for g in groups for v in g)
    dfb, dfe = k - 1, k * (n - 1)
    return {"F": (ssb / dfb) / (sse / dfe), "df": [dfb, dfe],
            "between_sd": st.stdev([st.mean(g) for g in groups]),
            "within_sd": (sse / dfe) ** 0.5}


def blocked(data, module, seeds, tag=""):
    M = [[cell(data, module, p, s, tag) for s in seeds] for p in PLACEMENTS]
    k, n = len(M), len(seeds)
    grand = st.mean([v for row in M for v in row])
    rm = [st.mean(r) for r in M]
    cm = [st.mean(M[i][j] for i in range(k)) for j in range(n)]
    ssb = n * sum((r - grand) ** 2 for r in rm)      # n replicates per placement
    sse = sum((M[i][j] - rm[i] - cm[j] + grand) ** 2
              for i in range(k) for j in range(n))
    dfb, dfe = k - 1, (k - 1) * (n - 1)
    return {"F": (ssb / dfb) / (sse / dfe), "df": [dfb, dfe],
            "between_sd": st.stdev(rm), "interaction_sd": (sse / dfe) ** 0.5}


def per_seed_rank(data, module, seeds, tag=""):
    out = {}
    for s in seeds:
        vals = {p: cell(data, module, p, s, tag) for p in PLACEMENTS}
        best = max(vals, key=vals.get)
        worst = min(vals, key=vals.get)
        out[str(s)] = {"best": best, "worst": worst,
                       "spread": vals[best] - vals[worst]}
    return out


def analyse(data, seeds, tag=""):
    res = {"seeds": list(seeds), "modules": {}}
    maxdiff = 0.0
    for mod in MODULES:
        ow = one_way(data, mod, seeds, tag)
        bl = blocked(data, mod, seeds, tag)
        ow["significant"] = ow["F"] > FCRIT.get(tuple(ow["df"]), 9e9)
        bl["significant"] = bl["F"] > FCRIT.get(tuple(bl["df"]), 9e9)
        means = {p: st.mean([cell(data, mod, p, s, tag) for s in seeds])
                 for p in PLACEMENTS}
        maxdiff = max(maxdiff, max(means.values()) - min(means.values()))
        res["modules"][mod] = {"one_way": ow, "blocked": bl,
                               "means": means,
                               "per_seed": per_seed_rank(data, mod, seeds, tag)}
    res["largest_placement_spread"] = maxdiff
    return res


def show(tag, r):
    print(f"\n===== {tag}: {len(r['seeds'])} seeds, placement spread "
          f"{r['largest_placement_spread']:.3f} pp =====")
    for mod, m in r["modules"].items():
        ow, bl = m["one_way"], m["blocked"]
        print(f"  {mod:4s} one-way F={ow['F']:5.2f} (df {ow['df'][0]},{ow['df'][1]})"
              f" {'SIG ' if ow['significant'] else 'ns  '}"
              f"| blocked F={bl['F']:5.2f} (df {bl['df'][0]},{bl['df'][1]})"
              f" {'SIG ' if bl['significant'] else 'ns  '}"
              f"| between-SD {bl['between_sd']:.3f} vs interaction-SD "
              f"{bl['interaction_sd']:.3f} (ratio {bl['between_sd'] / bl['interaction_sd']:.2f})")


def main():
    full = load("placement_sweep_det_fullres.json")
    low = load("placement_sweep_det.json")
    nfull = sum(1 for k in full if k.startswith("se__") and "__seed" in k) // 5
    nlow = sum(1 for k in low if k.startswith("se__") and "__seed" in k) // 5
    out = {
        "full_resolution": analyse(full, range(nfull)),
        "low_resolution": analyse(low, range(nlow)),
    }
    show("full resolution", out["full_resolution"])
    show("32x32", out["low_resolution"])

    # the reduced-budget probe answers a different question, but it is the same
    # statistic, so run it with the same code once its file exists
    for ep in (3, 6):
        name = f"placement_sweep_det_ep{ep}.json"
        if not os.path.exists(os.path.join(RESULTS, name)):
            continue
        d = load(name)
        n = sum(1 for k in d if k.startswith("se__") and "__seed" in k) // 5
        key = f"budget_{ep}_epochs"
        out[key] = analyse(d, range(n))
        show(f"{ep}-epoch budget (224 input)", out[key])

    # the data-limited probe: same statistic, tenth of the training set between
    # the model and the same validation split (keys carry a "frac0.1__" prefix)
    if os.path.exists(os.path.join(RESULTS, "placement_databudget.json")):
        d = load("placement_databudget.json")
        tag = "frac0.1__"
        n = sum(1 for k in d if k.startswith(tag + "se__") and "__seed" in k) // 5
        out["data_limited_10pct"] = analyse(d, range(n), tag)
        show("10% of training set (224 input)", out["data_limited_10pct"])
    path = os.path.join(RESULTS, "anova.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
