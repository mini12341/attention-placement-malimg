# -*- coding: utf-8 -*-
"""Equivalence testing for the "no placement effect" claim.

A null hypothesis test can only fail to reject; it cannot show that two
configurations are the same. To claim that placements are *equivalent* we use
two one-sided tests (TOST) on the seed-paired differences, against an a priori
margin DELTA: differences smaller than DELTA are treated as too small to matter
for a practice decision. DELTA = 0.5 percentage points.

Pairs are compared within seed, because every placement sees the same train /
validation split for a given seed; the seed is therefore the unit of
replication and the pairing removes split-to-split variation.

    python stats_equivalence.py [DELTA]

Writes results/equivalence.json and prints, per module and resolution, the pairs
that are NOT equivalent at DELTA.
"""
import json
import math
import os
import statistics as st
import sys

from paths import RESULTS

DELTA = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
MODULES = ["se", "cbam", "eca"]
PLACEMENTS = ["all", "4", "3,4", "2,3,4", "1,4"]

# 90% two-sided critical values, df = n - 1 (TOST uses one-sided 5% at each end).
T90 = {1: 6.314, 2: 2.920, 3: 2.353, 4: 2.132, 5: 2.015, 6: 1.943, 7: 1.895}


def load(name):
    with open(os.path.join(RESULTS, name)) as f:
        return json.load(f)


def pairs():
    for i in range(len(PLACEMENTS)):
        for j in range(i + 1, len(PLACEMENTS)):
            yield PLACEMENTS[i], PLACEMENTS[j]


def equivalence(data, seeds, module, a, b, tag=""):
    """Paired TOST on the seed-mean accuracy difference (percentage points)."""
    diffs = [100 * (max(data[f"{tag}{module}__{a}__seed{s}"])
                    - max(data[f"{tag}{module}__{b}__seed{s}"])) for s in seeds]
    n = len(diffs)
    mean = st.mean(diffs)
    sd = st.stdev(diffs) if n > 1 else 0.0
    half = T90[n - 1] * sd / math.sqrt(n)
    lo, hi = mean - half, mean + half
    return {"mean": mean, "sd": sd, "ci90": [lo, hi],
            "equivalent": lo > -DELTA and hi < DELTA, "diffs": diffs}


def analyse(title, filename, seeds, tag=""):
    data = load(filename)
    out = {"delta": DELTA, "seeds": seeds, "modules": {}}
    print(f"\n===== {title} ({len(seeds)} seeds, delta = {DELTA} pp) =====")
    worst = 0.0
    halves = []
    for mod in MODULES:
        rows = {}
        for a, b in pairs():
            r = equivalence(data, seeds, mod, a, b, tag)
            rows[f"{a} vs {b}"] = r
            worst = max(worst, abs(r["mean"]))
            halves.append((r["ci90"][1] - r["ci90"][0]) / 2)
        out["modules"][mod] = rows
        bad = {k: v for k, v in rows.items() if not v["equivalent"]}
        span = max(abs(v["mean"]) for v in rows.values())
        print(f"  {mod:4s} largest seed-mean difference {span:.3f} pp; "
              f"equivalent for {len(rows) - len(bad)}/{len(rows)} pairs")
        for k, v in bad.items():
            print(f"      NOT equivalent: {k}: {v['mean']:+.3f} pp")
    out["largest_abs_difference"] = worst
    # interval width is what decides whether a null can be certified, so report
    # the typical and the widest half-width explicitly
    out["median_half_width"] = st.median(halves)
    out["max_half_width"] = max(halves)
    print(f"  interval half-width: median {out['median_half_width']:.3f} pp, "
          f"max {out['max_half_width']:.3f} pp")
    return out


def main():
    result = {
        "full_resolution": analyse("Full resolution", "placement_sweep_det_fullres.json",
                                   [0, 1, 2, 3, 4]),
        "low_resolution": analyse("32x32", "placement_sweep_det.json",
                                  [0, 1, 2, 3, 4]),
    }
    # data-limited probe: does a lower ceiling let the effect surface? Keys in
    # that file are prefixed with the training-set fraction.
    if os.path.exists(os.path.join(RESULTS, "placement_databudget.json")):
        result["data_limited_10pct"] = analyse(
            "10% of training set", "placement_databudget.json",
            [0, 1, 2, 3, 4], "frac0.1__")
    path = os.path.join(RESULTS, "equivalence.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=1)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
