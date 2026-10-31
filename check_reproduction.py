# -*- coding: utf-8 -*-
"""Check that the separately trained gate-analysis models reproduce the sweep.

The gate study (gate_stats_fullres.py) trains its own copies of the three
modules at the "all" placement, seeds 0-2, and stores the per-epoch validation
curve next to the gate statistics. Under the deterministic protocol these
curves must equal the corresponding curves in the main sweep
(placement_sweep_det_fullres.json) epoch by epoch.

The `sweep_match` field stored in gate_stats_fullres.json was computed when the
gate runs finished, before the final 5-seed sweep file existed, so it is false
for every model even though the curves match. This script recomputes the check
from the two files as they are distributed.

Usage:
    python check_reproduction.py
"""
import json
import os

from paths import RESULTS


def main():
    gate = json.load(open(os.path.join(RESULTS, "gate_stats_fullres.json")))
    sweep = json.load(open(os.path.join(RESULTS, "placement_sweep_det_fullres.json")))
    ok = 0
    total = 0
    for key, rec in sorted(gate.items()):
        module, seed = key.split("__seed")
        skey = f"{module}__all__seed{seed}"
        total += 1
        same = rec["accs"] == sweep[skey]
        ok += same
        print(f"{key:12s} vs {skey:18s} per-epoch identical: {same}")
    print(f"\n{ok}/{total} gate models reproduce the main sweep exactly")


if __name__ == "__main__":
    main()
