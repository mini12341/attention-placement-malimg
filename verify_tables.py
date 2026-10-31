# -*- coding: utf-8 -*-
"""Recompute every table cell in the manuscript from the result JSONs.

The paper's tables are all derived, not measured by hand: each cell is a
statistic over the per-epoch accuracies or per-image gate activations stored in
results/*.json. This script recomputes each cell and compares it, number by
number, against the printed value in main.tex.

    python verify_tables.py                          # ../latex/jisa/main.tex
    python verify_tables.py path/to/main.tex

Exit code 0 means every printed number is backed by the JSONs. Any mismatch or
unparseable cell is reported individually.

Tolerance is half a unit in the last printed decimal, so the check fails only on
a genuine discrepancy, not on rounding.
"""
import json
import os
import re
import statistics as st
import sys

from paths import RESULTS, ROOT

TEX = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(ROOT), "latex", "jisa", "main.tex")

MODULES = ["se", "cbam", "eca"]
MODNAME = {"se": "SE", "cbam": "CBAM", "eca": "ECA"}
STAGES = "1234"

# Canonical placement keys, in the order they appear in every accuracy table.
PLACEMENTS = [("baseline", "baseline (none)"), ("all", "all"),
              ("4", "att4 only"), ("3,4", "att3+4"),
              ("2,3,4", "att2+3+4"), ("1,4", "att1+4")]

# Labels as used in the appendix's per-seed table.
SHORT = {"baseline (none)": "none", "all": "all", "att4 only": "att4",
         "att4": "att4", "att3+4": "att3+4", "att2+3+4": "att2+3+4",
         "att1+4": "att1+4"}

problems = []
checked = 0


def load(name):
    with open(os.path.join(RESULTS, name)) as f:
        return json.load(f)


TEX_TEXT = ""


def table_block(label):
    """Rows of the tabular belonging to \\label{label}, as lists of raw cells."""
    tex = TEX_TEXT
    i = tex.index("\\label{%s}" % label)
    start = tex.index("\\midrule", i) + len("\\midrule")
    end = tex.index("\\bottomrule", start)
    rows = []
    for chunk in tex[start:end].split("\\\\"):
        chunk = " ".join(chunk.split())
        # a block may start with a rule that shares its chunk with the next row
        if chunk.startswith("\\midrule"):
            chunk = chunk[len("\\midrule"):].strip()
        if "&" not in chunk or chunk.startswith("\\bottomrule"):
            continue
        rows.append([c.strip() for c in chunk.split("&")])
    return rows


FLOAT = re.compile(r"-?\d+(?:\.\d+)?")


def numbers(cell):
    """Floats printed inside a math cell; [] for a label cell."""
    if "$" not in cell:
        return []
    body = cell[cell.index("$") + 1:cell.rindex("$")]
    return [float(x) for x in FLOAT.findall(body)]


def close(a, b, decimals):
    return abs(a - b) <= 0.5 * 10 ** (-decimals) + 1e-9


def check(where, got, printed, decimals):
    """got/printed are equal-length tuples; decimals is a scalar or per-column."""
    global checked
    checked += 1
    ds = decimals if isinstance(decimals, (list, tuple)) else [decimals] * len(got)
    ok = all(close(g, p, d) for g, p, d in zip(got, printed, ds))
    if not ok:
        problems.append(f"{where}: printed {printed}, recomputed "
                        f"{tuple(round(g, max(ds) + 1) for g in got)}")


def accuracy_stats(data, seeds, module, label):
    vals = [100 * max(data[f"{module}__{label}__seed{s}"]) for s in seeds]
    return (st.mean(vals), st.stdev(vals) if len(vals) > 1 else 0.0)


def baseline_stats(data, seeds):
    vals = [100 * max(data[f"baseline__seed{s}"]) for s in seeds]
    return (st.mean(vals), st.stdev(vals) if len(vals) > 1 else 0.0)


def seeds_of(data):
    """Seed indices present in an accuracy JSON, so the check follows the data."""
    import re as _re
    return sorted({int(m.group(1)) for k in data
                   for m in [_re.search(r"__seed(\d+)$", k)] if m})


def check_accuracy_table(label_tex, filename, seeds=None):
    data = load(filename) if isinstance(filename, str) else filename
    if seeds is None:
        seeds = seeds_of(data)
    return _check_accuracy_table(label_tex, data, seeds)


def _check_accuracy_table(label_tex, data, seeds):
    rows = table_block(label_tex)
    for row in rows:
        head = row[0]
        key = next((k for k, disp in PLACEMENTS if head.startswith(disp.split(" ")[0])), None)
        if key is None:
            continue
        if key == "baseline":
            got = baseline_stats(data, seeds)
            check(f"{label_tex} baseline", got, tuple(numbers(row[1])), 2)
            continue
        for j, mod in enumerate(MODULES):
            cell = row[1 + j]
            if "$" not in cell:
                continue
            check(f"{label_tex} {mod}/{key}",
                  accuracy_stats(data, seeds, mod, key),
                  tuple(numbers(cell)), 2)


def gate_stats(data, seeds, module, stage):
    """gate_stats_*.json: one record per seed, all four stages inside it."""
    means, pcts = [], []
    for s in seeds:
        g = data[f"{module}__seed{s}"]["gates"][stage]
        means.append(g["mean"])
        pcts.append(100 * g["frac_gt_0.9"])
    return (st.mean(means), st.stdev(means) if len(means) > 1 else 0.0,
            st.mean(pcts))


def gate_stats_flat(data, seeds, module, stage):
    """att234_stats_*.json: one record per (module, stage, seed)."""
    means, pcts = [], []
    for s in seeds:
        g = data[f"{module}/att{stage}/seed{s}"]
        means.append(g["mean"])
        pcts.append(100 * g["frac_gt_0.9"])
    return (st.mean(means), st.stdev(means) if len(means) > 1 else 0.0,
            st.mean(pcts))


def check_gate_table(label_tex, filename, seeds, stages=STAGES, flat=False):
    """Cross-seed gate table: mean +- sd across seeds / percentage over 0.9."""
    data = load(filename)
    stat = gate_stats_flat if flat else gate_stats
    for row in table_block(label_tex):
        mod = {v: k for k, v in MODNAME.items()}.get(row[0])
        if mod is None:
            continue
        for j, stage in enumerate(stages):
            check(f"{label_tex} {mod}/att{stage}",
                  stat(data, seeds, mod, stage),
                  tuple(numbers(row[1 + j])), [3, 3, 1])


def check_seed_rank_table(label_tex, full, low):
    """Appendix: best and worst placement per (module, seed, resolution)."""
    global checked
    cur = None
    seeds_full, seeds_low = set(seeds_of(full)), set(seeds_of(low))
    for row in table_block(label_tex):
        name = next((m for m in ("CBAM", "ECA", "SE") if m in row[0]), "")
        if name:
            cur = name.lower()
        if cur is None or len(row) < 6:
            continue
        try:
            seed = int(row[1])
        except ValueError:
            continue
        for data, off, res, have in ((full, 2, "full", seeds_full),
                                     (low, 4, "32", seeds_low)):
            printed_cell = row[off].replace("{", "").replace("}", "").strip()
            if (printed_cell == "---") != (seed not in have):
                problems.append(f"{label_tex} {cur}/seed{seed}/{res}: printed "
                                f"{printed_cell!r} but seed {'is' if seed in have else 'is not'} "
                                f"in the results")
                continue
            if seed not in have:
                continue
            vals = {}
            for key, disp in PLACEMENTS:
                if key == "baseline":
                    continue
                vals[disp] = 100 * max(data[f"{cur}__{key}__seed{seed}"])
            best_v = max(vals.values())
            worst_v = min(vals.values())
            # ties are common at this precision; any placement attaining the
            # extreme value is a correct entry
            ties = {SHORT[k] for k, v in vals.items()
                    if abs(v - best_v) < 1e-9 or abs(v - worst_v) < 1e-9}
            for j in (off, off + 1):
                printed = row[j].replace("{", "").replace("}", "").strip()
                checked += 1
                if printed not in ties:
                    want = SHORT[max(vals, key=vals.get)] if j == off else \
                        SHORT[min(vals, key=vals.get)]
                    problems.append(f"{label_tex} {cur}/seed{seed}/{res}: printed "
                                    f"{printed!r}, recomputed {want!r}")


def run(name, fn, *args, **kwargs):
    """Run one table check; a missing JSON entry is a problem, not a crash."""
    try:
        fn(*args, **kwargs)
    except (KeyError, FileNotFoundError) as e:
        problems.append(f"{name}: result data missing ({e!r}) - run the "
                        f"corresponding script first")


def main():
    global TEX_TEXT
    TEX_TEXT = open(TEX, encoding="utf-8").read()

    full = load("placement_sweep_det_fullres.json")
    low = load("placement_sweep_det.json")

    run("tab1", check_accuracy_table, "tab1", "placement_sweep_det_fullres.json")
    run("tab2", check_gate_table, "tab2", "gate_stats_fullres.json", range(3))
    run("tab3", check_accuracy_table, "tab3", "placement_sweep_det.json")
    run("tab4", check_gate_table, "tab4", "gate_stats_32.json", range(3))
    run("tab5", check_gate_table, "tab5", "att234_stats_32.json", range(3),
        stages="234", flat=True)
    run("tab6", check_gate_table, "tab6", "att234_stats_fullres.json", range(3),
        stages="234", flat=True)
    run("appendix", check_seed_rank_table, "tab:seedrank", full, low)

    print(f"checked {checked} numeric cells against {os.path.basename(TEX)}")
    if problems:
        print(f"\n{len(problems)} MISMATCH(ES):")
        for p in problems:
            print("  -", p)
        return 1
    print("all printed numbers match the result JSONs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
