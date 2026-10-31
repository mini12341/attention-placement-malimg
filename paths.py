# -*- coding: utf-8 -*-
"""Filesystem layout for this release.

Every script imports its paths from here, so nothing is hard-coded and the code
runs unmodified on any machine. Any entry can be overridden with an environment
variable:

    MALIMG_FULLRES   full-resolution Malimg (one folder per family, original PNGs)
    MALIMG_32        32x32 Malimg (one folder per family)
    RESULTS_DIR      where result JSONs are written  (default: ./results)
    FIGURES_DIR      where figures and gate dumps go  (default: ./figures)
    CKPT_DIR         where checkpoints are written    (default: ./checkpoints)
"""

import os

ROOT = os.path.dirname(os.path.abspath(__file__))
_env = os.environ.get

# --- datasets -----------------------------------------------------------------
# Full resolution: 9,339 byteplot images, variable size, as distributed by
#   https://huggingface.co/datasets/ikrambenabd/malimg-original
DATA_FULL = _env("MALIMG_FULLRES", os.path.join(ROOT, "data", "malimg_fullres"))
# 32x32 control: produced from malimg.npz by convert_malimg.py
DATA_32 = _env("MALIMG_32", os.path.join(ROOT, "data", "malimg_32"))

# --- outputs ------------------------------------------------------------------
RESULTS = _env("RESULTS_DIR", os.path.join(ROOT, "results"))
FIGURES = _env("FIGURES_DIR", os.path.join(ROOT, "figures"))
CHECKPOINTS = _env("CKPT_DIR", os.path.join(ROOT, "checkpoints"))
CKPT_FULLRES = os.path.join(CHECKPOINTS, "fullres")


def ensure_dirs():
    for d in (RESULTS, FIGURES, CHECKPOINTS, CKPT_FULLRES):
        os.makedirs(d, exist_ok=True)
