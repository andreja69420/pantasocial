"""Add the new feature modules (fx_*.py outputs, w_<name>/l_<name> on the matches index) to a copy of a build:
OUT_NEW/{tour}_f2.parquet = OUT_IN f2 + every module column renamed w_nx_<name> / l_nx_<name> (float32), so a
training run can select them with the regex 'nx_'. Matches table and odds files are linked, not copied.

usage: fx_merge.py OUT_IN OUT_NEW FX_DIR [MODULES=ratings,score,context,pbp]
"""
import glob
import os
import sys

import pandas as pd

from design import read_f2

OUT_IN, OUT_NEW, FX = sys.argv[1:4]
MODS = (sys.argv[4] if len(sys.argv) > 4 else "ratings,score,context,pbp").split(",")
os.makedirs(OUT_NEW, exist_ok=True)
for f in glob.glob(os.path.join(OUT_IN, "*")):
    dst = os.path.join(OUT_NEW, os.path.basename(f))
    if not f.endswith(("_f2.parquet", "_f2_parts")) and not os.path.basename(f).startswith("preds3_") and not os.path.exists(dst):
        os.symlink(os.path.abspath(f), dst)

for tour in ("wta", "atp"):
    F = read_f2(os.path.join(OUT_IN, f"{tour}_f2.parquet"))  # build + rows appended by features_incr.py
    added = {}
    for m in MODS:
        X = pd.read_parquet(os.path.join(FX, f"{tour}_{m}.parquet"))
        assert X.index.equals(F.index), f"{tour} {m}: index differs from the build"
        X = X.rename(columns=lambda c: f"{c[:2]}nx_{c[2:]}").astype("float32")
        dup = set(X.columns) & set(F.columns)
        assert not dup, f"{tour} {m}: duplicate columns {sorted(dup)[:5]}"
        F = F.join(X)
        added[m] = X.shape[1] // 2
    F.to_parquet(os.path.join(OUT_NEW, f"{tour}_f2.parquet"))
    print(tour, "added per-player features:", added, "total", sum(added.values()), "->", F.shape, flush=True)
