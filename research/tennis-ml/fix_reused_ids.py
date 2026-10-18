"""Split Sackmann ids that hold two different players (same name, e.g. Miguel Tobon b. 1970s and Tobon Miguel Jr b. 2006).

usage: fix_reused_ids.py SACKMANN_RAW_DIR [--apply]

The birth year implied by every row (date - age) clusters per real person; an id whose clusters lie more than 4 years apart is
split: the biggest cluster keeps the id, each other one gets 600000000 + k*1000000 + id (a real, non-synthetic id). Rows without an
age follow the cluster of the nearest dated row. Files are rewritten as plain strings (nothing else changes); a symlink is replaced
by a fixed copy, the target stays as it was. Without --apply it only reports. Idempotent.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

D = sys.argv[1]
APPLY = "--apply" in sys.argv
for tour in ("atp", "wta"):
    files = sorted(glob.glob(f"{D}/{tour}__{tour}_matches_*.csv"))
    fr = {f: pd.read_csv(f, dtype=str, keep_default_na=False) for f in files}
    parts = []
    for f, d in fr.items():
        for s in ("winner", "loser"):
            if f"{s}_age" not in d:
                continue
            i = pd.to_numeric(d[f"{s}_id"], errors="coerce")
            dt = pd.to_datetime(d.tourney_date.str[:8], format="%Y%m%d", errors="coerce")
            parts.append(pd.DataFrame({"f": f, "s": s, "row": np.arange(len(d)), "id": i.values, "date": dt.values,
                                       "age": pd.to_numeric(d[f"{s}_age"], errors="coerce").values}))
    a = pd.concat(parts)
    a = a[(a.id < 9e8) & a.date.notna()].copy()
    yr = a.date.dt.year + (a.date.dt.dayofyear - 1) / 365.25
    a["birth"] = yr - a.age
    k = a.dropna(subset=["birth"]).groupby("id").birth.agg(lambda b: b.max() - b.min())
    bad = k[k > 6].index
    print(f"{tour}: {len(bad)} ids with a birth spread over 6 years")
    new_id = {}
    for i in bad:
        x = a[a.id == i]
        kn = x.dropna(subset=["birth"]).sort_values("birth")
        cl = (kn.birth.diff() > 4).cumsum()
        kn = kn.assign(c=cl.values)
        size = kn.c.value_counts()
        order = {c: n for n, c in enumerate([size.idxmax()] + [c for c in size.index if c != size.idxmax()])}
        kn["k"] = kn.c.map(order)
        unk = x[x.birth.isna()].assign(k=0)
        kd = kn.sort_values("date")
        if len(unk):
            pos = np.abs(unk.date.values[:, None] - kd.date.values[None, :]).argmin(axis=1)
            unk = unk.assign(k=kd.k.values[pos])
        y = pd.concat([kn[["f", "s", "row", "k", "date", "birth"]], unk[["f", "s", "row", "k", "date", "birth"]]])
        print(f"  id {int(i)}: " + ", ".join(f"cluster {n}: {int((y.k == n).sum())} rows, born {y[y.k == n].birth.median():.0f}, "
                                              f"{y[y.k == n].date.min().year}-{y[y.k == n].date.max().year}" for n in sorted(y.k.unique())))
        for r in y[y.k > 0].itertuples():
            new_id[(r.f, r.s, r.row)] = 600000000 + r.k * 1000000 + int(i)
    if not APPLY:
        continue
    for f in {f for f, _, _ in new_id}:
        d = fr[f]
        for (ff, s, row), v in new_id.items():
            if ff == f:
                d.iat[row, d.columns.get_loc(f"{s}_id")] = str(v)
        if os.path.islink(f):
            os.remove(f)
        d.to_csv(f + ".tmp", index=False)
        os.replace(f + ".tmp", f)
    print(f"{tour}: rewrote {len({f for f, _, _ in new_id})} files, {len(new_id)} player rows")
