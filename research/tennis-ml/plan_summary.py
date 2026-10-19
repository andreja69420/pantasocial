"""Plan rule (blend EV>5%, odds<=5, no ITF) on eval_levels2026.py --bets parquet files: bets, flat ROI, lower 90% bound (cluster
bootstrap by match day), per period. usage: plan_summary.py NAME=BETS_PARQUET [NAME=...] [--split=2026-06-08]"""
import sys

import numpy as np
import pandas as pd

sys.argv_ = sys.argv[:]
split = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--split=")), None)
runs = [a.split("=", 1) for a in sys.argv[1:] if not a.startswith("--")]
sys.argv = ["x"]
ns = {}
exec(open("kelly_cap.py").read().split("rows = []")[0], ns)
rng = np.random.default_rng(1)
for name, p in runs:
    b = ns["plan"](p)
    b["d"] = pd.to_datetime(b.day)
    parts = [("all", b)] + ([("before " + split, b[b.d < split]), ("from " + split, b[b.d >= split])] if split else [])
    for lab, x in parts:
        g = x.groupby("d").pnl.agg(["sum", "count"])
        s, c = g["sum"].values, g["count"].values
        boot = [s[i].sum() / c[i].sum() for i in (rng.integers(0, len(s), len(s)) for _ in range(1000))]
        print(f"{name:12s} {lab:22s} {len(x):5d} bets  ROI {x.pnl.mean():+.3f}  lower90 {np.percentile(boot, 5):+.3f}")
