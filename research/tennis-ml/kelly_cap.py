"""Quarter Kelly from the current bank with a cap per bet, on the plan rule (blend EV>5%, odds<=5, no ITF / Davis Cup /
BJK Cup) of eval_levels2026.py --bets=... runs (they carry the match start). Stakes are set from the bank at the start
of each day and settled at the end of it. Each run starts from START; ALL chains the runs into one bank.

usage: kelly_cap.py NAME=BETS_PARQUET [NAME=BETS_PARQUET ...] [--start=100000] [--cap=30000] [--frac=0.25] [--ev=0.05]
"""
import re
import sys

import numpy as np
import pandas as pd

opt = lambda k, d: float(next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{k}=")), d))
START, CAP, FRAC, EV = opt("start", 100000), opt("cap", 30000), opt("frac", 0.25), opt("ev", 0.05)
runs = [a.split("=", 1) for a in sys.argv[1:] if not a.startswith("--")]


def plan(path):
    S = pd.read_parquet(path)
    b = S[(S.evb > EV) & (S.o <= 5) & ~S.lvl.astype(str).str.match(r"^(\d|S$|D$)")].copy()
    b["day"] = pd.to_datetime(b.start, unit="s").dt.tz_localize("UTC").dt.tz_convert("Europe/Belgrade").dt.date
    b["f"] = FRAC * b.evb / (b.o - 1)  # Kelly fraction of the blend P at the taken price, quartered
    return b.sort_values("start")


def sim(b, bank):
    path, stakes = [bank], []
    for _, g in b.groupby("day"):
        st = np.minimum(g.f.values * bank, CAP)
        stakes += list(st)
        bank += float((st * g.pnl.values).sum())
        path.append(bank)
    p = np.array(path)
    peak = np.maximum.accumulate(p)
    return bank, p, peak, np.array(stakes)


rows = []
for name, path in runs + [("ALL", None)]:
    b = pd.concat([plan(p) for _, p in runs]).sort_values("start") if path is None else plan(path)
    end, p, peak, st = sim(b, START)
    rows.append({"run": name, "bets": len(b), "from": b.day.min(), "to": b.day.max(), "end bank": round(end),
                 "profit": round(end - START), "largest drop": round((peak - p).max()),
                 "largest drop %": round(100 * ((peak - p) / peak).max(), 1), "lowest bank": round(p.min()),
                 "median stake": round(float(np.median(st))), "bets at cap": int((st >= CAP).sum())})
pd.set_option("display.width", 250)
print(f"EV>{EV:.0%}, start {START:,.0f}, {FRAC:g} Kelly of the current bank, at most {CAP:,.0f} a bet")
print(pd.DataFrame(rows).to_string(index=False))
