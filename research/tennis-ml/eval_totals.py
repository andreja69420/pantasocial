"""Evaluate the total-games model against Bet365 / Unibet / Ladbrokes over-under lines (closing), 2014-2025.
Retired matches are void. Explore = 2014-2019, confirm = 2020-2025.

usage: eval_totals.py DATA_DIR TOUR [--open]
"""
import sys

import numpy as np
import pandas as pd

D, TOUR = sys.argv[1:3]
SUF = "_o" if "--open" in sys.argv else ""  # --open: price at the opening over/under odds (same line)
T = pd.read_parquet(f"{D}/out/tot_{TOUR}.parquet")
E = pd.read_parquet(f"{D}/out/{TOUR}_te_extra.parquet").reindex(T.index)
lvl = pd.read_parquet(f"{D}/out/{TOUR}_f2.parquet", columns=["tourney_level"]).tourney_level.astype(str).reindex(T.index)
pools = {}


def p_over_under(mu, key, line):
    if key not in pools:
        y, b = key.split("_bo")
        try:
            pools[key] = np.load(f"{D}/out/tot_res_{TOUR}_{y}_bo{b}.npy")
        except FileNotFoundError:
            pools[key] = np.load(f"{D}/out/tot_res_{TOUR}_{y}_bo3.npy")
    sim = mu + pools[key]
    return np.mean(sim > line), np.mean(sim < line)


rows = []
for bk in ("b365", "ub", "lb"):
    L, oo, ou = E[f"{bk}_tline"].values, E[f"{bk}_over{SUF}"].values, E[f"{bk}_under{SUF}"].values
    ok = ~np.isnan(L) & (oo > 1) & (ou > 1)
    for i in np.flatnonzero(ok):
        po, pu = p_over_under(T.mu.values[i], T.res_key.values[i], L[i])
        tot = T.total.values[i]
        for side, p, o, win in (("over", po, oo[i], tot > L[i]), ("under", pu, ou[i], tot < L[i])):
            push = 1 - po - pu
            pnl = 0.0 if np.isnan(tot) or tot == L[i] else (o - 1 if win else -1.0)
            rows.append((T.index[i], bk, side, T.year.values[i], lvl.values[i], L[i], o, p, p * o + push - 1, pnl,
                         1 / o, 1 / oo[i] + 1 / ou[i] - 1, not np.isnan(tot)))
S = pd.DataFrame(rows, columns=["mi", "bk", "side", "year", "lvl", "line", "o", "p", "ev", "pnl", "imp", "margin", "done"])
S["seg"] = np.where(S.lvl.str.match(r"^\d") | (S.lvl == "S"), "ITF", np.where(S.lvl == "C", "Challenger/125", "Tour"))
S["fair"] = S.imp / (1 + S.margin)
print(f"{TOUR}: {S.mi.nunique()} matches with lines; median margin by book "
      + ", ".join(f"{b} {g.margin.median():.1%}" for b, g in S.groupby("bk")))
# calibration check on completed matches, Bet365 over side: model vs market log-loss
c = S[(S.bk == "b365") & (S.side == "over") & S.done & (S.pnl != 0)]
y = (c.pnl > 0).astype(float)
ll = lambda q: -np.mean(y * np.log(np.clip(q, 1e-4, 1)) + (1 - y) * np.log(np.clip(1 - q, 1e-4, 1)))
print(f"over outcome log-loss: market {ll(c.fair.values):.4f}  model {ll(c.p.values):.4f}  (n={len(c)})")
rng = np.random.default_rng(0)
for seg, G in S.groupby("seg"):
    for bk in ("b365", "best"):
        if bk == "best":  # per match and side, the book with the highest model EV
            H = G.sort_values("ev", ascending=False).drop_duplicates(["mi", "side"])
        else:
            H = G[G.bk == bk]
        if len(H) < 1000:
            continue
        print(f"\n== {TOUR.upper()} {seg} @{bk}: {H.mi.nunique()} matches")
        for thr in (0.0, 0.03, 0.05, 0.10):
            for side in ("both", "over", "under"):
                b = H[(H.ev > thr) & (H.o <= 2.5) & ((H.side == side) if side != "both" else True)]
                if len(b) < 200:
                    continue
                e, f = b[b.year < 2020].pnl, b[b.year >= 2020].pnl
                se = b.pnl.std() / np.sqrt(len(b))
                by = b.groupby("year").pnl.mean()
                pool = H.pnl[(H.side == side) if side != "both" else slice(None)].values
                rnd = np.mean([rng.choice(pool, len(b)).mean() for _ in range(100)])
                print(f"  EV>{thr:4.0%} {side:5s} bets {len(b):6d} ROI {b.pnl.mean():+6.2%} 90%lo {b.pnl.mean() - 1.645 * se:+5.1%} "
                      f"yrs+ {(by > 0).sum()}/{len(by)} random {rnd:+5.1%} | 2014-19 {e.mean():+6.1%} (n={len(e)}) 2020-25 {f.mean():+6.1%} (n={len(f)})")
