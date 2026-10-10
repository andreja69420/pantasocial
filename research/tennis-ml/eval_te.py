"""Evaluate predictions settled at TennisExplorer odds (2014-2025): Bet365, Betfair and best price, by level."""
import sys

import numpy as np
import pandas as pd

D, TAG, TOUR = sys.argv[1], sys.argv[2], sys.argv[3]
Y = "2014-2015-2016-2017-2018-2019-2020-2021-2022-2023-2024-2025"
P = pd.read_parquet(f"{D}/out/preds3_{TOUR}_ens_{TAG}_te_{Y}.parquet")
for c in ["p", "y"]:
    P[c] = P[c].astype(float)
O = pd.read_parquet(f"{D}/out/{TOUR}_te_odds.parquet").reindex(P.index)
lvl = pd.read_parquet(f"{D}/out/{TOUR}_f2.parquet", columns=["tourney_level"]).tourney_level.astype(str).reindex(P.index)
if TOUR == "wta":
    P["seg"] = np.where(lvl.str.match(r"^\d"), "ITF", np.where(lvl == "C", "WTA125", "Tour"))
else:
    P["seg"] = np.where(lvl == "C", "Challenger", np.where(lvl.str.match(r"^\d") | (lvl == "S"), "ITF", "Tour"))
w1 = P.y == 1
rng = np.random.default_rng(0)


def run(G, ow, ol, thr=0.05, mo=2.0):
    o1 = np.where(G.y == 1, G[ow], G[ol]).astype(float)
    o2 = np.where(G.y == 1, G[ol], G[ow]).astype(float)
    ok = (o1 > 1) & (o2 > 1)
    pnl, yr, pool = [], [], []
    for s_, o in ((1, o1), (2, o2)):
        q = G.p.values if s_ == 1 else 1 - G.p.values
        win = (G.y == 1).values if s_ == 1 else (G.y == 0).values
        res = np.where(G.stands.values, np.where(win, o - 1, -1.0), 0.0)
        pick = ok & (q * o - 1 > thr) & (o <= mo)
        pnl.append(res[pick]); yr.append(G.year.values[pick]); pool.append(res[ok & (o <= mo)])
    pnl, yr, pool = np.concatenate(pnl), np.concatenate(yr), np.concatenate(pool)
    return pnl, yr, pool, ok


for seg, G in P.groupby("seg"):
    G = G.join(O)
    print(f"\n===== {TOUR.upper()} {seg}: {len(G)} matches with predictions")
    for name, ow, ol in [("Bet365", "b365_w", "b365_l"), ("Betfair", "bf_w", "bf_l"), ("Best", "best_w", "best_l")]:
        if ow not in G or G[ow].notna().sum() < 500:
            continue
        o1 = np.where(G.y == 1, G[ow], G[ol]).astype(float); o2 = np.where(G.y == 1, G[ol], G[ow]).astype(float)
        ok = (o1 > 1) & (o2 > 1)
        pm = (1 / o1) / (1 / o1 + 1 / o2)
        q = np.clip(G.p.values, 1e-6, 1 - 1e-6)
        ll = lambda p: -np.mean(G.y.values[ok] * np.log(p[ok]) + (1 - G.y.values[ok]) * np.log(1 - p[ok]))
        print(f"  {name:8s} n={ok.sum():6d} margin {np.median((1 / o1 + 1 / o2 - 1)[ok]):5.2%}  logloss market "
              f"{ll(np.clip(pm, 1e-6, 1 - 1e-6)):.4f} model {ll(q):.4f}")
        for thr, mo in [(0.05, 2.0), (0.02, 2.0), (0.10, 2.0), (0.05, 100.0)]:
            pnl, yr, pool, _ = run(G, ow, ol, thr, mo)
            if len(pnl) < 50:
                continue
            boot = np.array([pnl[rng.integers(0, len(pnl), len(pnl))].mean() for _ in range(1000)])
            lo, hi = np.percentile(boot, [5, 95])
            rnd = np.array([rng.choice(pool, len(pnl), replace=False).mean() for _ in range(300)]) if len(pool) > len(pnl) else np.array([np.nan])
            by = pd.Series(pnl).groupby(yr).mean()
            tag = "  <- pre-registered" if (thr, mo) == (0.05, 2.0) else ""
            print(f"     EV>{thr:4.0%} odds<={mo:<3g} bets {len(pnl):6d} ROI {pnl.mean():+6.2%} 90%CI [{lo:+.1%},{hi:+.1%}] "
                  f"years+ {(by > 0).sum()}/{len(by)} | random same odds {np.nanmean(rnd):+.1%}{tag}")
        pnl, yr, _, _ = run(G, ow, ol)
        print("       by year (EV>5%, odds<=2): " + " ".join(f"{y}:{v:+.0%}" for y, v in pd.Series(pnl).groupby(yr).mean().items()))
