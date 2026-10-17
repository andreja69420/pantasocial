"""v2 vs v3 per tournament level: log-loss on all 2026 matches, best-opening-price ROI (blend EV>5%).
usage: seg_compare.py DATA_DIR   (DATA_DIR/out_fx: r6_v2 predictions, DATA_DIR/out_v3: r6_v3)"""
import sys
import numpy as np, pandas as pd
D = sys.argv[1]
NAMES = {"wta": {"G": "Grand Slam", "PM": "WTA 1000", "P": "WTA 500", "I": "WTA 250", "C": "WTA 125", "100": "W100", "75": "W75",
                 "50": "W50", "35": "W35", "15": "W15", "D": "BJK Cup"},
         "atp": {"G": "Grand Slam", "M": "Masters 1000", "A": "ATP 250/500", "C": "Challenger", "25": "M25", "15": "M15", "D": "Davis Cup"}}
lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
def shin(a, b):
    lo, hi = np.zeros_like(a), np.full_like(a, 0.5); B = a + b
    f = lambda z, x: (np.sqrt(z ** 2 + 4 * (1 - z) * x ** 2 / B) - z) / (2 * (1 - z))
    for _ in range(50):
        z = (lo + hi) / 2; s = f(z, a) + f(z, b); hi, lo = np.where(s < 1, z, hi), np.where(s < 1, lo, z)
    return f((lo + hi) / 2, a)
out = []
for tour in ("atp", "wta"):
    M = pd.read_parquet(f"{D}/out_v3/{tour}_matches.parquet", columns=["tourney_date", "winner_id", "loser_id", "tourney_level"])
    fs = pd.read_csv(f"{D}/fs_raw/{tour}__{tour}_matches_fs_2026.csv", usecols=["tourney_date", "winner_id", "loser_id", "fs_mid"])
    key = lambda d: d.tourney_date.astype(int).astype(str) + "|" + d.winner_id.astype(int).astype(str) + "|" + d.loser_id.astype(int).astype(str)
    M["fs_mid"] = key(M).map(dict(zip(key(fs), fs.fs_mid)))
    O = pd.read_parquet(f"{D}/fs_raw/{tour}_fs_odds_GB.parquet")
    for v, od in (("v2", "out_fx"), ("v3", "out_v3")):
        P = pd.read_parquet(f"{D}/{od}/preds3_{tour}_ens_r6_{v}_lower_2026.parquet").join(M, how="inner")
        P["lvl"] = P.tourney_level.astype(str)
        ll = -(P.y * np.log(P.p) + (1 - P.y) * np.log(1 - P.p))
        for l, g in ll.groupby(P.lvl):
            out.append({"tour": tour, "lvl": l, "v": v, "n": len(g), "ll": g.mean(), "acc": ((P.p[g.index] > .5) == P.y[g.index]).mean()})
        J = P.dropna(subset=["fs_mid"]).reset_index().merge(O, on="fs_mid")
        ow = np.where(J.home_won, J.open_h, J.open_a); ol = np.where(J.home_won, J.open_a, J.open_h)
        R = []
        for won in (True, False):
            q = np.where((J.y == 1) == won, J.p, 1 - J.p); o, oo = (ow, ol) if won else (ol, ow)
            R.append(pd.DataFrame({"lvl": J.lvl.values, "mi": J["index"].values, "q": q, "o": o, "oo": oo, "won": won,
                                   "pnl": np.where(J.stands.values, np.where(won, o - 1, -1.0), 0.0)}))
        S = pd.concat(R, ignore_index=True); S = S[(S.o > 1) & (S.oo > 1)].copy()
        S["evb"] = 1 / (1 + np.exp(-(0.44 * lg(S.q.values) + 0.64 * lg(shin(1 / S.o.values, 1 / S.oo.values))))) * S.o - 1
        best = S.sort_values("o", ascending=False).drop_duplicates(["mi", "won"])
        for cap in (2, 5):
            b = best[(best.evb > 0.05) & (best.o <= cap)]
            for l, g in b.groupby("lvl"):
                out.append({"tour": tour, "lvl": l, "v": v, f"bets{cap}": len(g), f"roi{cap}": g.pnl.mean(),
                            f"lo{cap}": g.pnl.mean() - 1.645 * g.pnl.std() / np.sqrt(len(g))})
T = pd.DataFrame(out).groupby(["tour", "lvl", "v"]).first().unstack("v")
T = T[[c for c in T.columns if c[0] in ("n", "ll", "acc", "bets5", "roi5", "lo5", "bets2", "roi2")]]
T.index = [f"{t} {NAMES[t].get(l, l)}" for t, l in T.index]
pd.set_option("display.width", 250)
print(T.round(4).to_string())
