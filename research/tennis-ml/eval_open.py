"""Moneyline at OPENING odds (Bet365, best, Betfair, Unibet, Ladbrokes) vs closing, by segment; modern = 2020-2025.
Betfair winnings pay 5% commission.

usage: eval_open.py DATA_DIR
"""
import sys

import numpy as np
import pandas as pd

D = sys.argv[1]
TE = "2014-2015-2016-2017-2018-2019-2020-2021-2022-2023-2024-2025"
rng = np.random.default_rng(0)
for tour, segs in (("atp", {"Tour": r"^[GMAF]", "Challenger": r"^C$"}), ("wta", {"ITF": r"^\d", "WTA125": r"^C$", "Tour": r"^[GPIF]"})):
    P = pd.read_parquet(f"{D}/out/preds3_{tour}_ens_r6_hist1991_ens_te_{TE}.parquet")
    E = pd.read_parquet(f"{D}/out/{tour}_te_extra.parquet").reindex(P.index)
    C = pd.read_parquet(f"{D}/out/{tour}_te_odds.parquet").reindex(P.index)
    lvl = pd.read_parquet(f"{D}/out/{tour}_f2.parquet", columns=["tourney_level"]).tourney_level.astype(str).reindex(P.index)
    for seg, rx in segs.items():
        G = P[lvl.str.match(rx).fillna(False).values]
        for src, (cw, cl) in {"b365 close": ("b365_w", "b365_l"), "b365 OPEN": ("b365o_w", "b365o_l"),
                              "best close": ("best_w", "best_l"), "best OPEN": ("besto_w", "besto_l"),
                              "bf close": ("bf_w", "bf_l"), "bf OPEN": ("bfo_w", "bfo_l"),
                              "ub OPEN": ("ubo_w", "ubo_l"), "lb OPEN": ("lbo_w", "lbo_l")}.items():
            T = E if ("OPEN" in src or src.startswith("bf")) else C
            com = 0.05 if src.startswith("bf") else 0.0
            w, l = T.loc[G.index, cw].astype(float).values, T.loc[G.index, cl].astype(float).values
            o1, o2 = np.where(G.y == 1, w, l), np.where(G.y == 1, l, w)
            ok = (o1 > 1) & (o2 > 1)
            rows = []
            for s, o, q, win in ((1, o1, G.p.values, G.y.values == 1), (2, o2, 1 - G.p.values, G.y.values == 0)):
                pick = ok & (q * o - 1 > 0.05) & (o <= 2.0)
                rows.append(pd.DataFrame({"year": G.year.values[pick], "pnl": np.where(G.stands.values, np.where(win, (o - 1) * (1 - com), -1.0), 0.0)[pick]}))
            b = pd.concat(rows)
            mod = b[b.year >= 2020].pnl
            se = mod.std() / np.sqrt(max(len(mod), 1))
            print(f"{tour.upper()} {seg:10s} {src:10s} margin {np.nanmedian(1 / w + 1 / l - 1):5.1%} | all yrs n={len(b):6d} ROI {b.pnl.mean():+6.2%} "
                  f"| 2020-25 n={len(mod):6d} ROI {mod.mean():+6.2%} 90%lo {mod.mean() - 1.645 * se:+.1%}")
