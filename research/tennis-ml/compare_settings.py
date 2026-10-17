"""Compare boosting budgets (learning rate / trees / early-stopping patience) on one dev year, same matches:
log-loss and Brier of the calibrated probabilities, and ROI of EV>5% & odds<=2 at Bet365 opening prices.

usage: compare_settings.py DATA_DIR YEAR TAG[,TAG..]   (baseline = r6_hist1991_ens from the 2014-2025 TE run)
"""
import sys

import numpy as np
import pandas as pd

D, YEAR, TAGS = sys.argv[1], int(sys.argv[2]), sys.argv[3].split(",")
TE = "2014-2015-2016-2017-2018-2019-2020-2021-2022-2023-2024-2025"
SEG = {"atp": ("Challenger", r"^C$"), "wta": ("Women ITF", r"^\d")}
for tour in ("wta", "atp"):
    runs = {"baseline (patience 60)": f"preds3_{tour}_ens_r6_hist1991_ens_te_{TE}.parquet"}
    runs.update({t: f"preds3_{tour}_ens_{t}_te_{YEAR}.parquet" for t in TAGS})
    P = {}
    for name, f in runs.items():
        try:
            P[name] = pd.read_parquet(f"{D}/out/{f}").query("year == @YEAR")
        except FileNotFoundError:
            print(f"{tour}: missing {f}")
    common = sorted(set.intersection(*[set(p.index) for p in P.values()]))
    E = pd.read_parquet(f"{D}/out/{tour}_te_extra.parquet").reindex(common)
    lvl = pd.read_parquet(f"{D}/out/{tour}_f2.parquet", columns=["tourney_level"]).tourney_level.astype(str).reindex(common)
    seg_name, rx = SEG[tour]
    print(f"\n{tour.upper()} {YEAR}: {len(common)} matches in all runs")
    for name, p in P.items():
        p = p.loc[common]
        q, y = p.p.clip(1e-4, 1 - 1e-4).values, (p.y == 1).values
        ll = -np.mean(np.where(y, np.log(q), np.log(1 - q)))
        mk = p.p_mkt.clip(1e-4, 1 - 1e-4).values
        llm = -np.mean(np.where(y, np.log(mk), np.log(1 - mk)))
        pnl = []
        seg = lvl.str.match(rx).fillna(False).values
        w, l = E.b365o_w.values, E.b365o_l.values
        for side in (1, 2):
            won = y if side == 1 else ~y
            o = np.where(y, w, l) if side == 1 else np.where(y, l, w)
            qs = q if side == 1 else 1 - q
            pick = seg & (o > 1) & (qs * o - 1 > 0.05) & (o <= 2)
            pnl.append(np.where(p.stands.values, np.where(won, o - 1, -1.0), 0.0)[pick])
        pnl = np.concatenate(pnl)
        print(f"  {name:24s} log-loss {ll:.5f} (market {llm:.5f})  Brier {np.mean((q - y) ** 2):.5f}  "
              f"{seg_name} b365 open EV>5%: {len(pnl)} bets ROI {pnl.mean():+.2%}")
