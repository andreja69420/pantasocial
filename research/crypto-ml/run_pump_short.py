"""Focused test of the one edge the attribution found: shorting coins that pumped into the liquid set.
Short leg on USDT-M perps (real funding, perp availability), hedged with a BTC long of equal notional."""
import sys

import numpy as np
import pandas as pd

D = sys.argv[1]
src = open("run_futures.py").read().split("for univ in [100, 50, 30]:")[0]
g = {"__name__": "lib"}
sys.argv = ["x", D]
exec(src, g)
from backtest import metrics, simulate  # noqa: E402

dates, F, ret_d, P, pu, liq = g["dates"], g["F"], g["ret_d"], g["P"], g["pu"], g["liq"]
close = P["close"].reindex(dates)
qv = P["quote_volume"].reindex(dates)
spike = np.log1p(qv.rolling(3).mean()) - np.log1p(qv.rolling(60, min_periods=20).mean())
r7 = np.log(close / close.shift(7))
in100 = liq.rank(axis=1, ascending=False) <= 100
perp = pu.notna() | g["perp_ok"].reindex(index=dates, columns=pu.columns).fillna(False).astype(bool)
perp = perp & in100
r_adj = (ret_d.fillna(0) - F.reindex(columns=ret_d.columns).fillna(0))

rows, curves = {}, {}


def short_book(score, k, cand):
    s = score.where(cand)
    rk = s.rank(axis=1, ascending=False)
    n = np.minimum(k, s.notna().sum(axis=1)).replace(0, np.nan)
    return -(rk <= k).astype(float).div(n, axis=0).fillna(0) * 0.5


def run(name, sw, reb, cost):
    w = sw.reindex(columns=ret_d.columns, fill_value=0).copy()
    w["BTCUSDT"] = w["BTCUSDT"] + (-w.sum(axis=1)).clip(lower=0)  # BTC long hedge sized to the short book
    sim = simulate(w, r_adj, reb, cost).iloc[:-2]
    m = metrics(sim["ret"])
    m["turnover/day"] = sim["turnover"].mean()
    m["avg_names"] = (sw < 0).sum(axis=1).mean()
    m.update({str(y): (1 + q).prod() - 1 for y, q in sim["ret"].groupby(sim.index.year)})
    rows[name] = m
    curves[name] = (1 + sim["ret"]).cumprod()


ml_score = -pu  # lowest prediction = strongest short
pumped = (spike > 0.7) & (r7 > 0.15) & perp
for k in [5, 10]:
    for reb in [1, 3, 7]:
        cost = 0.003
        run(f"ML short bottom{k} (top100 perps) reb{reb}d cost0.3%", short_book(ml_score, k, perp), reb, cost)
        run(f"RULE short pumped (vol spike & +15%/7d) top{k} by r7 reb{reb}d cost0.3%",
            short_book(r7, k, pumped), reb, cost)
        run(f"ML short among pumped {k} reb{reb}d cost0.3%", short_book(ml_score, k, pumped), reb, cost)
for cost in [0.005, 0.01]:
    run(f"ML short bottom10 reb3d cost{cost:.1%}", short_book(ml_score, 10, perp), 3, cost)

bh = ret_d["BTCUSDT"].iloc[:-2]
rows["BTC buy&hold"] = {**metrics(bh), **{str(y): (1 + q).prod() - 1 for y, q in bh.groupby(bh.index.year)}}
res = pd.DataFrame(rows).T
res.to_csv(f"{D}/results_pump_short.csv")
pd.DataFrame(curves).to_parquet(f"{D}/curves_pump_short.parquet")
pd.set_option("display.width", 250)
print(res.round(3).to_string())
