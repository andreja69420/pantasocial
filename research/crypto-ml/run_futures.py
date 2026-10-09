"""Long/short on USDT-M perps: XGBoost walk-forward signal, real funding, futures fees, slippage stress,
and an execution-delay variant priced off hourly bars."""
import json
import sys

import numpy as np
import pandas as pd

from backtest import metrics, simulate, top_k_weights
from features import load_panel

D = sys.argv[1]
END = "2026-09-30"  # last full month of funding archives
P = load_panel(D, "1d")
close = P["close"]
mapping = json.load(open(f"{D}/spot_to_perp.json"))
fr = pd.read_parquet(f"{D}/funding.parquet")
inv = {}
for s, (p, _) in mapping.items():
    inv[p] = s
fr["sym"] = fr["perp"].map(inv)
fr = fr.dropna(subset=["sym"])
# Funding settled during (t, t+1] is charged to the position held from close t to close t+1.
fr["day"] = (fr["time"] - pd.Timedelta(milliseconds=1)).dt.floor("D")
fund = fr.groupby(["day", "sym"])["rate"].sum().unstack("sym")
fund.index = fund.index - pd.Timedelta(days=1)
listed = fr.groupby("sym")["time"].agg(["min", "max"])
perp_ok = pd.DataFrame({s: (close.index >= listed.loc[s, "min"].floor("D") + pd.Timedelta(days=1)) &
                           (close.index <= listed.loc[s, "max"].floor("D") - pd.Timedelta(days=2))
                        for s in listed.index}, index=close.index).reindex(columns=close.columns, fill_value=False)

pred = pd.read_parquet(f"{D}/pred_quick_xgb.parquet")["pred"]
X = pd.read_parquet(f"{D}/X_1d_h1.parquet", columns=["log_qv_30"])
dates = pred.index.get_level_values(0).unique().sort_values()
dates = dates[dates <= END]
pu = pred.unstack("symbol").reindex(dates)
pu = pu.where(perp_ok.reindex(index=dates, columns=pu.columns).fillna(False).astype(bool))
liq = X["log_qv_30"].unstack("symbol").reindex(index=dates, columns=pu.columns)

ret_d = (close.ffill().shift(-1) / close.ffill() - 1).where(close.notna()).reindex(dates)
F = fund.reindex(index=dates, columns=ret_d.columns).fillna(0.0)

# Execute one hour after the daily signal, using hourly prices (available for the top-50 universe only).
# Daily bar t closes at t+1 00:00; the 1h bar opening at t+1 00:00 closes at t+1 01:00.
H = load_panel(D, "1h")["close"].ffill()
h01 = H[H.index.hour == 0]
h01.index = h01.index.floor("D") - pd.Timedelta(days=1)
ret_h = (h01.shift(-1) / h01 - 1).reindex(index=dates)

rows, curves = {}, {}


def run(name, scores, k, reb, cost, rets, smooth=1):
    s = scores.rolling(smooth, min_periods=1).mean() if smooth > 1 else scores
    w = top_k_weights(s.stack(), k, long_short=True).reindex(index=dates, columns=rets.columns, fill_value=0)
    # funding enters as an extra return: longs pay positive funding, shorts receive it
    r_adj = rets.fillna(0) - F.reindex(columns=rets.columns).fillna(0)
    sim = simulate(w, r_adj, reb, cost).iloc[:-2]
    m = metrics(sim["ret"])
    m["turnover/day"] = sim["turnover"].mean()
    w_held = w.iloc[::reb].reindex(dates).ffill()
    m["funding_pnl_ann"] = -(w_held * F).sum(axis=1).mean() * 365
    m.update({str(y): (1 + g).prod() - 1 for y, g in sim["ret"].groupby(sim.index.year)})
    rows[name] = m
    curves[name] = (1 + sim["ret"]).cumprod()


for univ in [100, 50, 30]:
    sc = pu.where(liq.rank(axis=1, ascending=False) <= univ)
    k = {100: 10, 50: 7, 30: 5}[univ]
    for cost in [0.001, 0.002, 0.003]:
        run(f"top{univ} LS {k}x{k} daily  cost{cost:.1%}", sc, k, 1, cost, ret_d)
    run(f"top{univ} LS {k}x{k} 3d-rebal smooth3 cost0.2%", sc, k, 3, 0.002, ret_d, smooth=3)
    run(f"top{univ} LS {k}x{k} weekly smooth7 cost0.2%", sc, k, 7, 0.002, ret_d, smooth=7)
    if univ <= 50:
        sch = sc.where(ret_h.notna())
        run(f"top{univ} LS {k}x{k} daily  cost0.2% EXEC +1h", sch, k, 1, 0.002, ret_h)
        run(f"top{univ} LS {k}x{k} 3d-rebal smooth3 cost0.2% EXEC +1h", sch, k, 3, 0.002, ret_h, smooth=3)

bh = ret_d["BTCUSDT"].iloc[:-2]
rows["BTC buy&hold"] = {**metrics(bh), **{str(y): (1 + g).prod() - 1 for y, g in bh.groupby(bh.index.year)}}
curves["BTC buy&hold"] = (1 + bh).cumprod()
res = pd.DataFrame(rows).T
res.to_csv(f"{D}/results_futures.csv")
pd.DataFrame(curves).to_parquet(f"{D}/curves_futures.parquet")
pd.set_option("display.width", 250)
print(res.round(3).to_string())
