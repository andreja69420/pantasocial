import sys

import numpy as np
import pandas as pd

from backtest import metrics, simulate, top_k_weights
from features import load_panel

D = sys.argv[1]
COST = 0.0015
P = load_panel(D, "1d")
close, qv = P["close"], P["quote_volume"]
rets = (close.ffill().shift(-1) / close.ffill() - 1).where(close.notna())
pred = pd.read_parquet(f"{D}/pred_quick_xgb.parquet")["pred"]
X = pd.read_parquet(f"{D}/X_1d_h1.parquet", columns=["log_qv_30", "fwd"])
dates = pred.index.get_level_values(0).unique().sort_values()
R = rets.reindex(dates)
rows = {}


def rec(name, w, reb=1, cost=COST):
    s = simulate(w.reindex(columns=R.columns, fill_value=0).reindex(dates).fillna(0), R, reb, cost).iloc[:-2]
    m = metrics(s["ret"])
    rows[name] = {k: m[k] for k in ["total_return", "CAGR", "Sharpe", "MaxDD"]}


k = 10
L = top_k_weights(pred, k)
LS = top_k_weights(pred, k, long_short=True)
rec("long top10 (baseline)", L)
rec("long top10, executed 1 day LATE", L.shift(1))
rec("long/short (baseline)", LS)
rec("long/short, executed 1 day LATE", LS.shift(1))
rec("short leg only", LS.clip(upper=0) * 2)
rec("long leg only", LS.clip(lower=0) * 2)
# restrict to the 30 most liquid names at each date
liq = X["log_qv_30"].unstack("symbol").reindex(dates)
top30 = liq.rank(axis=1, ascending=False) <= 30
p30 = pred.unstack("symbol").where(top30).stack()
rec("top30-liquid only: long top5", top_k_weights(p30, 5))
rec("top30-liquid only: long/short 5x5", top_k_weights(p30, 5, long_short=True))
rec("top30-liquid only: long/short 5x5, 1 day LATE", top_k_weights(p30, 5, long_short=True).shift(1))
pd.set_option("display.width", 200)
print(pd.DataFrame(rows).T.round(3).to_string())

ic = pd.concat([pred, X["fwd"]], axis=1).dropna().groupby(level=0).apply(lambda g: g.iloc[:, 0].corr(g.iloc[:, 1], method="spearman"))
print("IC by year:\n", ic.groupby(ic.index.year).mean().round(3).to_string())
