"""Intraday experiments on 1h bars: (a) cross-sectional XGBoost over the 30 most liquid coins,
(b) a long/flat single-asset XGBoost bot on BTC and ETH. All walk-forward, net of fees."""
import sys

import numpy as np
import pandas as pd

from backtest import daily_ic, metrics, simulate, top_k_weights, walk_forward
from features import load_panel

D = sys.argv[1]
TEST_START = "2023-01-01"
COST = 0.0015
H = int(sys.argv[2]) if len(sys.argv) > 2 else 4
UNIV = 30
PER_YEAR = 365 * 24

P = load_panel(D, "1h")
qv = P["quote_volume"]
qv30 = qv.rolling(24 * 30, min_periods=24 * 10).mean()
alive = P["close"].notna()
age = alive.cumsum().where(alive)
liq_rank = qv30.where(age >= 24 * 60).rank(axis=1, ascending=False)
keep = liq_rank.le(UNIV * 2).any()
keep["BTCUSDT"] = True
P = {k: v.loc[:, keep] for k, v in P.items()}
liq_rank = liq_rank.loc[:, keep]
c, h, l, qv, tq = P["close"], P["high"], P["low"], P["quote_volume"], P["taker_quote"]
print("symbols kept", c.shape, flush=True)
lr = np.log(c).diff()

f = {}
for n in [1, 2, 4, 8, 12, 24, 48, 72, 168, 336, 720]:
    f[f"ret_{n}h"] = np.log(c / c.shift(n))
f["vol_24"] = lr.rolling(24).std()
f["vol_168"] = lr.rolling(168).std()
f["vol_ratio"] = f["vol_24"] / f["vol_168"]
f["range_24"] = np.log(h / l).rolling(24).mean()
f["qv_trend"] = np.log1p(qv.rolling(4).mean()) - np.log1p(qv.rolling(168).mean())
f["log_qv_30d"] = np.log1p(qv.rolling(720, min_periods=240).mean())
f["taker_4"] = tq.rolling(4).sum() / qv.rolling(4).sum()
f["taker_24"] = tq.rolling(24).sum() / qv.rolling(24).sum()
f["dist_high_168"] = c / h.rolling(168).max() - 1
f["dist_low_168"] = c / l.rolling(168).min() - 1
f["ma_gap_24"] = c / c.rolling(24).mean() - 1
f["ma_gap_168"] = c / c.rolling(168).mean() - 1
f["corr_btc_168"] = lr.rolling(168).corr(lr["BTCUSDT"])

elig = (liq_rank <= UNIV) & alive
cf = c.ffill()
fwd = cf.shift(-H) / c - 1
rets = (cf.shift(-1) / cf - 1).where(c.notna())

mkt = pd.DataFrame({
    "btc_ret_4h": f["ret_4h"]["BTCUSDT"], "btc_ret_24h": f["ret_24h"]["BTCUSDT"],
    "btc_ret_168h": f["ret_168h"]["BTCUSDT"], "btc_vol_24": f["vol_24"]["BTCUSDT"],
    "hour": c.index.hour, "dow": c.index.dayofweek,
})

rows = []
for k, v in f.items():
    v = v.where(elig)
    rows.append(v.stack(future_stack=True).rename(k))
    rows.append((v.rank(axis=1, pct=True) - 0.5).stack(future_stack=True).rename(k + "_xr"))
X = pd.concat(rows, axis=1)
X.index.names = ["date", "symbol"]
X = X[X["ret_1h"].notna()].join(mkt, on="date")
X["fwd"] = fwd.stack(future_stack=True).reindex(X.index)
X["fwd_xr"] = X.groupby(level="date")["fwd"].rank(pct=True) - 0.5
X = X.dropna(subset=["ret_720h"])
print("X", X.shape, flush=True)
feats = [col for col in X.columns if col not in ("fwd", "fwd_xr")]

results, curves = {}, {}


def record(name, sim):
    sim = sim[sim.index >= TEST_START].iloc[:-H]
    m = metrics(sim["ret"], PER_YEAR)
    m["trades_per_day(turnover)"] = sim["turnover"].mean() * 24
    results[name] = m
    curves[name] = (1 + sim["ret"]).cumprod().resample("1D").last()
    print(name, {k: round(v, 3) for k, v in m.items()}, flush=True)


record("BTC buy&hold", pd.DataFrame({"ret": rets["BTCUSDT"], "turnover": 0.0}))

# (a) cross-sectional, retrain every 30 days; the training set is subsampled every 4th hour to keep it tractable.
Xs = X[X.index.get_level_values("date").hour % 4 == 0]
pred = walk_forward(Xs, feats, "fwd_xr", TEST_START, retrain_every=30 * 6, purge=np.timedelta64(H + 1, "h"),
                    model="xgb")
ic = daily_ic(pred, Xs["fwd"])
print("XS IC", ic.mean(), ic.mean() / ic.std() * np.sqrt(len(ic)), flush=True)
pu = pred.unstack("symbol")
R = rets.reindex(columns=pu.columns)
# Predictions exist every 4h; hold each basket for the 4 bars until the next one.
for k in [5]:
    w = top_k_weights(pred, k).reindex(rets.index).ffill(limit=H - 1).reindex(columns=R.columns, fill_value=0)
    w = w[w.index >= TEST_START]
    record(f"XGB XS {H}h long top{k}", simulate(w, R.reindex(w.index), H, COST))
    record(f"XGB XS {H}h long top{k} ZERO COST", simulate(w, R.reindex(w.index), H, 0.0))

# (b) single-asset long/flat bots
for sym in ["BTCUSDT", "ETHUSDT"]:
    Xa = X.xs(sym, level="symbol", drop_level=False)
    Xa = Xa[Xa.index.get_level_values("date").hour % H == 0]
    p = walk_forward(Xa, feats, "fwd", TEST_START, retrain_every=30 * (24 // H), purge=np.timedelta64(H + 1, "h"),
                     model="xgb").droplevel("symbol")
    for thr_name, thr in [("pred>0", 0.0), ("pred>fees", 2 * COST)]:
        pos = (p > thr).astype(float).reindex(rets.index).ffill(limit=H - 1).fillna(0)
        pos = pos[pos.index >= TEST_START]
        w = pd.DataFrame({sym: pos})
        record(f"XGB {sym} long/flat {H}h {thr_name}", simulate(w, rets[[sym]].reindex(w.index), 1, COST))
    hit = np.sign(p) == np.sign(Xa["fwd"].droplevel("symbol").reindex(p.index))
    print(sym, "direction hit-rate", hit.mean(), flush=True)

pd.DataFrame(results).T.to_csv(f"{D}/results_hourly_h{H}.csv")
pd.DataFrame(curves).to_parquet(f"{D}/curves_hourly_h{H}.parquet")
print(pd.DataFrame(results).T.round(3).to_string())
