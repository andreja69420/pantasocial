import glob
import os

import numpy as np
import pandas as pd


def load_panel(data_dir, interval="1d"):
    frames = {}
    now = pd.Timestamp.utcnow().tz_localize(None)
    for p in glob.glob(os.path.join(data_dir, interval, "*.parquet")):
        df = pd.read_parquet(p)
        df = df[df["close_time"] < now]  # drop the still-forming bar
        if len(df) < 30:
            continue
        frames[os.path.basename(p)[:-8]] = df.set_index("open_time")
    fields = ["open", "high", "low", "close", "volume", "quote_volume", "taker_quote", "trades"]
    return {f: pd.DataFrame({s: d[f] for s, d in frames.items()}).sort_index() for f in fields}


def _xs_rank(df):
    return df.rank(axis=1, pct=True) - 0.5


def build(panel, horizon, universe_size=100, min_age=90, bar_per_day=1):
    """Return long-format feature table. Features at row t only use bars <= t."""
    c, h, l = panel["close"], panel["high"], panel["low"]
    qv, tq = panel["quote_volume"], panel["taker_quote"]
    d = bar_per_day
    lr = np.log(c).diff()
    alive = c.notna()
    age = alive.cumsum().where(alive)

    feats = {}
    for n in [1, 3, 7, 14, 30, 60, 90]:
        feats[f"ret_{n}"] = np.log(c / c.shift(n * d))
    feats["vol_7"] = lr.rolling(7 * d).std()
    feats["vol_30"] = lr.rolling(30 * d).std()
    feats["vol_ratio"] = feats["vol_7"] / feats["vol_30"]
    feats["range_7"] = np.log(h / l).rolling(7 * d).mean()
    feats["skew_30"] = lr.rolling(30 * d).skew()
    feats["max_ret_30"] = lr.rolling(30 * d).max()
    feats["min_ret_30"] = lr.rolling(30 * d).min()
    qv30 = qv.rolling(30 * d, min_periods=10 * d).mean()
    feats["log_qv_30"] = np.log1p(qv30)
    feats["qv_trend"] = np.log1p(qv.rolling(3 * d).mean()) - np.log1p(qv30)
    feats["taker_7"] = tq.rolling(7 * d).sum() / qv.rolling(7 * d).sum()
    feats["taker_1"] = tq / qv
    feats["dist_high_30"] = c / h.rolling(30 * d).max() - 1
    feats["dist_low_30"] = c / l.rolling(30 * d).min() - 1
    feats["dist_high_90"] = c / h.rolling(90 * d).max() - 1
    feats["ma_gap_20"] = c / c.rolling(20 * d).mean() - 1
    feats["ma_gap_50"] = c / c.rolling(50 * d).mean() - 1
    feats["log_age"] = np.log1p(age)
    btc = lr["BTCUSDT"]
    cov = lr.rolling(30 * d).cov(btc)
    feats["beta_30"] = cov.div(btc.rolling(30 * d).var(), axis=0)
    feats["corr_30"] = lr.rolling(30 * d).corr(btc)
    feats["resid_7"] = feats["ret_7"] - feats["beta_30"] * np.log(c["BTCUSDT"] / c["BTCUSDT"].shift(7 * d)).values[:, None]

    # Eligible universe: enough history and among the most liquid names at time t.
    liq_rank = qv30.where(age >= min_age * d).rank(axis=1, ascending=False)
    elig = (liq_rank <= universe_size) & alive

    # Forward return. Prices are forward-filled so a coin delisted mid-holding exits at its last
    # traded price instead of silently vanishing from the PnL.
    cf = c.ffill(limit=None)
    last_valid = c.apply(lambda s: s.last_valid_index())
    cf = cf.where(cf.index.values[:, None] <= last_valid.values[None, :] + pd.Timedelta(days=horizon + 1))
    fwd = cf.shift(-horizon) / c - 1

    mkt = {}
    mkt["mkt_btc_ret_7"] = np.log(c["BTCUSDT"] / c["BTCUSDT"].shift(7 * d))
    mkt["mkt_btc_ret_30"] = np.log(c["BTCUSDT"] / c["BTCUSDT"].shift(30 * d))
    mkt["mkt_btc_vol_30"] = btc.rolling(30 * d).std()
    mkt["mkt_breadth_50"] = (feats["ma_gap_50"].where(elig) > 0).sum(axis=1) / elig.sum(axis=1)
    mkt["mkt_xs_disp_7"] = feats["ret_7"].where(elig).std(axis=1)
    mkt = pd.DataFrame(mkt)

    rows = []
    for k, v in feats.items():
        v = v.where(elig)
        rows.append(v.stack(future_stack=True).rename(k))
        rows.append(_xs_rank(v).stack(future_stack=True).rename(k + "_xr"))
    X = pd.concat(rows, axis=1)
    X = X[elig.stack(future_stack=True).reindex(X.index).fillna(False).astype(bool)]
    X.index.names = ["date", "symbol"]
    X = X.join(mkt, on="date")
    y = fwd.stack(future_stack=True).reindex(X.index)
    X["fwd"] = y
    X["fwd_xr"] = X.groupby(level="date")["fwd"].rank(pct=True) - 0.5
    return X.dropna(subset=["ret_30", "vol_30"]), fwd, elig
