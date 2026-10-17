"""Losing weeks of the plan backtest (2024-2026, Serbian books) and a search for anything that should not be there.

Per bet: closing price at the same book (CLV), Shin fair probability at the open and at the close, other books' opening
price, the match from the Flashscore results, and the live data checks of paper_trade.py (real ids, 10+ matches, a
model-vs-book gap over 0.30 needs 30+ matches and a match in the last 180 days). Reports the edge split into line
movement and beating the close, losing weeks against a simulation, ROI per bet type and year, bet lines posted with the
players swapped, outlier lines, and each month from 300,000 (quarter Kelly, at most 30,000) without the suspect bets.

usage: loss_weeks.py DATA_DIR > paper/loss_weeks.txt
"""
import sys

import numpy as np
import pandas as pd

B = sys.argv[1]
sys.argv = sys.argv[:1]
ns = {}
exec(open("kelly_cap.py").read().split("rows = []")[0], ns)  # plan(): the plan rule and the quarter-Kelly fraction
SRC = {2024: (f"{B}/fs_hist/y2024.parquet", f"{B}/fs_hist/2024"), 2025: (f"{B}/fs_hist/y2025.parquet", f"{B}/fs_hist/2025"),
       2026: (f"{B}/calib_exp/six_open_dated.parquet", f"{B}/fs_geo")}
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 200)


def shin(a, b):
    s, z = a + b, np.zeros_like(a)
    for _ in range(30):
        pa = (np.sqrt(z ** 2 + 4 * (1 - z) * a ** 2 / s) - z) / (2 * (1 - z))
        pb = (np.sqrt(z ** 2 + 4 * (1 - z) * b ** 2 / s) - z) / (2 * (1 - z))
        z = z + (pa + pb - 1) * 0.5
    return pa / (pa + pb)


hist = {}
for t in ("atp", "wta"):
    M = pd.read_parquet(f"{B}/out_v3/{t}_matches.parquet", columns=["winner_id", "loser_id", "tourney_date"])
    M["md"] = pd.to_datetime(M.tourney_date.astype("Int64").astype(str), format="%Y%m%d", errors="coerce")
    L = pd.concat([M[["winner_id", "md"]].set_axis(["id", "d"], axis=1), M[["loser_id", "md"]].set_axis(["id", "d"], axis=1)])
    hist[t] = (M, L.sort_values("d").groupby("id").d.apply(np.array).to_dict())

rows = []
for y, (bp, od) in SRC.items():
    b = ns["plan"](bp).assign(year=y)
    O = pd.concat([pd.read_parquet(f"{od}/{t}_fs_odds_RS.parquet") for t in ("atp", "wta")])
    b = b.merge(O[["fs_mid", "book", "home_won", "cur_h", "cur_a"]], on=["fs_mid", "book"], how="left")
    home = b.won == b.home_won  # the side we bet is the home side
    b["close"], b["close_o"] = np.where(home, b.cur_h, b.cur_a), np.where(home, b.cur_a, b.cur_h)
    Ow = O.merge(b[["fs_mid", "won", "home_won"]].drop_duplicates("fs_mid"), on="fs_mid", suffixes=("", "_b"))
    Ow["ours"] = np.where(Ow.won == Ow.home_won, Ow.open_h, Ow.open_a)
    b = b.merge(Ow.groupby("fs_mid").ours.agg(open_med="median", n_books="count"), on="fs_mid", how="left")
    F = pd.concat([pd.read_csv(f"{od}/{t}__{t}_matches_fs_{y}.csv", usecols=["fs_mid", "surface", "round", "winner_name", "loser_name"])
                   for t in ("atp", "wta")]).drop_duplicates("fs_mid")
    b = b.merge(F, on="fs_mid", how="left")
    for t, g in b.groupby("tour"):
        M, H = hist[t]
        J = M.loc[g.mi]
        for side, col in (("me", np.where(g.won, J.winner_id, J.loser_id)), ("op", np.where(g.won, J.loser_id, J.winner_id))):
            n, last = [], []
            for pid, d in zip(col, J.md.values):
                arr = H.get(pid, np.array([], dtype="datetime64[ns]"))
                prev = arr[arr < d]
                n.append(len(prev))
                last.append((d - prev.max()) / np.timedelta64(1, "D") if len(prev) else np.nan)
            b.loc[g.index, f"id_{side}"], b.loc[g.index, f"n_{side}"], b.loc[g.index, f"last_{side}"] = col, n, last
    rows.append(b)
A = pd.concat(rows, ignore_index=True)
A["player"] = np.where(A.won, A.winner_name, A.loser_name)
A["opp"] = np.where(A.won, A.loser_name, A.winner_name)
A["p_open"], A["p_close"] = shin(1 / A.o.values, 1 / A.oo.values), shin(1 / A.close.values, 1 / A.close_o.values)
A["clv"] = A.o / A.close - 1
A["beat"] = A.won.astype(float) - A.p_close
A["gap"] = A.q - A.p_open
A["day"] = pd.to_datetime(A.day)
A["wk"] = A.day.dt.to_period("W-SUN").astype(str)
# what the live checks would refuse
A["blocked"] = ((A.id_me >= 9e8) | (A.id_op >= 9e8) | (A.n_me < 10) | (A.n_op < 10)
                | ((A.gap > 0.30) & ((A.n_me < 30) | (A.n_op < 30) | (A.last_me > 180) | (A.last_op > 180))))
# a line posted with the players swapped: the opening favourite is the closing underdog at near-mirror prices
A["swap"] = (A.o > A.oo) & (A.close < A.close_o) & (np.abs(np.log(A.o / A.close_o)) < 0.08) & (np.abs(np.log(A.oo / A.close)) < 0.08)
A["outlier"] = (A.n_books >= 3) & (A.o > 1.3 * A.open_med)  # 30% above the median opening of the books listing it
A["keep"] = ~A.blocked & ~A.swap & ~A.outlier

print("== edge: line movement vs beating the close (exp = ROI if the closing fair price were the truth)")
print(A.assign(exp=A.p_close * A.o - 1).groupby("year").agg(n=("pnl", "size"), roi=("pnl", "mean"), exp=("exp", "mean"),
      beat_pp=("beat", "mean"), clv_median=("clv", "median"), clv_pos=("clv", lambda s: (s > 0).mean())).round(3).to_string())

print("\n== losing weeks against a simulation (outcomes drawn from the closing fair price + the year's beat)")
rng = np.random.default_rng(1)
for y, g in A.groupby("year"):
    W = g.groupby("wk").pnl.sum()
    p = np.clip(g.p_close.values + g.beat.mean(), 0, 1)
    sims = [(pd.Series(np.where(rng.random(len(g)) < p, g.o.values - 1, -1), index=g.wk.values).groupby(level=0).sum() < 0).sum()
            for _ in range(2000)]
    print(f"{y}: {len(W)} weeks, {(W < 0).sum()} negative; simulated {np.mean(sims):.1f} (90% {np.percentile(sims, 5):.0f}-"
          f"{np.percentile(sims, 95):.0f}); lag-1 autocorrelation {W.autocorr(1):.2f}")
W = A.groupby(["year", "wk"]).agg(units=("pnl", "sum"), n=("pnl", "size"), clv=("clv", "mean"), beat=("beat", "mean"),
                                  odds=("o", "mean"), ev=("evb", "mean"), win=("won", "mean"))
print(W.groupby(W.units < 0).mean().round(3).rename(index={False: "positive weeks", True: "negative weeks"}).to_string())

print("\n== ROI per bet type and year")
A["odds"] = pd.cut(A.o, [1, 1.5, 2, 2.5, 3, 4, 5])
A["ev_band"] = pd.cut(A.evb, [0.05, 0.08, 0.12, 0.2, 0.3, 5])
A["gap_band"] = pd.cut(A.gap, [-1, 0.1, 0.2, 0.3, 1])
A["month"] = A.day.dt.month
A["weekday"] = A.day.dt.dayofweek
for col in ("month", "surface", "odds", "ev_band", "gap_band", "weekday", "book"):
    T = A.groupby([col, "year"], observed=True).pnl.mean().unstack("year").round(3)
    print(f"\n{col}\n" + A.groupby(col, observed=True).agg(n=("pnl", "size"), roi=("pnl", "mean")).round(3).join(T).to_string())

print("\n== suspect bets: refused by the live checks, swapped lines, outlier lines")
print(A.groupby("year").agg(n=("pnl", "size"), blocked=("blocked", "sum"), swap=("swap", "sum"), outlier=("outlier", "sum"),
                            kept=("keep", "sum")).to_string())
print(A.groupby(["year", "keep"]).agg(n=("pnl", "size"), roi=("pnl", "mean"), clv=("clv", "mean")).round(3).to_string())
print("\nopponents behind the most refused big-gap bets (a split identity: the model sees a newcomer)")
print(A[A.blocked & (A.gap > 0.3)].opp.value_counts().head(10).to_string())
for y, g in A[A.keep].groupby("year"):
    Wk = g.groupby("wk").pnl.sum()
    print(f"kept {y}: ROI {g.pnl.mean():+.1%}, {len(Wk)} weeks, {(Wk < 0).sum()} negative")


def months(b):
    out = {}
    for m, gm in b.sort_values("start").groupby(b.day.dt.to_period("M")):
        bank = 300000.0
        for _, g in gm.groupby("day"):
            st = np.minimum(g.f.values * bank, 30000)
            bank += float((st * g.pnl.values).sum())
        out[str(m)] = round(bank - 300000)
    return pd.Series(out)


R = pd.DataFrame({"test": pd.concat([months(g) for _, g in A.groupby("year")]),
                  "live checks": pd.concat([months(g[~g.blocked]) for _, g in A.groupby("year")]),
                  "live checks + no suspect lines": pd.concat([months(g[g.keep]) for _, g in A.groupby("year")])})
print("\n== each month from 300,000 (quarter Kelly of the current capital, at most 30,000 a bet)")
print(R.to_string())
full = R.drop(index="2026-10")
print("\nfull months: mean", full.mean().round(0).to_dict(), "| negative", (full < 0).sum().to_dict())
print(full.groupby(full.index.str[:4]).sum().to_string())
