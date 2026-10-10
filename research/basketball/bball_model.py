"""Basketball win-probability model from results only (no odds in the features), walk-forward by season.

Features, all computed from games strictly before the one predicted: margin-of-victory Elo (season carry-over 75%),
rolling form over 3/5/10/20 games (win share, margin, points for/against, opponent-adjusted margin), season-to-date
record, home/away-specific form, rest, back-to-back, games in the last 4/7 days, road-trip / home-stand length,
head-to-head, and home-minus-away differences of everything.
Test season S: train on seasons before the latest earlier season V, early stopping + Platt calibration on V. XGB + LGBM + logistic(Elo).

usage: bball_model.py LEAGUE_DIR FIRST_TEST LAST_TEST     (LEAGUE_DIR holds games.csv from the scraper)
Writes LEAGUE_DIR/preds.parquet (mid, season, date, home, away, p_home, y).
"""
import sys
from collections import defaultdict, deque

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression

DIR, FIRST, LAST = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
RENAME = {"New Jersey Nets": "Brooklyn Nets", "Charlotte Bobcats": "Charlotte Hornets", "New Orleans Hornets": "New Orleans Pelicans",
          "Seattle SuperSonics": "Oklahoma City Thunder", "LA Clippers": "Los Angeles Clippers"}
G = pd.read_csv(f"{DIR}/games.csv", dtype={"mid": str})
G["home"], G["away"] = G.home.replace(RENAME), G.away.replace(RENAME)
G["date"] = pd.to_datetime(G.date)
G = G.drop_duplicates("mid").sort_values(["date", "mid"]).reset_index(drop=True)
G["margin"] = G.home_pts - G.away_pts
G = G[G.margin != 0].reset_index(drop=True)
WIN = (3, 5, 10, 20)
elo = defaultdict(lambda: 1500.0)
hist = defaultdict(lambda: deque(maxlen=20))       # (date, is_home, won, margin, pf, pa, adj_margin)
side_hist = defaultdict(lambda: deque(maxlen=10))  # per (team, is_home): margins
h2h = defaultdict(lambda: deque(maxlen=3))
season_rec = defaultdict(lambda: [0, 0, 0.0])      # (team, season): games, wins, margin sum
streak = defaultdict(lambda: (None, 0))            # team: (last venue is_home, run length)
last_season = {}
HOME_ADV = 70.0


def team_feats(t, season, d, is_home):
    f = {"elo": elo[t]}
    h = list(hist[t])
    for n in WIN:
        x = h[-n:]
        k = len(x)
        f[f"n{n}"] = k
        f[f"win{n}"] = np.mean([g[2] for g in x]) if k else np.nan
        f[f"mar{n}"] = np.mean([g[3] for g in x]) if k else np.nan
        f[f"pf{n}"] = np.mean([g[4] for g in x]) if k else np.nan
        f[f"pa{n}"] = np.mean([g[5] for g in x]) if k else np.nan
        f[f"adj{n}"] = np.mean([g[6] for g in x]) if k else np.nan
    s = season_rec[(t, season)]
    f["s_games"], f["s_win"], f["s_mar"] = s[0], s[1] / s[0] if s[0] else np.nan, s[2] / s[0] if s[0] else np.nan
    sh = list(side_hist[(t, is_home)])
    f["venue_mar10"] = np.mean(sh) if sh else np.nan
    dates = [g[0] for g in h]
    f["rest"] = min((d - dates[-1]).days, 10) if dates else 10
    f["b2b"] = float(bool(dates) and (d - dates[-1]).days == 1)
    f["g4"] = sum((d - x).days <= 4 for x in dates)
    f["g7"] = sum((d - x).days <= 7 for x in dates)
    v, run = streak[t]
    f["same_venue_run"] = run if v == is_home else 0
    f["away_run"] = run if v is False else 0
    return f


rows = []
for g in G.itertuples():
    for t in (g.home, g.away):
        if last_season.get(t) is not None and last_season[t] != g.season:
            elo[t] = 1500 + 0.75 * (elo[t] - 1500)
        last_season[t] = g.season
    fh, fa = team_feats(g.home, g.season, g.date, True), team_feats(g.away, g.season, g.date, False)
    r = {"mid": g.mid, "season": g.season, "date": g.date, "home": g.home, "away": g.away, "y": int(g.margin > 0),
         "playoff": float("play" in str(g.stage).lower() or "final" in str(g.stage).lower()), "month": g.date.month}
    for k in fh:
        r[f"h_{k}"], r[f"a_{k}"] = fh[k], fa[k]
        r[f"d_{k}"] = fh[k] - fa[k] if not (pd.isna(fh[k]) or pd.isna(fa[k])) else np.nan
    hh = list(h2h[frozenset((g.home, g.away))])
    r["h2h_mar"] = np.mean([m if w == g.home else -m for w, m in hh]) if hh else np.nan
    r["h2h_n"] = len(hh)
    pe = 1 / (1 + 10 ** (-(elo[g.home] + HOME_ADV - elo[g.away]) / 400))
    r["p_elo"] = pe
    rows.append(r)
    # updates after the game
    m = g.margin
    mult = np.log(abs(m) + 1) * 2.2 / ((elo[g.home] + HOME_ADV - elo[g.away]) * (1 if m > 0 else -1) * 0.001 + 2.2)
    delta = 20 * mult * ((m > 0) - pe)
    exp_m = (elo[g.home] + HOME_ADV - elo[g.away]) / 28.0
    elo[g.home] += delta
    elo[g.away] -= delta
    for t, is_home, sm in ((g.home, True, m), (g.away, False, -m)):
        pf, pa = (g.home_pts, g.away_pts) if is_home else (g.away_pts, g.home_pts)
        adj = sm - (exp_m if is_home else -exp_m)
        hist[t].append((g.date, is_home, float(sm > 0), sm, pf, pa, adj))
        side_hist[(t, is_home)].append(sm)
        s = season_rec[(t, g.season)]
        s[0], s[1], s[2] = s[0] + 1, s[1] + (sm > 0), s[2] + sm
        v, run = streak[t]
        streak[t] = (is_home, run + 1 if v == is_home else 1)
    h2h[frozenset((g.home, g.away))].append((g.home, m))
F = pd.DataFrame(rows)
X = F.drop(columns=["mid", "season", "date", "home", "away", "y"])
print(f"{DIR}: {len(F)} games, {X.shape[1]} features, seasons {F.season.min()}-{F.season.max()}", flush=True)
lgt = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
out = []
for S in range(FIRST, LAST + 1):
    prev = F.season[F.season < S].max()  # validation = latest earlier season (EuroLeague has no 2019-20)
    tr, va, te = (F.season < prev).values, (F.season == prev).values, (F.season == S).values
    if te.sum() == 0:
        continue
    pv, pt = [], []
    m = xgb.XGBClassifier(n_estimators=3000, learning_rate=0.03, max_depth=4, min_child_weight=20, subsample=0.8,
                          colsample_bytree=0.5, reg_lambda=5.0, tree_method="hist", n_jobs=2, early_stopping_rounds=200,
                          eval_metric="logloss")
    m.fit(X[tr], F.y[tr], eval_set=[(X[va], F.y[va])], verbose=False)
    pv.append(m.predict_proba(X[va])[:, 1]); pt.append(m.predict_proba(X[te])[:, 1])
    m = lgb.LGBMClassifier(n_estimators=3000, learning_rate=0.03, num_leaves=15, min_child_samples=40, subsample=0.8,
                           subsample_freq=1, colsample_bytree=0.5, reg_lambda=5.0, n_jobs=2, verbose=-1)
    m.fit(X[tr], F.y[tr], eval_set=[(X[va], F.y[va])], callbacks=[lgb.early_stopping(200, verbose=False)])
    pv.append(m.predict_proba(X[va])[:, 1]); pt.append(m.predict_proba(X[te])[:, 1])
    Z = lambda idx: lgt(F.p_elo[idx].values)[:, None]
    m = LogisticRegression(C=1.0).fit(Z(tr), F.y[tr])
    pv.append(m.predict_proba(Z(va))[:, 1]); pt.append(m.predict_proba(Z(te))[:, 1])
    v, t = np.mean(pv, axis=0), np.mean(pt, axis=0)
    cal = LogisticRegression(C=1e4).fit(lgt(v)[:, None], F.y[va])
    p = cal.predict_proba(lgt(t)[:, None])[:, 1]
    y = F.y[te].values
    ll = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    print(f"  season {S}: {te.sum()} games, log-loss {ll:.4f}, Elo-only {-np.mean(y * np.log(F.p_elo[te]) + (1 - y) * np.log(1 - F.p_elo[te])):.4f}", flush=True)
    out.append(pd.DataFrame({"mid": F.mid[te], "season": S, "date": F.date[te], "home": F.home[te], "away": F.away[te],
                             "p_home": p, "y": y}))
pd.concat(out).to_parquet(f"{DIR}/preds.parquet")
