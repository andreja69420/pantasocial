"""Screen ~50 betting tactics per segment (ATP Tour, Challenger, men's ITF, women's ITF) on walk-forward predictions.

Every tactic is scored on an explore half and a confirm half of the years. A tactic "survives" when it has >=300 bets,
is positive in both halves and its 90% lower bound is above zero. To size the false-positive rate, the same screen is
re-run with the model probabilities shuffled within year (a no-skill model) and the survivors are counted.

usage: tactics.py DATA_DIR
"""
import sys

import numpy as np
import pandas as pd

D = sys.argv[1]
TE = "2014-2015-2016-2017-2018-2019-2020-2021-2022-2023-2024-2025"
COLS = ["rest_days", "n_7d", "prev_minutes", "home", "entry_WC", "entry_Q", "entry_LL", "age", "rank", "n_career",
        "h2h_n", "h2h_wins", "welo_mom10", "left", "ret_365d", "seeded", "win_10"]
CTX = ["surface", "tourney_level", "rord", "best_of", "main", "date"]


def load(tour, preds, level_re, odds=None):
    P = pd.concat([pd.read_parquet(f"{D}/out/{f}") for f in preds])
    P = P[~P.index.duplicated()]
    F = pd.read_parquet(f"{D}/out/{tour}_f2.parquet", columns=CTX + [f"{s}_{c}" for s in "wl" for c in COLS]).reindex(P.index)
    keep = F.tourney_level.astype(str).str.match(level_re).fillna(False).values
    P, F = P[keep], F[keep]
    if odds:
        O = pd.read_parquet(f"{D}/out/{tour}_te_odds.parquet").reindex(P.index)
        w, l = O[f"{odds}_w"].astype(float).values, O[f"{odds}_l"].astype(float).values
    else:
        w = np.where(P.y == 1, P.o1, P.o2).astype(float)
        l = np.where(P.y == 1, P.o2, P.o1).astype(float)
    ok = (w > 1) & (l > 1)
    P, F, w, l = P[ok], F[ok], w[ok], l[ok]
    rows = []
    for won in (True, False):  # one row per side: the bet on the winner and the bet on the loser
        me, op = ("w", "l") if won else ("l", "w")
        q = np.where((P.y == 1) == won, P.p, 1 - P.p).astype(float)
        o, oo = (w, l) if won else (l, w)
        r = pd.DataFrame({"year": P.year.values, "q": q, "o": o, "oo": oo,
                          "pnl": np.where(P.stands, (o - 1) if won else -1.0, 0.0), "mi": P.index,
                          "p1": P.p.astype(float).values, "y1": (P.y == 1).values, "won": won})
        for c in COLS:
            r[f"me_{c}"], r[f"op_{c}"] = F[f"{me}_{c}"].astype(float).values, F[f"{op}_{c}"].astype(float).values
        for c in CTX:
            r[c] = F[c].values
        rows.append(r)
    S = pd.concat(rows, ignore_index=True)
    S["ev"] = S.q * S.o - 1
    S["month"] = pd.to_datetime(S.date).dt.month
    return S


def tactics(S):
    core = (S.ev > 0.05) & (S.o <= 2.0)
    T = {}
    for e in (0.02, 0.05, 0.10, 0.20):
        for c in (1.6, 2.0, 2.5):
            T[f"EV>{e:.0%} odds<={c}"] = (S.ev > e) & (S.o <= c)
    for x in (0.6, 0.7, 0.8, 0.9):
        T[f"model p>{x:.0%} & EV>0"] = (S.q > x) & (S.ev > 0)
    for a, b in ((1.0, 1.3), (1.3, 1.6), (1.6, 2.0), (2.0, 2.5), (2.5, 3.5)):
        T[f"EV>5% odds {a}-{b}"] = (S.ev > 0.05) & (S.o >= a) & (S.o < b)
    ctx = {
        "clay": S.surface == "Clay", "hard": S.surface == "Hard", "grass": S.surface == "Grass",
        "best of 5": S.best_of == 5, "qualifying": S.main == 0, "main draw early rounds": (S.main == 1) & (S.rord <= 7),
        "main draw later rounds": (S.main == 1) & (S.rord > 7),
        "opp played yesterday": S.op_rest_days <= 1, "me rested 3+ days": S.me_rest_days >= 3,
        "opp 3+ matches in 7d": S.op_n_7d >= 3, "opp long previous match 120+min": S.op_prev_minutes >= 120,
        "me home": S.me_home == 1, "opp home": S.op_home == 1,
        "opp wildcard": S.op_entry_WC == 1, "opp qualifier/LL": (S.op_entry_Q == 1) | (S.op_entry_LL == 1),
        "me under 23": S.me_age < 23, "me over 30": S.me_age > 30,
        "opp unranked": S.op_rank.isna(), "both top-300": (S.me_rank <= 300) & (S.op_rank <= 300),
        "me better ranked": S.me_rank.fillna(5000) < S.op_rank.fillna(5000),
        "me <30 career matches": S.me_n_career < 30, "both 100+ matches": (S.me_n_career >= 100) & (S.op_n_career >= 100),
        "me leads h2h": (S.me_h2h_n > 0) & (S.me_h2h_wins * 2 > S.me_h2h_n),
        "me rising elo": S.me_welo_mom10 > 0, "me lefty": S.me_left == 1,
        "opp retires often": S.op_ret_365d > 0.05, "me seeded": S.me_seeded == 1,
        "me hot form win10>=70%": S.me_win_10 >= 0.7,
        "Jan-Apr": S.month <= 4, "May-Aug": (S.month >= 5) & (S.month <= 8), "Sep-Dec": S.month >= 9,
        "market favourite": S.o < S.oo, "market underdog": S.o > S.oo,
    }
    for k, m in ctx.items():
        T[f"core & {k}"] = core & m.fillna(False).values
    return T


def score(S, T, cut):
    out = []
    for name, m in T.items():
        b = S.loc[m.values if hasattr(m, "values") else m]
        if len(b) < 50:
            continue
        roi, se = b.pnl.mean(), b.pnl.std() / np.sqrt(len(b))
        by = b.groupby("year").pnl.mean()
        e, c = b[b.year < cut].pnl, b[b.year >= cut].pnl
        out.append({"tactic": name, "bets": len(b), "roi": roi, "lo90": roi - 1.645 * se, "yrs_pos": f"{(by > 0).sum()}/{len(by)}",
                    "explore": e.mean() if len(e) else np.nan, "confirm": c.mean() if len(c) else np.nan,
                    "max_odds": b.o.max()})
    R = pd.DataFrame(out)
    R["survives"] = (R.bets >= 300) & (R.explore > 0) & (R.confirm > 0) & (R.lo90 > 0)
    return R


SEGS = [
    ("ATP Tour @Bet365", "atp", [f"preds3_atp_ens_r6_hist1991_ens_te_{TE}.parquet"], r"^[GMAF]", "b365"),
    ("ATP Tour @best", "atp", [f"preds3_atp_ens_r6_hist1991_ens_te_{TE}.parquet"], r"^[GMAF]", "best"),
    ("Challenger @Bet365", "atp", [f"preds3_atp_ens_r6_hist1991_ens_te_{TE}.parquet"], r"^C$", "b365"),
    ("Challenger @best", "atp", [f"preds3_atp_ens_r6_hist1991_ens_te_{TE}.parquet"], r"^C$", "best"),
    ("Men ITF 2012-18 @OddsPortal", "atp", ["preds3_atp_ens_r6_hist1991_ens_lower.parquet",
                                            "preds3_atp_ens_r6_hist1991_ens_lower_2012-2013.parquet"], r"^(\d|S)", None),
    ("Women ITF @Bet365", "wta", [f"preds3_wta_ens_r6_hist1991_ens_te_{TE}.parquet"], r"^\d", "b365"),
    ("Women ITF @best", "wta", [f"preds3_wta_ens_r6_hist1991_ens_te_{TE}.parquet"], r"^\d", "best"),
]
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 200)
rng = np.random.default_rng(0)
summary = []
for name, tour, preds, lvl, odds in SEGS:
    S = load(tour, preds, lvl, odds)
    years = sorted(S.year.unique())
    cut = years[len(years) // 2]
    R = score(S, tactics(S), cut)
    null = []
    for _ in range(5):  # no-skill control: shuffle model probabilities within each year (sides stay paired)
        Sn = S.copy()
        n = len(S) // 2
        p1 = S.p1.values[:n].copy()
        for y in years:  # p1/p2 order is random, so a shuffled p1 probability carries no information on the winner
            i = np.where(S.year.values[:n] == y)[0]
            p1[i] = p1[rng.permutation(i)]
        qw = np.where(S.y1.values[:n], p1, 1 - p1)
        Sn["q"] = np.r_[qw, 1 - qw]
        Sn["ev"] = Sn.q * Sn.o - 1
        null.append(score(Sn, tactics(Sn), cut).survives.sum())
    print(f"\n######## {name}: {len(S) // 2} matches, years {years[0]}-{years[-1]}, explore < {cut} <= confirm; "
          f"median margin {np.median(1 / S.o[:len(S) // 2] + 1 / S.oo[:len(S) // 2] - 1):.1%}")
    print(f"  tactics tested {len(R)}, survivors {R.survives.sum()} | no-skill model survivors (5 shuffles): {null}")
    show = R[R.bets >= 300].sort_values("confirm", ascending=False).head(12)
    fmt = show.assign(**{c: show[c].map(lambda v: f"{v:+.1%}") for c in ["roi", "lo90", "explore", "confirm"]})
    print(fmt.to_string(index=False))
    for _, r in R[R.survives].iterrows():
        summary.append({"segment": name, **r.to_dict()})
print("\n======== ALL SURVIVORS (>=300 bets, + in both halves, 90% lower bound > 0)")
if summary:
    Z = pd.DataFrame(summary).sort_values("lo90", ascending=False)
    print(Z.assign(**{c: Z[c].map(lambda v: f"{v:+.1%}") for c in ["roi", "lo90", "explore", "confirm"]})
          .drop(columns="survives").to_string(index=False))
