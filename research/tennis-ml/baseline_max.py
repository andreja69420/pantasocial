"""Is the edge at best-available odds the model's, or just the price? Compare the model against naive rules
settled at the same Max odds: every favourite, every player <= 3.0, random side, and 'Pinnacle fair' value."""
import sys

import numpy as np
import pandas as pd

D, MLT, TAG = sys.argv[1], sys.argv[2], sys.argv[3]
rng = np.random.default_rng(1)
for tour in ["atp", "wta"]:
    P = pd.read_parquet(f"{D}/out/preds3_{tour}_ens_{TAG}_dev.parquet")
    P = P[P.year <= 2019]
    f2 = pd.read_parquet(f"{D}/out/{tour}_f2.parquet", columns=["date", "odds_w", "odds_l"]).loc[P.index]
    f2.index.name = "index"
    s = pd.read_csv(f"{MLT}/df_{tour}.csv", low_memory=False)
    s["Date"] = pd.to_datetime(s["Date"], errors="coerce")
    for c in ["B365W", "B365L", "PSW", "PSL", "MaxW", "MaxL"]:
        s[c] = pd.to_numeric(s[c], errors="coerce")
    s = s.dropna(subset=["B365W", "B365L"])
    key = lambda w, l: np.round(w, 3).astype(str) + "|" + np.round(l, 3).astype(str)
    s["k"], f2["k"] = key(s.B365W, s.B365L), key(f2.odds_w, f2.odds_l)
    j = f2.reset_index().merge(s[["k", "Date", "PSW", "PSL", "MaxW", "MaxL"]], on="k")
    j = j[(j.Date - j.date).dt.days.between(-2, 20)].drop_duplicates("index").set_index("index")
    P = P.join(j[["PSW", "PSL", "MaxW", "MaxL"]], how="inner").dropna(subset=["MaxW", "MaxL", "PSW", "PSL"])
    w1 = P.y == 1
    M1, M2 = np.where(w1, P.MaxW, P.MaxL), np.where(w1, P.MaxL, P.MaxW)
    S1, S2 = np.where(w1, P.PSW, P.PSL), np.where(w1, P.PSL, P.PSW)
    ps_fair1 = (1 / S1) / (1 / S1 + 1 / S2)

    def roi(pick1, pick2, cap=3.0):
        pick1, pick2 = pick1 & (M1 <= cap), pick2 & (M2 <= cap)
        pnl = np.r_[np.where(P.stands, np.where(P.y == 1, M1 - 1, -1.0), 0)[pick1],
                    np.where(P.stands, np.where(P.y == 0, M2 - 1, -1.0), 0)[pick2]]
        se = pnl.std() / np.sqrt(len(pnl))
        return f"bets {len(pnl):5d}  ROI {pnl.mean():+6.2%} ± {1.65 * se:.1%}"

    print(f"\n{tour.upper()} 2014-2019 at best-available odds (cap 3.0), n={len(P)}")
    print("  model EV>5%              ", roi(P.p * M1 > 1.05, (1 - P.p) * M2 > 1.05))
    print("  every favourite          ", roi(M1 < M2, M2 < M1))
    print("  every underdog <=3       ", roi(M1 > M2, M2 > M1))
    print("  both players             ", roi(np.ones(len(P), bool), np.ones(len(P), bool)))
    r = rng.random(len(P)) < 0.5
    print("  random side              ", roi(r, ~r))
    print("  Pinnacle-fair value >2%  ", roi(ps_fair1 * M1 > 1.02, (1 - ps_fair1) * M2 > 1.02))
    # same number of bets as the model, but chosen at random among the same odds range
    n_model = int(((P.p * M1 > 1.05) & (M1 <= 3)).sum() + (((1 - P.p) * M2 > 1.05) & (M2 <= 3)).sum())
    sims = []
    for _ in range(300):
        a, b = rng.random(len(P)) < 0.5, rng.random(len(P)) < 0.5
        idx1 = np.flatnonzero(a & (M1 <= 3)); idx2 = np.flatnonzero(~a & (M2 <= 3))
        pnl = np.r_[np.where(P.stands, np.where(P.y == 1, M1 - 1, -1.0), 0)[idx1], np.where(P.stands, np.where(P.y == 0, M2 - 1, -1.0), 0)[idx2]]
        sims.append(rng.choice(pnl, n_model, replace=False).mean())
    print(f"  random bets, same count  ROI mean {np.mean(sims):+.2%}, 95th pct {np.percentile(sims, 95):+.2%}, max {np.max(sims):+.2%}")
