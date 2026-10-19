"""Build pre-match features for every match since 1991 (all levels: main tour, qualifying, challengers,
futures/ITF) with a single chronological pass, then attach Bet365 odds to the tour-level matches."""
import glob
import math
import os
import re
import sys
import unicodedata
from collections import defaultdict, deque

import numpy as np
import pandas as pd

RAW, OUT, TOUR = sys.argv[1], sys.argv[2], sys.argv[3]  # TOUR in {atp, wta}
ROUND_ORDER = {"Q1": 0, "Q2": 1, "Q3": 2, "Q4": 3, "ER": 4, "R128": 5, "R64": 6, "R32": 7, "R16": 8, "RR": 9,
               "QF": 10, "SF": 11, "BR": 12, "F": 13}


def load_matches():
    files = sorted(glob.glob(f"{RAW}/sackmann_raw/{TOUR}__{TOUR}_matches_*.csv"))
    files = [f for f in files if re.search(r"_(\d{4})\.csv$", f) and int(re.search(r"_(\d{4})\.csv$", f).group(1)) >= 1991]
    dfs = []
    for f in files:
        d = pd.read_csv(f, low_memory=False)
        d["src"] = "main" if re.search(rf"{TOUR}_matches_\d{{4}}", f) else "lower"
        dfs.append(d)
    m = pd.concat(dfs, ignore_index=True)
    m = m.dropna(subset=["winner_id", "loser_id", "tourney_date"])
    m["date"] = pd.to_datetime(m["tourney_date"].astype(int).astype(str), format="%Y%m%d")
    m["rord"] = m["round"].map(ROUND_ORDER).fillna(5)
    m["score"] = m["score"].fillna("").astype(str)
    m["walkover"] = m["score"].str.contains("W/O|DEF|Walkover|unfinished", case=False) | (m["score"].str.strip() == "")
    m["retired"] = m["score"].str.contains("RET", case=False)
    m["surface"] = m["surface"].fillna("Hard")
    return m.sort_values(["date", "tourney_id", "rord", "match_num"]).reset_index(drop=True)


def k_factor(n):
    return 250.0 / (n + 5) ** 0.4


def elo_p(a, b):
    return 1.0 / (1.0 + 10 ** ((b - a) / 400.0))


def build_features(m):
    elo = defaultdict(lambda: 1500.0)
    selo = defaultdict(lambda: 1500.0)
    n = defaultdict(int)
    ns = defaultdict(int)
    last = {}
    hist = defaultdict(lambda: deque(maxlen=80))  # (day, won, surface, svw, svn, rtw, rtn, minutes)
    h2h = defaultdict(int)
    W, L = m["winner_id"].values, m["loser_id"].values
    S = m["surface"].values
    D = (m["date"].values.astype("datetime64[D]").astype(np.int64)).tolist()
    wo, ret = m["walkover"].values, m["retired"].values
    stat_cols = ["w_svpt", "w_1stWon", "w_2ndWon", "l_svpt", "l_1stWon", "l_2ndWon", "minutes"]
    st = np.nan_to_num(m[stat_cols].to_numpy(dtype=float), nan=-1.0).tolist()
    out = np.full((len(m), 28), np.nan)
    nan = float("nan")
    for i in range(len(m)):
        w, l, s, d = W[i], L[i], S[i], D[i]
        for j, (p, o) in enumerate(((w, l), (l, w))):
            h = hist[p]
            n52 = won52 = n14 = s52 = sw52 = 0
            min7 = 0.0
            svw = svn = rtw = rtn = 0.0
            k = 0
            for x in reversed(h):
                age = d - x[0]
                if age <= 365:
                    n52 += 1; won52 += x[1]
                    if age <= 14:
                        n14 += 1
                    if age <= 7 and x[7] > 0:
                        min7 += x[7]
                    if x[2] == s:
                        s52 += 1; sw52 += x[1]
                if k < 30 and x[4] > 0:
                    svw += x[3]; svn += x[4]; rtw += x[5]; rtn += x[6]
                k += 1
            last10 = list(h)[-10:]
            b = j * 14
            out[i, b:b + 14] = (
                elo[p], selo[(p, s)], n[p], ns[(p, s)],
                d - last[p] if p in last else 999,
                sum(x[1] for x in last10) / len(last10) if last10 else nan,
                won52 / n52 if n52 else nan, n52, n14,
                sw52 / s52 if s52 else nan,
                svw / svn if svn >= 100 else nan,
                rtw / rtn if rtn >= 100 else nan,
                min7, h2h[(p, o)],
            )
        if wo[i]:
            continue
        # Features above are strictly pre-match; only now fold the result into the state.
        mult = 0.5 if ret[i] else 1.0
        ew = elo_p(elo[w], elo[l])
        elo[w] += mult * k_factor(n[w]) * (1 - ew)
        elo[l] -= mult * k_factor(n[l]) * (1 - ew)
        esw = elo_p(selo[(w, s)], selo[(l, s)])
        selo[(w, s)] += mult * k_factor(ns[(w, s)]) * (1 - esw)
        selo[(l, s)] -= mult * k_factor(ns[(l, s)]) * (1 - esw)
        n[w] += 1; n[l] += 1; ns[(w, s)] += 1; ns[(l, s)] += 1
        last[w] = d; last[l] = d
        r = st[i]
        w_sw, l_sw = r[1] + r[2], r[4] + r[5]
        ok = r[0] > 0 and r[3] > 0
        hist[w].append((d, 1, s, w_sw if ok else 0, r[0] if ok else 0, r[3] - l_sw if ok else 0, r[3] if ok else 0, r[6]))
        hist[l].append((d, 0, s, l_sw if ok else 0, r[3] if ok else 0, r[0] - w_sw if ok else 0, r[0] if ok else 0, r[6]))
        h2h[(w, l)] += 1
    names = ["elo", "selo", "n", "ns", "rest", "form10", "win52", "n52", "n14", "surf52", "svpct", "rtpct", "min7", "h2h"]
    cols = [f"w_{x}" for x in names] + [f"l_{x}" for x in names]
    return pd.DataFrame(out, columns=cols, index=m.index)


def norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z ]", " ", s.replace("-", " ").replace("'", "")).split()


def attach_odds(m, F):
    fname = "atp_tennis.csv" if TOUR == "atp" else "wta.csv"
    o = pd.read_csv(f"{RAW}/odds_raw/{fname}", low_memory=False)
    for c in ["Odd_1", "Odd_2", "Rank_1", "Rank_2"]:
        o[c] = pd.to_numeric(o[c], errors="coerce")
    o["date"] = pd.to_datetime(o["Date"], errors="coerce")
    o = o[(o.Odd_1 > 1) & (o.Odd_2 > 1) & o.date.notna()].reset_index(drop=True)
    main = m[m["src"] == "main"]
    full = {}
    for c in ["winner", "loser"]:
        for pid, nm in zip(main[f"{c}_id"], main[f"{c}_name"]):
            full[pid] = norm(nm)
    by_token = defaultdict(list)
    for idx, wid, lid in zip(main.index, main["winner_id"], main["loser_id"]):
        for t in set(full[wid]) | set(full[lid]):
            by_token[t].append(idx)
    mdate = m["date"]

    def name_ok(td_name, pid):
        parts = str(td_name).strip().split(" ")
        initials = parts[-1] if parts[-1].endswith(".") else ""
        sur = norm(" ".join(parts[:-1]) if initials else td_name)
        toks = full.get(pid, [])
        if not sur or not toks or not set(sur) <= set(toks):
            return False
        return not initials or toks[0][0] == norm(initials)[0][0] if norm(initials) else True

    hit = []
    for r in o.itertuples():
        a, b = r.Player_1, r.Player_2
        sur = norm(str(a).rsplit(" ", 1)[0])
        if not sur:
            hit.append(-1); continue
        cands = by_token.get(max(sur, key=len), [])
        found = -1
        for idx in cands:
            dd = (r.date - mdate[idx]).days
            if dd < -2 or dd > 20:
                continue
            wid, lid = m.at[idx, "winner_id"], m.at[idx, "loser_id"]
            if (name_ok(a, wid) and name_ok(b, lid)) or (name_ok(a, lid) and name_ok(b, wid)):
                found = idx; break
        hit.append(found)
    o["midx"] = hit
    print(f"odds rows {len(o)}, matched {np.mean(np.array(hit) >= 0):.1%}", flush=True)
    o = o[o.midx >= 0].drop_duplicates("midx")
    mm = m.loc[o.midx.values]
    w_name_is_p1 = [name_ok(a, wid) for a, wid in zip(o.Player_1, mm.winner_id)]
    o["odds_w"] = np.where(w_name_is_p1, o.Odd_1, o.Odd_2)
    o["odds_l"] = np.where(w_name_is_p1, o.Odd_2, o.Odd_1)
    # sanity: the odds file's Winner must agree with Sackmann's winner
    o["winner_agrees"] = [name_ok(wn, wid) for wn, wid in zip(o.Winner, mm.winner_id)]
    return o[["midx", "odds_w", "odds_l", "winner_agrees", "date", "Tournament"]].rename(columns={"date": "match_date"})


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    m = load_matches()
    print(TOUR, len(m), "matches", m.date.min().date(), m.date.max().date(), flush=True)
    F = build_features(m)
    keep = ["tourney_id", "tourney_name", "tourney_level", "surface", "date", "round", "rord", "best_of", "src",
            "winner_id", "loser_id", "winner_name", "loser_name", "winner_rank", "loser_rank", "winner_rank_points",
            "loser_rank_points", "winner_age", "loser_age", "winner_ht", "loser_ht", "winner_hand", "loser_hand",
            "walkover", "retired", "score"]
    m["tourney_level"] = m["tourney_level"].astype(str)
    out = pd.concat([m[keep], F], axis=1)
    o = attach_odds(m, F)
    out = out.join(o.set_index("midx"))
    print("odds attached", out.odds_w.notna().sum(), "winner agrees", out.winner_agrees.mean(), flush=True)
    out.to_parquet(f"{OUT}/{TOUR}_features.parquet")
