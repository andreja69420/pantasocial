"""Vectorized feature engine (>150 features, no odds). Per-player rolling stats are computed from prior
matches only, using cumulative sums over (group, chronological order) and searchsorted for time windows.

usage: features2.py RAW_DIR OUT_DIR TOUR START_YEAR
"""
import glob
import re
import sys
import unicodedata
from collections import defaultdict

import numpy as np
import pandas as pd

RAW, OUT, TOUR, START = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
ROUND_ORDER = {"Q1": 0, "Q2": 1, "Q3": 2, "Q4": 3, "ER": 4, "R128": 5, "R64": 6, "R32": 7, "R16": 8, "RR": 9,
               "QF": 10, "SF": 11, "BR": 12, "F": 13}


# ---------------------------------------------------------------- loading
def load_matches():
    dfs = []
    for f in sorted(glob.glob(f"{RAW}/sackmann_raw/{TOUR}__{TOUR}_matches_*.csv")):
        yr = re.search(r"_(\d{4})\.csv$", f)
        if not yr or int(yr.group(1)) < START:
            continue
        d = pd.read_csv(f, low_memory=False)
        d["main"] = int(bool(re.search(rf"{TOUR}_matches_\d{{4}}", f)))
        dfs.append(d)
    m = pd.concat(dfs, ignore_index=True).dropna(subset=["winner_id", "loser_id", "tourney_date"])
    m["date"] = pd.to_datetime(m["tourney_date"].astype(int).astype(str), format="%Y%m%d")
    m["rord"] = m["round"].map(ROUND_ORDER).fillna(5)
    m["score"] = m["score"].fillna("").astype(str)
    m["walkover"] = m["score"].str.contains("W/O|DEF|Walkover|unfinished", case=False) | (m["score"].str.strip() == "")
    m["retired"] = m["score"].str.contains("RET", case=False)
    m["surface"] = m["surface"].fillna("Hard")
    m["tourney_level"] = m["tourney_level"].astype(str)
    m = m[~m["walkover"]].sort_values(["date", "tourney_id", "rord", "match_num"]).reset_index(drop=True)
    m["order"] = np.arange(len(m))
    return m


_score_cache = {}


def parse_score(s):
    if s in _score_cache:
        return _score_cache[s]
    gw = gl = sw = sl = tbw = tbl = b0w = b0l = 0
    sets = re.findall(r"(\d+)-(\d+)", s.split("RET")[0])
    complete = 0
    for a, b in sets:
        a, b = int(a), int(b)
        gw += a; gl += b
        if max(a, b) >= 6 and (abs(a - b) >= 2 or max(a, b) == 7):
            complete += 1
            sw += a > b; sl += b > a
            if {a, b} == {7, 6}:
                tbw += a > b; tbl += b > a
            if (a, b) == (6, 0):
                b0w += 1
            if (a, b) == (0, 6):
                b0l += 1
    r = (gw, gl, sw, sl, tbw, tbl, b0w, b0l, complete)
    _score_cache[s] = r
    return r


# ---------------------------------------------------------------- sequential ratings
def elo_p(a, b):
    return 1.0 / (1.0 + 10 ** ((b - a) / 400.0))


def ratings(m, games_share):
    """Several Elo variants, recorded pre-match for winner and loser."""
    E, SE, FE, GE, ME = (defaultdict(lambda: 1500.0) for _ in range(5))
    n, ns = defaultdict(int), defaultdict(int)
    W, L, S = m.winner_id.values, m.loser_id.values, m.surface.values
    main, ret = m.main.values, m.retired.values
    out = np.zeros((len(m), 10))
    for i in range(len(m)):
        w, l, s = W[i], L[i], S[i]
        out[i] = (E[w], E[l], SE[(w, s)], SE[(l, s)], FE[w], FE[l], GE[w], GE[l], ME[w], ME[l])
        mult = 0.5 if ret[i] else 1.0
        kw, kl = 250 / (n[w] + 5) ** 0.4, 250 / (n[l] + 5) ** 0.4
        e = elo_p(E[w], E[l]); E[w] += mult * kw * (1 - e); E[l] -= mult * kl * (1 - e)
        ksw, ksl = 250 / (ns[(w, s)] + 5) ** 0.4, 250 / (ns[(l, s)] + 5) ** 0.4
        e = elo_p(SE[(w, s)], SE[(l, s)]); SE[(w, s)] += mult * ksw * (1 - e); SE[(l, s)] -= mult * ksl * (1 - e)
        e = elo_p(FE[w], FE[l]); FE[w] += mult * 48 * (1 - e); FE[l] -= mult * 48 * (1 - e)  # fast, form-like
        # games-based Elo: outcome is the share of games won, so margin of victory matters
        g = games_share[i]
        if not np.isnan(g):
            e = elo_p(GE[w], GE[l]); GE[w] += mult * 1.5 * kw * (g - e); GE[l] -= mult * 1.5 * kl * (g - e)
        if main[i]:
            e = elo_p(ME[w], ME[l]); ME[w] += mult * kw * (1 - e); ME[l] -= mult * kl * (1 - e)
        n[w] += 1; n[l] += 1; ns[(w, s)] += 1; ns[(l, s)] += 1
    return out


# ---------------------------------------------------------------- vectorized prior-window machinery
class Prior:
    """Sums of columns over a player's previous rows within a group (row-count and day windows)."""

    def __init__(self, gkey, order, day):
        self.idx = np.lexsort((order, gkey))
        g = gkey[self.idx]
        self.day = day[self.idx]
        self.start = np.zeros(len(g), dtype=np.int64)
        brk = np.r_[0, np.flatnonzero(g[1:] != g[:-1]) + 1]
        lens = np.diff(np.r_[brk, len(g)])
        self.start = np.repeat(brk, lens)
        self.pos = np.arange(len(g)) - self.start  # number of prior rows in group
        gcode = np.repeat(np.arange(len(brk)), lens).astype(np.int64)
        self.key = gcode * 1_000_000 + self.day
        self.inv = np.empty_like(self.idx); self.inv[self.idx] = np.arange(len(self.idx))

    def _cs(self, v):
        v = np.nan_to_num(v[self.idx].astype(float))
        return np.r_[0.0, np.cumsum(v)]  # cs[i] = sum of first i rows (global, sorted)

    def rows(self, v, k):
        cs = self._cs(v)
        i = np.arange(len(cs) - 1)
        lo = np.maximum(i - k, self.start)
        return (cs[i] - cs[lo])[self.inv]

    def days(self, v, d):
        cs = self._cs(v)
        i = np.arange(len(cs) - 1)
        lo = np.searchsorted(self.key, self.key - d, side="right")
        lo = np.maximum(lo, self.start)
        return (cs[i] - cs[np.minimum(lo, i)])[self.inv]

    def all(self, v):
        cs = self._cs(v)
        i = np.arange(len(cs) - 1)
        return (cs[i] - cs[self.start])[self.inv]

    def lag(self, v, k):
        vs = v[self.idx].astype(float)
        i = np.arange(len(vs))
        j = i - k
        out = np.where(j >= self.start, vs[np.maximum(j, 0)], np.nan)
        return out[self.inv]

    def count_prior(self):
        return self.pos[self.inv].astype(float)

    def first_day(self):
        return self.day[self.start][self.inv]

    def last_day(self):
        i = np.arange(len(self.day))
        return np.where(self.pos > 0, self.day[np.maximum(i - 1, 0)], np.nan)[self.inv]


def ratio(a, b, min_den=1):
    return np.where(b >= min_den, a / np.where(b == 0, 1, b), np.nan)


# ---------------------------------------------------------------- build
def build():
    m = load_matches()
    print(TOUR, len(m), "matches", m.date.min().date(), m.date.max().date(), flush=True)
    sc = np.array([parse_score(s) for s in m.score.values], dtype=float)
    gshare = np.where(sc[:, 0] + sc[:, 1] > 0, sc[:, 0] / np.maximum(sc[:, 0] + sc[:, 1], 1), np.nan)
    R = ratings(m, gshare)
    print("ratings done", flush=True)

    # long table: one row per player per match
    def side(prefix, oprefix, won):
        d = pd.DataFrame({
            "mi": m.index.values, "order": m.order.values, "won": won,
            "pid": m[f"{prefix}_id"].values.astype(np.int64), "opp": m[f"{oprefix}_id"].values.astype(np.int64),
            "day": m.date.values.astype("datetime64[D]").astype(np.int64),
            "surface": m.surface.values, "main": m.main.values, "tname": m.tourney_name.fillna("").str.lower().values,
            "tid": m.tourney_id.values, "minutes": m.minutes.values,
            "rank": m[f"{prefix}_rank"].values, "pts": m[f"{prefix}_rank_points"].values,
            "age": m[f"{prefix}_age"].values, "ht": m[f"{prefix}_ht"].values,
            "left": (m[f"{prefix}_hand"] == "L").astype(int).values,
            "entry": m[f"{prefix}_entry"].fillna("").astype(str).values,
            "seed": pd.to_numeric(m[f"{prefix}_seed"], errors="coerce").values,
        })
        a, b = ("w", "l") if won == 1 else ("l", "w")
        sv = {k: pd.to_numeric(m[f"{a}_{k}"], errors="coerce").values for k in
              ["svpt", "ace", "df", "1stIn", "1stWon", "2ndWon", "SvGms", "bpSaved", "bpFaced"]}
        osv = {k: pd.to_numeric(m[f"{b}_{k}"], errors="coerce").values for k in
               ["svpt", "1stWon", "2ndWon", "SvGms", "bpSaved", "bpFaced"]}
        ok = (sv["svpt"] > 0) & (osv["svpt"] > 0)
        d["st"] = ok.astype(float)
        f = lambda v: np.where(ok, v, 0.0)
        d["svpt"], d["ace"], d["df"] = f(sv["svpt"]), f(sv["ace"]), f(sv["df"])
        d["in1"], d["won1"], d["won2"] = f(sv["1stIn"]), f(sv["1stWon"]), f(sv["2ndWon"])
        d["svwon"] = d["won1"] + d["won2"]
        d["sv2pt"] = d["svpt"] - d["in1"]
        d["svg"], d["bps"], d["bpf"] = f(sv["SvGms"]), f(sv["bpSaved"]), f(sv["bpFaced"])
        d["held"] = d["svg"] - (d["bpf"] - d["bps"])
        d["rtpt"] = f(osv["svpt"]); d["rtwon"] = f(osv["svpt"] - osv["1stWon"] - osv["2ndWon"])
        d["rtg"] = f(osv["SvGms"]); d["bpc"] = f(osv["bpFaced"]); d["bpw"] = f(osv["bpFaced"] - osv["bpSaved"])
        gi = (0, 1, 2, 3, 4, 5, 6, 7) if won == 1 else (1, 0, 3, 2, 5, 4, 7, 6)
        for name, j in zip(["gw", "gl", "sw", "sl", "tbw", "tbl", "b0w", "b0l"], gi):
            d[name] = sc[:, j]
        d["sets"] = sc[:, 8]
        d["decider"] = ((sc[:, 2] + sc[:, 3]) == m.best_of.fillna(3).values).astype(float)
        d["decw"] = d["decider"] * won
        d["ret_me"] = (m.retired.values & (won == 0)).astype(float)
        cols = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
        me = [0, 2, 4, 6, 8] if won == 1 else [1, 3, 5, 7, 9]
        op = [1, 3, 5, 7, 9] if won == 1 else [0, 2, 4, 6, 8]
        for name, a_, b_ in zip(["elo", "selo", "felo", "gelo", "melo"], me, op):
            d[name], d[f"o_{name}"] = R[:, a_], R[:, b_]
        d["exp"] = elo_p(d["elo"].values, d["o_elo"].values)
        d["resid"] = won - d["exp"]
        d["gresid"] = np.nan_to_num(np.where(won == 1, gshare, 1 - gshare)) - elo_p(d["gelo"].values, d["o_gelo"].values)
        return d

    Lw, Ll = side("winner", "loser", 1), side("loser", "winner", 0)
    L = pd.concat([Lw, Ll], ignore_index=True)
    L["one"] = 1.0
    L["dur_known"] = L.minutes.notna().astype(float)
    L["minutes"] = L.minutes.fillna(0)
    L["upset_win"] = ((L.won == 1) & (L.exp < 0.4)).astype(float)
    L["bad_loss"] = ((L.won == 0) & (L.exp > 0.6)).astype(float)
    F = {}

    order, day = L.order.values, L.day.values
    P = Prior(L.pid.values, order, day)
    surf_code = L.surface.map({"Hard": 0, "Clay": 1, "Grass": 2, "Carpet": 3}).fillna(0).astype(int).values
    PS = Prior(L.pid.values * 4 + surf_code, order, day)
    PM = Prior(L.pid.values * 2 + L.main.values.astype(int), order, day)
    PT = Prior(pd.factorize(L.pid.astype(str) + "|" + L.tname)[0], order, day)
    PH = Prior(pd.factorize(L.pid.astype(str) + "|" + L.opp.astype(str))[0], order, day)
    PHS = Prior(pd.factorize(L.pid.astype(str) + "|" + L.opp.astype(str) + "|" + L.surface)[0], order, day)
    PE = Prior(pd.factorize(L.pid.astype(str) + "|" + L.tid.astype(str))[0], order, day)
    PLH = Prior(L.pid.values * 2 + np.where(L.won == 1, Ll.set_index("mi").left.reindex(L.mi).values, Lw.set_index("mi").left.reindex(L.mi).values).astype(int), order, day)
    bo5 = m.best_of.fillna(3).values[L.mi.values] == 5
    PB5 = Prior(L.pid.values * 2 + bo5.astype(int), order, day)
    # Venue stats must come from earlier *matches*: on the long table both players of the current match share
    # the venue group, so one side's row would see the other side's stats from this very match.
    Mv = L.groupby("mi").agg(order=("order", "first"), day=("day", "first"), tname=("tname", "first"),
                             ace=("ace", "sum"), svpt=("svpt", "sum"), svwon=("svwon", "sum"),
                             tb=("tbw", "sum"), sets=("sets", "first"))
    PTN = Prior(pd.factorize(Mv.tname)[0], Mv.order.values, Mv.day.values)
    venue = pd.DataFrame({
        "venue_ace_rate": ratio(PTN.days(Mv.ace.values, 1500), PTN.days(Mv.svpt.values, 1500), 500),
        "venue_sv_won": ratio(PTN.days(Mv.svwon.values, 1500), PTN.days(Mv.svpt.values, 1500), 500),
        "venue_tb_rate": ratio(PTN.days(Mv.tb.values, 1500), PTN.days(Mv.sets.values, 1500), 50),
    }, index=Mv.index).reindex(L.mi.values)

    # ratings and their dynamics
    for r in ["elo", "selo", "felo", "gelo", "melo"]:
        F[r] = L[r].values
    F["elo_mom5"] = L.elo.values - P.lag(L.elo.values, 5)
    F["elo_mom20"] = L.elo.values - P.lag(L.elo.values, 20)
    F["elo_vs_avg20"] = L.elo.values - P.rows(L.elo.values, 20) / np.maximum(P.rows(L.one.values, 20), 1)
    F["selo_minus_elo"] = L.selo.values - L.elo.values
    F["n_career"] = P.count_prior()
    F["n_surface"] = PS.count_prior()
    F["log_n_career"] = np.log1p(F["n_career"])
    F["elo_uncert"] = 1 / np.sqrt(F["n_career"] + 1)
    F["years_active"] = (day - P.first_day()) / 365.25
    F["rest_days"] = np.clip(day - P.last_day(), 0, 400)
    F["long_layoff"] = (F["rest_days"] > 60).astype(float)

    # form over the last k matches
    for k in [5, 10, 20, 50]:
        cnt = P.rows(L.one.values, k)
        F[f"win_{k}"] = ratio(P.rows(L.won.values, k), cnt)
        F[f"resid_{k}"] = ratio(P.rows(L.resid.values, k), cnt)
        F[f"gshare_{k}"] = ratio(P.rows(L.gw.values, k), P.rows(L.gw.values + L.gl.values, k))
        F[f"sshare_{k}"] = ratio(P.rows(L.sw.values, k), P.rows(L.sw.values + L.sl.values, k))
    for k in [10, 30]:
        F[f"gresid_{k}"] = ratio(P.rows(L.gresid.values, k), P.rows(L.one.values, k))
        F[f"oppelo_{k}"] = ratio(P.rows(L.o_elo.values, k), P.rows(L.one.values, k))
        F[f"upsets_{k}"] = P.rows(L.upset_win.values, k)
        F[f"badloss_{k}"] = P.rows(L.bad_loss.values, k)
    # day windows: workload and results
    for d in [3, 7, 14, 30, 90, 365]:
        F[f"n_{d}d"] = P.days(L.one.values, d)
    for d in [7, 14, 30]:
        F[f"min_{d}d"] = P.days(L.minutes.values, d)
        F[f"sets_{d}d"] = P.days(L.sets.values, d)
    for d in [90, 365]:
        F[f"win_{d}d"] = ratio(P.days(L.won.values, d), P.days(L.one.values, d))
        F[f"resid_{d}d"] = P.days(L.resid.values, d)
        F[f"ret_{d}d"] = P.days(L.ret_me.values, d)
    # serve / return stats (matches with stats only)
    for k in [10, 30]:
        g = lambda c: P.rows(L[c].values, k)
        F[f"ace_rate_{k}"] = ratio(g("ace"), g("svpt"), 30)
        F[f"df_rate_{k}"] = ratio(g("df"), g("svpt"), 30)
        F[f"first_in_{k}"] = ratio(g("in1"), g("svpt"), 30)
        F[f"first_won_{k}"] = ratio(g("won1"), g("in1"), 30)
        F[f"second_won_{k}"] = ratio(g("won2"), g("sv2pt"), 30)
        F[f"sv_won_{k}"] = ratio(g("svwon"), g("svpt"), 30)
        F[f"bp_saved_{k}"] = ratio(g("bps"), g("bpf"), 5)
        F[f"hold_{k}"] = ratio(g("held"), g("svg"), 5)
        F[f"rt_won_{k}"] = ratio(g("rtwon"), g("rtpt"), 30)
        F[f"bp_conv_{k}"] = ratio(g("bpw"), g("bpc"), 5)
        F[f"break_{k}"] = ratio(g("bpw"), g("rtg"), 5)
        F[f"dominance_{k}"] = ratio(g("rtwon"), g("rtpt"), 30) / np.maximum(1 - ratio(g("svwon"), g("svpt"), 30), 1e-3)
        F[f"tb_win_{k}"] = ratio(g("tbw"), g("tbw") + g("tbl"))
        F[f"stats_n_{k}"] = g("st")
    F["decider_win_30"] = ratio(P.rows(L.decw.values, 30), P.rows(L.decider.values, 30))
    F["bagels_for_30"] = P.rows(L.b0w.values, 30)
    F["bagels_against_30"] = P.rows(L.b0l.values, 30)
    # surface-specific
    for k in [10, 30]:
        cnt = PS.rows(L.one.values, k)
        F[f"surf_win_{k}"] = ratio(PS.rows(L.won.values, k), cnt)
        F[f"surf_resid_{k}"] = ratio(PS.rows(L.resid.values, k), cnt)
        F[f"surf_sv_won_{k}"] = ratio(PS.rows(L.svwon.values, k), PS.rows(L.svpt.values, k), 30)
        F[f"surf_rt_won_{k}"] = ratio(PS.rows(L.rtwon.values, k), PS.rows(L.rtpt.values, k), 30)
        F[f"surf_gshare_{k}"] = ratio(PS.rows(L.gw.values, k), PS.rows(L.gw.values + L.gl.values, k))
    F["surf_win_365d"] = ratio(PS.days(L.won.values, 365), PS.days(L.one.values, 365))
    # same level (main vs lower)
    F["lvl_win_20"] = ratio(PM.rows(L.won.values, 20), PM.rows(L.one.values, 20))
    F["lvl_resid_20"] = ratio(PM.rows(L.resid.values, 20), PM.rows(L.one.values, 20))
    F["lvl_n"] = PM.count_prior()
    # this tournament: history and current run
    F["tourn_hist_n"] = PT.count_prior()
    F["tourn_hist_win"] = ratio(PT.all(L.won.values), PT.all(L.one.values))
    F["event_matches"] = PE.count_prior()
    F["event_minutes"] = PE.all(L.minutes.values)
    F["event_sets"] = PE.all(L.sets.values)
    # opponent handedness, best-of-5, previous match load, venue speed
    F["vs_hand_win_30"] = ratio(PLH.rows(L.won.values, 30), PLH.rows(L.one.values, 30))
    F["vs_hand_resid_30"] = ratio(PLH.rows(L.resid.values, 30), PLH.rows(L.one.values, 30))
    F["bo_win_30"] = ratio(PB5.rows(L.won.values, 30), PB5.rows(L.one.values, 30))
    F["bo_resid_30"] = ratio(PB5.rows(L.resid.values, 30), PB5.rows(L.one.values, 30))
    F["bo_n"] = PB5.count_prior()
    F["prev_minutes"] = P.lag(L.minutes.values, 1)
    F["prev_sets"] = P.lag(L.sets.values, 1)
    F["prev_won"] = P.lag(L.won.values, 1)
    for c in venue.columns:
        F[c] = venue[c].values
    F["sv_vs_venue_30"] = F["sv_won_30"] - F["venue_sv_won"]
    # head to head
    F["h2h_n"] = PH.count_prior()
    F["h2h_wins"] = PH.all(L.won.values)
    F["h2h_share"] = ratio(F["h2h_wins"], F["h2h_n"])
    F["h2h_gshare"] = ratio(PH.all(L.gw.values), PH.all(L.gw.values + L.gl.values))
    F["h2h_last"] = PH.lag(L.won.values, 1)
    F["h2h_surf_wins"] = PHS.all(L.won.values)
    F["h2h_surf_n"] = PHS.count_prior()
    # ranking and bio (known before the match)
    rank = L["rank"].values.astype(float)
    F["rank"] = rank
    F["log_rank"] = np.log(np.nan_to_num(rank, nan=3000))
    F["log_pts"] = np.log1p(np.nan_to_num(L.pts.values.astype(float)))
    F["rank_chg10"] = np.log(np.nan_to_num(P.lag(rank, 10), nan=3000)) - F["log_rank"]
    r_sorted, pid_sorted = pd.Series(rank[P.idx]), L.pid.values[P.idx]
    best = r_sorted.groupby(pid_sorted).cummin().groupby(pid_sorted).shift().values
    F["best_rank_seen"] = np.log(np.nan_to_num(best[P.inv], nan=3000))
    F["age"] = L.age.values.astype(float)
    F["age2"] = (F["age"] - 26) ** 2
    F["ht"] = L.ht.values.astype(float)
    F["left"] = L.left.values.astype(float)
    F["seeded"] = (~np.isnan(L.seed.values)).astype(float)
    for e in ["Q", "WC", "LL", "PR"]:
        F[f"entry_{e}"] = (L.entry.values == e).astype(float)

    feat = pd.DataFrame(F)
    feat["mi"], feat["won"] = L.mi.values, L.won.values
    W = feat[feat.won == 1].set_index("mi").drop(columns="won")
    Lo = feat[feat.won == 0].set_index("mi").drop(columns="won")
    print("per-player features:", W.shape[1], flush=True)
    return m, W.sort_index(), Lo.sort_index()


def norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z ]", " ", s.replace("-", " ").replace("'", "")).split()


def attach_odds(m):
    o = pd.read_csv(f"{RAW}/odds_raw/{'atp_tennis.csv' if TOUR == 'atp' else 'wta.csv'}", low_memory=False)
    for c in ["Odd_1", "Odd_2"]:
        o[c] = pd.to_numeric(o[c], errors="coerce")
    o["date"] = pd.to_datetime(o["Date"], errors="coerce")
    o = o[(o.Odd_1 > 1) & (o.Odd_2 > 1) & (o.date.dt.year >= START + 2)].reset_index(drop=True)
    main = m[m.main == 1]
    full = {}
    for c in ["winner", "loser"]:
        full.update({pid: norm(nm) for pid, nm in zip(main[f"{c}_id"], main[f"{c}_name"])})
    by_token = defaultdict(list)
    for idx, wid, lid in zip(main.index, main.winner_id, main.loser_id):
        for t in set(full[wid]) | set(full[lid]):
            by_token[t].append(idx)

    def ok(td, pid):
        parts = str(td).strip().split(" ")
        ini = parts[-1] if parts[-1].endswith(".") else ""
        sur = norm(" ".join(parts[:-1]) if ini else td)
        toks = full.get(pid, [])
        if not sur or not toks or not set(sur) <= set(toks):
            return False
        ni = norm(ini)
        return not ni or toks[0][0] == ni[0][0]

    hit = []
    for r in o.itertuples():
        sur = norm(str(r.Player_1).rsplit(" ", 1)[0])
        found = -1
        for idx in by_token.get(max(sur, key=len), []) if sur else []:
            dd = (r.date - m.at[idx, "date"]).days
            if -2 <= dd <= 20:
                wid, lid = m.at[idx, "winner_id"], m.at[idx, "loser_id"]
                if (ok(r.Player_1, wid) and ok(r.Player_2, lid)) or (ok(r.Player_1, lid) and ok(r.Player_2, wid)):
                    found = idx; break
        hit.append(found)
    o["mi"] = hit
    print(f"odds rows {len(o)}, matched {np.mean(np.array(hit) >= 0):.1%}", flush=True)
    o = o[o.mi >= 0].drop_duplicates("mi")
    mm = m.loc[o.mi.values]
    p1w = np.array([ok(a, w) for a, w in zip(o.Player_1, mm.winner_id)])
    o["odds_w"] = np.where(p1w, o.Odd_1, o.Odd_2)
    o["odds_l"] = np.where(p1w, o.Odd_2, o.Odd_1)
    o["agree"] = [ok(wn, w) for wn, w in zip(o.Winner, mm.winner_id)]
    return o.set_index("mi")[["odds_w", "odds_l", "agree"]]


if __name__ == "__main__":
    m, W, Lo = build()
    ctx = m[["date", "surface", "tourney_level", "rord", "best_of", "draw_size", "main", "retired", "score", "tourney_name"]].copy()
    ctx = ctx.join(attach_odds(m))
    W.add_prefix("w_").join(Lo.add_prefix("l_")).join(ctx).to_parquet(f"{OUT}/{TOUR}_f2.parquet")
    print("saved", flush=True)
