"""Score-pattern features (results only, strictly pre-match): clean game shares, set/tiebreak/comeback habits,
retirement history, on-court load and fatigue. Returns w_<name>/l_<name> columns for the match winner and loser.
Only matches with a smaller 'order' feed a row (rolling sums via cumulative sums over (group, order)).

usage: fx_score.py MATCHES_PARQUET OUT_PARQUET
"""
import re
import sys

import numpy as np
import pandas as pd

TRAIN_END = np.datetime64("2020-01-01")
FALLBACK = {"q1": (-1.66, 0.9), "q2": (1.66, 0.9), "p1set": (0.0, 1.0), "luck": (0.0, 7.0)}
_TOK = re.compile(r"^\[?(\d+)-(\d+)\]?(?:\(\d+(?:-\d+)?\))?$")
FIELDS = ["gw", "gl", "sw", "sl", "ncs", "mw", "ml", "load", "s1c", "s1w", "tbw", "tbl", "clw", "cll", "bfw", "bfl",
          "dgl", "nwn", "dgw", "nln", "rpwn", "rpwa", "rpln", "rpla", "g1w", "g1l", "mar"]
F = {k: i for i, k in enumerate(FIELDS)}


def parse_score(s):
    """Per-match scalars from the winner's view. Complete set = max>=6 and (diff>=2 or max==7); MTB = bracketed or
    max>=10 (one set, 3 load games, 0 games in clean stats); unfinished set counts only in load."""
    v = np.zeros(len(FIELDS))
    prev = None
    last_comp = None
    k = -1
    for t in s.replace("RET", " ").split():
        m = _TOK.match(t)
        if not m:
            continue
        k += 1
        a, b = int(m.group(1)), int(m.group(2))
        hi = max(a, b)
        if t.startswith("[") or hi >= 10:
            kind = 2
        elif hi >= 6 and (abs(a - b) >= 2 or hi == 7):
            kind = 1
        else:
            kind = 0
        v[F["load"]] += 3 if kind == 2 else a + b
        if kind == 0:
            prev = None
            continue
        v[F["sw"]] += a > b
        v[F["sl"]] += b > a
        if k == 0:
            v[F["s1c"]] = 1
            v[F["s1w"]] = a > b
        last_comp = (kind, a + b)
        if kind == 2:
            v[F["mw"]] += a > b
            v[F["ml"]] += b > a
            prev = a > b
            continue
        v[F["gw"]] += a
        v[F["gl"]] += b
        v[F["ncs"]] += 1
        if k == 0:
            v[F["g1w"]], v[F["g1l"]] = a, b
        if hi == 7 and min(a, b) == 6:
            v[F["tbw"]] += a > b
            v[F["tbl"]] += b > a
        if (hi, min(a, b)) in ((7, 5), (7, 6)):
            v[F["clw"]] += a > b
            v[F["cll"]] += b > a
        if hi == 6 and min(a, b) <= 1:
            v[F["bfw"]] += a > b
            v[F["bfl"]] += b > a
        if a > b:
            v[F["dgl"]] += b
            v[F["nwn"]] += 1
        else:
            v[F["dgw"]] += a
            v[F["nln"]] += 1
        if prev is not None:
            if prev:
                v[F["rpln"]] += 1
                v[F["rpla"]] += b > a
            else:
                v[F["rpwn"]] += 1
                v[F["rpwa"]] += a > b
        prev = a > b
    if last_comp is not None and (last_comp[0] == 2 or last_comp[1] >= 13):
        v[F["mar"]] = 1
    return v


def elo_exp(W, L, ret):
    """Pre-match standard Elo win probability of the winner (K=250/(n+5)^0.4, half weight for retirements)."""
    E = {}
    N = {}
    out = np.empty((len(W), 2))
    for i in range(len(W)):
        w, l = W[i], L[i]
        ew, el = E.get(w, 1500.0), E.get(l, 1500.0)
        e = 1.0 / (1.0 + 10 ** ((el - ew) / 400.0))
        out[i, 0], out[i, 1] = ew, el
        mult = 0.5 if ret[i] else 1.0
        nw, nl = N.get(w, 0), N.get(l, 0)
        E[w] = ew + mult * 250 / (nw + 5) ** 0.4 * (1 - e)
        E[l] = el - mult * 250 / (nl + 5) ** 0.4 * (1 - e)
        N[w], N[l] = nw + 1, nl + 1
    return out


def fit_logit(x, y):
    """p = sigmoid(a + b*x) for fractional y by Newton steps."""
    a = b = 0.0
    for _ in range(30):
        p = 1 / (1 + np.exp(-(a + b * x)))
        w = p * (1 - p) + 1e-9
        g = np.array([(y - p).sum(), ((y - p) * x).sum()])
        H = np.array([[w.sum(), (w * x).sum()], [(w * x).sum(), (w * x * x).sum()]])
        d = np.linalg.solve(H, g)
        a, b = a + d[0], b + d[1]
        if np.abs(d).max() < 1e-8:
            break
    return a, b


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


class Grp:
    """Rows grouped by key and ordered by 'order'; windows over the rows strictly before each row."""

    def __init__(self, key, order):
        self.idx = np.lexsort((order, key))
        k = key[self.idx]
        m = len(k)
        self.pos = np.arange(m)
        self.first = np.maximum.accumulate(np.where(np.r_[True, k[1:] != k[:-1]], self.pos, 0))
        self.key_sorted = k
        self._lo = {}

    def lo(self, n):
        if n not in self._lo:
            self._lo[n] = self.first if n is None else np.maximum(self.pos - n, self.first)
        return self._lo[n]

    def sum(self, a, n=None):
        s = np.concatenate(([0.0], np.cumsum(a[self.idx], dtype=np.float64)))
        out = np.empty(len(a))
        out[self.idx] = s[self.pos] - s[self.lo(n)]
        return out

    def count(self, n=None):
        out = np.empty(len(self.pos))
        out[self.idx] = self.pos - self.lo(n)
        return out

    def lag(self, a, k=1):
        s = a[self.idx]
        out = np.full(len(a), np.nan)
        ok = self.pos - k >= self.first
        out[self.idx[ok]] = s[self.pos[ok] - k]
        return out


def window_share_stats(G, share, n=20, min_valid=6):
    """Std (ddof=1) and 25th percentile of per-match values over the last n rows, valid values only."""
    s = share[G.idx]
    m = len(s)
    sd = np.full(m, np.nan)
    q25 = np.full(m, np.nan)
    ar = np.arange(1, n + 1)
    for c0 in range(0, m, 200000):
        c1 = min(m, c0 + 200000)
        j = G.pos[c0:c1, None] - ar[None, :]
        ok = j >= G.first[c0:c1, None]
        X = np.where(ok, s[np.maximum(j, 0)], np.nan)
        k = (~np.isnan(X)).sum(1)
        mu = np.nansum(X, 1) / np.maximum(k, 1)
        var = np.nansum((X - mu[:, None]) ** 2, 1) / np.maximum(k - 1, 1)
        X.sort(1)
        p = 0.25 * (k - 1)
        lo = np.floor(p).astype(int)
        hi = np.ceil(p).astype(int)
        r = np.arange(c1 - c0)
        q = X[r, lo] + (p - lo) * (X[r, hi] - X[r, lo])
        good = k >= min_valid
        sd[c0:c1] = np.where(good, np.sqrt(var), np.nan)
        q25[c0:c1] = np.where(good, q, np.nan)
    out_sd = np.empty(m)
    out_q = np.empty(m)
    out_sd[G.idx] = sd
    out_q[G.idx] = q25
    return out_sd, out_q


class Out(dict):
    def __setitem__(self, k, v):
        super().__setitem__(k, np.asarray(v, dtype=np.float32))


def sr(x, n, p0, m):
    return (x + m * p0) / (n + m)


def compute(M, women=None):
    """Self-play rows (winner_id == loser_id, a source data error) take no part in any state and get NaN."""
    bad = M["winner_id"].values == M["loser_id"].values
    if bad.any():
        if women is None:
            women = bool(M["tourney_level"].isin(["P", "PM", "I", "T1"]).any())
        return _compute(M[~bad], women).reindex(M.index)
    return _compute(M, women)


def _compute(M, women=None):
    n = len(M)
    order = M["order"].values.astype(np.int64)
    if women is None:
        women = bool(M["tourney_level"].isin(["P", "PM", "I", "T1"]).any())
    def two(a, b):
        c = np.concatenate([a, b])
        return c.astype(np.float32) if c.dtype == np.float64 else c

    W, L = M["winner_id"].values, M["loser_id"].values
    pc, _ = pd.factorize(np.concatenate([W, L]))
    wc, lc = pc[:n], pc[n:]
    ret = M["retired"].values.astype(bool)
    day = M["date"].values.astype("datetime64[D]").astype(np.int64)
    bo = M["best_of"].values

    elo = elo_exp(wc, lc, ret)
    e_w = 1.0 / (1.0 + 10 ** ((elo[:, 1] - elo[:, 0]) / 400.0))

    codes, uniq = pd.factorize(M["score"].fillna("").astype(str).values)
    PS = np.array([parse_score(s) for s in uniq]).reshape(len(uniq), len(FIELDS))[codes].astype(np.float32)
    f = lambda k: PS[:, F[k]]

    lvl = M["tourney_level"].astype(str)
    ch = lvl.isin(["C", "CC"]).values
    itf = lvl.str.fullmatch(r"\d+|S|T\d").values
    tour = ~ch & ~itf
    full_m = (f("sw") + f("sl") == bo) & np.isin(bo, (3, 5))

    ordL = two(order, order)
    GP = Grp(pc, ordL)
    won = two(np.ones(n), np.zeros(n))
    lostm = 1 - won
    gw, gl = two(f("gw"), f("gl")), two(f("gl"), f("gw"))
    games = gw + gl
    sw, sl = two(f("sw"), f("sl")), two(f("sl"), f("sw"))
    cs_all = sw + sl
    ncs = two(f("ncs"), f("ncs"))
    mtb_won, mtb_pl = two(f("mw"), f("ml")), two(f("mw") + f("ml"), f("mw") + f("ml"))
    load = two(f("load"), f("load"))
    s1c = two(f("s1c"), f("s1c"))
    s1won = two(f("s1w"), 1 - f("s1w")) * s1c
    tbp = two(f("tbw") + f("tbl"), f("tbw") + f("tbl"))
    tbwon = two(f("tbw"), f("tbl"))
    clp = two(f("clw") + f("cll"), f("clw") + f("cll"))
    clwon = two(f("clw"), f("cll"))
    bf_for, bf_ag = two(f("bfw"), f("bfl")), two(f("bfl"), f("bfw"))
    dom_gl, dom_n = two(f("dgl"), f("dgw")), two(f("nwn"), f("nln"))
    fight_gw, fight_n = two(f("dgw"), f("dgl")), two(f("nln"), f("nwn"))
    rp_n, rp_a = two(f("rpwn"), f("rpln")), two(f("rpwa"), f("rpla"))
    g1w, g1 = two(f("g1w"), f("g1l")), two(f("g1w") + f("g1l"), f("g1w") + f("g1l"))
    retL = two(np.zeros(n), ret.astype(float))
    ret_any = two(ret, ret).astype(float)
    nr = 1 - ret_any
    full = two(full_m, full_m).astype(float)
    mar = two(f("mar") * full_m, f("mar") * full_m)
    boL = two(bo, bo)
    dayL = two(day, day)
    e_row = two(e_w, 1 - e_w)
    own_elo, opp_elo = np.r_[elo[:, 0], elo[:, 1]], np.r_[elo[:, 1], elo[:, 0]]
    tr = two(M["date"].values < TRAIN_END, M["date"].values < TRAIN_END)
    wom = np.full(2 * n, women)
    ch2, itf2, tour2 = two(ch, ch), two(itf, itf), two(tour, tour)
    ct2 = ch2 | tour2

    def coef(name, x, y, mask):
        mk = mask & tr
        if mk.sum() < 5000:
            return FALLBACK[name]
        return fit_logit(x[mk].astype(float), y[mk].astype(float))

    lg = logit(e_row)
    a1, b1 = coef("q1", lg, won, (s1c == 1) & (s1won == 0))
    a2, b2 = coef("q2", lg, won, (s1c == 1) & (s1won == 1))
    a3, b3 = coef("p1set", lg, s1won, s1c == 1)
    q1, q2, p1s = sigmoid(a1 + b1 * lg), sigmoid(a2 + b2 * lg), sigmoid(a3 + b3 * lg)

    out = Out()
    nwin = lambda k: GP.count(k)

    g30, gm30 = GP.sum(gw, 30), GP.sum(games, 30)
    gshare30 = np.where(gm30 >= 12, g30 / np.maximum(gm30, 1), np.nan)
    out["gshare_clean_30"] = gshare30

    lost1 = (s1c == 1) & (s1won == 0)
    won1 = (s1c == 1) & (s1won == 1)
    out["comeback_resid_100"] = GP.sum(lost1 * (won - q1), 100) / (GP.sum(lost1.astype(float), 100) + 4)
    out["closeout_resid_100"] = GP.sum(won1 * (won - q2), 100) / (GP.sum(won1.astype(float), 100) + 6)
    s = sr(GP.sum(sw, 100), GP.sum(cs_all, 100), 0.5, 10)
    out["rebound_set_100"] = sr(GP.sum(rp_a, 100), GP.sum(rp_n, 100), s, 4) - s
    dwn = full * nr
    out["dec_win_resid_100"] = GP.sum(dwn * (won - e_row), 100) / (GP.sum(dwn, 100) + 5)
    p0 = np.where(wom & itf2, 0.26, np.where(ct2, 0.31, 0.28))
    out["dec_share_50"] = sr(GP.sum(dwn, 50), GP.sum(nr, 50), p0, 8)
    p0 = 0.10 if women else 0.15
    out["tb_play_rate_100"] = sr(GP.sum(tbp, 100), GP.sum(ncs, 100), p0, 15)
    out["tb_win_shr_100"] = (GP.sum(tbwon, 100) + 4) / (GP.sum(tbp, 100) + 8)
    out["close_set_win_100"] = (GP.sum(clwon, 100) + 4) / (GP.sum(clp, 100) + 8)
    nw_ = won * nr
    p0 = np.where(ct2, 0.66, 0.72)
    out["straight_win_share_50"] = sr(GP.sum(nw_ * (sl == 0), 50), GP.sum(nw_, 50), p0, 6)
    nl_ = lostm * nr
    p0 = np.where(wom & itf2, 0.27, np.where(ch2, 0.34, 0.30))
    out["loss_resist_50"] = sr(GP.sum(nl_ * (sw >= 1), 50), GP.sum(nl_, 50), p0, 6)
    out["dom_gl_per_set_won_50"] = sr(GP.sum(dom_gl, 50), GP.sum(dom_n, 50), 3.1, 8)
    out["fight_gw_per_set_lost_50"] = sr(GP.sum(fight_gw, 50), GP.sum(fight_n, 50), 3.2, 8)
    p0 = 0.12 if women else 0.07
    out["blowout_for_rate_100"] = sr(GP.sum(bf_for, 100), GP.sum(ncs, 100), p0, 15)
    out["blowout_against_rate_100"] = sr(GP.sum(bf_ag, 100), GP.sum(ncs, 100), p0, 15)

    nm = nwin(40)
    w40 = sr(GP.sum(won, 40), nm, 0.5, 4)
    g40 = sr(GP.sum(gw, 40), GP.sum(games, 40), 0.5, 40)
    ok = (nm >= 5) & tr
    la, lb = fit_logit(g40[ok].astype(float) - 0.5, w40[ok].astype(float)) if ok.sum() >= 5000 else FALLBACK["luck"]
    out["luck_match_40"] = np.where(nm >= 5, w40 - sigmoid(la + lb * (g40 - 0.5)), 0.0)
    sset = sr(GP.sum(sw, 40), GP.sum(cs_all, 40), 0.5, 8)
    out["luck_set_40"] = np.where(nm >= 5, sset - np.clip(0.5 + 1.7 * (g40 - 0.5), 0, 1), 0.0)

    share = np.where(games >= 8, gw / np.maximum(games, 1), np.nan)
    out["gshare_std_20"], out["gshare_q25_20"] = window_share_stats(GP, share, 20, 6)

    g_all = np.where(GP.sum(games, 50) > 0, GP.sum(gw, 50) / np.maximum(GP.sum(games, 50), 1), 0.5)
    for nm_, fl in (("up", opp_elo > own_elo), ("down", opp_elo < own_elo)):
        out[f"gshare_{nm_}_50"] = sr(GP.sum(gw * fl, 50), GP.sum(games * fl, 50), g_all, 30)

    out["fs_win_resid_50"] = GP.sum(s1c * (s1won - p1s), 50) / (GP.sum(s1c, 50) + 5)
    r1w, r1 = GP.sum(g1w, 50), GP.sum(g1, 50)
    out["start_delta_50"] = sr(r1w, r1, 0.5, 20) - sr(GP.sum(gw, 50) - r1w, GP.sum(games, 50) - r1, 0.5, 20)
    out["mtb_win_career"] = (GP.sum(mtb_won) + 1.5) / (GP.sum(mtb_pl) + 3)
    out["ret_me_rate_100"] = sr(GP.sum(retL, 100), nwin(100), 0.03, 25)
    out["prev_ret_me"] = np.nan_to_num(GP.lag(retL))

    flag_s = retL[GP.idx] > 0
    sret_pos = np.maximum.accumulate(np.where(flag_s, GP.pos, -1))
    prev_pos = np.r_[-1, sret_pos[:-1]]
    has = (prev_pos >= GP.first) & (prev_pos >= 0)
    d_ret = np.where(has, dayL[GP.idx] - dayL[GP.idx][np.maximum(prev_pos, 0)], 60)
    tmp = np.empty(2 * n)
    tmp[GP.idx] = np.clip(d_ret, 0, 60)
    out["days_since_own_ret"] = tmp
    out["prev_win_by_ret"] = np.nan_to_num(GP.lag(won * ret_any))

    tid, _ = pd.factorize(M["tourney_id"].values)
    GE = Grp(pc.astype(np.int64) * (tid.max() + 1) + two(tid, tid), ordL)
    eg = GE.sum(load)
    out["event_games"] = eg
    out["event_deciders"] = GE.sum(full)
    egm = GE.sum(games)
    out["event_gshare_delta"] = sr(GE.sum(gw), egm, np.nan_to_num(gshare30, nan=0.5), 12) - np.nan_to_num(gshare30, nan=0.5)

    out["prev_games"] = GP.lag(load)
    out["prev_gshare_clean"] = GP.lag(np.where(games > 0, gw / np.maximum(games, 1), np.nan))

    def unsort(a):
        o = np.empty(len(a))
        o[GP.idx] = a
        return o

    kd = pc.astype(np.int64) * 1_000_000 + dayL
    kds = kd[GP.idx]
    ds, ls = dayL[GP.idx], load[GP.idx].astype(np.float64)
    heavy = np.where(boL == 5, load >= 40, load >= 28).astype(float) + 0.1 * np.maximum(0, load - 22)
    hs = heavy[GP.idx]
    cs_load = np.concatenate(([0.0], np.cumsum(ls)))
    cs_h = np.concatenate(([0.0], np.cumsum(hs)))
    left = {}

    def L_(j):
        if j not in left:
            left[j] = np.searchsorted(kds, kds - j, "left")
        return left[j]

    def decay(days, rate):
        tot = cs_load[GP.pos] - cs_load[L_(0)]
        for d in range(1, days + 1):
            tot = tot + rate ** d * (cs_load[L_(d - 1)] - cs_load[L_(d)])
        return tot

    day_feats = {"games_3d": cs_load[L_(0)] - cs_load[L_(3)], "games_10d": cs_load[L_(0)] - cs_load[L_(10)],
                 "fatigue_games_decay3d": decay(3, 0.75), "fatigue_games_decay7d": decay(7, 0.85),
                 "heavy_matches_7d": cs_h[GP.pos] - cs_h[L_(6)]}
    # live-appended rows can have a date earlier than rows before them in 'order'; the sorted-key searchsorted is
    # then invalid for that player only, so those players are recomputed exactly (earlier order AND day window)
    bad = np.nonzero(kds[1:] < kds[:-1])[0] + 1
    for g0 in np.unique(GP.first[bad]):
        g1 = g0 + np.searchsorted(GP.first[g0:], g0 + 1, "left")
        dd = ds[g0:g1, None] - ds[None, g0:g1]
        prior = np.tri(g1 - g0, k=-1, dtype=bool)
        lg_, hg_ = ls[g0:g1], hs[g0:g1]
        w_in = lambda lo, hi: prior & (dd >= lo) & (dd <= hi)
        day_feats["games_3d"][g0:g1] = w_in(1, 3) @ lg_
        day_feats["games_10d"][g0:g1] = w_in(1, 10) @ lg_
        for nm_, days, rate in (("fatigue_games_decay3d", 3, 0.75), ("fatigue_games_decay7d", 7, 0.85)):
            day_feats[nm_][g0:g1] = (w_in(0, days) * float(rate) ** np.maximum(dd, 0)) @ lg_
        day_feats["heavy_matches_7d"][g0:g1] = w_in(0, 6) @ hg_
    for k, v in day_feats.items():
        out[k] = unsort(v)
    out["prev_marathon"] = np.nan_to_num(GP.lag(mar))

    out["games_per_set_30"] = sr(GP.sum(games * nr, 30), GP.sum(ncs * nr, 30), 9.5, 4)

    ok_m = 1 - ret_any
    sel = np.nonzero(ok_m[GP.idx] > 0)[0]
    cnt_before = np.cumsum(ok_m[GP.idx]) - ok_m[GP.idx]
    first_cnt = cnt_before[GP.first]
    lo_n = np.maximum(cnt_before - 50, first_cnt)
    cw = np.concatenate(([0.0], np.cumsum((won * ok_m)[GP.idx][sel])))
    wn = cw[cnt_before.astype(int)] - cw[lo_n.astype(int)]
    nn = cnt_before - lo_n
    res = np.where(nn >= 3, (wn + 1.5) / (nn + 3), 0.5)
    out["win_clean_50"] = unsort(res)

    pair = np.minimum(wc, lc).astype(np.int64) * (pc.max() + 1) + np.maximum(wc, lc)
    GH = Grp(pair, order)
    gsw = np.where(f("gw") + f("gl") > 0, f("gw") / np.maximum(f("gw") + f("gl"), 1), np.nan)
    prev_g, prev_w = GH.lag(gsw), GH.lag(wc.astype(float))
    h_w = np.where(prev_w == wc, prev_g, 1 - prev_g) - 0.5
    h_l = np.where(prev_w == lc, prev_g, 1 - prev_g) - 0.5
    h_w[np.isnan(prev_g)] = np.nan
    h_l[np.isnan(prev_g)] = np.nan
    out["h2h_last_gshare"] = two(h_w, h_l)

    mainL = two(M["main"].values, M["main"].values).astype(np.int64)
    GM = Grp(pc.astype(np.int64) * 2 + mainL, ordL)
    cm = GM.count(30)
    out["lvl_gshare_clean_30"] = np.where(cm > 0, sr(GM.sum(gw, 30), GM.sum(games, 30), 0.5, 24), np.nan)

    res = {}
    for k, v in out.items():
        res[f"w_{k}"], res[f"l_{k}"] = v[:n], v[n:]
    return pd.DataFrame(res, index=M.index)


if __name__ == "__main__":
    M = pd.read_parquet(sys.argv[1])
    name = sys.argv[1].lower()
    compute(M, True if "wta" in name else False if "atp" in name else None).to_parquet(sys.argv[2])
