"""Rating-system, opponent-quality and common-opponent features from results only (no odds, no stats).

Every value for a match is recorded before that match touches any state and only uses rows with a smaller
'order' (vectorised parts use exclusive per-group cumulative sums over (group, order); loops record first, then update).
Columns are w_<name> / l_<name> = value for the match winner / loser (float32), same index as the input.
Needs columns: winner_id loser_id date surface tourney_level score retired main order (rows sorted by order).
Time windows use the running maximum of 'date', so a row dated before an earlier row (live tail) counts as that later day.

usage: fx_ratings.py MATCHES_PARQUET OUT_PARQUET
"""
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import os
import re
import sys
from collections import deque
from math import erfc, exp, log, log1p, pi, sqrt

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import ndtr

LN2 = log(2.0)
BIG = 10 ** 6
SC = 173.7178
SURF = {"Hard": 0, "Clay": 1, "Grass": 2, "Carpet": 3}
LG1 = {"10", "15", "25", "S"}
LG2 = {"35", "35+H", "40", "50", "50+H", "60", "75", "80", "100"}
_G = {}


# ---------------------------------------------------------------- inputs
_score_cache = {}


def parse_score(s):
    """(games won, games lost, sets with a tiebreak, completed sets) over completed non-match-tiebreak sets."""
    if s in _score_cache:
        return _score_cache[s]
    gw = gl = tb = cs = 0
    for a, b in re.findall(r"(\d+)-(\d+)", s.split("RET")[0]):
        a, b = int(a), int(b)
        hi, lo = max(a, b), min(a, b)
        if hi >= 10 or hi < 6 or not (hi - lo >= 2 or hi == 7):
            continue
        gw += a; gl += b; cs += 1
        tb += hi == 7 and lo == 6
    _score_cache[s] = (gw, gl, tb, cs)
    return _score_cache[s]


def _prep(M):
    n = len(M)
    codes, ids = pd.factorize(np.r_[M.winner_id.values, M.loser_id.values])
    day = np.maximum.accumulate(M["date"].values.astype("datetime64[D]").astype(np.int64))  # live tail rows can be out of date order
    lvl = M["tourney_level"].astype(str)
    sc_codes, uniq = pd.factorize(M["score"].fillna("").astype(str))
    ps = np.array([parse_score(s) for s in uniq], dtype=float).reshape(-1, 4)[sc_codes]
    gtot = ps[:, 0] + ps[:, 1]
    md = pd.DatetimeIndex(day.astype("datetime64[D]"))
    G = dict(
        n=n, nP=int(codes.max()) + 1, w=codes[:n], l=codes[n:], ids=np.asarray(ids),
        day=day,
        surf=M["surface"].astype(str).str.capitalize().map(SURF).fillna(0).astype(int).values,
        lg=np.where(lvl.isin(LG1), 0, np.where(lvl.isin(LG2), 1, np.where(lvl == "C", 2, 3))),
        ret=M["retired"].values.astype(bool), main=M["main"].values.astype(int),
        offs=(md.month * 100 + md.day >= 1015),
        mg=np.where(gtot > 0, (ps[:, 0] - ps[:, 1]) / np.maximum(gtot, 1), np.nan), tbs=ps[:, 2], cmp=ps[:, 3])
    return G


def _std_elo(G):
    """Standard Elo exactly as features2.ratings (K=250/(n+5)^0.4, retirements x0.5), recorded pre/post match."""
    n, nP = G["n"], G["nP"]
    w, l, ret = G["w"].tolist(), G["l"].tolist(), G["ret"].tolist()
    E, N = [1500.0] * nP, [0] * nP
    ew, el, pw, ewp, elp, nw, nl = ([0.0] * n for _ in range(7))
    for i in range(n):
        a, b = w[i], l[i]
        ra, rb, na, nb = E[a], E[b], N[a], N[b]
        e = 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))
        m = 0.5 if ret[i] else 1.0
        ra2 = ra + m * 250.0 / (na + 5) ** 0.4 * (1 - e)
        rb2 = rb - m * 250.0 / (nb + 5) ** 0.4 * (1 - e)
        E[a], E[b], N[a], N[b] = ra2, rb2, na + 1, nb + 1
        ew[i], el[i], pw[i], ewp[i], elp[i], nw[i], nl[i] = ra, rb, e, ra2, rb2, na, nb
    G.update(ew=np.array(ew), el=np.array(el), pw=np.array(pw), ewp=np.array(ewp), elp=np.array(elp),
             nw=np.array(nw), nl=np.array(nl))


def _split(prefix_pairs):
    out = {}
    for name, (a, b) in prefix_pairs.items():
        out["w_" + name] = np.asarray(a, dtype=np.float32); out["l_" + name] = np.asarray(b, dtype=np.float32)
    return out


# ---------------------------------------------------------------- vectorised per-group windows
class Long:
    """Rows sorted by (group key, order) with exclusive per-group cumulative sums and day-window lookups."""

    def __init__(self, key, order, day):
        self.srt = np.lexsort((order, key))
        self.inv = np.empty_like(self.srt); self.inv[self.srt] = np.arange(len(key))
        k = key[self.srt]
        brk = np.r_[0, np.flatnonzero(k[1:] != k[:-1]) + 1]
        lens = np.diff(np.r_[brk, len(k)])
        self.start = np.repeat(brk, lens)
        self.gid = np.repeat(np.arange(len(brk)), lens)
        self.ar = np.arange(len(k))
        self.pos = self.ar - self.start
        self.day = day[self.srt]
        self.d0 = self.day[self.start]
        self.dkey = self.gid.astype(np.int64) * BIG + self.day
        self._lo = {}

    def g(self, v):
        return np.asarray(v)[self.srt]

    def back(self, v):
        return v[self.inv]

    def excl(self, v):
        v = np.nan_to_num(np.asarray(v, dtype=float))
        c = pd.Series(v).groupby(self.gid).cumsum().to_numpy()
        ex = np.r_[0.0, c[:-1]]
        ex[self.pos == 0] = 0.0
        return ex

    def lo(self, window):
        if window is None:
            return self.start
        if window not in self._lo:
            self._lo[window] = np.searchsorted(self.dkey, self.dkey - window, side="left")
        return self._lo[window]

    def rows(self, v, k):
        ex = self.excl(v)
        return ex - ex[np.maximum(self.ar - k, self.start)]

    def dsum(self, v, window):
        ex = self.excl(v)
        return ex - ex[self.lo(window)]

    def wdsum(self, v, lam, window=None):
        sc = np.exp(lam * (self.day - self.d0))
        ex = self.excl(np.nan_to_num(v) * sc)
        return (ex - ex[self.lo(window)]) / sc

    def rows_decay(self, v, k, halflife):
        sc = np.exp(LN2 / halflife * self.pos)
        ex = self.excl(np.nan_to_num(v) * sc)
        return (ex - ex[np.maximum(self.ar - k, self.start)]) / sc

    def shift(self, v, k):
        out = np.full(len(v), np.nan)
        out[k:] = v[:-k]
        out[self.pos < k] = np.nan
        return out


# ---------------------------------------------------------------- Glicko-2
def _f(x, a, d2, phi2, v):
    ex = exp(x)
    return ex * (d2 - phi2 - v - ex) / (2.0 * (phi2 + v + ex) ** 2) - (x - a) * 4.0


def _g2_update(mu, phi, sig, muo, phio, s):
    g = 1.0 / sqrt(1.0 + 3.0 * phio * phio / (pi * pi))
    E = 1.0 / (1.0 + exp(-g * (mu - muo)))
    v = 1.0 / (g * g * E * (1.0 - E))
    delta = v * g * (s - E)
    phi2, d2 = phi * phi, delta * delta
    a = log(sig * sig)
    A = a
    if d2 > phi2 + v:
        B = log(d2 - phi2 - v)
    else:
        k = 1
        while _f(a - k * 0.5, a, d2, phi2, v) < 0 and k < 100:
            k += 1
        B = a - k * 0.5
    fA, fB = _f(A, a, d2, phi2, v), _f(B, a, d2, phi2, v)
    for _ in range(80):
        if abs(B - A) <= 1e-5:
            break
        C = A + (A - B) * fA / (fB - fA)
        fC = _f(C, a, d2, phi2, v)
        if fC * fB <= 0:
            A, fA = B, fB
        else:
            fA /= 2.0
        B, fB = C, fC
    sig2 = exp(A / 2.0)
    phin = 1.0 / sqrt(1.0 / (phi2 + sig2 * sig2) + 1.0 / v)
    return mu + phin * phin * g * (s - E), phin, sig2


def _task_g2():
    G = _G
    n, nP = G["n"], G["nP"]
    w, l, day, sf, ret = G["w"].tolist(), G["l"].tolist(), G["day"].tolist(), G["surf"].tolist(), G["ret"].tolist()
    phi0 = 350.0 / SC
    mu, phi, sig, last = [0.0] * nP, [phi0] * nP, [0.06] * nP, [-1] * nP
    S = {}
    R = np.zeros((n, 8))
    for i in range(n):
        a, b, d, s = w[i], l[i], day[i], sf[i]
        pa = phi[a] if last[a] < 0 else min(sqrt(phi[a] ** 2 + 0.005 * (d - last[a])), phi0)
        pb = phi[b] if last[b] < 0 else min(sqrt(phi[b] ** 2 + 0.005 * (d - last[b])), phi0)
        ma, mb = mu[a], mu[b]
        ka, kb = a * 4 + s, b * 4 + s
        sa = S.get(ka)
        sa = [ma, max(pa, 1.0), 0.06, d] if sa is None else sa
        sb = S.get(kb)
        sb = [mb, max(pb, 1.0), 0.06, d] if sb is None else sb
        psa = min(sqrt(sa[1] ** 2 + 0.005 * (d - sa[3])), phi0)
        psb = min(sqrt(sb[1] ** 2 + 0.005 * (d - sb[3])), phi0)
        R[i] = (ma, mb, pa, pb, sig[a], sig[b], (sa[0] * SC + 1500) - 2 * psa * SC, (sb[0] * SC + 1500) - 2 * psb * SC)
        sw, sl = (0.75, 0.25) if ret[i] else (1.0, 0.0)
        mu[a], phi[a], sig[a] = _g2_update(ma, pa, sig[a], mb, pb, sw)
        mu[b], phi[b], sig[b] = _g2_update(mb, pb, sig[b], ma, pa, sl)
        x = _g2_update(sa[0], psa, sa[2], sb[0], psb, sw)
        y = _g2_update(sb[0], psb, sb[2], sa[0], psa, sl)
        S[ka] = [x[0], x[1], x[2], d]
        S[kb] = [y[0], y[1], y[2], d]
        last[a] = last[b] = d
    mw, ml, pw_, pl_, vw, vl, sw_, sl_ = R.T
    rw, rl = mw * SC + 1500, ml * SC + 1500
    dw, dl = pw_ * SC, pl_ * SC
    z = (rw - rl) / np.sqrt(dw ** 2 + dl ** 2)
    gg = 1.0 / np.sqrt(1.0 + 3.0 * (pw_ ** 2 + pl_ ** 2) / pi ** 2)
    p = 1.0 / (1.0 + np.exp(-gg * (mw - ml)))
    return _split({"g2_rating": (rw, rl), "g2_rd": (dw, dl), "g2_lb": (rw - 2 * dw, rl - 2 * dl),
                   "g2_vol": (vw, vl), "g2_surf_lb": (sw_, sl_), "g2_z": (z, -z), "g2_prob": (p, 1 - p)})


# ---------------------------------------------------------------- TrueSkill and level-aware Elo variants
def _task_ts_lvl():
    G = _G
    n, nP = G["n"], G["nP"]
    w, l, day, ret, lgc = G["w"].tolist(), G["l"].tolist(), G["day"].tolist(), G["ret"].tolist(), G["lg"].tolist()
    nw, nl, ew = G["nw"].tolist(), G["nl"].tolist(), G["ew"].tolist()
    s0 = 25.0 / 3
    v0, b2, t2 = s0 * s0, (s0 / 2) ** 2, (s0 / 100) ** 2
    tmu, tvar, tlast = [25.0] * nP, [v0] * nP, [-1] * nP
    Eo = [1500.0] * nP
    El = [None] * nP
    Sg = [1500.0] * 4
    LGE, LGN = {}, {}
    T = np.zeros((n, 4)); O = np.zeros((n, 2)); V = np.zeros((n, 3)); Q = np.zeros((n, 2))
    for i in range(n):
        a, b, d, g = w[i], l[i], day[i], lgc[i]
        na, nb = nw[i], nl[i]
        m = 0.5 if ret[i] else 1.0
        ka, kb = 250.0 / (na + 5) ** 0.4, 250.0 / (nb + 5) ** 0.4
        # TrueSkill
        va = tvar[a] if tlast[a] < 0 else min(tvar[a] + t2 + 0.00048 * (d - tlast[a]), v0)
        vb = tvar[b] if tlast[b] < 0 else min(tvar[b] + t2 + 0.00048 * (d - tlast[b]), v0)
        ma, mb = tmu[a], tmu[b]
        T[i] = (ma, mb, sqrt(va), sqrt(vb))
        c2 = 2 * b2 + va + vb
        c = sqrt(c2)
        t = (ma - mb) / c
        vv = -t if t < -30 else exp(-0.5 * t * t) / sqrt(2 * pi) / (0.5 * erfc(-t / sqrt(2.0)))
        ww = vv * (vv + t)
        tmu[a], tmu[b] = ma + va / c * vv, mb - vb / c * vv
        tvar[a], tvar[b] = va * (1 - va / c2 * ww), vb * (1 - vb / c2 * ww)
        tlast[a] = tlast[b] = d
        # opponent-experience weighted Elo
        ra, rb = Eo[a], Eo[b]
        O[i] = (ra, rb)
        e = 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))
        Eo[a] = ra + m * ka * min(1.0, (nb + 2) / 15.0) * (1 - e)
        Eo[b] = rb - m * kb * min(1.0, (na + 2) / 15.0) * (1 - e)
        # Elo whose debutants start at the level-group mean - 75
        sg = Sg[g]
        ra = El[a] if El[a] is not None else sg - 75.0
        rb = El[b] if El[b] is not None else sg - 75.0
        V[i] = (ra, rb, sg)
        e = 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))
        El[a] = ra + m * ka * (1 - e)
        El[b] = rb - m * kb * (1 - e)
        if na >= 10:
            sg += 0.0005 * (ra - sg)
        if nb >= 10:
            sg += 0.0005 * (rb - sg)
        Sg[g] = sg
        # Elo per level group, initialised from the overall Elo
        qa, qb = a * 4 + g, b * 4 + g
        ra = LGE.get(qa)
        ra = ew[i] if ra is None else ra
        rb = LGE.get(qb)
        rb = G["el"][i] if rb is None else rb
        Q[i] = (ra, rb)
        e = 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))
        ca, cb = LGN.get(qa, 0), LGN.get(qb, 0)
        LGE[qa] = ra + m * 250.0 / (ca + 5) ** 0.4 * (1 - e)
        LGE[qb] = rb - m * 250.0 / (cb + 5) ** 0.4 * (1 - e)
        LGN[qa], LGN[qb] = ca + 1, cb + 1
    ew_, el_ = G["ew"], G["el"]
    dm = T[:, 0] - T[:, 1]
    z = dm / np.sqrt(T[:, 2] ** 2 + T[:, 3] ** 2)
    p = ndtr(dm / np.sqrt(2 * b2 + T[:, 2] ** 2 + T[:, 3] ** 2))
    return _split({
        "ts_mu": (T[:, 0], T[:, 1]), "ts_sigma": (T[:, 2], T[:, 3]), "ts_z": (z, -z), "ts_prob": (p, 1 - p),
        "ow_elo": (O[:, 0], O[:, 1]), "lvl_elo": (V[:, 0], V[:, 1]),
        "elo_vs_level_mean": (ew_ - V[:, 2], el_ - V[:, 2]),
        "lg_elo": (Q[:, 0], Q[:, 1]), "lg_elo_minus_elo": (Q[:, 0] - ew_, Q[:, 1] - el_)})


RHO = [[1, .75, .65, .85], [.75, 1, .5, .6], [.65, .5, 1, .75], [.85, .6, .75, 1]]


def _task_cs_layoff_mf():
    G = _G
    n, nP = G["n"], G["nP"]
    w, l, day, ret, sf = G["w"].tolist(), G["l"].tolist(), G["day"].tolist(), G["ret"].tolist(), G["surf"].tolist()
    main, offs = G["main"].tolist(), G["offs"].tolist()
    nw, nl = G["nw"].tolist(), G["nl"].tolist()
    ewl, ell = G["ew"].tolist(), G["el"].tolist()
    CS = [None] * nP
    LO, llast, loff, since = [None] * nP, [-1] * nP, [False] * nP, [99] * nP
    init = np.array([np.random.default_rng((7, int(p))).normal(0, 0.1, 8) for p in G["ids"]])  # keyed by player id, not data span
    U, Vv = np.ascontiguousarray(init[:, :4]), np.ascontiguousarray(init[:, 4:])
    cm = log(10.0) / 400.0
    C = np.zeros((n, 2)); Lo = np.zeros((n, 2)); F = np.zeros(n)
    for i in range(n):
        a, b, d, s = w[i], l[i], day[i], sf[i]
        na, nb = nw[i], nl[i]
        m = 0.5 if ret[i] else 1.0
        # cross-surface Elo
        xa = CS[a] = CS[a] or [1500.0] * 4
        xb = CS[b] = CS[b] or [1500.0] * 4
        C[i] = (xa[s], xb[s])
        e = 1.0 / (1.0 + 10.0 ** ((xb[s] - xa[s]) / 400.0))
        fa, fb = m * 250.0 / (na + 5) ** 0.4 * (1 - e), m * 250.0 / (nb + 5) ** 0.4 * (1 - e)
        rho = RHO[s]
        for j in range(4):
            xa[j] += rho[j] * fa
            xb[j] -= rho[j] * fb
        # layoff-aware Elo
        for p in (a, b):
            if LO[p] is None:
                LO[p] = 1500.0 if main[i] else 1250.0
            elif d - llast[p] >= 56 and not loff[p]:
                LO[p] -= min(100.0 + 1.5 * (d - llast[p] - 56) / 7.0, 200.0)
                since[p] = 0
        ra, rb = LO[a], LO[b]
        Lo[i] = (ra, rb)
        e = 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))
        LO[a] = ra + m * 250.0 / (na + 5) ** 0.4 * (1 + 0.55 * max(0.0, 1 - since[a] / 20.0)) * (1 - e)
        LO[b] = rb - m * 250.0 / (nb + 5) ** 0.4 * (1 + 0.55 * max(0.0, 1 - since[b] / 20.0)) * (1 - e)
        since[a] += 1; since[b] += 1
        llast[a] = llast[b] = d
        loff[a] = loff[b] = offs[i]
        # latent matchup factors (logit = Elo term + u_i.v_j - u_j.v_i), one SGD step after the match
        ua, va, ub, vb = U[a], Vv[a], U[b], Vv[b]
        x = ua @ vb - ub @ va
        F[i] = x if (na >= 5 and nb >= 5) else 0.0
        gr = 1.0 - 1.0 / (1.0 + exp(-(cm * (ewl[i] - ell[i]) + x)))
        da, db_ = gr * vb - 0.05 * ua, -gr * va - 0.05 * ub
        dva, dvb = -gr * ub - 0.05 * va, gr * ua - 0.05 * vb
        U[a] = ua + 0.05 * da; U[b] = ub + 0.05 * db_
        Vv[a] = va + 0.05 * dva; Vv[b] = vb + 0.05 * dvb
    return _split({"cs_elo": (C[:, 0], C[:, 1]), "layoff_elo": (Lo[:, 0], Lo[:, 1]), "mf_latent_matchup": (F, -F)})


# ---------------------------------------------------------------- opponents' current Elo vs Elo when they played
def _task_hind():
    G = _G
    n, nP = G["n"], G["nP"]
    w, l = G["w"].tolist(), G["l"].tolist()
    ew, el, ewp, elp = G["ew"].tolist(), G["el"].tolist(), G["ewp"].tolist(), G["elp"].tolist()
    Ec = np.full(nP, 1500.0)
    ho, ht = [[] for _ in range(nP)], [[] for _ in range(nP)]
    H = np.full((n, 4), np.nan)
    for i in range(n):
        a, b = w[i], l[i]
        for j, p in ((0, a), (2, b)):
            h = ho[p]
            if h:
                dif = Ec[h[-30:]] - np.array(ht[p][-30:])
                H[i, j] = dif.mean()
                H[i, j + 1] = dif[-10:].mean()
        ho[a].append(b); ht[a].append(el[i])
        ho[b].append(a); ht[b].append(ew[i])
        Ec[a], Ec[b] = ewp[i], elp[i]
    return _split({"hind_30": (H[:, 0], H[:, 2]), "hind_10": (H[:, 1], H[:, 3])})


# ---------------------------------------------------------------- common opponents and two-hop chains
def _task_co():
    G = _G
    n, nP = G["n"], G["nP"]
    w, l, day, mgl, pwl = G["w"].tolist(), G["l"].tolist(), G["day"].tolist(), G["mg"].tolist(), G["pw"].tolist()
    dq, by = [None] * nP, [None] * nP
    k180, k270, k365 = LN2 / 180.0, LN2 / 270.0, 1.0 / 365.0
    out = np.zeros((n, 5))

    def window(p, lim):
        q = dq[p]
        if q is None:
            return None
        b = by[p]
        while q and q[0][0] < lim:
            o = q.popleft()[1]
            r = b[o]
            r.popleft()
            if not r:
                del b[o]
        return b

    for i in range(n):
        a, b, d = w[i], l[i], day[i]
        lim = d - 730
        A, B = window(a, lim), window(b, lim)
        if A and B:
            small, big = (A, B) if len(A) < len(B) else (B, A)
            common = [c for c in small if c in big]
            nc = len(common)
            if nc:
                out[i, 0] = log1p(nc)
                wd = md = ed = ws = es = 0.0
                c1 = c2 = 0.0
                for c in common:
                    la, lb = A[c], B[c]
                    st = []
                    for lst in (la, lb):
                        sw = sn = ms = mw = rs = rw = 0.0
                        mn = 0
                        wins, losses = [], []
                        for dd, won, mg, r in lst:
                            age = d - dd
                            x = exp(-age * k180)
                            sw += x; sn += x * won
                            if mg == mg:
                                ms += x * mg; mw += x; mn += 1
                            y = exp(-age * k270)
                            rs += y * r; rw += y
                            (wins if won else losses).append(age)
                        st.append(((sn + 0.5) / (sw + 1.0), ms / mw if mw else None, mn, rs / rw, len(lst), wins, losses))
                    sa, sb = st
                    wd += sa[0] - sb[0]
                    if sa[1] is not None and sb[1] is not None:
                        x = min(sa[2], sb[2], 3)
                        md += x * (sa[1] - sb[1]); ws += x
                    x = min(sa[4], sb[4], 3)
                    ed += x * (sa[3] - sb[3]); es += x
                    for u in sa[5]:
                        for v in sb[6]:
                            c1 += exp(-max(u, v) * k365)
                    for u in sb[5]:
                        for v in sa[6]:
                            c2 += exp(-max(u, v) * k365)
                if nc >= 2:
                    out[i, 1] = wd / nc
                    out[i, 2] = md / ws if ws else 0.0
                    out[i, 3] = ed / es if es else 0.0
                out[i, 4] = (c1 - c2) / (c1 + c2 + 2.0) if (c1 + c2) > 0 else 0.0
        mg, r = mgl[i], 1.0 - pwl[i]
        for p, o, won, sg in ((a, b, 1, 1.0), (b, a, 0, -1.0)):
            if dq[p] is None:
                dq[p], by[p] = deque(), {}
            dq[p].append((d, o))
            by[p].setdefault(o, deque()).append((d, won, sg * mg, sg * r))
    return _split({"co_count": (out[:, 0], out[:, 0]), "co_win_diff": (out[:, 1], -out[:, 1]),
                   "co_margin_diff": (out[:, 2], -out[:, 2]), "co_elo_adj_diff": (out[:, 3], -out[:, 3]),
                   "trans_2hop": (out[:, 4], -out[:, 4])})


# ---------------------------------------------------------------- PageRank snapshots
def _task_pr():
    G = _G
    n, nP = G["n"], G["nP"]
    w, l, day = G["w"], G["l"], G["day"]
    PRg = np.zeros(nP)
    loc = np.full(nP, -1)
    ln15 = log(0.15)
    prw, prl = np.full(n, ln15), np.full(n, ln15)
    pcw, pcl = np.full(n, 0.15), np.full(n, 0.15)
    ud, us = np.unique(day, return_index=True)
    ends = np.r_[us[1:], n]
    for d, s0, s1 in zip(ud, us, ends):
        lo = np.searchsorted(day, d - 365, side="left")
        if s0 <= lo:
            continue
        ww, ll = w[lo:s0], l[lo:s0]
        m = len(ww)
        wt = np.exp(-(d - day[lo:s0]) / 120.0)
        nodes, inv = np.unique(np.r_[ww, ll], return_inverse=True)
        N = len(nodes)
        iw, il = inv[:m], inv[m:]
        outw = np.bincount(il, weights=wt, minlength=N)
        P = sparse.csr_matrix((wt / outw[il], (iw, il)), shape=(N, N))
        dang = outw == 0
        x = PRg[nodes]
        x = np.where(x > 0, x, 1.0 / N)
        x = x / x.sum()
        for _ in range(30):
            x = 0.85 * (P @ x + x[dang].sum() / N) + 0.15 / N
        PRg[nodes] = x
        lo90 = np.searchsorted(day, d - 90, side="left") - lo
        act = np.zeros(N, bool)
        act[iw[lo90:]] = True; act[il[lo90:]] = True
        ref = np.sort(x[act])
        lv = np.log(x * N)
        loc[nodes] = np.arange(N)
        for ids, po, pc in ((w[s0:s1], prw, pcw), (l[s0:s1], prl, pcl)):
            j = loc[ids]
            f = j >= 0
            xs = x[j[f]]
            sl = slice(s0, s1)
            po[sl][f] = lv[j[f]]
            if len(ref):
                pc[sl][f] = (np.searchsorted(ref, xs, "left") + np.searchsorted(ref, xs, "right")) / (2.0 * len(ref))
        loc[nodes] = -1
    return _split({"pagerank_365": (prw, prl), "pagerank_pct90": (pcw, pcl)})


# ---------------------------------------------------------------- long-format vectorised features
def _task_long():
    G = _G
    n = G["n"]
    w, l, day, pw = G["w"], G["l"], G["day"], G["pw"]
    ew, el = G["ew"], G["el"]
    mi = np.r_[np.arange(n), np.arange(n)]
    pl, dayl = np.r_[w, l], np.r_[day, day]
    own_u, opp_u = np.r_[ew, el], np.r_[el, ew]
    won_u = np.r_[np.ones(n), np.zeros(n)]
    res_u = np.r_[1 - pw, -(1 - pw)]
    oppn_u = np.r_[G["nl"], G["nw"]]
    L = Long(pl, mi, dayl)
    own, opp, won, res = L.g(own_u), L.g(opp_u), L.g(won_u), L.g(res_u)
    pos, ar, st = L.pos, L.ar, L.start
    ones = np.ones(len(pos))
    first = pos == 0
    out = {}

    def put(name, v, lng=L):
        u = lng.back(v)
        out["w_" + name], out["l_" + name] = u[:n].astype(np.float32), u[n:].astype(np.float32)

    cnt20 = np.minimum(pos, 20)
    prov = L.rows((L.g(oppn_u) < 5).astype(float), 20) / np.maximum(cnt20, 1)
    put("prov_opp_20", np.where(first, np.nan, prov))

    oe10 = np.where(first, np.nan, L.rows(opp, 10) / np.maximum(np.minimum(pos, 10), 1))
    oth = L.inv[(L.srt + n) % (2 * n)]
    oe_o = oe10[oth]
    cv = L.rows((~np.isnan(oe_o)).astype(float), 10)
    put("sos2_10", np.where(cv > 0, L.rows(oe_o, 10) / np.maximum(cv, 1), np.nan))

    bw, wl = np.full(len(pos), np.nan), np.full(len(pos), np.nan)
    aw, al = np.where(won == 1, opp, np.nan), np.where(won == 0, opp, np.nan)
    for k in range(1, 21):
        bw, wl = np.fmax(bw, L.shift(aw, k)), np.fmin(wl, L.shift(al, k))
    put("bestwin_20", bw); put("worstloss_20", wl)

    for name, win in (("elo_chg_90d", 90), ("elo_chg_365d", 365)):
        j = np.searchsorted(L.dkey, L.dkey - win, side="right") - 1
        ref = np.where(j >= st, own[np.maximum(j, 0)], own[st])
        put(name, np.where(first, np.nan, own - ref))

    x = (L.day - L.d0) / 30.0
    y = own - 1500.0
    lo = L.lo(180)
    k = (ar - lo) + 1
    sx, sy = L.dsum(x, 180) + x, L.dsum(y, 180) + y
    sxy, sxx = L.dsum(x * y, 180) + x * y, L.dsum(x * x, 180) + x * x
    den = k * sxx - sx ** 2
    ok = (k >= 4) & (L.day - L.day[lo] >= 21) & (den > 1e-9)
    put("elo_slope_180d", np.where(ok, (k * sxy - sx * sy) / np.where(ok, den, 1.0), np.nan))

    dd = np.where(first, 0.0, own - np.r_[0.0, own[:-1]])
    lo20 = np.maximum(ar - 19, st + 1)
    c = np.minimum(pos, 20)
    ex = L.excl(dd)
    ex2 = L.excl(dd * dd)
    s1, s2 = ex + dd - ex[np.minimum(lo20, ar)], ex2 + dd * dd - ex2[np.minimum(lo20, ar)]
    cc = np.maximum(c, 1)
    var = np.maximum(s2 / cc - (s1 / cc) ** 2, 0.0)
    put("elo_vol_20", np.where(c >= 3, np.sqrt(var), np.nan))
    put("elo_frompeak", own - pd.Series(own).groupby(L.gid).cummax().to_numpy())

    lam = LN2 / 120.0
    put("ewm_resid_120", L.wdsum(res, lam, 730) / (L.wdsum(ones, lam, 730) + 3.0))

    for name, cond in (("vs_higher_resid", opp >= own + 50), ("vs_lower_resid", opp <= own - 50)):
        cf = cond.astype(float)
        ct = L.rows(cf, 40)
        num, den = L.rows_decay(cf * res, 40, 20), L.rows_decay(cf, 40, 20)
        put(name, np.where(ct >= 3, num / (den + 4.0), 0.0))

    cnt90 = L.dsum(ones, 90)
    lam = 1.0 / 45.0
    gap = L.wdsum(opp, lam, 90) / np.maximum(L.wdsum(ones, lam, 90), 1e-12) - own
    put("sos_gap_90d", np.where(cnt90 >= 2, gap, 0.0))

    lam = 1.0 / 180.0
    good = won * (opp >= own) - (1 - won) * (opp <= own - 100)
    put("quality_win_rate_365", L.wdsum(good, lam, 365) / (L.dsum(ones, 365) + 3.0))

    # z-score of opponent Elo against the level group's pooled Elo over the previous 365 days
    mean_m, sd_m, ok_m = np.zeros(n), np.ones(n), np.zeros(n, bool)
    for g in range(4):
        idx = np.flatnonzero(G["lg"] == g)
        if not len(idx):
            continue
        dg = day[idx]
        a, b = ew[idx] - 1500.0, el[idx] - 1500.0
        c1, c2 = np.r_[0.0, np.cumsum(a + b)], np.r_[0.0, np.cumsum(a * a + b * b)]
        lo_, hi_ = np.searchsorted(dg, dg - 365, side="left"), np.searchsorted(dg, dg, side="left")
        cn = 2.0 * (hi_ - lo_)
        mu_ = (c1[hi_] - c1[lo_]) / np.maximum(cn, 1)
        mean_m[idx] = mu_ + 1500.0
        sd_m[idx] = np.sqrt(np.maximum((c2[hi_] - c2[lo_]) / np.maximum(cn, 1) - mu_ ** 2, 1.0))
        ok_m[idx] = cn >= 40
    zl = np.where(np.r_[ok_m, ok_m], np.clip((opp_u - np.r_[mean_m, mean_m]) / np.r_[sd_m, sd_m], -3, 3), 0.0)
    wm = (won - 0.5) * L.g(zl)
    num = den = np.zeros(len(pos))
    for k in range(1, 11):
        v = (pos >= k) * 0.9 ** (k - 1)
        sh = L.shift(wm, k)
        num = num + v * np.nan_to_num(sh); den = den + v
    put("oqw_form_10", np.where(pos >= 3, num / np.maximum(den, 1e-12), 0.0))

    # same-band history: player x band of (own - opp Elo)
    band = np.searchsorted([-150.0, -50.0, 50.0, 150.0], own_u - opp_u, side="right")
    L2 = Long(pl * 5 + band, mi, dayl)
    r2, o2 = L2.g(res_u), np.ones(len(pl))
    lam = LN2 / 365.0
    c2_ = L2.dsum(o2, 1095)
    put("band_resid", np.where(c2_ >= 4, L2.wdsum(r2, lam, 1095) / (L2.wdsum(o2, lam, 1095) + 6.0), 0.0), L2)

    # residual history against opponents of the same tiebreak propensity bucket
    tb_u, cm_u = np.r_[G["tbs"], G["tbs"]], np.r_[G["cmp"], G["cmp"]]
    prop = (L.dsum(L.g(tb_u), 730) + 1.2) / (L.dsum(L.g(cm_u), 730) + 10.0)
    prop_u = L.back(prop)
    bucket = (prop_u[(np.arange(2 * n) + n) % (2 * n)] > 0.20).astype(int)
    L3 = Long(pl * 2 + bucket, mi, dayl)
    r3, o3 = L3.g(res_u), np.ones(len(pl))
    c3 = L3.dsum(o3, 730)
    put("res_vs_high_tb_opp", np.where(c3 >= 5, L3.dsum(r3, 730) / (c3 + 5.0), 0.0), L3)

    # head to head
    lo_id, hi_id = np.minimum(w, l), np.maximum(w, l)
    pk = lo_id.astype(np.int64) * G["nP"] + hi_id
    wl_ = w == lo_id
    won_lo = wl_.astype(float)
    r_lo = won_lo - np.where(wl_, pw, 1 - pw)
    Lp = Long(pk, np.arange(n), day)
    Ls = Long(pk * 4 + G["surf"], np.arange(n), day)
    h2h = Lp.back(Lp.excl(Lp.g(r_lo)) / (Lp.pos + 2.0))
    out["w_h2h_resid"], out["l_h2h_resid"] = np.where(wl_, h2h, -h2h).astype(np.float32), np.where(wl_, -h2h, h2h).astype(np.float32)
    lam = log(1.25) / 365.25
    o_ = np.ones(n)
    sa = Ls.back(Ls.wdsum(Ls.g(won_lo), lam)); ss = Ls.back(Ls.wdsum(Ls.g(o_), lam))
    aa = Lp.back(Lp.wdsum(Lp.g(won_lo), lam)); sw_ = Lp.back(Lp.wdsum(Lp.g(o_), lam))
    p_lo = (0.6 * aa + 0.4 * sa + 0.5) / (0.6 * sw_ + 0.4 * ss + 1.0)
    pwin = np.where(wl_, p_lo, 1 - p_lo)
    out["w_h2h_decay_shrunk"], out["l_h2h_decay_shrunk"] = pwin.astype(np.float32), (1 - pwin).astype(np.float32)
    return out


def _task_sim():
    G = _G
    n = G["n"]
    w, l, day, pw, ew, el = G["w"], G["l"], G["day"], G["pw"], G["ew"], G["el"]
    mi = np.r_[np.arange(n), np.arange(n)]
    L = Long(np.r_[w, l], mi, np.r_[day, day])
    opp = L.g(np.r_[el, ew]); res = L.g(np.r_[1 - pw, -(1 - pw)])
    lo = L.lo(730)
    val = np.zeros(len(opp))
    for q in np.flatnonzero(L.ar - lo >= 5):
        s = slice(lo[q], q)
        m = np.abs(opp[s] - opp[q]) <= 100.0
        c = m.sum()
        if c >= 5:
            val[q] = res[s][m].sum() / (c + 5.0)
    u = L.back(val)
    return {"w_sim_opp_resid": u[:n].astype(np.float32), "l_sim_opp_resid": u[n:].astype(np.float32)}


TASKS = [_task_g2, _task_co, _task_pr, _task_ts_lvl, _task_cs_layoff_mf, _task_long, _task_hind, _task_sim]


def _run(i):
    return TASKS[i]()


def compute(M, workers=None):
    """Features for every row of M (rows in 'order'); returns a float32 frame with M's index.
    Self-play rows (winner_id == loser_id, a source data error) take no part in any state and get NaN."""
    bad = M["winner_id"].values == M["loser_id"].values
    if bad.any():
        return _compute(M[~bad], workers).reindex(M.index)
    return _compute(M, workers)


def _compute(M, workers=None):
    perm = np.argsort(M["order"].values, kind="stable")
    Ms = M.iloc[perm] if (perm != np.arange(len(M))).any() else M
    _G.clear()
    _G.update(_prep(Ms))
    _std_elo(_G)
    workers = workers or int(os.environ.get("FX_WORKERS", min(3, os.cpu_count() or 1)))
    if workers > 1:
        with ProcessPoolExecutor(workers, mp_context=mp.get_context("fork")) as pool:
            parts = list(pool.map(_run, range(len(TASKS))))
    else:
        parts = [_run(i) for i in range(len(TASKS))]
    cols = {}
    for p in parts:
        cols.update(p)
    out = pd.DataFrame(cols, index=Ms.index)
    return out.loc[M.index] if Ms is not M else out


if __name__ == "__main__":
    cols = ["winner_id", "loser_id", "date", "surface", "tourney_level", "score", "retired", "main", "order"]
    F = compute(pd.read_parquet(sys.argv[1], columns=cols))
    F.to_parquet(sys.argv[2])
    print(F.shape, "->", sys.argv[2])
