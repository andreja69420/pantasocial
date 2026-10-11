"""Context features (results only, strictly pre-match): event country, travel and time zones, schedule density and
load, win/loss streaks, layoffs, surface and level transitions, career stage, qualifying path and last year's
depth at the same event. Returns w_<name>/l_<name> columns for the match winner and loser. Only matches with a
smaller 'order' feed a row (exclusive cumulative sums over (group, order), searchsorted for day windows).

Event country: parsed from tourney_id, else learned from earlier editions of the normalised tourney_name, else
modal ioc of the players already seen in the same event (row-time, earlier-ordered matches only). Day-level
quantities use est_day = tourney_date + min(6, rord - first rord of earlier matches in the event); per player the
tourney_date and est_day sequences are made non-decreasing (running max) so live-appended rows with an older
date cannot reach back to rows that come later in 'order'.

usage: fx_context.py MATCHES_PARQUET OUT_PARQUET
"""
import re
import sys
from collections import Counter

import numpy as np
import pandas as pd

EPOCH = np.datetime64("1960-01-01")
TRAIN_END = int((np.datetime64("2020-01-01") - EPOCH).astype(np.int64))
CENSOR_DAY = int((np.datetime64("1990-01-01") - EPOCH).astype(np.int64))
BIG = 1 << 17
FINAL = 13.0
MIN_SHARE = 0.35
_TOK = re.compile(r"^\[?(\d+)-(\d+)\]?(?:\(\d+(?:-\d+)?\))?$")
_PARSE_ID = re.compile(r"^\d{4}-[MW]-[A-Z0-9]+-([A-Z]{3})-")

# code lat lon utc continent(EU AS AF NA SA OC); utc '-' = multi-zone, longitude/15 of the population centroid
_TABLE = """
EU FRA 46.6 2.2 1|ITA 42.8 12.5 1|ESP 40.2 -3.6 1|GER 51.1 10.4 1|GBR 54.0 -2.5 0|IRL 53.2 -8.1 0|POR 39.6 -8.0 0|NED 52.2 5.3 1
EU BEL 50.6 4.7 1|LUX 49.8 6.1 1|SUI 46.8 8.2 1|AUT 47.6 14.1 1|CZE 49.8 15.5 1|SVK 48.7 19.5 1|POL 52.1 19.4 1|HUN 47.2 19.4 1
EU ROU 45.9 25.0 2|BUL 42.7 25.5 2|GRE 39.1 22.9 2|CRO 45.1 15.2 1|SLO 46.1 14.8 1|SRB 44.0 20.8 1|BIH 44.2 17.8 1|MNE 42.8 19.3 1
EU MKD 41.6 21.7 1|ALB 41.1 20.0 1|KOS 42.6 20.9 1|UKR 49.0 31.4 2|BLR 53.5 28.0 3|MDA 47.2 28.5 2|LTU 55.3 23.9 2|LAT 56.9 24.9 2
EU EST 58.7 25.5 2|FIN 64.5 26.3 2|SWE 62.8 16.7 1|NOR 61.1 8.7 1|DEN 56.0 10.0 1|ISL 64.9 -18.6 0|MLT 35.9 14.4 1|CYP 35.0 33.3 2
EU MON 43.7 7.4 1|AND 42.5 1.6 1|SMR 43.9 12.5 1|LIE 47.2 9.6 1|RUS 56.0 45.0 -|TUR 39.0 35.2 3|GEO 42.2 43.5 4|ARM 40.3 44.9 4
EU AZE 40.3 47.7 4|ISR 31.4 35.0 2|URS 56.0 45.0 3|CIS 56.0 45.0 3|YUG 44.0 20.8 1|SCG 44.0 20.8 1|TCH 49.2 17.5 1
AS KAZ 48.0 70.0 -|UZB 41.4 64.6 5|KGZ 41.5 74.5 6|TJK 38.9 71.0 5|TKM 39.1 59.4 5|IRI 32.5 54.3 3.5|IRQ 33.0 43.7 3|SYR 35.0 38.5 2
AS LBN 33.9 35.9 2|JOR 31.2 36.5 2|PLE 31.9 35.2 2|KSA 24.0 45.0 3|KUW 29.3 47.6 3|QAT 25.3 51.2 3|UAE 24.3 54.2 4|OMA 21.0 57.0 4
AS BRN 26.0 50.6 3|YEM 15.9 47.6 3|PAK 30.0 69.3 5|IND 21.0 78.0 5.5|SRI 7.6 80.7 5.5|BAN 23.8 90.3 6|NEP 28.2 84.0 5.75
AS CHN 35.0 103.0 8|HKG 22.3 114.2 8|MAC 22.2 113.5 8|TPE 23.7 121.0 8|JPN 36.5 138.0 9|KOR 36.5 127.9 9|MGL 46.9 103.8 8
AS THA 15.0 101.0 7|VIE 16.0 107.0 7|LAO 18.0 103.0 7|CAM 12.6 104.9 7|MAS 4.2 102.0 8|SGP 1.35 103.8 8|INA -5.0 112.0 -|PHI 12.9 122.0 8|BRU 4.5 114.7 8
AF EGY 26.8 30.8 2|MAR 31.8 -7.1 1|ALG 28.0 1.7 1|TUN 34.0 9.0 1|LBA 27.0 17.0 2|RSA -29.0 24.0 2|NGR 9.1 8.7 1|GHA 7.9 -1.0 0
AF CIV 7.5 -5.5 0|SEN 14.5 -14.5 0|KEN 0.0 38.0 3|UGA 1.4 32.3 3|TAN -6.4 34.9 3|RWA -1.9 29.9 2|BDI -3.4 29.9 2|ZIM -19.0 29.2 2
AF ZAM -13.1 27.8 2|BOT -22.3 24.7 2|NAM -22.0 17.0 2|MOZ -18.7 35.5 2|MAD -18.8 47.0 3|MRI -20.3 57.6 4|REU -21.1 55.5 4|TOG 8.6 0.8 0
AF BEN 9.3 2.3 1|CMR 7.4 12.4 1|GAB -0.8 11.6 1|CGO -0.2 15.8 1|COD -2.9 23.7 -|ANG -11.2 17.9 1|ETH 9.1 40.5 3|SUD 15.6 32.5 2
AF SLE 8.5 -11.8 0|MLI 17.6 -4.0 0|CAF 6.6 20.9 1|LES -29.6 28.2 2|MRT 21.0 -10.9 0|BUR 12.2 -1.6 0|ERI 15.2 39.8 3|DJI 11.8 42.6 3
AF SWZ -26.5 31.5 2|GAM 13.4 -15.3 0
NA USA 38.0 -94.0 -|CAN 47.0 -80.0 -|MEX 21.0 -100.0 -|GUA 15.8 -90.2 -6|ESA 13.8 -88.9 -6|HON 15.2 -86.2 -6|NCA 12.9 -85.2 -6
NA CRC 9.7 -83.8 -6|PAN 8.5 -80.8 -5|CUB 21.5 -77.8 -5|DOM 18.7 -70.2 -4|PUR 18.2 -66.5 -4|JAM 18.1 -77.3 -5|HAI 19.0 -72.4 -5
NA TTO 10.7 -61.2 -4|BAR 13.2 -59.5 -4|BAH 25.0 -77.4 -5|BER 32.3 -64.8 -4|ANT 17.1 -61.8 -4|GRN 12.1 -61.7 -4|KNA 17.3 -62.7 -4
NA LCA 13.9 -61.0 -4|VIN 13.2 -61.2 -4|ARU 12.5 -70.0 -4|AHO 12.2 -69.0 -4|CUW 12.2 -69.0 -4|CAY 19.3 -81.3 -5|ISV 18.3 -64.9 -4|GUD 16.2 -61.6 -4
SA BRA -15.0 -48.0 -|ARG -34.0 -64.0 -3|CHI -33.0 -71.0 -4|COL 4.6 -74.3 -5|PER -9.2 -75.0 -5|ECU -1.8 -78.2 -5|BOL -16.3 -63.6 -4
SA PAR -23.4 -58.4 -4|URU -32.8 -56.0 -3|VEN 6.4 -66.6 -4|SUR 4.0 -56.0 -3|GUY 5.0 -58.9 -4
OC AUS -33.0 148.0 -|NZL -41.0 174.0 12|FIJ -17.7 178.1 12|PNG -6.3 143.9 10|SAM -13.8 -172.1 13|SOL -9.6 160.2 11|VAN -15.4 166.9 11
OC NCL -21.3 165.5 11|GUM 13.4 144.8 10|NMI 15.2 145.7 10|POC -17.0 178.0 12|COK -21.2 -159.8 -10|ASM -14.3 -170.7 -11|PYF -17.7 -149.4 -10
OC MHL 7.1 171.2 12|HAW 21.3 -157.9 -10
"""
_ALIAS = {"DEU": "GER", "NLD": "NED", "GRC": "GRE", "BGR": "BUL", "CHL": "CHI", "PRY": "PAR", "LVA": "LAT", "CRI": "CRC",
          "PHL": "PHI", "TWN": "TPE", "SIN": "SGP", "KWT": "KUW", "BHS": "BAH", "SAU": "KSA", "ZWE": "ZIM", "NGA": "NGR",
          "AGO": "ANG", "COG": "CGO", "CAR": "CAF", "NIC": "NCA", "LIB": "LBN", "ECA": "ESA", "MNG": "MGL",
          "GTM": "GUA", "NPL": "NEP", "PRI": "PUR", "BWA": "BOT", "MDG": "MAD", "TRI": "TTO", "TGO": "TOG", "BHR": "BRN"}
_CONT = {"EU": 0, "AS": 1, "AF": 2, "NA": 3, "SA": 4, "OC": 5}
LVL_RANK = {"10": 1, "15": 1, "20": 1, "25": 2, "S": 2, "CC": 2, "35": 3, "35+H": 3, "40": 3, "50": 4, "50+H": 4, "60": 5,
            "75": 5, "80": 5, "100": 6, "C": 7, "P": 9, "A": 9, "W": 9, "I": 9, "D": 9, "T2": 9, "T3": 9, "T4": 9,
            "T5": 9, "PM": 10, "T1": 10, "M": 10, "G": 10, "F": 10, "O": 10}
LVL_GROUP = {**{k: 0 for k in ("10", "15", "20", "25", "S", "CC")},
             **{k: 1 for k in ("35", "35+H", "40", "50", "50+H", "60", "75", "80", "100")}, "C": 2}


def _build_table():
    codes, rows = [], []
    for line in _TABLE.strip().splitlines():
        cont, rest = line.split(" ", 1)
        for item in rest.split("|"):
            c, lat, lon, utc = item.split()
            codes.append(c)
            rows.append((float(lat), float(lon), float(lon) / 15 if utc == "-" else float(utc), _CONT[cont]))
    t = np.array(rows)
    return {c: i for i, c in enumerate(codes)}, t[:, 0], t[:, 1], t[:, 2], t[:, 3].astype(np.int64)


CODE, LAT, LON, UTC, CONT = _build_table()


def code_index(s):
    """ioc strings -> table index (-1 unknown); ISO/ITF spellings are mapped to the IOC code."""
    s = pd.Series(s).astype(str).str.upper()
    return s.map(lambda c: CODE.get(_ALIAS.get(c, c), -1)).fillna(-1).astype(np.int64).values


def haversine(a, b):
    la1, lo1, la2, lo2 = (np.radians(x) for x in (LAT[a], LON[a], LAT[b], LON[b]))
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 12742.0 * np.arcsin(np.sqrt(h))


def norm_name(s):
    """Event name key: lower case ascii, level tags (W15, M25+H, 25K, F6, H), ', ST' and trailing integers removed."""
    s = s.fillna("").astype(str).str.lower().str.normalize("NFKD").str.encode("ascii", "ignore").str.decode("ascii")
    s = s.str.replace(r",\s*st\b", " ", regex=True).str.replace(r"[^a-z0-9+$ ]", " ", regex=True)
    s = s.str.replace(r"\b[mw]\d{2,3}\+?h?\b|\$?\d+k\b|\bf\d+\b|\bitf\b|\bch\b|\+h\b|\bh\b", " ", regex=True)
    s = s.str.replace(r"[^a-z0-9 ]", " ", regex=True).str.replace(r"(\s+\d+)+\s*$", "", regex=True)
    return s.str.replace(r"\s+", " ", regex=True).str.strip()


def load_games(s):
    """Games on court from the score (C1): complete and unfinished sets count a+b, a match tiebreak counts 3."""
    v = 0
    for t in s.replace("RET", " ").split():
        m = _TOK.match(t)
        if not m:
            continue
        a, b = int(m.group(1)), int(m.group(2))
        v += 3 if t.startswith("[") or max(a, b) >= 10 else a + b
    return v


def elo_exp(W, L, ret):
    """Pre-match standard Elo (K=250/(n+5)^0.4, half weight for retirements): winner and loser rating."""
    E, N = {}, {}
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


class Out(dict):
    def __setitem__(self, k, v):
        super().__setitem__(k, np.asarray(v, dtype=np.float32))


class Seg:
    """Rows grouped by key (input order kept inside a group), or already contiguous groups given by their start."""

    def __init__(self, key=None, st=None):
        if key is None:
            self.p, self.st = None, st
            self.pos = np.arange(len(st))
        else:
            self.p = np.argsort(key, kind="stable")
            k = key[self.p]
            self.pos = np.arange(len(k))
            self.st = np.maximum.accumulate(np.where(np.r_[True, k[1:] != k[:-1]], self.pos, 0))
            self.k = k

    def _to(self, v):
        return v if self.p is None else v[self.p]

    def _back(self, v):
        if self.p is None:
            return v
        out = np.empty_like(v)
        out[self.p] = v
        return out

    def cnt(self):
        return self._back((self.pos - self.st).astype(np.float64))

    def csum(self, v):
        s = np.concatenate(([0.0], np.cumsum(self._to(v), dtype=np.float64)))
        return self._back(s[self.pos] - s[self.st])

    def lag(self, v):
        x = self._to(v).astype(np.float64)
        out = np.full(len(x), np.nan)
        ok = self.pos > self.st
        out[ok] = x[self.pos[ok] - 1]
        return self._back(out)

    def first(self, v):
        return self._back(self._to(v)[self.st])


def gcummax(gid, v, scale=BIG):
    g = gid.astype(np.int64) * scale
    return np.maximum.accumulate(g + v.astype(np.int64)) - g


def _is_wta(M):
    return bool(M["tourney_level"].astype(str).isin(["PM", "T1", "T2", "10", "W"]).any())


def event_country(M, ev, td_m, nm_key, W, L, wio, lio):
    """Per-match event country index (-1 unknown): parsed id, else learned from earlier editions, else row-time modal ioc."""
    n = len(M)
    nev = int(ev.max()) + 1
    first = np.unique(ev, return_index=True)[1]
    tid = M["tourney_id"].values[first]
    parsed = code_index(pd.Series(tid).str.extract(_PARSE_ID)[0].fillna(""))
    ev_td = td_m[first]
    ev_key = nm_key[first]
    pl = np.concatenate([ev, ev]).astype(np.int64) * (1 << 24)
    pid2 = pd.factorize(np.concatenate([W, L]))[0].astype(np.int64)
    io2 = np.concatenate([wio, lio])
    code = pl + pid2
    _, u = np.unique(code, return_index=True)
    e_u, i_u = pl[u] >> 24, io2[u]
    ok = i_u >= 0
    pair, cnt = np.unique(e_u[ok] * 512 + i_u[ok], return_counts=True)
    pe, pi = pair // 512, pair % 512
    ptr = np.searchsorted(pe, np.arange(nev + 1))
    npl = np.bincount(e_u, minlength=nev)
    stats = {}
    out = np.full(nev, -1, dtype=np.int64)
    eo = np.argsort(ev_td, kind="stable")
    d_sorted = ev_td[eo]
    bounds = np.r_[0, np.flatnonzero(d_sorted[1:] != d_sorted[:-1]) + 1, nev]
    for b in range(len(bounds) - 1):
        grp = eo[bounds[b]:bounds[b + 1]]
        for e in grp:
            if parsed[e] >= 0:
                out[e] = parsed[e]
                continue
            st = stats.get(ev_key[e])
            if st is None or ev_key[e] < 0:
                continue
            if st[3]:
                out[e] = st[3].most_common(1)[0][0]
            elif (st[0] >= 2 or st[1] >= 8) and st[2]:
                c, k = st[2].most_common(1)[0]
                if k / sum(st[2].values()) >= MIN_SHARE:
                    out[e] = c
        for e in grp:
            k = ev_key[e]
            if k < 0:
                continue
            st = stats.setdefault(k, [0, 0, Counter(), Counter()])
            st[0] += 1
            st[1] += npl[e]
            for c, m in zip(pi[ptr[e]:ptr[e + 1]], cnt[ptr[e]:ptr[e + 1]]):
                st[2][c] += m
            if parsed[e] >= 0:
                st[3][parsed[e]] += 1
    cty = out[ev]
    idx = np.flatnonzero(cty < 0)
    seen = {}
    Wl, Ll, wl, ll, evl = W[idx].tolist(), L[idx].tolist(), wio[idx].tolist(), lio[idx].tolist(), ev[idx].tolist()
    res = np.full(len(idx), -1, dtype=np.int64)
    for j in range(len(idx)):
        s = seen.get(evl[j])
        if s is None:
            s = seen[evl[j]] = (set(), Counter())
        pset, c = s
        if len(pset) >= 6 and c:
            m, k = c.most_common(1)[0]
            if k / sum(c.values()) >= MIN_SHARE:
                res[j] = m
        for p, io in ((Wl[j], wl[j]), (Ll[j], ll[j])):
            if p not in pset:
                pset.add(p)
                if io >= 0:
                    c[io] += 1
    cty[idx] = res
    return cty, parsed[ev]


def player_rows(n, wta, ordv, W, L, td_m, est_m, ev, rord, load_m, isQ_m, isF_m, surf_m, lvl_m, lg_m, cty_m, wio, lio, wa, la, e_w, elo, mean_a, std_a, lg_elo_mean, ev_mean, min_main_prev, indoor_m, phase, nk_codes):
    """Features on stacked player rows sorted by (player, order); returns the feature dict and the row permutation."""
    N = 2 * n
    # ---- stacked player rows sorted by (player, order)
    pid = np.concatenate([W, L])
    perm = np.lexsort((np.concatenate([ordv, ordv]), pid))
    mi = perm % n
    isw = perm < n
    won = isw.astype(np.float64)
    pid_s = pid[perm]
    newp = np.r_[True, pid_s[1:] != pid_s[:-1]]
    gid = np.cumsum(newp) - 1
    pos = np.arange(N)
    st = np.maximum.accumulate(np.where(newp, pos, 0))
    P = Seg(st=st)
    gB = gid.astype(np.int64) * BIG
    td_raw = td_m[mi]
    td = gcummax(gid, td_raw)
    est = gcummax(gid, est_m[mi])
    kTD, kEST = gB + td, gB + est
    ev_s = ev[mi]
    rord_s, load_s = rord[mi], load_m[mi]
    isQ, isF = isQ_m[mi], isF_m[mi]
    surf, lvl, lg, cty = surf_m[mi], lvl_m[mi], lg_m[mi], cty_m[mi]
    pioc = np.where(isw, wio[mi], lio[mi])
    age = np.where(isw, wa[mi], la[mi])
    e_p = np.where(isw, e_w[mi], 1 - e_w[mi])
    resid = won - e_p
    elo_s = np.where(isw, elo[mi, 0], elo[mi, 1])
    n_prev = (pos - st).astype(np.float64)
    out = Out()

    # ---- runs (player x event) and the previous event (last row with a strictly smaller tourney_date)
    newrun = np.r_[True, newp[1:] | (ev_s[1:] != ev_s[:-1])]
    rid = np.cumsum(newrun) - 1
    rfirst = np.flatnonzero(newrun)
    rlast = np.r_[rfirst[1:] - 1, N - 1]
    bs = rfirst[rid]
    in_run = (pos - bs).astype(np.float64)
    pp = np.searchsorted(kTD, kTD, "left") - 1
    okp = (pp >= 0) & (gid[np.maximum(pp, 0)] == gid)
    pp = np.maximum(pp, 0)
    prun = rid[pp]
    gap = td - est[pp]
    has21, has60 = okp & (gap <= 21), okp & (gap <= 60)
    r_gid, r_td, r_rf, r_rl = gid[rfirst], td[rfirst], rord_s[rfirst], rord_s[rlast]
    r_rmax = np.maximum.reduceat(rord_s, rfirst)
    r_games = np.add.reduceat(load_s, rfirst)
    r_cty, r_surf, r_lvl = cty[rlast], surf[rfirst], lvl[rfirst]
    r_title = (won[rlast] > 0) & isF[rlast]
    r_estl = est[rlast]
    den = r_rf - FINAL
    r_depth = np.where(r_rl >= FINAL, 1.0, np.where(den < 0, np.clip((r_rf - r_rl) / np.where(den < 0, den, -1), 0, 1) + 0.0, 0.0))
    nr = len(rfirst)

    # country change / travel / time zones
    c_prev, c_cur = r_cty[prun], cty
    both = has60 & (c_prev >= 0) & (c_cur >= 0)
    chg = np.where(both, (c_prev != c_cur).astype(np.float64), np.nan)
    out["ctx_ev_country_change"] = np.where(has60, chg, 0.0)
    out["ctx_no_prev_event"] = (~has60).astype(np.float64)
    ok21 = has21 & (c_prev >= 0) & (c_cur >= 0)
    cp, cc = np.maximum(c_prev, 0), np.maximum(c_cur, 0)
    travel = np.where(ok21, np.log1p(haversine(cp, cc)), np.nan)
    dtz = np.where(ok21, UTC[cc] - UTC[cp], 0.0)
    out["ctx_tz_abs"] = np.abs(dtz)
    out["ctx_tz_east"] = dtz
    home = (pioc >= 0) & (c_cur >= 0)
    dist_home = np.where(home, np.log1p(haversine(np.maximum(pioc, 0), cc)), np.nan)
    out["ctx_same_continent"] = (home & (CONT[np.maximum(pioc, 0)] == CONT[cc])).astype(np.float64)

    # consecutive events in the same country
    rgap = r_td[1:] - r_estl[:-1]
    link = np.r_[False, (r_gid[1:] == r_gid[:-1]) & (r_cty[1:] == r_cty[:-1]) & (r_cty[1:] >= 0) & (rgap <= 14)]
    ridx = np.arange(nr)
    chain = ridx - np.maximum.accumulate(np.where(link, 0, ridx)) + 1
    out["ctx_country_streak_weeks"] = np.where(okp & (gap <= 14) & (c_prev == c_cur) & (c_cur >= 0),
                                               np.minimum(chain[prun], 8), 0).astype(np.float64)

    # familiarity with the current event country
    SC = Seg(key=gid.astype(np.int64) * 512 + (cty + 1))
    kc = SC.k.astype(np.int64) * BIG + SC._to(pos - st)
    lo30 = np.searchsorted(kc, SC.k.astype(np.int64) * BIG + np.maximum(SC._to(pos - st) - 30, 0), "left")
    in30 = SC._back((SC.pos - lo30).astype(np.float64))
    out["ctx_country_share_30"] = np.where((n_prev >= 5) & (cty >= 0), in30 / np.minimum(n_prev, 30).clip(1), 0.0)
    out["ctx_country_resid_shrunk"] = np.where(cty >= 0, SC.csum(resid) / (SC.cnt() + 6), 0.0)
    del SC, kc, lo30

    # schedule density
    consec = np.zeros(N)
    alive = np.ones(N, dtype=bool)
    active8 = np.zeros(N)
    for k in range(1, 13):
        a = np.searchsorted(kEST, gB + td - 7 * k, "left")
        b = np.minimum(np.searchsorted(kEST, gB + td - 7 * k + 6, "right"), pos)
        present = np.maximum(b - np.maximum(a, st), 0) > 0
        alive &= present
        consec += alive
        if k <= 8:
            active8 += present
    out["ctx_consec_weeks"] = consec
    out["ctx_weeks_active_8"] = active8
    kR = r_gid.astype(np.int64) * BIG + r_td
    n90 = np.searchsorted(kR, kR, "left") - np.searchsorted(kR, kR - 90, "left")
    out["ctx_n_events_90d"] = n90[rid].astype(np.float64)

    # previous event
    out["ctx_prev_ev_deep"] = np.where(has21, r_depth[prun], 0.0)
    out["ctx_prev_ev_games"] = np.where(has21, r_games[prun], 0.0)
    out["ctx_prev_ev_title"] = np.where(has21, r_title[prun], 0).astype(np.float64)
    estl_prev = np.r_[0.0, load_s[:-1]]
    dt = np.r_[0.0, np.diff(est)].clip(0)
    f4, f10 = np.exp(-dt / 4), np.exp(-dt / 10)
    o4, o10 = np.zeros(N), np.zeros(N)
    a = b = 0.0
    stl, lpl, f4l, f10l = st.tolist(), estl_prev.tolist(), f4.tolist(), f10.tolist()
    for r in range(N):
        if stl[r] == r:
            a = b = 0.0
        else:
            a = (a + lpl[r]) * f4l[r]
            b = (b + lpl[r]) * f10l[r]
        o4[r], o10[r] = a, b
    out["ctx_load_ewm4"], out["ctx_load_ewm10"] = o4 / 10, o10 / 10
    del stl, lpl, f4l, f10l, f4, f10, o4, o10, estl_prev, dt

    # streaks
    chg_r = np.r_[True, newp[1:] | (won[1:] != won[:-1])]
    rl = pos - np.maximum.accumulate(np.where(chg_r, pos, 0)) + 1
    prev_len, prev_won = P.lag(rl.astype(np.float64)), P.lag(won)
    out["ctx_win_streak"] = np.where(prev_won == 1, np.minimum(prev_len, 15), 0.0)
    out["ctx_loss_streak"] = np.where(prev_won == 0, np.minimum(prev_len, 15), 0.0)
    for nm, flag in (("loss", won == 0), ("win", won == 1)):
        lp = np.r_[-1, np.maximum.accumulate(np.where(flag, pos, -1))[:-1]]
        ok = lp >= st
        d = np.where(ok, est - est[np.maximum(lp, 0)], 365)
        out[f"ctx_days_since_{nm}"] = np.log1p(np.minimum(d, 365))

    # layoffs
    gap_m = est - np.r_[0, est[:-1]]
    lay = (~newp) & (gap_m >= 42)
    lpos = np.maximum.accumulate(np.where(lay, pos, -1))
    hasl = lpos >= st
    idx_l = np.where(hasl, pos - lpos, 8)
    out["ctx_post_layoff_idx"] = np.minimum(idx_l, 8).astype(np.float64)
    out["ctx_last_layoff_len_log"] = np.where(hasl, np.log1p(gap_m[np.maximum(lpos, 0)]), 0.0)
    fl = (hasl & (idx_l <= 1)).astype(np.float64)
    out["ctx_comeback_resid_hist"] = P.csum(resid * fl) / (P.csum(fl) + 4)

    # surface
    SS = Seg(key=gid.astype(np.int64) * 4 + surf)
    ls = SS.lag(est.astype(np.float64))
    out["ctx_surf_days_since"] = np.log1p(np.minimum(np.where(np.isnan(ls), 730, est - ls), 730))
    out["ctx_surf_switch"] = (has21 & (r_surf[prun] != surf)).astype(np.float64)
    ks = SS.k.astype(np.int64) * BIG + SS._to(est)
    lo_s = np.searchsorted(ks, ks - 365, "right")
    n_s = SS._back((SS.pos - np.maximum(lo_s, SS.st)).astype(np.float64))
    lo_a = np.maximum(np.searchsorted(kEST, kEST - 365, "right"), st)
    n_all = (pos - lo_a).astype(np.float64)
    out["ctx_surf_share_365d"] = np.where(n_all > 0, n_s / np.maximum(n_all, 1), np.nan)
    del SS, ks

    out["ctx_indoor"] = indoor_m[mi]
    out["ctx_season_phase"] = phase[mi]
    out["ctx_lvl_rank_cur"] = lvl

    # level transitions and career stage
    cmax = gcummax(gid, lvl.astype(np.int64), 16)
    bm = bs - 1
    hist = bm >= st
    cprev = np.where(hist, cmax[np.maximum(bm, 0)], 0).astype(np.float64)
    out["ctx_lvl_delta_prev"] = np.where(has21, lvl - r_lvl[prun], 0.0)
    out["ctx_lvl_step_above_max"] = np.where(hist, lvl - cprev, 0.0)
    SL = Seg(key=gid.astype(np.int64) * 4 + lg)
    cnt_lg = SL.cnt()
    out["ctx_first_at_level"] = ((cnt_lg - in_run) <= 0).astype(np.float64)
    out["ctx_years_at_level"] = np.where(cnt_lg > 0, (est - SL.first(est.astype(np.float64))) / 365, 0.0)
    del SL
    yrs = (est - est[st]) / 365.25
    first_lvl = lvl[st]
    out["ctx_career_slope"] = np.where(hist, (cprev - first_lvl) / np.maximum(0.5, yrs), 0.0)
    amin = gcummax(gid, np.where(np.isnan(age), 0, np.round((100 - np.nan_to_num(age)) * 1000)))
    amin_prev = np.where(newp, 0, np.r_[0, amin[:-1]])
    first_age = np.where(amin_prev > 0, 100 - amin_prev / 1000, np.nan)
    debut = np.where(np.isnan(first_age), age - yrs, first_age)
    debut = np.clip(debut, 13, 35)
    debut = np.where(td[st] < CENSOR_DAY, np.nan, debut)
    out["ctx_age_peak_gap"] = np.where(np.isnan(age), 0.0, age - (24.5 if wta else 26.5))
    out["ctx_age_vs_level_z"] = np.where(np.isnan(age) | np.isnan(mean_a[mi]) | (std_a[mi] <= 0), 0.0,
                                         (age - mean_a[mi]) / np.where(std_a[mi] > 0, std_a[mi], 1))
    em = np.where(np.isnan(ev_mean[mi]), lg_elo_mean[mi], ev_mean[mi])
    out["ctx_elo_vs_event"] = elo_s - em
    del elo_s, em, e_p, resid, age, pioc, rl, chg_r

    # qualifying path
    qf = isQ.astype(np.float64)
    RN = Seg(st=bs)
    out["ctx_q_matches"] = np.minimum(RN.csum(qf), 4)
    out["ctx_q_games"] = RN.csum(qf * load_s)
    lq = np.r_[-1, np.maximum.accumulate(np.where(isQ, pos, -1))[:-1]]
    okq = (lq >= bs) & (pos > bs)
    out["ctx_lucky_loser"] = (okq & (won[np.maximum(lq, 0)] == 0) & ~isQ).astype(np.float64)
    out["ctx_bye_proxy"] = ((pos == bs) & ~isQ & (min_main_prev[mi] < rord_s)).astype(np.float64)

    # same event last year (300-430 days earlier)
    key_ev = np.where(nk_codes >= 0, nk_codes.astype(np.int64) * 4 + lg_m, -1)[mi][rfirst]
    kK = (r_gid.astype(np.int64) * (int(key_ev.max()) + 2) + key_ev + 1) * BIG
    so = np.argsort(kK + r_td, kind="stable")
    sk = (kK + r_td)[so]
    q = np.searchsorted(sk, kK + r_td - 300, "right") - 1
    cand = so[np.maximum(q, 0)]
    okc = (q >= 0) & (key_ev >= 0) & (kK[cand] == kK) & (r_td[cand] >= r_td - 430)
    r_def = np.where(okc, r_depth[cand], -1.0)
    r_bey = np.where(okc, r_rmax[cand], np.nan)
    out["ctx_def_depth"] = r_def[rid]
    last_max = r_bey[rid]
    out["ctx_beyond_last_year"] = np.where(np.isnan(last_max), -1.0, (rord_s > last_max).astype(np.float64))

    ttl = (won * isF).astype(np.float64)
    cs_t = np.concatenate(([0.0], np.cumsum(ttl)))
    out["ctx_titles_365d"] = cs_t[pos] - cs_t[lo_a]

    # training medians for unknown values
    tr = td_raw < TRAIN_END

    def fill_med(v):
        ok = ~np.isnan(v)
        sel = tr & ok if (tr & ok).any() else ok
        return np.where(ok, v, np.nanmedian(v[sel]) if sel.any() else 0.0)

    del cty, kTD, kEST, gB, td, est, rord_s, load_s
    out["ctx_travel_km_log"] = fill_med(travel)
    out["ctx_dist_home_km_log"] = fill_med(dist_home)
    out["ctx_age_debut"] = fill_med(debut)
    return out, perm


def compute(M, wta=None):
    """Self-play rows (winner_id == loser_id, a source data error) take no part in any state and get NaN."""
    bad = M["winner_id"].values == M["loser_id"].values
    if bad.any():
        if wta is None:
            wta = _is_wta(M)
        return _compute(M[~bad], wta).reindex(M.index)
    return _compute(M, wta)


def _compute(M, wta=None):
    n = len(M)
    N = 2 * n
    if wta is None:
        wta = _is_wta(M)
    ordv = M["order"].values
    if n > 1 and not (np.diff(ordv) > 0).all():
        raise ValueError("M must be sorted by 'order'")
    td_m = (M["date"].values.astype("datetime64[D]") - EPOCH).astype(np.int64)
    W = M["winner_id"].values.astype(np.int64)
    L = M["loser_id"].values.astype(np.int64)
    ret = M["retired"].values.astype(bool)
    rord = pd.to_numeric(M["rord"], errors="coerce").fillna(5).values.astype(np.float64)
    rnd = M["round"].astype(str).str.upper()
    isQ_m = rnd.str.startswith("Q").values
    isF_m = (rnd == "F").values
    surf_s = M["surface"].fillna("Hard").astype(str).str.lower()
    carpet_m = (surf_s == "carpet").values
    surf_m = surf_s.map({"clay": 1, "grass": 2}).fillna(0).astype(np.int64).values
    lev = M["tourney_level"].astype(str)
    nm_codes, nm_uni = pd.factorize(M["tourney_name"].fillna("").astype(str))
    nm_low = pd.Series(nm_uni).str.lower()
    lvl_m = lev.map(LVL_RANK).fillna(9).values.astype(np.float64)
    c8 = nm_low.str.contains(r"100|125|175", regex=True).values[nm_codes] & (lev.values == "C")
    lvl_m = np.where(c8, 8.0, lvl_m)
    lg_m = lev.map(LVL_GROUP).fillna(3).astype(np.int64).values
    nk_uni = norm_name(pd.Series(nm_uni))
    nk_codes = pd.factorize(nk_uni)[0]
    nk_codes = np.where(nk_uni.values == "", -1, nk_codes)[nm_codes]
    ev = pd.factorize(M["tourney_id"].astype(str).values)[0].astype(np.int64) * BIG + td_m
    ev = pd.factorize(ev)[0].astype(np.int64)
    wio, lio = code_index(M["winner_ioc"].fillna("")), code_index(M["loser_ioc"].fillna(""))
    cty_m, _ = event_country(M, ev, td_m, nk_codes, W, L, wio, lio)

    # match-level: load games, est_day, Elo
    sc, su = pd.factorize(M["score"].fillna("").astype(str))
    load_m = np.array([load_games(x) for x in su], dtype=np.float64)[sc]
    rmin_prev = pd.Series(rord).groupby(ev).cummin().groupby(ev).shift(1).values
    est_m = td_m + np.clip(np.nan_to_num(rord - rmin_prev, nan=0.0), 0, 6).astype(np.int64)
    elo = elo_exp(W, L, ret)
    e_w = 1.0 / (1.0 + 10 ** ((elo[:, 1] - elo[:, 0]) / 400.0))

    # match-level context: indoor, season phase, level rank
    month = pd.DatetimeIndex(M["date"]).month.values
    isoweek = pd.DatetimeIndex(M["date"]).isocalendar().week.values.astype(np.int64)
    lat_c = np.where(cty_m >= 0, LAT[np.maximum(cty_m, 0)], np.nan)
    cont_c = np.where(cty_m >= 0, CONT[np.maximum(cty_m, 0)], -1)
    name_in = (nm_low.str.contains(r"\b(?:indoor|hall|halle|arena|dome|palais)\b", regex=True)
               & ~nm_low.str.contains("hall of fame")).values[nm_codes] & (surf_m != 2)
    winter = np.isin(month, (10, 11, 12, 1, 2, 3))
    indoor_m = (name_in | carpet_m | ((surf_m != 2) & (lat_c >= 44) & winter)).astype(np.float64)
    wk = isoweek
    phase = np.full(n, 8.0)
    for val, cond in reversed([
            (0, wk <= 8), (7, (wk <= 14) & (surf_m == 1) & (cont_c == 4)), (1, (wk >= 9) & (wk <= 14)),
            (2, (wk >= 15) & (wk <= 23) & (surf_m == 1)), (3, (wk >= 24) & (wk <= 28)), (4, (wk >= 29) & (wk <= 36)),
            (5, (wk >= 37) & (wk <= 42) & (cont_c == 1)), (6, (wk >= 37) & (wk <= 48) & ((cont_c == 0) | (indoor_m > 0))),
            (8, wk >= 49)]):
        phase[cond] = val

    # match-level windows over the level group: age mean/std and mean Elo of players over the previous 365 days
    wa, la = M["winner_age"].values.astype(np.float64), M["loser_age"].values.astype(np.float64)
    ak = np.where(np.isnan(wa), 0, 1) + np.where(np.isnan(la), 0, 1)
    a1 = np.nan_to_num(wa) + np.nan_to_num(la)
    a2 = np.nan_to_num(wa) ** 2 + np.nan_to_num(la) ** 2
    LG = Seg(key=lg_m)
    tdl = LG._to(td_m)
    kl = np.maximum.accumulate(LG.k.astype(np.int64) * BIG + tdl)
    lo = np.maximum(np.searchsorted(kl, kl - 365, "right"), LG.st)

    def lg_window(v, min_matches=200):
        cs = np.concatenate(([0.0], np.cumsum(v[LG.p], dtype=np.float64)))
        recent = cs[LG.pos] - cs[lo]
        alltime = cs[LG.pos] - cs[LG.st]
        use_all = (LG.pos - lo) < min_matches
        out = np.where(use_all, alltime, recent)
        res = np.empty(n)
        res[LG.p] = out
        return res

    s_k, s_a1, s_a2 = lg_window(ak.astype(np.float64)), lg_window(a1), lg_window(a2)
    mean_a = np.where(s_k > 0, s_a1 / np.maximum(s_k, 1), np.nan)
    std_a = np.sqrt(np.maximum(s_a2 / np.maximum(s_k, 1) - mean_a ** 2, 0))
    s_e = lg_window(elo[:, 0] + elo[:, 1])
    s_m = lg_window(np.full(n, 2.0))
    lg_elo_mean = np.where(s_m > 0, s_e / np.maximum(s_m, 1), 1500.0)

    # event running mean Elo of distinct players seen so far
    ev_mean = np.full(n, np.nan)
    seen_ev = {}
    evl, Wl, Ll, e0, e1 = ev.tolist(), W.tolist(), L.tolist(), elo[:, 0].tolist(), elo[:, 1].tolist()
    for i in range(n):
        s = seen_ev.get(evl[i])
        if s is None:
            s = seen_ev[evl[i]] = [set(), 0.0]
        if len(s[0]) >= 6:
            ev_mean[i] = s[1] / len(s[0])
        if Wl[i] not in s[0]:
            s[0].add(Wl[i])
            s[1] += e0[i]
        if Ll[i] not in s[0]:
            s[0].add(Ll[i])
            s[1] += e1[i]
    del seen_ev, evl, Wl, Ll, e0, e1

    # first main-draw round already played in the event, strictly earlier in order
    rmain = np.where(isQ_m, np.inf, rord)
    min_main_prev = pd.Series(rmain).groupby(ev).cummin().groupby(ev).shift(1).values

    out, perm = player_rows(n, wta, ordv, W, L, td_m, est_m, ev, rord, load_m, isQ_m, isF_m, surf_m, lvl_m, lg_m, cty_m, wio, lio, wa, la, e_w, elo, mean_a, std_a, lg_elo_mean, ev_mean, min_main_prev, indoor_m, phase, nk_codes)
    res = {}
    for k, v in out.items():
        full = np.empty(N, dtype=np.float32)
        full[perm] = v
        res[f"w_{k}"], res[f"l_{k}"] = full[:n], full[n:]
    return pd.DataFrame(res, index=M.index)


if __name__ == "__main__":
    cols = ["tourney_id", "tourney_name", "surface", "tourney_level", "date", "order", "round", "rord", "score", "retired",
            "winner_id", "loser_id", "winner_ioc", "loser_ioc", "winner_age", "loser_age"]
    M = pd.read_parquet(sys.argv[1], columns=cols)
    compute(M).to_parquet(sys.argv[2])
