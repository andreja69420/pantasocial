"""Paper trading (simulation without money) of the opening-odds strategy on upcoming matches.

Each run (meant to run every 10 min, so opening lines are caught while they still stand):
  0. results of the last days appended to the feature build in seconds (fs_update.py)
  1. every scheduled singles match Flashscore lists (today and as many days ahead as are published) with every UK
     bookmaker's opening and current home/away odds (fs_odds.py --fixtures)
  2. fixture rows in Sackmann format (players mapped to ids, bio from their latest match) -> features in seconds
     with features_incr.py (dry run, the base build is not changed)
  3. production model (train_final.py) -> P(home); betting-stage blend with the Shin fair opening price
  4. the plan rule (blend EV>5%, odds<=5) on every book's CURRENT price (blend with the Shin fair price of that book's
     current pair): the best price that passes is logged once per match as a paper bet at that price
  4b. checks before a bet is logged (otherwise it goes to DATA_DIR/paper/skipped.csv with the reason): both players
     mapped to a real id, at least 10 matches of history each, the Flashscore name shares a name with the history of
     that id, and when the model is 30+ points above the book (fair price) both have 30+ matches and played in the
     last 180 days (a big gap is where a split or wrong history shows up first)
  5. pending bets: the book's latest price before the start is kept as the closing price (CLV); the kick-off time is
     taken again from Flashscore (fixtures and the daily feeds, yesterday to tomorrow) and changed when it moved;
     finished matches are settled from Flashscore's daily results feed
Ledger: DATA_DIR/paper/ledger.csv. Summary printed per rule.

usage: paper_trade.py DATA_DIR [DAYS_AHEAD=7]   (env SKIP_UPDATE=1: no results update)
env LIVE_OUT=feature build dir (default out_live), MODELS=model dir (default models), NX=1: add the new-feature modules
(fx_live.py) to the fixture rows, MODEL_TAG=label stored with each logged bet, GEO=Flashscore odds region (default GB),
BOOKS=comma-separated bookmakers to bet with (default: every book of the region), SKIP_CATS=comma-separated Flashscore
categories not bet (e.g. ITF MEN - SINGLES,ITF WOMEN - SINGLES)
"""
import json
import os
import pickle
import re
import subprocess
import sys
import time
from difflib import SequenceMatcher
import urllib.request

import numpy as np
import pandas as pd

from fs_common import UA, NameIds, apply_aliases, event_id, key, level, player_name, qual_rounds, records, round_code, tournament
import fs_rank

D = sys.argv[1]
DAYS = int(sys.argv[2]) if len(sys.argv) > 2 else 7
HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.join(D, "paper")
os.makedirs(PAPER, exist_ok=True)
LEDGER = os.path.join(PAPER, "ledger.csv")
LIVE_OUT = os.environ.get("LIVE_OUT", "out_live")
MODELS = os.environ.get("MODELS", "models")
NX = os.environ.get("NX") == "1"
TAG = os.environ.get("MODEL_TAG", "")
GEO = os.environ.get("GEO", "GB")  # Flashscore odds region
BOOKS = [b for b in os.environ.get("BOOKS", "").split(",") if b]  # only these bookmakers (all of the region if empty)
SKIP_CATS = [c for c in os.environ.get("SKIP_CATS", "").split(",") if c]  # Flashscore categories not bet, e.g. ITF MEN - SINGLES
RULES = {"EV>3% odds<=5": (0.03, 5.0)}  # 5% until 2026-10-18 (whole-2026 test: EV>3% gives about twice the bets at similar ROI at later prices)
CACHE_TTL = float(os.environ.get("CACHE_TTL", 6 * 3600))  # seconds a stored prediction is trusted without a change of its inputs
MIN_HIST, BIG_GAP, BIG_HIST, BIG_DAYS = 10, 0.30, 30, 180  # the plan rule only (whole-2026 test)
lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))


def shin(a, b):
    lo, hi = np.zeros_like(a), np.full_like(a, 0.5)
    B = a + b
    f = lambda z, x: (np.sqrt(z ** 2 + 4 * (1 - z) * x ** 2 / B) - z) / (2 * (1 - z))
    for _ in range(50):
        z = (lo + hi) / 2
        s = f(z, a) + f(z, b)
        hi, lo = np.where(s < 1, z, hi), np.where(s < 1, lo, z)
    return f((lo + hi) / 2, a)


t0 = time.time()
# in parallel: odds of all upcoming fixtures for both tours, and the results update of the feature build
jobs = {t: subprocess.Popen([sys.executable, os.path.join(HERE, "fs_odds.py"), os.path.join(D, "fs_raw"), t, GEO, "--fixtures", str(DAYS)],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) for t in ("wta", "atp")}
if not os.environ.get("SKIP_UPDATE"):
    jobs["update"] = subprocess.Popen([sys.executable, os.path.join(HERE, "fs_update.py"), D, "3"],
                                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for name, j in jobs.items():
    out = j.communicate()[0].strip()
    print(f"[{name}] " + (out.splitlines()[-1] if j.returncode == 0 and name != "update" else out[-800:]), flush=True)
print(f"[time] odds and update {time.time() - t0:.0f}s", flush=True)
ledger = pd.read_csv(LEDGER) if os.path.exists(LEDGER) else pd.DataFrame()
def predict_rows(tour, rows, model_path, ts):
    """P(home) of the given fixture rows: features (features_incr, fx_live) and the production model; None when a step fails"""
    fcsv = os.path.join(PAPER, f"{tour}__{tour}_matches_fs_fixtures.csv")
    pd.DataFrame(rows).to_csv(fcsv, index=False)
    fout = os.path.join(PAPER, f"{tour}_fixture_features.parquet")
    fnew = os.path.join(PAPER, f"{tour}_fixture_rows.parquet")
    env = dict(os.environ, DRY_OUT=fout, DRY_MATCHES=fnew)
    tp = time.time()
    r = subprocess.run([sys.executable, os.path.join(HERE, "features_incr.py"), os.path.join(D, "live_raw"), os.path.join(D, LIVE_OUT),
                        tour, fcsv], env=env, capture_output=True, text=True)
    if r.returncode:
        print(f"{tour}: features_incr failed\n{r.stderr[-1500:]}")
        return None
    Fx = pd.read_parquet(fout)
    tf = time.time()
    if NX:
        fnx = os.path.join(PAPER, f"{tour}_fixture_nx.parquet")
        r = subprocess.run([sys.executable, os.path.join(HERE, "fx_live.py"), os.path.join(D, LIVE_OUT), tour, fnew, fnx],
                           capture_output=True, text=True)
        if r.returncode:
            print(f"{tour}: fx_live failed\n{r.stderr[-1500:]}")
            return None
        print(r.stdout.strip().splitlines()[-1], flush=True)
        Fx = Fx.join(pd.read_parquet(fnx))
    sys.path.insert(0, HERE)
    from design import build_X
    M = pickle.load(open(model_path, "rb"))
    X, _, flip = build_X(Fx, tour)
    X["level"] = pd.Categorical(Fx.tourney_level.astype(str), categories=M["levels"]).codes
    X = X[M["features"]]
    p = np.mean([M["models"]["xgb"].predict_proba(X)[:, 1], M["models"]["lgbm"].predict_proba(X)[:, 1],
                 M["models"]["logit"].predict_proba(np.nan_to_num(lg(X[M["pcols"]].values.astype(float))))[:, 1]], axis=0)
    p = M["cal"].predict_proba(lg(p)[:, None])[:, 1]
    p_home = pd.Series(np.where(flip, 1 - p, p), index=Fx.fs_mid.values)
    print(f"[time] {tour}: fixtures {tp - ts:.0f}s, features_incr {tf - tp:.0f}s, fx_live and model {time.time() - tf:.0f}s", flush=True)
    return p_home


new_bets, seen, checks = [], [], {}
for tour in ("wta", "atp"):
    ts = time.time()
    model_path = os.path.join(D, MODELS, f"{tour}_final.pkl")
    if not os.path.exists(model_path):
        print(f"{tour}: no production model yet ({model_path})")
        continue
    O = pd.read_parquet(os.path.join(D, "fs_raw", f"{tour}_fs_fixture_odds_{GEO}.parquet"))
    if BOOKS:
        O = O[O.book.isin(BOOKS)]
    if SKIP_CATS:
        O = O[~O.tournament.str.split(":").str[0].str.strip().isin(SKIP_CATS)]
    if O.empty:
        print(f"{tour}: no fixtures with odds yet")
        continue
    seen.append(O.assign(tour=tour))
    O = O[O.start > time.time()]  # not started
    base = pd.read_parquet(os.path.join(D, LIVE_OUT, f"{tour}_matches.parquet"))
    ids = NameIds()
    nk = {c: base[f"{c}_name"].map({n: key(n) for n in base[f"{c}_name"].unique()}) for c in ("winner", "loser")}
    names = pd.concat([pd.DataFrame({"k": nk[c], "id": base[f"{c}_id"], "d": base.tourney_date.astype(float)})
                       for c in ("winner", "loser")])
    # a later result can give a known player a synthetic id (>= 9e8); the real id with the full history must win
    names = names.assign(real=names.id < 9e8).sort_values(["real", "d"])
    ids.add(names.k, names.id, names.d.values)
    apply_aliases(ids)
    hist = pd.concat([pd.DataFrame({"id": base[f"{c}_id"], "date": base.date, "name": nk[c]}) for c in ("winner", "loser")])
    n_hist, last_hist = hist.groupby("id").size(), hist.groupby("id").date.max()
    id_names = hist.groupby("id").name.agg(set)
    bio = pd.concat([pd.DataFrame({"id": base[f"{c}_id"], "date": base.date, "hand": base[f"{c}_hand"], "ht": base[f"{c}_ht"],
                                   "ioc": base[f"{c}_ioc"], "age": base[f"{c}_age"]}) for c in ("winner", "loser")])
    bio = bio.dropna(subset=["id"]).sort_values("date").groupby("id").last()
    tstart = base.assign(tid=base.tourney_id.astype(str)).groupby("tid").tourney_date.min()
    fx = O.drop_duplicates("fs_mid").copy()
    QF = os.path.join(D, "fs_raw", "qual_rounds.json")
    nq = qual_rounds((({"ZA": z}, {"ER": e}) for z, e in zip(fx.tournament, fx["round"])),
                     json.load(open(QF)) if os.path.exists(QF) else {})
    rows = []
    for r in fx.itertuples():
        d = {"WU": r.home, "WV": r.away}
        nh, na = player_name(d, "h"), player_name(d, "a")
        hid, aid = ids.get(key(nh)), ids.get(key(na))
        if hid is None or aid is None:  # debutants have no history: no model price
            continue
        day = pd.Timestamp(int(r.start), unit="s")
        chk = {}
        for s_, nm, pid in (("h", nh, hid), ("a", na, aid)):
            toks = [t for kk in id_names.get(pid, ()) for t in kk.split()]
            chk[f"n_{s_}"] = int(n_hist.get(pid, 0))
            chk[f"days_{s_}"] = int((day - last_hist[pid]).days) if pid in last_hist.index else 9999
            chk[f"name_{s_}"] = any(SequenceMatcher(None, t, u).ratio() >= 0.8 for t in key(nm).split() for u in toks)
            chk[f"syn_{s_}"] = pid >= 9e8
            chk[f"id_{s_}"] = int(pid)
        checks[r.fs_mid] = chk
        cat, slug, tname, surf = tournament({"ZA": r.tournament, "ZL": r.path})
        tid = event_id(cat, slug, day.year)
        row = {"tourney_id": tid, "tourney_name": tname, "surface": surf, "draw_size": "", "tourney_level": level(cat, tname),
               "tourney_date": int(tstart.get(tid, int(day.strftime("%Y%m%d")))), "match_num": 999,
               "winner_id": hid, "winner_name": nh.title(), "loser_id": aid, "loser_name": na.title(),
               "score": "6-0 6-0", "best_of": 3, "round": round_code({"ER": r.round}, r.tournament, nq), "fs_mid": r.fs_mid}
        for side, pid in (("winner", hid), ("loser", aid)):
            if pid in bio.index:
                b = bio.loc[pid]
                row.update({f"{side}_hand": b.hand, f"{side}_ht": b.ht, f"{side}_ioc": b.ioc,
                            f"{side}_age": b.age + (day - pd.Timestamp(b.date)).days / 365.25})
            rk, pt = fs_rank.ranks_asof(D, tour, [pid], [day])  # rank / points of the latest Flashscore list (fs_rank.py)
            row.update({f"{side}_rank": rk[0], f"{side}_rank_points": pt[0]})
        rows.append(row)
    if not rows:
        print(f"{tour}: no priceable fixtures")
        continue
    # prediction cache: a fixture goes through features and model only when it is new, when its inputs changed (a new result
    # of either player, rank, round, kick-off day, model) or when the entry is older than CACHE_TTL (ratings of third
    # players drift a little with every result); otherwise the stored P(home) is used
    cpath = os.path.join(PAPER, f"pred_cache_{tour}.pkl")
    cache = pickle.load(open(cpath, "rb")) if os.path.exists(cpath) else {}
    mkey = f"{MODELS}|{TAG}|{LIVE_OUT}|{os.path.getmtime(model_path):.0f}"
    sig = {r_["fs_mid"]: repr((mkey, checks[r_["fs_mid"]], sorted((k_, f"{v_:.1f}" if k_.endswith("_age") else str(v_)) for k_, v_ in r_.items() if k_ != "fs_mid")))
           for r_ in rows}
    now_ = time.time()
    need = [r_ for r_ in rows if cache.get(r_["fs_mid"], (None, 0, 0))[0] != sig[r_["fs_mid"]] or now_ - cache[r_["fs_mid"]][2] > CACHE_TTL]
    print(f"{tour}: {len(rows) - len(need)} of {len(rows)} predictions from the cache, {len(need)} to compute", flush=True)
    if os.environ.get("SIGDEBUG"):
        with open(os.path.join(PAPER, f"sig_debug_{tour}.txt"), "w") as f_:
            for r_ in need[:5]:
                f_.write(f"{r_['fs_mid']}\nOLD {cache.get(r_['fs_mid'], ('none',))[0]}\nNEW {sig[r_['fs_mid']]}\n\n")
    if need:
        pn = predict_rows(tour, need, model_path, ts)
        if pn is None:
            continue
        cache.update({m: (sig[m], float(v), now_) for m, v in pn.items()})
    cache = {m: v for m, v in cache.items() if m in sig}
    pickle.dump(cache, open(cpath, "wb"))
    p_home = pd.Series({m: v[1] for m, v in cache.items()})
    for r in O[O.fs_mid.isin(p_home.index)].itertuples():
        for side, o, oo, cur, cuo in (("home", r.open_h, r.open_a, r.cur_h, r.cur_a), ("away", r.open_a, r.open_h, r.cur_a, r.cur_h)):
            if not (cur > 1 and cuo > 1):
                continue
            # the price on offer now: blend with the Shin fair price of this book's current pair
            q = p_home[r.fs_mid] if side == "home" else 1 - p_home[r.fs_mid]
            qb = 1 / (1 + np.exp(-(0.44 * lg(q) + 0.64 * lg(shin(np.array([1 / cur]), np.array([1 / cuo]))[0]))))
            new_bets.append({"fs_mid": r.fs_mid, "tour": tour, "tournament": r.tournament, "home": r.home, "away": r.away,
                             "start": pd.Timestamp(int(r.start), unit="s"), "side": side, "book": r.book, "odds": cur, "open": o,
                             "q_model": q, "q_blend": qb, "ev": qb * cur - 1, "model": TAG,
                             "fair": shin(np.array([1 / cur]), np.array([1 / cuo]))[0]})
C = pd.DataFrame(new_bets)
if not C.empty:
    C.to_csv(os.path.join(PAPER, "candidates.csv"), index=False)  # every priced side of the pass, to see how close the matches are to the EV threshold
logged, skipped = 0, []


def problems(k, r):
    """why this bet must not be logged ('' = the data checks pass)"""
    if not k:
        return "no data check"
    out = []
    for s_, who in (("h", r.home), ("a", r.away)):
        if k[f"syn_{s_}"] and os.environ.get("SKIP_SYNTHETIC"):
            out.append(f"{who}: no real id (history may be split)")
        elif k[f"n_{s_}"] < MIN_HIST:
            out.append(f"{who}: only {k[f'n_{s_}']} matches of history")
        if not k[f"name_{s_}"]:
            out.append(f"{who}: name does not match the history of the id")
    gap = r.q_model - r.fair
    if gap > BIG_GAP:
        for s_, who in (("h", r.home), ("a", r.away)):
            if k[f"n_{s_}"] < BIG_HIST or k[f"days_{s_}"] > BIG_DAYS:
                out.append(f"model {gap:.0%} above the book and {who} has {k[f'n_{s_}']} matches, last {k[f'days_{s_}']} days ago")
    return "; ".join(out)


# REVIEW=1: a bet that passes the checks is only proposed (DATA_DIR/paper/proposals.csv); it goes to the ledger after a
# manual look at both players (review.py ok / no)
REVIEW = bool(os.environ.get("REVIEW"))
PROP = os.path.join(PAPER, "proposals.csv")
SKIPPED = os.path.join(PAPER, "skipped.csv")
props = pd.read_csv(PROP) if REVIEW and os.path.exists(PROP) else pd.DataFrame(columns=["fs_mid"])
done_ids = set(props.fs_mid) | (set(pd.read_csv(SKIPPED).fs_mid) if REVIEW and os.path.exists(SKIPPED) else set())
proposed = []
if not C.empty:
    for rule, (thr, cap) in RULES.items():
        c = C[(C.ev > thr) & (C.odds <= cap)].sort_values("odds", ascending=False).drop_duplicates(["fs_mid"])
        for r in c.itertuples():
            if not ledger.empty and (ledger.fs_mid == r.fs_mid).any():  # one bet per match, whatever the rule was called
                continue
            if r.fs_mid in done_ids:  # already proposed, or rejected
                continue
            why = problems(checks.get(r.fs_mid, {}), r)
            if why:
                skipped.append({"fs_mid": r.fs_mid, "home": r.home, "away": r.away, "start": r.start, "side": r.side, "book": r.book,
                                "odds": r.odds, "q_model": r.q_model, "ev": r.ev, "reason": why,
                                "seen_at": pd.Timestamp.now("UTC").tz_localize(None)})
                continue
            k = checks[r.fs_mid]
            row = {**r._asdict(), "hist_home": k["n_h"], "hist_away": k["n_a"], "rule": rule,
                   "logged_at": pd.Timestamp.now('UTC').tz_localize(None), "result": "", "pnl": np.nan}
            row.pop("Index")
            if REVIEW:
                proposed.append({**row, "id_home": k["id_h"], "id_away": k["id_a"]})
                print(f"proposed {r.home} - {r.away}: {r.side} {r.odds} {r.book}, model {r.q_model:.0%}, book {r.fair:.0%}")
                continue
            ledger = pd.concat([ledger, pd.DataFrame([row])], ignore_index=True)
            logged += 1
if proposed:
    pd.concat([props, pd.DataFrame(proposed)]).to_csv(PROP, index=False)
if skipped:
    sk = os.path.join(PAPER, "skipped.csv")
    K = pd.concat([pd.read_csv(sk), pd.DataFrame(skipped)]) if os.path.exists(sk) else pd.DataFrame(skipped)
    K.drop_duplicates(["fs_mid", "reason"]).to_csv(sk, index=False)
    for x in skipped:
        print(f"not logged {x['home']} - {x['away']} ({x['side']} {x['odds']}): {x['reason']}")
def move_start(ledger, starts):
    """starts: fs_mid -> Flashscore start (unix s); open bets whose start differs get the new one."""
    for i in ledger.index[ledger.result.fillna("") == ""]:
        t = starts.get(ledger.at[i, "fs_mid"])
        if not t:
            continue
        new = str(pd.Timestamp(int(t), unit="s"))
        if new != str(pd.Timestamp(ledger.at[i, "start"])):
            print(f"start moved {ledger.at[i, 'home']} - {ledger.at[i, 'away']}: {ledger.at[i, 'start']} -> {new} UTC")
            ledger.at[i, "start"] = new


# pending bets: the book's latest price while the match has not started is the closing price so far (CLV)
if not ledger.empty and seen:
    S = pd.concat(seen)
    S = S[S.start > time.time()]
    cur = {(r.fs_mid, r.book): (r.cur_h, r.cur_a) for r in S.itertuples()}
    # kick-off times move (rain, a long match before, a new order of play): every pass takes Flashscore's current one
    move_start(ledger, dict(zip(pd.concat(seen).fs_mid, pd.concat(seen).start)))
    for c in ("close", "close_seen_at"):
        if c not in ledger:
            ledger[c] = np.nan
    ledger["close_seen_at"] = ledger.close_seen_at.astype(object)
    for i in ledger.index[ledger.result.fillna("") == ""]:
        p = cur.get((ledger.at[i, "fs_mid"], ledger.at[i, "book"]))
        if p:
            ledger.at[i, "close"] = p[0] if ledger.at[i, "side"] == "home" else p[1]
            ledger.at[i, "close_seen_at"] = str(pd.Timestamp.now('UTC').tz_localize(None))[:16]
    # every book's latest price for our side while the match has not started (px_<book>, shown in the Excel sheet)
    for bk in BOOKS:
        c = f"px_{bk}"
        if c not in ledger:
            ledger[c] = np.nan
        for i in ledger.index[ledger.result.fillna("") == ""]:
            p = cur.get((ledger.at[i, "fs_mid"], bk))
            if p:
                v = p[0] if ledger.at[i, "side"] == "home" else p[1]
                ledger.at[i, c] = v if v > 1 else np.nan
# settle finished matches from the daily results feeds (today and the 3 days before)
if not ledger.empty:
    done, starts = {}, {}
    for day in range(-3, 2):
        try:
            with urllib.request.urlopen(urllib.request.Request(f"https://www.flashscore.com/x/feed/f_2_{day}_1_en_1", headers=UA),
                                        timeout=30) as resp:
                text = resp.read().decode("utf-8", "ignore")
            for head, d in records(text):
                if d.get("AD"):
                    starts[d["AA"]] = d["AD"]
                if d.get("AB") == "3":
                    hs, as_ = int(d.get("AG", 0) or 0), int(d.get("AH", 0) or 0)
                    done[d["AA"]] = ("void" if d.get("AC") not in ("3",) else ("home" if hs > as_ else "away"))
        except Exception as e:
            print("settlement feed error", e)
    move_start(ledger, starts)  # also matches already under way or no longer listed with odds
    open_bets = ledger.result.fillna("") == ""
    # the daily feeds lag the match's own feed by up to ~20 min; ask each started open bet directly
    for mid in ledger.fs_mid[open_bets & (pd.to_datetime(ledger.start) < pd.Timestamp.now("UTC").tz_localize(None))]:
        if mid in done:
            continue
        try:
            h = dict(UA); h["x-fsign"] = "SW9D1eZo"
            with urllib.request.urlopen(urllib.request.Request(f"https://www.flashscore.com/x/feed/dc_1_{mid}", headers=h),
                                        timeout=20) as resp:
                d = dict(re.findall(r"(\w+)÷([^¬]*)", resp.read().decode("utf-8", "ignore")))
            if d.get("DA") == "3":
                hs, as_ = int(d.get("DE", 0) or 0), int(d.get("DF", 0) or 0)
                done[mid] = "void" if d.get("DB") != "3" else ("home" if hs > as_ else "away")
        except Exception as e:
            print("match feed error", mid, e)
    # when the loop first saw the result: from then on the bank includes this bet (stakes of later bets only)
    ledger["settled_at"] = ledger["settled_at"].astype(object) if "settled_at" in ledger else pd.Series(index=ledger.index, dtype=object)
    ledger["result"] = ledger["result"].astype(object)  # all empty until the first settlement, so read_csv makes it float
    for i in ledger.index[open_bets]:
        res = done.get(ledger.at[i, "fs_mid"])
        if res:
            ledger.at[i, "result"] = res
            ledger.at[i, "settled_at"] = pd.Timestamp.now("UTC").strftime("%Y-%m-%d %H:%M:%S")
            ledger.at[i, "pnl"] = 0.0 if res == "void" else (ledger.at[i, "odds"] - 1 if res == ledger.at[i, "side"] else -1.0)
    ledger.to_csv(LEDGER, index=False)
if seen:
    S = pd.concat(seen)
    S["day"] = pd.to_datetime(S.start, unit="s").dt.strftime("%m-%d")
    S["at_open"] = (S.open_h == S.cur_h) & (S.open_a == S.cur_a)
    S = S[S.start > time.time()]
    cov = S.groupby("day").agg(matches=("fs_mid", "nunique"))
    cov["some_book_at_open"] = S[S.at_open].groupby("day").fs_mid.nunique()
    cov["priced_by_model"] = C.drop_duplicates("fs_mid").assign(day=lambda x: x.start.dt.strftime("%m-%d")).groupby("day").size() \
        if not C.empty else 0
    print("upcoming matches with odds:\n" + cov.fillna(0).astype(int).to_string())
print(f"paper trading run {pd.Timestamp.now('UTC'):%Y-%m-%d %H:%M} UTC: {logged} new paper bets, {len(proposed)} proposed for review, {time.time() - t0:.0f}s")
if not ledger.empty:
    ledger["clv"] = ledger.odds / ledger.close - 1 if "close" in ledger else np.nan
    s = ledger.groupby("rule").agg(bets=("fs_mid", "size"), settled=("pnl", "count"), units=("pnl", "sum"), roi=("pnl", "mean"),
                                   clv=("clv", "mean"))
    print(s.round(3).to_string())
    # every competition is in the plan (since 2026-10-11 evening); summary per Flashscore category
    seg = ledger.tournament.str.split(":").str[0].str.strip().map({"ITF WOMEN - SINGLES": "women ITF",
          "CHALLENGER MEN - SINGLES": "men Challenger", "ITF MEN - SINGLES": "men ITF", "CHALLENGER WOMEN - SINGLES": "WTA 125",
          "WTA - SINGLES": "WTA tour", "ATP - SINGLES": "ATP tour"}).fillna("other")
    s = ledger.assign(seg=seg).groupby(["seg", "rule"]).agg(bets=("fs_mid", "size"), settled=("pnl", "count"),
                                                            units=("pnl", "sum"), roi=("pnl", "mean"), clv=("clv", "mean"))
    print(s.round(3).to_string())
