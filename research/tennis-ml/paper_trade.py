"""Paper trading (simulation without money) of the opening-odds strategy on upcoming matches.

Each run (meant to run every 30 min, so opening lines are caught while they still stand):
  0. results of the last days appended to the feature build in seconds (fs_update.py)
  1. every scheduled singles match Flashscore lists (today and as many days ahead as are published) with every UK
     bookmaker's opening and current home/away odds (fs_odds.py --fixtures)
  2. fixture rows in Sackmann format (players mapped to ids, bio from their latest match) -> features in seconds
     with features_incr.py (dry run, the base build is not changed)
  3. production model (train_final.py) -> P(home); betting-stage blend with the Shin fair opening price
  4. for every rule, the best book whose CURRENT price still equals its OPENING price and passes the rule is logged
     once per match as a 1-unit paper bet at that price
  5. pending bets: the book's latest price before the start is kept as the closing price (CLV); finished matches are
     settled from Flashscore's daily results feed
Ledger: DATA_DIR/paper/ledger.csv. Summary printed per rule.

usage: paper_trade.py DATA_DIR [DAYS_AHEAD=7]   (env SKIP_UPDATE=1: no results update)
env LIVE_OUT=feature build dir (default out_live), MODELS=model dir (default models), NX=1: add the new-feature modules
(fx_live.py) to the fixture rows, MODEL_TAG=label stored with each logged bet
"""
import os
import pickle
import subprocess
import sys
import time
import urllib.request

import numpy as np
import pandas as pd

from fs_common import UA, NameIds, key, level, player_name, records, round_code, tournament

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
RULES = {"EV>15% odds<=5": (0.15, 5.0), "EV>10% odds<=2": (0.10, 2.0), "EV>5% odds<=5": (0.05, 5.0), "EV>5% odds<=2": (0.05, 2.0)}
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
jobs = {t: subprocess.Popen([sys.executable, os.path.join(HERE, "fs_odds.py"), os.path.join(D, "fs_raw"), t, "GB", "--fixtures", str(DAYS)],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) for t in ("wta", "atp")}
if not os.environ.get("SKIP_UPDATE"):
    jobs["update"] = subprocess.Popen([sys.executable, os.path.join(HERE, "fs_update.py"), D, "3"],
                                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for name, j in jobs.items():
    out = j.communicate()[0].strip()
    print(f"[{name}] " + (out.splitlines()[-1] if j.returncode == 0 and name != "update" else out[-800:]), flush=True)
ledger = pd.read_csv(LEDGER) if os.path.exists(LEDGER) else pd.DataFrame()
new_bets, seen = [], []
for tour in ("wta", "atp"):
    model_path = os.path.join(D, MODELS, f"{tour}_final.pkl")
    if not os.path.exists(model_path):
        print(f"{tour}: no production model yet ({model_path})")
        continue
    O = pd.read_parquet(os.path.join(D, "fs_raw", f"{tour}_fs_fixture_odds_GB.parquet"))
    if O.empty:
        print(f"{tour}: no fixtures with odds yet")
        continue
    seen.append(O.assign(tour=tour))
    O = O[O.start > time.time()]  # not started
    base = pd.read_parquet(os.path.join(D, LIVE_OUT, f"{tour}_matches.parquet"))
    ids = NameIds()
    for c in ("winner", "loser"):
        ids.add(base[f"{c}_name"].map(key), base[f"{c}_id"], base.tourney_date.astype(float).values)
    bio = pd.concat([pd.DataFrame({"id": base[f"{c}_id"], "date": base.date, "hand": base[f"{c}_hand"], "ht": base[f"{c}_ht"],
                                   "ioc": base[f"{c}_ioc"], "age": base[f"{c}_age"]}) for c in ("winner", "loser")])
    bio = bio.dropna(subset=["id"]).sort_values("date").groupby("id").last()
    tstart = base.assign(tid=base.tourney_id.astype(str)).groupby("tid").tourney_date.min()
    fx = O.drop_duplicates("fs_mid").copy()
    rows = []
    for r in fx.itertuples():
        d = {"WU": r.home, "WV": r.away}
        nh, na = player_name(d, "h"), player_name(d, "a")
        hid, aid = ids.get(key(nh)), ids.get(key(na))
        if hid is None or aid is None:  # debutants have no history: no model price
            continue
        if os.environ.get("SKIP_SYNTHETIC") and max(hid, aid) >= 9e8:  # player whose history may be split over two ids
            continue
        cat, slug, tname, surf = tournament({"ZA": r.tournament, "ZL": r.path})
        day = pd.Timestamp(int(r.start), unit="s")
        tid = f"fs-{slug}-{day.year}"
        row = {"tourney_id": tid, "tourney_name": tname, "surface": surf, "draw_size": "", "tourney_level": level(cat, tname),
               "tourney_date": int(tstart.get(tid, int(day.strftime("%Y%m%d")))), "match_num": 999,
               "winner_id": hid, "winner_name": nh.title(), "loser_id": aid, "loser_name": na.title(),
               "score": "6-0 6-0", "best_of": 3, "round": round_code({"ER": r.round}), "fs_mid": r.fs_mid}
        for side, pid in (("winner", hid), ("loser", aid)):
            if pid in bio.index:
                b = bio.loc[pid]
                row.update({f"{side}_hand": b.hand, f"{side}_ht": b.ht, f"{side}_ioc": b.ioc,
                            f"{side}_age": b.age + (day - pd.Timestamp(b.date)).days / 365.25})
        rows.append(row)
    if not rows:
        print(f"{tour}: no priceable fixtures")
        continue
    fcsv = os.path.join(PAPER, f"{tour}__{tour}_matches_fs_fixtures.csv")
    pd.DataFrame(rows).to_csv(fcsv, index=False)
    fout = os.path.join(PAPER, f"{tour}_fixture_features.parquet")
    fnew = os.path.join(PAPER, f"{tour}_fixture_rows.parquet")
    env = dict(os.environ, DRY_OUT=fout, DRY_MATCHES=fnew)
    r = subprocess.run([sys.executable, os.path.join(HERE, "features_incr.py"), os.path.join(D, "live_raw"), os.path.join(D, LIVE_OUT),
                        tour, fcsv], env=env, capture_output=True, text=True)
    if r.returncode:
        print(f"{tour}: features_incr failed\n{r.stderr[-1500:]}")
        continue
    Fx = pd.read_parquet(fout)
    if NX:
        fnx = os.path.join(PAPER, f"{tour}_fixture_nx.parquet")
        r = subprocess.run([sys.executable, os.path.join(HERE, "fx_live.py"), os.path.join(D, LIVE_OUT), tour, fnew, fnx],
                           capture_output=True, text=True)
        if r.returncode:
            print(f"{tour}: fx_live failed\n{r.stderr[-1500:]}")
            continue
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
    for r in O[O.fs_mid.isin(p_home.index)].itertuples():
        for side, o, oo, cur, cuo in (("home", r.open_h, r.open_a, r.cur_h, r.cur_a), ("away", r.open_a, r.open_h, r.cur_a, r.cur_h)):
            if not (o > 1 and oo > 1):
                continue
            q = p_home[r.fs_mid] if side == "home" else 1 - p_home[r.fs_mid]
            qb = 1 / (1 + np.exp(-(0.44 * lg(q) + 0.64 * lg(shin(np.array([1 / o]), np.array([1 / oo]))[0]))))
            new_bets.append({"fs_mid": r.fs_mid, "tour": tour, "tournament": r.tournament, "home": r.home, "away": r.away,
                             "start": pd.Timestamp(int(r.start), unit="s"), "side": side, "book": r.book, "open": o, "current": cur,
                             "at_open": abs(cur - o) < 1e-9, "q_model": q, "q_blend": qb, "ev": qb * o - 1, "model": TAG,
                             "cur_other": cuo})
C = pd.DataFrame(new_bets)
ALL = C.copy()  # every book's price, also the ones that moved since opening (re-check of pending bets below)
C = C.drop(columns="cur_other", errors="ignore")
logged = 0
if not C.empty:
    C = C[C.at_open]
    for rule, (thr, cap) in RULES.items():
        c = C[(C.ev > thr) & (C.open <= cap)].sort_values("open", ascending=False).drop_duplicates(["fs_mid"])
        for r in c.itertuples():
            if not ledger.empty and ((ledger.fs_mid == r.fs_mid) & (ledger.rule == rule)).any():
                continue
            ledger = pd.concat([ledger, pd.DataFrame([{**r._asdict(), "rule": rule, "logged_at": pd.Timestamp.now('UTC').tz_localize(None),
                                                       "result": "", "pnl": np.nan}]).drop(columns=["Index"])], ignore_index=True)
            logged += 1
# pending bets: the book's latest price while the match has not started is the closing price so far (CLV)
if not ledger.empty and seen:
    S = pd.concat(seen)
    S = S[S.start > time.time()]
    cur = {(r.fs_mid, r.book): (r.cur_h, r.cur_a) for r in S.itertuples()}
    for c in ("close", "close_seen_at"):
        if c not in ledger:
            ledger[c] = np.nan
    ledger["close_seen_at"] = ledger.close_seen_at.astype(object)
    for i in ledger.index[ledger.result.fillna("") == ""]:
        p = cur.get((ledger.at[i, "fs_mid"], ledger.at[i, "book"]))
        if p:
            ledger.at[i, "close"] = p[0] if ledger.at[i, "side"] == "home" else p[1]
            ledger.at[i, "close_seen_at"] = str(pd.Timestamp.now('UTC').tz_localize(None))[:16]
# pending bets re-priced now: the current model (fed with every result up to now) and the current prices of every book.
# q_blend_now: blend with the Shin fair price of the bet's book now (the median over books if it no longer lists it);
# ev_hold: what the bet already placed at its entry odds is worth now; best_now / ev_now / enter_now: would the rule
# log this side now at the best current price (EV>5%, odds<=5; the opening-price condition left out).
# ev_entry_new / enter_entry: the same bet as the test rule sees it, the current model at the entry odds with the Shin
# price of the entry book's opening pair (backed out of the logged blend): would the current model have taken this bet?
# A bet logged by the old model that it would not take leaves the plan (Excel); the last check before kick-off counts.
if not ledger.empty and not ALL.empty:
    A = ALL[(ALL.current > 1) & (ALL.cur_other > 1)].copy()
    A["qb_now"] = 1 / (1 + np.exp(-(0.44 * lg(A.q_model.values) + 0.64 * lg(shin(1 / A.current.values, 1 / A.cur_other.values)))))
    for c in ("q_now", "q_blend_now", "ev_hold", "best_now", "book_now", "ev_now", "enter_now", "rechecked_at", "ev_entry_new",
              "enter_entry"):
        if c not in ledger:
            ledger[c] = np.nan
    for c in ("book_now", "enter_now", "rechecked_at", "enter_entry"):
        ledger[c] = ledger[c].astype(object)
    started = pd.to_datetime(ledger.start) <= pd.Timestamp.now("UTC").tz_localize(None)
    for i in ledger.index[(ledger.result.fillna("") == "") & ~started]:
        a = A[(A.fs_mid == ledger.at[i, "fs_mid"]) & (A.side == ledger.at[i, "side"])]
        if a.empty:
            continue
        own = a[a.book == ledger.at[i, "book"]]
        qb = float(own.qb_now.iloc[0]) if len(own) else float(a.qb_now.median())
        best = a.sort_values("current", ascending=False).iloc[0]
        ev_now = qb * best.current - 1
        ledger.loc[i, ["q_now", "q_blend_now", "ev_hold", "best_now", "book_now", "ev_now", "enter_now", "rechecked_at"]] = [
            float(a.q_model.iloc[0]), qb, qb * ledger.at[i, "open"] - 1, float(best.current), best.book, ev_now,
            "yes" if ev_now > 0.05 and best.current <= 5 else "no", str(pd.Timestamp.now("UTC").tz_localize(None))[:16]]
        shin_lo = (lg(ledger.at[i, "q_blend"]) - 0.44 * lg(ledger.at[i, "q_model"])) / 0.64
        ev_entry = 1 / (1 + np.exp(-(0.44 * lg(ledger.at[i, "q_now"]) + 0.64 * shin_lo))) * ledger.at[i, "open"] - 1
        thr = float(ledger.at[i, "rule"].split("%")[0].split(">")[1]) / 100
        ledger.loc[i, ["ev_entry_new", "enter_entry"]] = [ev_entry, "yes" if ev_entry > thr else "no"]
# settle finished matches from the daily results feeds (today and the 3 days before)
if not ledger.empty:
    done = {}
    for day in range(-3, 1):
        try:
            with urllib.request.urlopen(urllib.request.Request(f"https://www.flashscore.com/x/feed/f_2_{day}_1_en_1", headers=UA),
                                        timeout=30) as resp:
                text = resp.read().decode("utf-8", "ignore")
            for head, d in records(text):
                if d.get("AB") == "3":
                    hs, as_ = int(d.get("AG", 0) or 0), int(d.get("AH", 0) or 0)
                    done[d["AA"]] = ("void" if d.get("AC") not in ("3",) else ("home" if hs > as_ else "away"))
        except Exception as e:
            print("settlement feed error", e)
    open_bets = ledger.result.fillna("") == ""
    # when the loop first saw the result: from then on the bank includes this bet (stakes of later bets only)
    ledger["settled_at"] = ledger["settled_at"].astype(object) if "settled_at" in ledger else pd.Series(index=ledger.index, dtype=object)
    for i in ledger.index[open_bets]:
        res = done.get(ledger.at[i, "fs_mid"])
        if res:
            ledger.at[i, "result"] = res
            ledger.at[i, "settled_at"] = pd.Timestamp.now("UTC").strftime("%Y-%m-%d %H:%M:%S")
            ledger.at[i, "pnl"] = 0.0 if res == "void" else (ledger.at[i, "open"] - 1 if res == ledger.at[i, "side"] else -1.0)
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
print(f"paper trading run {pd.Timestamp.now('UTC'):%Y-%m-%d %H:%M} UTC: {logged} new paper bets, {time.time() - t0:.0f}s")
if not ledger.empty:
    ledger["clv"] = ledger.open / ledger.close - 1 if "close" in ledger else np.nan
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
