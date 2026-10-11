"""Paper trading (simulation without money) of the opening-odds strategy on upcoming matches.

Each run:
  1. upcoming singles fixtures with every UK bookmaker's opening and current home/away odds (fs_odds.py --fixtures)
  2. fixture rows in Sackmann format (players mapped to ids, bio from their latest match) -> features in seconds
     with features_incr.py (dry run, the base build is not changed)
  3. production model (train_final.py) -> P(home); betting-stage blend with the Shin fair opening price
  4. for every rule, the best book whose CURRENT price still equals its OPENING price and passes the rule is logged
     once per match as a 1-unit paper bet at that price
  5. earlier paper bets are settled from Flashscore's daily results feed
Ledger: DATA_DIR/paper/ledger.csv. Summary printed per rule.

usage: paper_trade.py DATA_DIR [DAYS_AHEAD=1]
"""
import json
import os
import re
import subprocess
import sys
import time
import unicodedata
import urllib.request

import numpy as np
import pandas as pd

D = sys.argv[1]
DAYS = int(sys.argv[2]) if len(sys.argv) > 2 else 1
HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.join(D, "paper")
os.makedirs(PAPER, exist_ok=True)
LEDGER = os.path.join(PAPER, "ledger.csv")
RULES = {"EV>15% odds<=5": (0.15, 5.0), "EV>10% odds<=2": (0.10, 2.0), "EV>5% odds<=5": (0.05, 5.0), "EV>5% odds<=2": (0.05, 2.0)}
SEGMENTS = {"ITF WOMEN - SINGLES": ("wta", "itf"), "CHALLENGER WOMEN - SINGLES": ("wta", "P"), "WTA - SINGLES": ("wta", "P"),
            "ITF MEN - SINGLES": ("atp", "itf"), "CHALLENGER MEN - SINGLES": ("atp", "C"), "ATP - SINGLES": ("atp", "A")}
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36",
      "x-fsign": "SW9D1eZo"}
lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))


def key(name):
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    return " ".join(sorted(re.sub(r"[^a-z]+", " ", s).split()))


def shin(a, b):
    lo, hi = np.zeros_like(a), np.full_like(a, 0.5)
    B = a + b
    f = lambda z, x: (np.sqrt(z ** 2 + 4 * (1 - z) * x ** 2 / B) - z) / (2 * (1 - z))
    for _ in range(50):
        z = (lo + hi) / 2
        s = f(z, a) + f(z, b)
        hi, lo = np.where(s < 1, z, hi), np.where(s < 1, lo, z)
    return f((lo + hi) / 2, a)


def feed(day):
    with urllib.request.urlopen(urllib.request.Request(f"https://www.flashscore.com/x/feed/f_2_{day}_1_en_1", headers=UA), timeout=30) as r:
        text = r.read().decode("utf-8", "ignore")
    head = None
    for rec in text.split("~"):
        d = dict(x.split("÷", 1) for x in rec.split("¬") if "÷" in x)
        if "ZA" in d:
            head = d
        elif "AA" in d and head is not None:
            yield head, d


t0 = time.time()
ledger = pd.read_csv(LEDGER) if os.path.exists(LEDGER) else pd.DataFrame()
new_bets = []
for tour in ("wta", "atp"):
    model_path = os.path.join(D, "models", f"{tour}_final.pkl")
    if not os.path.exists(model_path):
        print(f"{tour}: no production model yet ({model_path})")
        continue
    subprocess.run([sys.executable, os.path.join(HERE, "fs_odds.py"), os.path.join(D, "fs_raw"), tour, "GB", "--fixtures", str(DAYS)],
                   check=True, capture_output=True)
    O = pd.read_parquet(os.path.join(D, "fs_raw", f"{tour}_fs_fixture_odds_GB.parquet"))
    if O.empty:
        print(f"{tour}: no fixtures with odds yet")
        continue
    O = O[O.tournament.str.split(":").str[0].str.strip().isin(SEGMENTS)]
    O = O[O.start > time.time()]  # not started
    base = pd.read_parquet(os.path.join(D, "out_live", f"{tour}_matches.parquet"))
    ids = {}
    for c in ("winner", "loser"):
        ids.update(zip(base[f"{c}_name"].map(key), base[f"{c}_id"]))
    bio = pd.concat([pd.DataFrame({"id": base[f"{c}_id"], "date": base.date, "hand": base[f"{c}_hand"], "ht": base[f"{c}_ht"],
                                   "ioc": base[f"{c}_ioc"], "age": base[f"{c}_age"]}) for c in ("winner", "loser")])
    bio = bio.dropna(subset=["id"]).sort_values("date").groupby("id").last()
    tstart = base.assign(tid=base.tourney_id.astype(str)).groupby("tid").tourney_date.min()
    fx = O.drop_duplicates("fs_mid").copy()
    rows = []
    for r in fx.itertuples():
        nh, na = (re.sub(r"(\s\d+)+$", "", str(n).replace("-", " ")) for n in (r.home, r.away))
        hid, aid = ids.get(key(nh)), ids.get(key(na))
        if hid is None or aid is None:  # debutants have no history: no model price
            continue
        za = r.tournament.split(":", 1)[-1].strip()
        tname = re.sub(r"\s*\(.*$", "", za).strip()
        surf = za.rsplit(",", 1)[-1].strip().capitalize() if "," in za else "Hard"
        cat, lvl = SEGMENTS[r.tournament.split(":")[0].strip()]
        m = re.match(r"[WM](\d+)", tname)
        level = (m.group(1) if m else "15") if lvl == "itf" else lvl
        day = pd.Timestamp(int(r.start), unit="s")
        slug = r.path.strip("/").split("/")[-1] if r.path else re.sub(r"[^a-z0-9]+", "-", tname.lower())
        tid = f"fs-{slug}-{day.year}"
        row = {"tourney_id": tid, "tourney_name": tname, "surface": surf if surf in ("Clay", "Grass", "Carpet") else "Hard",
               "draw_size": "", "tourney_level": level, "tourney_date": int(tstart.get(tid, int(day.strftime("%Y%m%d")))),
               "match_num": 999, "winner_id": hid, "winner_name": nh.title(), "loser_id": aid, "loser_name": na.title(),
               "score": "6-0 6-0", "best_of": 3, "round": "R32", "fs_mid": r.fs_mid}
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
    env = dict(os.environ, DRY_OUT=fout)
    subprocess.run([sys.executable, os.path.join(HERE, "features_incr.py"), os.path.join(D, "live_raw"), os.path.join(D, "out_live"),
                    tour, fcsv], check=True, env=env, capture_output=True)
    Fx = pd.read_parquet(fout)
    sys.path.insert(0, HERE)
    from design import build_X
    M = __import__("pickle").load(open(model_path, "rb"))
    X, _, flip = build_X(Fx, tour)
    X["level"] = pd.Categorical(Fx.tourney_level.astype(str), categories=M["levels"]).codes
    X = X[M["features"]]
    p = np.mean([M["models"]["xgb"].predict_proba(X)[:, 1], M["models"]["lgbm"].predict_proba(X)[:, 1],
                 M["models"]["logit"].predict_proba(np.nan_to_num(lg(X[M["pcols"]].values.astype(float))))[:, 1]], axis=0)
    p = M["cal"].predict_proba(lg(p)[:, None])[:, 1]
    p_home = pd.Series(np.where(flip, 1 - p, p), index=Fx.fs_mid.values)
    for r in O[O.fs_mid.isin(p_home.index)].itertuples():
        for side, o, oo, cur in (("home", r.open_h, r.open_a, r.cur_h), ("away", r.open_a, r.open_h, r.cur_a)):
            if not (o > 1 and oo > 1):
                continue
            q = p_home[r.fs_mid] if side == "home" else 1 - p_home[r.fs_mid]
            qb = 1 / (1 + np.exp(-(0.44 * lg(q) + 0.64 * lg(shin(np.array([1 / o]), np.array([1 / oo]))[0]))))
            new_bets.append({"fs_mid": r.fs_mid, "tour": tour, "tournament": r.tournament, "home": r.home, "away": r.away,
                             "start": pd.Timestamp(int(r.start), unit="s"), "side": side, "book": r.book, "open": o, "current": cur,
                             "at_open": abs(cur - o) < 1e-9, "q_model": q, "q_blend": qb, "ev": qb * o - 1})
C = pd.DataFrame(new_bets)
logged = 0
if not C.empty:
    C = C[C.at_open]
    for rule, (thr, cap) in RULES.items():
        c = C[(C.ev > thr) & (C.open <= cap)].sort_values("open", ascending=False).drop_duplicates(["fs_mid"])
        for r in c.itertuples():
            if not ledger.empty and ((ledger.fs_mid == r.fs_mid) & (ledger.rule == rule)).any():
                continue
            ledger = pd.concat([ledger, pd.DataFrame([{**r._asdict(), "rule": rule, "logged_at": pd.Timestamp.utcnow().tz_localize(None),
                                                       "result": "", "pnl": np.nan}]).drop(columns=["Index"])], ignore_index=True)
            logged += 1
# settle finished matches from the daily results feeds (today and the 3 days before)
if not ledger.empty:
    done = {}
    for day in range(-3, 1):
        try:
            for head, d in feed(day):
                if d.get("AB") == "3":
                    hs, as_ = int(d.get("AG", 0) or 0), int(d.get("AH", 0) or 0)
                    done[d["AA"]] = ("void" if d.get("AC") not in ("3",) else ("home" if hs > as_ else "away"))
        except Exception as e:
            print("settlement feed error", e)
    open_bets = ledger.result.fillna("") == ""
    for i in ledger.index[open_bets]:
        res = done.get(ledger.at[i, "fs_mid"])
        if res:
            ledger.at[i, "result"] = res
            ledger.at[i, "pnl"] = 0.0 if res == "void" else (ledger.at[i, "open"] - 1 if res == ledger.at[i, "side"] else -1.0)
    ledger.to_csv(LEDGER, index=False)
print(f"paper trading run {pd.Timestamp.utcnow():%Y-%m-%d %H:%M} UTC: {logged} new paper bets, {time.time() - t0:.0f}s")
if not ledger.empty:
    s = ledger.groupby("rule").agg(bets=("fs_mid", "size"), settled=("pnl", "count"), units=("pnl", "sum"), roi=("pnl", "mean"))
    print(s.round(3).to_string())
