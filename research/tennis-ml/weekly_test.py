"""Weekly / monthly / yearly tables of the plan test for the dashboard "Tenis test po nedeljama".
Plan rule (blend EV>5%, odds<=5, no ITF) from the eval_levels2026.py --bets parquet files; quarter Kelly, each month starts
from 300,000, at most 30,000 per bet (kelly_cap.py `plan`). 2024 / 2025 come from fs_hist, 2026 from the holdout of the
current model.

usage: weekly_test.py DATA_DIR BETS_2026_PARQUET FS_2026_DIR OUT_DIR
(writes nedelje.json, meseci.json, godine.json and opklade.json, one row per bet with players, tournament, odds, stake, result)
"""
import json
import os
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
B, P26, FS26, OUT = sys.argv[1:5]
sys.argv = ["x"]
ns = {}
exec(open(os.path.join(HERE, "kelly_cap.py")).read().split("rows = []")[0], ns)
START, CAP = 300000.0, 30000.0
books = {2024: 3, 2025: 5, 2026: 6}
weeks, months, years, bets = [], [], [], []
LVL = {"atp": {"G": "ATP Grand Slam", "M": "ATP Masters 1000", "A": "ATP 250/500", "C": "ATP Challenger", "F": "ATP Finals", "O": "Olimpijske igre"},
       "wta": {"G": "WTA Grand Slam", "PM": "WTA 1000", "P": "WTA 500", "I": "WTA 250", "C": "WTA 125", "F": "WTA Finals", "W": "WTA Finals", "O": "Olimpijske igre"}}


def flip(score):
    out = []
    for t in str(score).split():
        m = re.match(r"^(\d+)-(\d+)(\(\d+\))?$", t)
        out.append(f"{m.group(2)}-{m.group(1)}{m.group(3) or ''}" if m else t)
    return " ".join(out)

for y, p, d in [(2024, f"{B}/fs_hist/y2024.parquet", f"{B}/fs_hist/2024"), (2025, f"{B}/fs_hist/y2025.parquet", f"{B}/fs_hist/2025"),
                (2026, P26, FS26)]:
    b = ns["plan"](p)
    mt = pd.concat([pd.read_csv(f"{d}/{t}__{t}_matches_fs_{y}.csv", usecols=["fs_mid", "tourney_name", "round", "score", "winner_name", "loser_name"])
                    for t in ("atp", "wta")]).drop_duplicates("fs_mid")
    b = b.merge(mt, on="fs_mid", how="left")
    assert b.winner_name.notna().all()
    b["day"] = pd.to_datetime(b.day)
    b["mon"] = b.day.dt.to_period("M")
    b["stake"] = 0.0
    b["bank"] = 0.0
    last_month = str(b.mon.max())
    for m, gm in b.groupby("mon"):
        bank = low = START
        for d, g in gm.groupby("day"):
            st = np.minimum(g.f.values * bank, CAP)
            b.loc[g.index, "stake"] = st
            bank += float((st * g.pnl.values).sum())
            low = min(low, bank)
            b.loc[g.index, "bank"] = bank
        st = b.loc[gm.index]
        months.append(dict(godina=y, mesec=str(m), opklade=len(gm), dobijeno=int((gm.pnl > 0).sum()), ulozeno=round(st.stake.sum()),
                           dobit=round(bank - START), roi=round((bank - START) / st.stake.sum(), 4), kraj=round(bank), najnize=round(low),
                           pun=bool(gm.day.max().day >= 25 or y < 2026 or str(m) != last_month)))
    b["wk"] = b.day.dt.to_period("W-SUN")
    for (m, w), g in b.groupby(["mon", "wk"]):
        dob = float((g.stake * g.pnl).sum())
        weeks.append(dict(godina=y, mesec=str(m), nedelja=str(g.day.min().date()), do=str(g.day.max().date()), opklade=len(g),
                          dobijeno=int((g.pnl > 0).sum()), izgubljeno=int((g.pnl <= 0).sum()), pogodak=round(float((g.pnl > 0).mean()), 4),
                          prosecna_kvota=round(float(g.o.mean()), 2), ulozeno=round(g.stake.sum()), dobit=round(dob),
                          roi=round(dob / g.stake.sum(), 4), banka=round(float(g.bank.iloc[-1]))))
    wk_first = b.groupby(["mon", "wk"]).day.transform("min")
    for i, r in b.iterrows():
        won = bool(r.pnl > 0)
        bets.append(dict(id=r.fs_mid + ("_w" if won else "_l"), godina=y, mesec=str(r.mon), nedelja=str(wk_first[i].date()), dan=str(r.day.date()),
                         vreme=pd.to_datetime(r.start, unit="s", utc=True).tz_convert("Europe/Belgrade").strftime("%H:%M"),
                         kat=LVL[r.tour].get(str(r.lvl), str(r.lvl)), turnir=r.tourney_name, runda=r["round"],
                         igrac=r.winner_name if won else r.loser_name, protivnik=r.loser_name if won else r.winner_name,
                         kvota=round(float(r.o), 2), kladionica=str(r.book).replace(".rs", "").replace("Mozzartbetrs", "Mozzart"),
                         model=round(float(r.q), 4), p=round(float((r.evb + 1) / r.o), 4), ev=round(float(r.evb), 4),
                         ulog=round(float(r.stake)), ishod="D" if won else "I", dobit=round(float(r.stake * r.pnl)),
                         skor=str(r.score) if won else flip(r.score), kapital=round(float(r.bank))))
    M = [x for x in months if x["godina"] == y]
    Wk = [x for x in weeks if x["godina"] == y]
    full = [x for x in M if x["pun"]]
    ww = min([x for x in Wk if x["opklade"] >= 10], key=lambda x: x["dobit"])
    wm, bm = min(full, key=lambda x: x["dobit"]), max(full, key=lambda x: x["dobit"])
    years.append(dict(godina=y, kladionice=books[y], opklade=len(b), dobijeno=int((b.pnl > 0).sum()), dobit=sum(x["dobit"] for x in M),
                      ulozeno=sum(x["ulozeno"] for x in M), plus_meseci=sum(x["dobit"] > 0 for x in full), meseci=len(full),
                      prosek_mesec=round(np.mean([x["dobit"] for x in full])), najgori_mesec=wm["mesec"], najgori_mesec_dobit=wm["dobit"],
                      najbolji_mesec=bm["mesec"], najbolji_mesec_dobit=bm["dobit"], plus_nedelje=sum(x["dobit"] > 0 for x in Wk),
                      nedelje=len(Wk), najgora_nedelja=ww["nedelja"], najgora_dobit=ww["dobit"]))
for x in years:
    x["roi"] = round(x["dobit"] / x["ulozeno"], 4)
    print(x)
assert len({w["nedelja"] for w in weeks}) == len(weeks)
os.makedirs(OUT, exist_ok=True)
assert len({x["id"] for x in bets}) == len(bets)
for name, obj in [("nedelje", weeks), ("meseci", months), ("godine", years), ("opklade", bets)]:
    json.dump(obj, open(f"{OUT}/{name}.json", "w"))
print(len(weeks), "weeks")
