"""Weekly / monthly / yearly tables of the plan test for the dashboard "Tenis test po nedeljama".
Plan rule (blend EV>5%, odds<=5, no ITF) from the eval_levels2026.py --bets parquet files; quarter Kelly, each month starts
from 300,000, at most 30,000 per bet (kelly_cap.py `plan`). 2024 / 2025 come from fs_hist, 2026 from the holdout of the
current model.

usage: weekly_test.py DATA_DIR BETS_2026_PARQUET OUT_DIR   (writes nedelje.json, meseci.json, godine.json)
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
B, P26, OUT = sys.argv[1:4]
sys.argv = ["x"]
ns = {}
exec(open(os.path.join(HERE, "kelly_cap.py")).read().split("rows = []")[0], ns)
START, CAP = 300000.0, 30000.0
books = {2024: 3, 2025: 5, 2026: 6}
weeks, months, years = [], [], []
for y, p in [(2024, f"{B}/fs_hist/y2024.parquet"), (2025, f"{B}/fs_hist/y2025.parquet"), (2026, P26)]:
    b = ns["plan"](p)
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
for name, obj in [("nedelje", weeks), ("meseci", months), ("godine", years)]:
    json.dump(obj, open(f"{OUT}/{name}.json", "w"))
print(len(weeks), "weeks")
