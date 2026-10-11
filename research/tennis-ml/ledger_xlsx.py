"""Excel sheet of the paper bets for following along: one row per match, quarter-Kelly stake from a bank cell,
kick-off in Belgrade time, result and profit filled in as matches finish (formulas, so the bank can be changed).

usage: ledger_xlsx.py LEDGER_CSV OUT_XLSX [BANK=1000]
"""
import sys

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.formatting.rule import CellIsRule
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from fs_common import player_name

SRC, OUT = sys.argv[1], sys.argv[2]
BANK = float(sys.argv[3]) if len(sys.argv) > 3 else 1000.0
KELLY = 0.25
SEG = {"ITF WOMEN - SINGLES": ("Ženski ITF", "DA (glavno)"), "CHALLENGER MEN - SINGLES": ("Muški Challenger", "DA"),
       "ITF MEN - SINGLES": ("Muški ITF", "DA"), "CHALLENGER WOMEN - SINGLES": ("WTA 125", "ne"),
       "WTA - SINGLES": ("WTA Tour", "ne"), "ATP - SINGLES": ("ATP Tour", "ne")}

L = pd.read_csv(SRC)
# one row per match and side: the broadest rule that logged it (EV>5% odds<=5), stake sized by Kelly anyway
L["thr"] = L.rule.str.extract(r"EV>(\d+)%").astype(float)[0]
L = L.sort_values(["thr", "logged_at"]).drop_duplicates(["fs_mid", "side"]).sort_values("start")
now = pd.Timestamp.now("UTC").tz_localize(None)

wb = Workbook()
ws = wb.active
ws.title = "Opklade"
bold, head_fill = Font(bold=True), PatternFill("solid", fgColor="DDEBF7")
ws["A1"], ws["B1"] = "Banka", BANK
ws["A1"].font = bold
ws["B1"].fill = PatternFill("solid", fgColor="FFF2CC")
ws["C1"] = "<- upiši svoju banku; ulozi se preračunaju sami (1/4 Kelly)"
cols = ["#", "Početak (Beograd)", "Kategorija", "U planu", "Turnir", "Igramo na", "Protivnik", "Kladionica", "Kvota (početna)",
        "Model %", "Prednost (EV)", "1/4 Kelly % banke", "Ulog", "Kvota sada / pred meč", "Rezultat (W/L/V)", "Dobit", "Status"]
for j, c in enumerate(cols, 1):
    cell = ws.cell(row=3, column=j, value=c)
    cell.font, cell.fill, cell.alignment = bold, head_fill, Alignment(wrap_text=True, vertical="center")
r0 = 4
for i, b in enumerate(L.itertuples()):
    r = r0 + i
    start = pd.Timestamp(b.start)
    pick, opp = (b.home, b.away) if b.side == "home" else (b.away, b.home)
    cat, plan = SEG.get(b.tournament.split(":")[0].strip(), ("?", "ne"))
    p, o = float(b.q_blend), float(b.open)
    kelly = KELLY * max(0.0, (p * o - 1) / (o - 1))
    res = {"void": "V"}.get(str(b.result), "W" if b.result == b.side else ("L" if str(b.result) in ("home", "away") else ""))
    status = "završen" if res else ("u toku / počeo" if start <= now else "čeka")
    close = getattr(b, "close", np.nan)
    vals = [i + 1, start.tz_localize("UTC").tz_convert("Europe/Belgrade").tz_localize(None).to_pydatetime(), cat, plan,
            b.tournament.split(":", 1)[-1].strip(), player_name({"WU": pick}, "h").title(), player_name({"WU": opp}, "h").title(),
            b.book, o, p, p * o - 1, kelly, f"=ROUND($B$1*L{r},2)", None if pd.isna(close) else float(close), res,
            f'=IF(O{r}="W",M{r}*(I{r}-1),IF(O{r}="L",-M{r},IF(O{r}="V",0,"")))', status]
    for j, v in enumerate(vals, 1):
        ws.cell(row=r, column=j, value=v)
    ws.cell(row=r, column=2).number_format = "dd.mm. HH:MM"
    for j in (10, 11, 12):
        ws.cell(row=r, column=j).number_format = "0.0%"
    for j in (13, 16):
        ws.cell(row=r, column=j).number_format = "0.00"
last = r0 + max(len(L), 1) - 1
green, red = PatternFill("solid", fgColor="C6EFCE"), PatternFill("solid", fgColor="FFC7CE")
ws.conditional_formatting.add(f"P{r0}:P{last}", CellIsRule(operator="greaterThan", formula=["0"], fill=green))
ws.conditional_formatting.add(f"P{r0}:P{last}", CellIsRule(operator="lessThan", formula=["0"], fill=red))
for j, w in enumerate([4, 13, 16, 11, 34, 22, 22, 13, 9, 8, 9, 10, 9, 11, 9, 9, 13], 1):
    ws.column_dimensions[get_column_letter(j)].width = w
ws.freeze_panes = "A4"

s = wb.create_sheet("Ukupno")
rng = lambda c: f"Opklade!{c}{r0}:{c}{last}"
rows = [("Opklada ukupno", f"=COUNTA({rng('F')})"), ("Završeno", f'=COUNTIF({rng("O")},"W")+COUNTIF({rng("O")},"L")+COUNTIF({rng("O")},"V")'),
        ("Dobijeno", f'=COUNTIF({rng("O")},"W")'), ("Izgubljeno", f'=COUNTIF({rng("O")},"L")'),
        ("Uloženo (završene)", f'=SUMPRODUCT(({rng("O")}="W")+({rng("O")}="L"),{rng("M")})'),
        ("Dobit", f"=SUM({rng('P')})"), ("ROI", "=IF(B6>0,B7/B6,\"\")"), ("Banka sada", "=Opklade!B1+B7"),
        ("Samo plan (ženski ITF, muški Challenger/ITF) - dobit", f'=SUMIF({rng("D")},"DA*",{rng("P")})')]
for i, (a, f) in enumerate(rows, 2):
    s.cell(row=i, column=1, value=a).font = bold
    s.cell(row=i, column=2, value=f)
s["B8"].number_format = "0.0%"
s.column_dimensions["A"].width = 52
s.column_dimensions["B"].width = 14
s["A12"] = ("Pravilo: kvota se uzima samo dok je još početna; P = σ(0,44·logit(model) + 0,64·logit(fer kvota)); "
            "igra se kad je P·kvota − 1 > 5% i kvota ≤ 5. Ulog = 1/4 Kelly = 0,25·(P·kvota − 1)/(kvota − 1) banke.")
wb.save(OUT)
print(f"{len(L)} bets -> {OUT}")
