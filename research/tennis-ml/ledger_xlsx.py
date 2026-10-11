"""Excel sheet of the paper bets, kept to the basics: summary on top (bank, profit, ROI, record), profit per category,
then one row per bet: kick-off (Belgrade time), category, tournament, who we back, opponent, odds, bookmaker, stake,
result, profit. Stake = quarter Kelly of the bank at the moment the bet was logged = starting bank + profit of every bet
already settled then (settled = when the loop first saw the result), so a settled bet changes only the stakes of later
bets. Stakes are computed here (values); profits and the summary are formulas. LibreOffice (if installed) stores the
computed values for viewers that do not evaluate formulas.

usage: ledger_xlsx.py LEDGER_CSV OUT_XLSX [BANK=100000]   (dinars)
"""
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.formatting.rule import CellIsRule
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side

from fs_common import player_name

SRC, OUT = sys.argv[1], sys.argv[2]
BANK = float(sys.argv[3]) if len(sys.argv) > 3 else 100000.0
KELLY = 0.25
SEG = {"ITF WOMEN - SINGLES": "Ženski ITF", "CHALLENGER MEN - SINGLES": "Muški Challenger", "ITF MEN - SINGLES": "Muški ITF",
       "CHALLENGER WOMEN - SINGLES": "WTA 125", "WTA - SINGLES": "WTA Tour", "ATP - SINGLES": "ATP Tour"}

L = pd.read_csv(SRC)
# one row per match and side: the broadest rule that logged it (EV>5% odds<=5), stake sized by Kelly anyway
L["thr"] = L.rule.str.extract(r"EV>(\d+)%").astype(float)[0]
L = L.sort_values(["thr", "logged_at"]).drop_duplicates(["fs_mid", "side"])
L = L[L.tournament.str.split(":").str[0].str.strip().isin(SEG)]
now = pd.Timestamp.now("UTC").tz_localize(None)
bg = lambda t: pd.Timestamp(t).tz_localize("UTC").tz_convert("Europe/Belgrade").tz_localize(None)

L["res"] = [{"void": "Povraćaj"}.get(str(r), "Dobijeno" if r == s else ("Izgubljeno" if str(r) in ("home", "away") else ""))
            for r, s in zip(L.result, L.side)]
# settled before settlement times were recorded: kick-off + 3 h
L["settled"] = [pd.Timestamp(a) if isinstance(a, str) and a else pd.Timestamp(st) + pd.Timedelta(hours=3)
                for a, st in zip(L.get("settled_at", pd.Series(np.nan, index=L.index)), L.start)]
L["logged"] = pd.to_datetime(L.logged_at)
# stakes in the order the bets were logged: a bet settled before a later bet was logged is in that bet's bank
stake, profit = {}, {}
for i, b in L.sort_values("logged").iterrows():
    bank = BANK + sum(v for j, v in profit.items() if L.at[j, "settled"] <= b.logged)
    p, o = float(b.q_blend), float(b.open)
    stake[i] = round(bank * KELLY * max(0.0, (p * o - 1) / (o - 1)))
    if b.res:
        profit[i] = stake[i] * (o - 1) if b.res == "Dobijeno" else (-stake[i] if b.res == "Izgubljeno" else 0.0)
L["stake"] = pd.Series(stake)
L = L.sort_values("start")

INK, MUTED, LINE, HEAD, BAND = "1F2933", "7B8794", "D9DEE3", "2F3B48", "F5F7F9"
font = lambda **k: Font(**{"name": "Calibri", "size": 10, "color": INK, **k})
thin = Side(style="thin", color=LINE)
COLS = [("Početak", 13, "dd.mm. hh:mm"), ("Kategorija", 16, "@"), ("Turnir", 34, "@"), ("Igramo na", 22, "@"),
        ("Protivnik", 22, "@"), ("Kvota", 7, "0.00"), ("Kladionica", 13, "@"), ("Ulog", 9, "#,##0"), ("Rezultat", 11, "@"),
        ("Dobit", 10, "#,##0;[Red]-#,##0")]
RIGHT = {"Kvota", "Ulog", "Dobit"}

wb = Workbook()
ws = wb.active
ws.title = "Opklade"
ws.sheet_view.showGridLines = False
r0 = 10
last = r0 + max(len(L), 1) - 1
col = lambda c: f"${c}${r0}:${c}${last}"
ws["A1"] = "Opklade"
ws["A1"].font = font(size=14, bold=True)
staked = f'(SUMIF({col("I")},"Dobijeno",{col("H")})+SUMIF({col("I")},"Izgubljeno",{col("H")}))'
summary = [("Početna banka", BANK, "#,##0"), ("Banka sada", f"=A3+SUM({col('J')})", "#,##0"),
           ("Dobit", f"=SUM({col('J')})", "#,##0;[Red]-#,##0"), ("ROI", f'=IFERROR(SUM({col("J")})/{staked},"")', "0.0%;[Red]-0.0%"),
           ("Opklade", f"=COUNTA({col('D')})", "0"), ("Dobijeno", f'=COUNTIF({col("I")},"Dobijeno")', "0"),
           ("Izgubljeno", f'=COUNTIF({col("I")},"Izgubljeno")', "0"),
           ("Čeka", f'=COUNTIF({col("I")},"Čeka")+COUNTIF({col("I")},"U toku")', "0")]
for j, (lab, v, fmt) in enumerate(summary, 1):
    a, b = ws.cell(row=2, column=j, value=lab), ws.cell(row=3, column=j, value=v)
    a.font, b.font = font(size=9, color=MUTED), font(size=12, bold=True)
    b.number_format = fmt
    a.alignment = b.alignment = Alignment(horizontal="left")
# profit per category (rows 5-7): name, profit, ROI and record
for j, cat in enumerate(SEG.values(), 1):
    crit = f'{col("B")},"{cat}"'
    won, lost = f'COUNTIFS({crit},{col("I")},"Dobijeno")', f'COUNTIFS({crit},{col("I")},"Izgubljeno")'
    st = f'(SUMIFS({col("H")},{crit},{col("I")},"Dobijeno")+SUMIFS({col("H")},{crit},{col("I")},"Izgubljeno"))'
    a = ws.cell(row=5, column=j, value=cat)
    b = ws.cell(row=6, column=j, value=f'=SUMIFS({col("J")},{crit})')
    c = ws.cell(row=7, column=j, value=f'=IF({st}>0,"ROI "&TEXT(SUMIFS({col("J")},{crit})/{st},"0.0%")&" · ","")'
                                       f'&{won}&"–"&{lost}&" od "&COUNTIFS({crit})')
    a.font, b.font, c.font = font(size=9, color=MUTED), font(size=12, bold=True), font(size=9, color=MUTED)
    b.number_format = "#,##0;[Red]-#,##0"
    for x in (a, b, c):
        x.alignment = Alignment(horizontal="left", vertical="top", wrap_text=True)
ws.row_dimensions[7].height = 26

for j, (c, w, _) in enumerate(COLS, 1):
    cell = ws.cell(row=r0 - 1, column=j, value=c)
    cell.font, cell.fill = font(bold=True, color="FFFFFF"), PatternFill("solid", fgColor=HEAD)
    cell.alignment = Alignment(horizontal="right" if c in RIGHT else "left", vertical="center")
    ws.column_dimensions[cell.column_letter].width = w
ws.row_dimensions[r0 - 1].height = 20
for i, b in enumerate(L.itertuples()):
    r = r0 + i
    start = pd.Timestamp(b.start)
    pick, opp = (b.home, b.away) if b.side == "home" else (b.away, b.home)
    vals = [bg(start).to_pydatetime(), SEG[b.tournament.split(":")[0].strip()], b.tournament.split(":", 1)[-1].strip(),
            player_name({"WU": pick}, "h").title(), player_name({"WU": opp}, "h").title(), float(b.open), b.book, int(b.stake),
            b.res or ("U toku" if start <= now else "Čeka"),
            f'=IF(I{r}="Dobijeno",H{r}*(F{r}-1),IF(I{r}="Izgubljeno",-H{r},IF(I{r}="Povraćaj",0,"")))']
    for j, v in enumerate(vals, 1):
        cell = ws.cell(row=r, column=j, value=v)
        cell.number_format = COLS[j - 1][2]
        cell.font = font(color=MUTED) if j == 9 and not b.res else font()
        cell.border = Border(bottom=thin)
        cell.alignment = Alignment(horizontal="right" if COLS[j - 1][0] in RIGHT else "left", vertical="center")
        if i % 2:
            cell.fill = PatternFill("solid", fgColor=BAND)
    ws.row_dimensions[r].height = 18
ws.conditional_formatting.add(f"J{r0}:J{last}", CellIsRule(operator="greaterThan", formula=["0"],
                                                         font=Font(name="Calibri", size=10, color="1E7B34")))
ws.freeze_panes = ws.cell(row=r0, column=1)
ws.auto_filter.ref = f"A{r0 - 1}:J{last}"
ws.page_setup.orientation = "landscape"
ws.page_setup.fitToWidth, ws.page_setup.fitToHeight = 1, 0
ws.sheet_properties.pageSetUpPr.fitToPage = True
wb.save(OUT)
if shutil.which("soffice"):
    with tempfile.TemporaryDirectory() as tmp:
        r = subprocess.run(["soffice", f"-env:UserInstallation=file://{tmp}/profile", "--headless", "--calc",
                            "--convert-to", "xlsx", "--outdir", tmp, os.path.abspath(OUT)], capture_output=True, timeout=180)
        done = os.path.join(tmp, os.path.basename(OUT))
        if r.returncode == 0 and os.path.exists(done):
            shutil.move(done, OUT)
        else:
            print("LibreOffice recalculation failed, formulas without stored values:", r.stderr.decode()[-300:])
print(f"{len(L)} bets -> {OUT}")
