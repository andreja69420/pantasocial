"""Excel sheet of the plan's paper bets: summary on top (bank, profit, ROI, record), then one row per bet with kick-off
in Belgrade time, odds, stake, result and profit. Stake = quarter Kelly of the bank at the moment the bet was logged =
starting bank + profit of every bet already settled then (settled = when the loop first saw the result), so a settled
bet changes only the stakes of later bets. Formulas, so changing the starting bank (B3) recomputes everything;
LibreOffice (if installed) stores the computed values for viewers that do not evaluate formulas.

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
from openpyxl.utils import get_column_letter

from fs_common import player_name

SRC, OUT = sys.argv[1], sys.argv[2]
BANK = float(sys.argv[3]) if len(sys.argv) > 3 else 100000.0
KELLY = 0.25
# "ne" = outside the plan: on the second sheet, same rules and its own starting bank, for comparison only
SEG = {"ITF WOMEN - SINGLES": ("Ženski ITF", "DA (glavno)"), "CHALLENGER MEN - SINGLES": ("Muški Challenger", "DA"),
       "ITF MEN - SINGLES": ("Muški ITF", "DA"), "CHALLENGER WOMEN - SINGLES": ("WTA 125", "ne"),
       "WTA - SINGLES": ("WTA Tour", "ne"), "ATP - SINGLES": ("ATP Tour", "ne")}

L = pd.read_csv(SRC)
# one row per match and side: the broadest rule that logged it (EV>5% odds<=5), stake sized by Kelly anyway
L["thr"] = L.rule.str.extract(r"EV>(\d+)%").astype(float)[0]
L = L.sort_values(["thr", "logged_at"]).drop_duplicates(["fs_mid", "side"]).sort_values("start")
in_plan = L.tournament.str.split(":").str[0].str.strip().map(lambda c: SEG.get(c, ("", "ne"))[1] != "ne")
now = pd.Timestamp.now("UTC").tz_localize(None)
bg = lambda t: pd.Timestamp(t).tz_localize("UTC").tz_convert("Europe/Belgrade").tz_localize(None).to_pydatetime()

INK, MUTED, LINE, HEAD, BAND = "1F2933", "7B8794", "D9DEE3", "2F3B48", "F5F7F9"
font = lambda **k: Font(**{"name": "Calibri", "size": 10, "color": INK, **k})
thin = Side(style="thin", color=LINE)


def sheet(ws, L, title):
    ws.sheet_view.showGridLines = False
    # summary block: row 2 labels, row 3 values from column B (B3 = starting bank, editable)
    r0 = 6
    last = r0 + max(len(L), 1) - 1
    col = lambda c: f"${c}${r0}:${c}${last}"
    ws["A1"] = title
    ws["A1"].font = font(size=14, bold=True)
    summary = [("Početna banka", BANK, "#,##0"), ("Banka sada", f"=B3+SUM({col('M')})", "#,##0"),
               ("Dobit", f"=SUM({col('M')})", "#,##0;[Red]-#,##0"),
               ("ROI", f'=IFERROR(SUM({col("M")})/(SUMIF({col("L")},"Dobijeno",{col("K")})+SUMIF({col("L")},"Izgubljeno",{col("K")})),"")', "0.0%;[Red]-0.0%"),
               ("Opklade", f"=COUNTA({col('E')})", "0"), ("Dobijeno", f'=COUNTIF({col("L")},"Dobijeno")', "0"),
               ("Izgubljeno", f'=COUNTIF({col("L")},"Izgubljeno")', "0")]
    for j, (lab, v, fmt) in enumerate(summary, 2):
        a, b = ws.cell(row=2, column=j, value=lab), ws.cell(row=3, column=j, value=v)
        a.font, b.font = font(size=9, color=MUTED), font(size=12, bold=True)
        b.number_format = fmt
        a.alignment = b.alignment = Alignment(horizontal="left")

    cols = [("#", 4, "0"), ("Početak", 13, "dd.mm. hh:mm"), ("Kategorija", 15, "@"), ("Turnir", 42, "@"), ("Igramo na", 22, "@"),
            ("Protivnik", 22, "@"), ("Kladionica", 13, "@"), ("Kvota", 7, "0.00"), ("Verovatnoća", 11, "0.0%"),
            ("Prednost", 9, "0.0%"), ("Ulog", 9, "#,##0"), ("Rezultat", 11, "@"), ("Dobit", 9, "#,##0;[Red]-#,##0"),
            ("Banka pri opkladi", 19, "#,##0"),
            # helpers for the formulas, hidden
            ("Kelly", 8, "0.00%"), ("Upisano", 13, "dd.mm. hh:mm"), ("Obračunato", 13, "dd.mm. hh:mm"),
            # which production model logged the bet
            ("Model", 8, "@")]
    for j, (c, w, _) in enumerate(cols, 1):
        cell = ws.cell(row=r0 - 1, column=j, value=c)
        cell.font, cell.fill = font(bold=True, color="FFFFFF"), PatternFill("solid", fgColor=HEAD)
        cell.alignment = Alignment(horizontal="right" if 8 <= j <= 17 else "left", vertical="center")
        ws.column_dimensions[get_column_letter(j)].width = w
    for j in (15, 16, 17):
        ws.column_dimensions[get_column_letter(j)].hidden = True
    ws.row_dimensions[r0 - 1].height = 20

    for i, b in enumerate(L.itertuples()):
        r = r0 + i
        start = pd.Timestamp(b.start)
        pick, opp = (b.home, b.away) if b.side == "home" else (b.away, b.home)
        cat = SEG.get(b.tournament.split(":")[0].strip(), ("?", "ne"))[0]
        p, o = float(b.q_blend), float(b.open)
        kelly = KELLY * max(0.0, (p * o - 1) / (o - 1))
        res = {"void": "Povraćaj"}.get(str(b.result), "Dobijeno" if b.result == b.side else
                                       ("Izgubljeno" if str(b.result) in ("home", "away") else ""))
        s_at = getattr(b, "settled_at", np.nan)
        # settled before settlement times were recorded: kick-off + 3 h
        settled = (bg(s_at) if isinstance(s_at, str) and s_at else bg(start) + pd.Timedelta(hours=3)) if res else None
        vals = [i + 1, bg(start), cat, b.tournament.split(":", 1)[-1].strip(),
                player_name({"WU": pick}, "h").title(), player_name({"WU": opp}, "h").title(), b.book, o, p, p * o - 1,
                f"=ROUND(N{r}*O{r},0)", res or ("U toku" if start <= now else "Čeka"),
                f'=IF(L{r}="Dobijeno",K{r}*(H{r}-1),IF(L{r}="Izgubljeno",-K{r},IF(L{r}="Povraćaj",0,"")))',
                f'=$B$3+SUMIFS({col("M")},{col("Q")},"<="&P{r})', kelly, bg(b.logged_at), settled,
                "novi" if getattr(b, "model", "") == "novi" else "stari"]
        for j, v in enumerate(vals, 1):
            cell = ws.cell(row=r, column=j, value=v)
            cell.number_format = cols[j - 1][2]
            cell.font = font(color=MUTED) if j == 12 and not res else font()
            cell.border = Border(bottom=thin)
            cell.alignment = Alignment(horizontal="right" if 8 <= j <= 17 else "left", vertical="center")
            if i % 2:
                cell.fill = PatternFill("solid", fgColor=BAND)
        ws.row_dimensions[r].height = 18
    green = Font(name="Calibri", size=10, color="1E7B34")
    ws.conditional_formatting.add(f"M{r0}:M{last}", CellIsRule(operator="greaterThan", formula=["0"], font=green))
    ws.freeze_panes = ws.cell(row=r0, column=1)
    ws.auto_filter.ref = f"A{r0 - 1}:R{last}"
    ws.page_setup.orientation = "landscape"
    ws.page_setup.fitToWidth, ws.page_setup.fitToHeight = 1, 0
    ws.sheet_properties.pageSetUpPr.fitToPage = True


wb = Workbook()
ws = wb.active
ws.title = "Opklade"
sheet(ws, L[in_plan], "Opklade")
sheet(wb.create_sheet("Van plana"), L[~in_plan], "Van plana")
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
print(f"{int(in_plan.sum())} plan bets, {int((~in_plan).sum())} outside the plan -> {OUT}")
