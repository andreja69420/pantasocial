"""Live scores of the open paper bets from Sofascore, merged into the rows of the dashboard JSON.

Per bet that has kicked off (or is about to) and is not settled: the Sofascore event is found in the live list, or by
searching "<our surname> <opponent surname>" and taking the event within 14 h of the kick-off; its id is kept in
paper/sofa_ids.json. Fields added to a row: live ("live" | "ended" | "soon" | None), score (sets from our player's side, "6-3 2-2"),
point (current game from our player's side), lead ("ahead" | "behind" | "level"), sets_us / sets_them, sofa (event link),
live_at (Belgrade HH:MM of the check).

usage: live_scores.py OPKLADE_JSON      (patches the file in place; ledger_xlsx.py also calls merge() after writing it)
"""
import json
import os
import re
import subprocess
import sys
import unicodedata
import urllib.parse
from datetime import datetime, timedelta, timezone

API = "https://api.sofascore.com/api/v1/"
OPEN = ("Čeka", "U toku")
ID_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper", "sofa_ids.json")


def get(path, timeout=15):
    r = subprocess.run(["curl", "-sS", "-m", str(timeout), "-A", "Mozilla/5.0 Chrome/124", "-H", "Referer: https://www.sofascore.com/",
                        API + path], capture_output=True, text=True)
    try:
        return json.loads(r.stdout)
    except ValueError:
        return {}


def toks(name):
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    return set(re.sub(r"[^a-z ]", " ", s.replace("-", " ")).split())


def same(a, b):
    a, b = toks(a), toks(b)
    return bool(a and b) and (a == b or (min(len(a), len(b)) >= 2 and (a <= b or b <= a)))


def side_of(ev, pick, opp):
    h, a = ev["homeTeam"]["name"], ev["awayTeam"]["name"]
    if same(h, pick) and same(a, opp):
        return "home"
    if same(a, pick) and same(h, opp):
        return "away"
    return None


def periods(sc):
    return [sc.get(f"period{i}") for i in range(1, 6)]


def fill(row, ev, side):
    mine, theirs = (ev["homeScore"], ev["awayScore"]) if side == "home" else (ev["awayScore"], ev["homeScore"])
    pm, pt = periods(mine), periods(theirs)
    sets = [(x, y) for x, y in zip(pm, pt) if x is not None and y is not None]
    kind = ev["status"]["type"]
    row["live"] = {"inprogress": "live", "finished": "ended"}.get(kind, "soon" if kind == "notstarted" else None)
    row["score"] = " ".join(f"{x}-{y}" for x, y in sets) or None
    row["sets_us"], row["sets_them"] = mine.get("current"), theirs.get("current")
    p1, p2 = mine.get("point"), theirs.get("point")
    row["point"] = f"{p1}-{p2}" if row["live"] == "live" and p1 is not None and p2 is not None else None
    if row["live"] == "ended":
        w = ev.get("winnerCode")
        row["lead"] = None if w not in (1, 2) else ("ahead" if (w == 1) == (side == "home") else "behind")
    elif sets:
        won_us, won_them = mine.get("current") or 0, theirs.get("current") or 0
        last_us, last_them = sets[-1]
        d = (won_us - won_them, last_us - last_them)
        row["lead"] = "ahead" if d > (0, 0) else "behind" if d < (0, 0) else "level"
    else:
        row["lead"] = None
    row["sofa"] = f"https://www.sofascore.com/event/{ev['id']}"


def merge(rows, now=None):
    now = now or datetime.now(timezone.utc)
    belgrade = now.replace(tzinfo=None) + timedelta(hours=2)
    try:
        ids = json.load(open(ID_FILE))
    except (OSError, ValueError):
        ids = {}
    live = None
    for row in rows:
        row.update({"live": None, "score": None, "point": None, "lead": None, "sets_us": None, "sets_them": None, "sofa": None,
                    "live_at": f"{belgrade:%H:%M}"})
        start = datetime.strptime(row["start"], "%Y-%m-%dT%H:%M")
        if row["status"] not in OPEN and not (row["profit"] is not None and belgrade - start < timedelta(hours=8)):
            continue
        if start - belgrade > timedelta(minutes=45):
            continue
        ev, side = None, None
        if row["id"] in ids:
            ev = get(f"event/{ids[row['id']]}").get("event")
        if ev is None:
            if live is None:
                live = get("sport/tennis/events/live").get("events", [])
            for e in live:
                s = side_of(e, row["pick"], row["opponent"])
                if s:
                    ev = e
                    break
        if ev is None:
            q = urllib.parse.quote(f"{row['pick'].split()[0]} {row['opponent'].split()[0]}")
            for r in get(f"search/all?q={q}").get("results", []):
                e = r["entity"]
                if r["type"] != "event" or "homeTeam" not in e:
                    continue
                kick = datetime.fromtimestamp(e["startTimestamp"], timezone.utc).replace(tzinfo=None) + timedelta(hours=2)
                if abs(kick - start) < timedelta(hours=14) and side_of(e, row["pick"], row["opponent"]):
                    ev = get(f"event/{e['id']}").get("event")
                    break
        if ev is None:
            continue
        side = side_of(ev, row["pick"], row["opponent"])
        if side is None:
            continue
        ids[row["id"]] = ev["id"]
        fill(row, ev, side)
    try:
        json.dump(ids, open(ID_FILE, "w"))
    except OSError:
        pass
    return rows


if __name__ == "__main__":
    path = sys.argv[1]
    d = json.load(open(path))
    merge(d["rows"])
    json.dump(d, open(path, "w"), ensure_ascii=False, indent=1)
    for r in d["rows"]:
        if r["live"]:
            print(r["pick"], "-", r["opponent"], r["live"], r["score"], r["point"], r["lead"])
