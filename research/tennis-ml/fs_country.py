"""Event country of Flashscore tournaments. The category header of every match line names it ("ATP - SINGLES: Shanghai
(China), hard"); the feature build (fx_context.event_country) needs it for events whose id carries no country, i.e. all
Flashscore events (fs-<slug>-<year>, fs-ch-<slug>-<year>). A new venue otherwise has to be guessed from the players' flags.

event_country.json (this directory, merged and rewritten by fs_update.py / fs_stats.py): {"fs-<slug>" or "fs-ch-<slug>": IOC code},
the year left out because a venue keeps its country.
"""
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
PATH = os.environ.get("EVENT_COUNTRY", os.path.join(HERE, "event_country.json"))

# Flashscore country name -> IOC code (the codes of fx_context._TABLE)
NAME2IOC = {
    "Albania": "ALB", "Algeria": "ALG", "Andorra": "AND", "Angola": "ANG", "Argentina": "ARG", "Armenia": "ARM", "Aruba": "ARU",
    "Australia": "AUS", "Austria": "AUT", "Azerbaijan": "AZE", "Bahamas": "BAH", "Bahrain": "BRN", "Bangladesh": "BAN",
    "Barbados": "BAR", "Belarus": "BLR", "Belgium": "BEL", "Benin": "BEN", "Bermuda": "BER", "Bolivia": "BOL",
    "Bosnia and Herzegovina": "BIH", "Botswana": "BOT", "Brazil": "BRA", "Bulgaria": "BUL", "Burkina Faso": "BUR",
    "Burundi": "BDI", "Cambodia": "CAM", "Cameroon": "CMR", "Canada": "CAN", "Chile": "CHI", "China": "CHN",
    "Chinese Taipei": "TPE", "Colombia": "COL", "Costa Rica": "CRC", "Croatia": "CRO", "Cuba": "CUB", "Curacao": "CUW",
    "Cyprus": "CYP", "Czech Republic": "CZE", "Czechia": "CZE", "Denmark": "DEN", "Dominican Republic": "DOM",
    "DR Congo": "COD", "Ecuador": "ECU", "Egypt": "EGY", "El Salvador": "ESA", "Estonia": "EST", "Eswatini": "SWZ",
    "Ethiopia": "ETH", "Fiji": "FIJ", "Finland": "FIN", "France": "FRA", "Gabon": "GAB", "Gambia": "GAM", "Georgia": "GEO",
    "Germany": "GER", "Ghana": "GHA", "Great Britain": "GBR", "Greece": "GRE", "Guam": "GUM", "Guatemala": "GUA",
    "Guyana": "GUY", "Haiti": "HAI", "Honduras": "HON", "Hong Kong": "HKG", "Hungary": "HUN", "Iceland": "ISL", "India": "IND",
    "Indonesia": "INA", "Iran": "IRI", "Iraq": "IRQ", "Ireland": "IRL", "Israel": "ISR", "Italy": "ITA", "Ivory Coast": "CIV",
    "Jamaica": "JAM", "Japan": "JPN", "Jordan": "JOR", "Kazakhstan": "KAZ", "Kenya": "KEN", "Kosovo": "KOS", "Kuwait": "KUW",
    "Kyrgyzstan": "KGZ", "Laos": "LAO", "Latvia": "LAT", "Lebanon": "LBN", "Libya": "LBA", "Liechtenstein": "LIE",
    "Lithuania": "LTU", "Luxembourg": "LUX", "Macau": "MAC", "Madagascar": "MAD", "Malaysia": "MAS", "Mali": "MLI",
    "Malta": "MLT", "Mauritius": "MRI", "Mexico": "MEX", "Moldova": "MDA", "Monaco": "MON", "Mongolia": "MGL",
    "Montenegro": "MNE", "Morocco": "MAR", "Mozambique": "MOZ", "Namibia": "NAM", "Nepal": "NEP", "Netherlands": "NED",
    "New Zealand": "NZL", "Nicaragua": "NCA", "Nigeria": "NGR", "North Macedonia": "MKD", "Norway": "NOR", "Oman": "OMA",
    "Pakistan": "PAK", "Palestine": "PLE", "Panama": "PAN", "Paraguay": "PAR", "Peru": "PER", "Philippines": "PHI",
    "Poland": "POL", "Portugal": "POR", "Puerto Rico": "PUR", "Qatar": "QAT", "Romania": "ROU", "Russia": "RUS",
    "Rwanda": "RWA", "Saudi Arabia": "KSA", "Senegal": "SEN", "Serbia": "SRB", "Singapore": "SGP", "Slovakia": "SVK",
    "Slovenia": "SLO", "South Africa": "RSA", "South Korea": "KOR", "Spain": "ESP", "Sri Lanka": "SRI", "Sudan": "SUD",
    "Suriname": "SUR", "Sweden": "SWE", "Switzerland": "SUI", "Syria": "SYR", "Taiwan": "TPE", "Tajikistan": "TJK",
    "Tanzania": "TAN", "Thailand": "THA", "Togo": "TOG", "Trinidad and Tobago": "TTO", "Tunisia": "TUN", "Turkey": "TUR",
    "Turkiye": "TUR", "Turkmenistan": "TKM", "UAE": "UAE", "United Arab Emirates": "UAE", "Uganda": "UGA", "Ukraine": "UKR",
    "United Kingdom": "GBR", "Uruguay": "URU", "USA": "USA", "Uzbekistan": "UZB", "Venezuela": "VEN", "Vietnam": "VIE",
    "Yemen": "YEM", "Zambia": "ZAM", "Zimbabwe": "ZIM",
}


def header_country(za):
    """IOC code from a category header "ATP - SINGLES: Shanghai (China) - Qualification, hard", None if absent / unknown."""
    m = re.search(r"\(([^()]+)\)", za.split(":", 1)[-1])
    return NAME2IOC.get(m.group(1).strip()) if m else None


def event_key(tid):
    """Map key of an event id: the year at the end left out ("fs-ch-shanghai-2026" -> "fs-ch-shanghai")."""
    return re.sub(r"-\d{4}$", "", str(tid))


def load(path=None):
    p = path or PATH
    return json.load(open(p)) if os.path.exists(p) else {}


def update(pairs, ev_id, path=None):
    """Merge the countries of the feed headers into the map. pairs: (head, record) of the daily feed; ev_id(head) -> the
    event id of that head (fs_common.event_id) or None. Returns the number of new events."""
    p = path or PATH
    cur = load(p)
    new = 0
    for h, _ in pairs:
        c, tid = header_country(h.get("ZA", "")), ev_id(h)
        if c and tid and cur.get(event_key(tid)) != c:
            new += event_key(tid) not in cur
            cur[event_key(tid)] = c
    if new or not os.path.exists(p):
        json.dump(dict(sorted(cur.items())), open(p + ".tmp", "w"), indent=0)
        os.replace(p + ".tmp", p)
    return new
