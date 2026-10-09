"""List every USDT-M perpetual that ever had funding data on data.binance.vision (delisted included)."""
import json
import re
import sys
import urllib.parse
import urllib.request

out = sys.argv[1]
base = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
syms, marker = [], ""
while True:
    q = urllib.parse.urlencode({"delimiter": "/", "prefix": "data/futures/um/monthly/fundingRate/", "marker": marker})
    x = urllib.request.urlopen(f"{base}?{q}", timeout=60).read().decode()
    found = re.findall(r"<Prefix>data/futures/um/monthly/fundingRate/([^<]+)/</Prefix>", x)
    syms += found
    if "<IsTruncated>true</IsTruncated>" not in x or not found:
        break
    marker = re.findall(r"<NextMarker>([^<]+)</NextMarker>", x)[0]
syms = sorted(set(syms))
json.dump(syms, open(out, "w"))
print(len(syms), "perp symbols;", sum(s.endswith("USDT") for s in syms), "USDT")
