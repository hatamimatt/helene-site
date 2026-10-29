#!/usr/bin/env python3
"""Power infrastructure of the US Southeast for the globe: lines, substations, plants, data centres.

For Alabama, Florida, Georgia, Mississippi, North Carolina, South Carolina and Tennessee:
  - power lines of 44 kV and above (transmission and sub-transmission), transmission
    substations (115 kV and above) and data centres from OpenStreetMap through the
    Overpass API (the data behind Open Infrastructure Map)
  - operating power plants from EIA-860M, the US Energy Information Administration's
    monthly generator inventory, summed per plant (capacity, main fuel)

Writes data/grid/: grid.json (plants, substations, data centres), lines_<class>.json
per voltage class so the globe loads only the classes a zoom level shows (coordinates
rounded to 0.001 degree, lines simplified), and meta.json. Raw Overpass replies are
cached in .cache/grid/. The data changes slowly: rerun every few months.
OpenStreetMap data is ODbL: credit "OpenStreetMap contributors". EIA data is public domain.
Needs openpyxl.

    python prepare_grid.py
"""
import argparse, collections, io, json, os, re, sys, time, urllib.parse, urllib.request

import openpyxl

STATES = {"AL": "Alabama", "FL": "Florida", "GA": "Georgia", "MS": "Mississippi",
          "NC": "North Carolina", "SC": "South Carolina", "TN": "Tennessee"}
OVERPASS = "https://overpass-api.de/api/interpreter"
EIA = "https://www.eia.gov/electricity/data/eia860m/xls/{month}_generator{year}.xlsx"
MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august",
          "september", "october", "november", "december"]
UA = "helene-site/1.0 (https://github.com/hatamimatt/helene-site)"
MIN_KV = 44                  # lines: transmission and sub-transmission
MIN_SUB_KV = 115             # substations: transmission level only
MIN_PLANT_MW = 5
CLASSES = [(500, "500"), (200, "230"), (100, "115"), (MIN_KV, "69")]   # lower bound (kV), class name
SIMPLIFY_DEG = 0.004                                     # about 400 m
FUELS = ["gas", "coal", "nuclear", "hydro", "solar", "wind", "oil", "battery", "bio", "other"]


def fuel_of(tech):
    t = tech.lower()
    for key, words in [("gas", ["natural gas", "other gases"]), ("coal", ["coal"]), ("nuclear", ["nuclear"]),
                       ("hydro", ["hydro"]), ("solar", ["solar"]), ("wind", ["wind"]),
                       ("oil", ["petroleum"]), ("battery", ["batter", "flywheel"]),
                       ("bio", ["wood", "biomass", "landfill", "municipal", "waste"])]:
        if any(w in t for w in words):
            return key
    return "other"


def overpass(query, cache_path=None, tries=6):
    if cache_path and os.path.exists(cache_path):
        return json.load(open(cache_path))
    body = urllib.parse.urlencode({"data": query}).encode()
    for k in range(tries):
        try:
            req = urllib.request.Request(OVERPASS, data=body, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=400) as r:
                elements = json.load(r)["elements"]
            if cache_path:
                json.dump(elements, open(cache_path, "w"))
            return elements
        except Exception as e:  # Overpass answers 429/504 when busy
            if k == tries - 1:
                raise
            print(f"    Overpass busy ({e}); retrying", flush=True)
            time.sleep(20 * (k + 1))


def kv(tags):
    vals = [int(v) for v in re.findall(r"\d+", tags.get("voltage", "")) if int(v) >= 1000]
    return max(vals) // 1000 if vals else None


def simplify(pts, tol):
    """Douglas-Peucker on a list of (lon, lat)."""
    if len(pts) < 3:
        return pts
    keep = [False] * len(pts); keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        a, b = stack.pop()
        (x0, y0), (x1, y1) = pts[a], pts[b]
        dx, dy = x1 - x0, y1 - y0; n = (dx * dx + dy * dy) ** 0.5 or 1e-12
        best, idx = 0, None
        for i in range(a + 1, b):
            d = abs(dy * (pts[i][0] - x0) - dx * (pts[i][1] - y0)) / n
            if d > best:
                best, idx = d, i
        if idx is not None and best > tol:
            keep[idx] = True; stack += [(a, idx), (idx, b)]
    return [p for p, k in zip(pts, keep) if k]


def centre(el):
    c = el.get("center") or el
    return round(c["lon"], 3), round(c["lat"], 3)


def osm(state_codes, cache):
    """Lines per class as {"ops": [operator names], "lines": [[kv, operator index, lon, lat, lon, lat, ...]]}."""
    lines = {c: {"ops": [], "lines": []} for _, c in CLASSES}; subs, dcs = {}, {}
    seen = set()
    for code in state_codes:
        area = f'area["ISO3166-2"="US-{code}"][admin_level=4]->.a;'
        print(f"  OpenStreetMap {STATES[code]}…", flush=True)
        cached = lambda kind: os.path.join(cache, f"osm_{code}_{kind}.json")
        for el in overpass(f'[out:json][timeout:300];{area}way["power"="line"]["voltage"](area.a);out tags geom;', cached("lines")):
            t = el.get("tags", {}); v = kv(t)
            if el["id"] in seen or v is None or v < MIN_KV or "geometry" not in el:
                continue
            seen.add(el["id"])
            pts = simplify([(g["lon"], g["lat"]) for g in el["geometry"]], SIMPLIFY_DEG)
            group = lines[next(c for lo, c in CLASSES if v >= lo)]
            op = t.get("operator", "")
            if op not in group["ops"]:
                group["ops"].append(op)
            group["lines"].append([v, group["ops"].index(op)] + [round(x, 3) for p in pts for x in p])
        for el in overpass(f'[out:json][timeout:300];{area}nwr["power"="substation"](area.a);out tags center;', cached("substations")):
            t = el.get("tags", {}); v = kv(t)
            if (v or 0) >= MIN_SUB_KV or t.get("substation") == "transmission":
                subs[(el["type"], el["id"])] = [*centre(el), v or 0, t.get("name", ""), t.get("operator", "")]
        for el in overpass(f'[out:json][timeout:300];{area}(nwr["telecom"="data_center"](area.a);nwr["building"="data_center"](area.a););out tags center;', cached("datacenters")):
            t = el.get("tags", {})
            dcs[(el["type"], el["id"])] = [*centre(el), t.get("name", ""), t.get("operator", "")]
    return lines, list(subs.values()), list(dcs.values())


def eia(state_codes, cache):
    """Latest EIA-860M generator file, operating generators summed per plant."""
    os.makedirs(cache, exist_ok=True)
    path = os.path.join(cache, "eia860m.xlsx")
    year, month = time.gmtime().tm_year, time.gmtime().tm_mon
    for _ in range(6):
        month -= 1
        if month == 0:
            year, month = year - 1, 12
        url = EIA.format(month=MONTHS[month - 1], year=year)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=120) as r:
                if r.geturl() != url:   # EIA redirects missing files to a landing page
                    continue
                data = r.read()
            open(path, "wb").write(data)
            break
        except urllib.error.HTTPError:
            continue
    else:
        raise SystemExit("No EIA-860M file found for the last six months")
    print(f"  EIA-860M {MONTHS[month - 1]} {year}", flush=True)
    ws = openpyxl.load_workbook(io.BytesIO(data), read_only=True)["Operating"]
    rows = ws.iter_rows(values_only=True)
    header = next(r for r in rows if r and r[0] == "Entity ID")
    col = {name: i for i, name in enumerate(header)}
    plants = {}
    for r in rows:
        if not r or r[col["Plant State"]] not in state_codes:
            continue
        mw = r[col["Nameplate Capacity (MW)"]]
        if not isinstance(mw, (int, float)):
            continue
        p = plants.setdefault(r[col["Plant ID"]], {"name": r[col["Plant Name"]], "state": r[col["Plant State"]],
                                                   "lon": r[col["Longitude"]], "lat": r[col["Latitude"]],
                                                   "mw": 0.0, "by": collections.Counter()})
        p["mw"] += mw; p["by"][fuel_of(r[col["Technology"]])] += mw
    out = [[round(p["lon"], 3), round(p["lat"], 3), round(p["mw"]), FUELS.index(p["by"].most_common(1)[0][0]), p["name"], p["state"]]
           for p in plants.values() if p["mw"] >= MIN_PLANT_MW and isinstance(p["lon"], float)]
    return sorted(out, key=lambda p: -p[2]), f"{MONTHS[month - 1].title()} {year}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join("data", "grid"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    codes = list(STATES)

    print(f"Power infrastructure for {', '.join(codes)}", flush=True)
    plants, eia_month = eia(codes, os.path.join(".cache", "grid"))
    lines, subs, dcs = osm(codes, os.path.join(".cache", "grid"))

    grid = {"plants": plants, "substations": subs, "datacenters": dcs, "fuels": FUELS, "lineClasses": [c for _, c in CLASSES]}
    with open(os.path.join(args.out, "grid.json"), "w") as fh:
        json.dump(grid, fh, separators=(",", ":"))
    for c, group in lines.items():
        with open(os.path.join(args.out, f"lines_{c}.json"), "w") as fh:
            json.dump(group, fh, separators=(",", ":"))
    meta = {
        "created": time.strftime("%Y-%m-%d", time.gmtime()), "states": codes,
        "lineClassesKv": {c: lo for lo, c in CLASSES}, "minSubstationKv": MIN_SUB_KV, "minPlantMw": MIN_PLANT_MW, "eiaMonth": eia_month,
        "counts": {"lines": {c: len(v["lines"]) for c, v in lines.items()}, "plants": len(plants),
                   "substations": len(subs), "datacenters": len(dcs)},
        "source": "OpenStreetMap contributors (ODbL) via Overpass API; EIA-860M (US Energy Information Administration)",
    }
    with open(os.path.join(args.out, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    sizes = {f: os.path.getsize(os.path.join(args.out, f)) / 1e6 for f in sorted(os.listdir(args.out)) if f.endswith(".json")}
    print(f"Done: {json.dumps(meta['counts'])}\n  " + ", ".join(f"{f} {mb:.1f} MB" for f, mb in sizes.items()))
    print(f"  plant capacity {sum(p[2] for p in plants) / 1000:.0f} GW; largest: " + "; ".join(f"{p[4]} ({p[2]} MW)" for p in plants[:3]))


if __name__ == "__main__":
    sys.exit(main())
