#!/usr/bin/env python3
"""Build the data folder for helene_viewer.html from real ERA5 fields.

Data come from the public ERA5 copy on Google Cloud (ARCO-ERA5, raw daily
netCDF files). No account or API key is needed. Each file is a global day of
one variable, but the script reads only the rows of your box with HTTP range
requests, so an event costs a few hundred MB of traffic, not several GB.

Needs only numpy and the standard library.

    python prepare_event.py                      # Hurricane Helene, default box
    python prepare_event.py --out data/helene    # same, explicit output

Output (what the page expects in data/<event>/):
    meta.json        grid, times, layer file names, markers, default point
    <layer>.json     {"vmin", "vmax", "data"}: base64 uint8, 0..254 scaled
                     between vmin and vmax, 255 = missing; frames in time
                     order, rows from south to north
"""
import argparse, base64, datetime as dt, json, os, struct, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np

BASE = "https://storage.googleapis.com/gcp-public-data-arco-era5/raw"

EVENTS = {
    "helene": dict(
        title="Hurricane Helene",
        subtitle="23–29 September 2024, southeastern United States",
        start="2024-09-23T00", end="2024-09-29T21", step_hours=3,
        lat0=16.0, lat1=40.0, lon0=-98.0, lon1=-70.0,
        point={"name": "Asheville, NC", "lat": 35.60, "lon": -82.55},
        markers=[
            {"time": "2024-09-24T12:00", "title": "A storm organizes", "view": "moisture",
             "text": "Helene takes shape over the warm northwestern Caribbean, embedded in a deep reservoir of tropical moisture."},
            {"time": "2024-09-26T00:00", "title": "Rain arrives before the storm", "view": "rainfall",
             "text": "A predecessor rain event sets up over the southern Appalachians, soaking soils a day before Helene's core arrives."},
            {"time": "2024-09-27T03:00", "title": "Landfall in the Big Bend", "view": "synoptic",
             "text": "Helene makes landfall in Florida's Big Bend as a major hurricane and races north toward a cutoff low over the Tennessee Valley."},
            {"time": "2024-09-27T12:00", "title": "Flooding in western North Carolina", "view": "rainfall",
             "text": "Moist southeasterly flow is forced up the Blue Ridge escarpment. Rain on already saturated ground drives catastrophic flooding."},
            {"time": "2024-09-28T00:00", "title": "Absorbed by the cutoff low", "view": "synoptic",
             "text": "Helene's remnants merge with the upper-level low and the rain shield weakens."},
        ],
    ),
}

# key in the page -> (ARCO folder, file name, unit conversion)
SOURCES = {
    "tp":    ("single_level", "total_precipitation", "surface", lambda a: np.maximum(a * 1000.0, 0)),  # m -> mm
    "pw":    ("single_level", "total_column_water_vapour", "surface", lambda a: a),                    # kg m-2 = mm
    "t2m":   ("single_level", "2m_temperature", "surface", lambda a: a - 273.15),
    "d2m":   ("single_level", "2m_dewpoint_temperature", "surface", lambda a: a - 273.15),
    "cape":  ("single_level", "convective_available_potential_energy", "surface", lambda a: a),
    "ivtu":  ("single_level", "vertical_integral_of_eastward_water_vapour_flux", "surface", lambda a: a),
    "ivtv":  ("single_level", "vertical_integral_of_northward_water_vapour_flux", "surface", lambda a: a),
    "u850":  ("pressure_level", "u_component_of_wind", "850", lambda a: a),
    "v850":  ("pressure_level", "v_component_of_wind", "850", lambda a: a),
    "u250":  ("pressure_level", "u_component_of_wind", "250", lambda a: a),
    "v250":  ("pressure_level", "v_component_of_wind", "250", lambda a: a),
    "z500":  ("pressure_level", "geopotential", "500", lambda a: a / 9.80665),                          # m2 s-2 -> m
}

# ---------- minimal netCDF3 header reader (classic and 64-bit offset) ----------
NC_TYPES = {1: ("b", 1), 2: ("S1", 1), 3: (">i2", 2), 4: (">i4", 4), 5: (">f4", 4), 6: (">f8", 8)}


def http_range(url, start, length, tries=5):
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{start + length - 1}"})
    for k in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
            if len(data) != length:
                raise IOError(f"short read {len(data)} of {length}")
            return data
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(1.5 * (k + 1))


def parse_header(buf):
    pos = 0

    def u32():
        nonlocal pos
        v = struct.unpack(">I", buf[pos:pos + 4])[0]; pos += 4; return v

    def u64():
        nonlocal pos
        v = struct.unpack(">Q", buf[pos:pos + 8])[0]; pos += 8; return v

    def name():
        nonlocal pos
        n = u32(); s = buf[pos:pos + n].decode(); pos += n + (-n % 4); return s

    def attrs():
        nonlocal pos
        u32(); n = u32(); out = {}
        for _ in range(n):
            nm = name(); typ = u32(); cnt = u32()
            dtype, size = NC_TYPES[typ]
            raw = buf[pos:pos + cnt * size]; pos += cnt * size + (-(cnt * size) % 4)
            out[nm] = raw.decode(errors="replace") if typ == 2 else np.frombuffer(raw, dtype)
        return out

    if buf[:3] != b"CDF":
        raise ValueError("not a netCDF3 file")
    version = buf[3]; pos = 4
    numrecs = u32()
    u32(); ndims = u32()
    dims = []
    for _ in range(ndims):
        dims.append((name(), u32()))
    attrs()
    u32(); nvars = u32()
    variables = {}
    for _ in range(nvars):
        nm = name(); nd = u32(); ids = [u32() for _ in range(nd)]
        at = attrs(); typ = u32(); vsize = u32(); begin = u64() if version == 2 else u32()
        variables[nm] = dict(dims=[dims[i] for i in ids], attrs=at, type=typ, vsize=vsize, begin=begin)
    for v in variables.values():
        v["record"] = bool(v["dims"]) and v["dims"][0][1] == 0
    recvars = [v for v in variables.values() if v["record"]]
    if len(recvars) == 1:
        v = recvars[0]
        recsize = int(np.prod([d[1] for d in v["dims"][1:]])) * NC_TYPES[v["type"]][1]
    else:
        recsize = sum(v["vsize"] for v in recvars)
    return dict(numrecs=numrecs, vars=variables, recsize=recsize)


class RemoteNC:
    """One ARCO file; reads row blocks of the (time, latitude, longitude) data variable."""

    def __init__(self, url):
        self.url = url
        size = 65536
        while True:
            try:
                self.h = parse_header(http_range(url, 0, size)); break
            except (struct.error, IndexError, UnicodeDecodeError):
                size *= 4
                if size > 4 << 20:
                    raise
        data = [k for k, v in self.h["vars"].items() if len(v["dims"]) == 3]
        if len(data) != 1:
            raise ValueError(f"{url}: expected one 3-D variable, found {data}")
        self.v = self.h["vars"][data[0]]
        (_, nt), (_, self.ny), (_, self.nx) = self.v["dims"]
        self.nt = self.h["numrecs"] if nt == 0 else nt
        self.dtype, self.size = NC_TYPES[self.v["type"]]
        a = self.v["attrs"]
        self.scale = float(a["scale_factor"][0]) if "scale_factor" in a else 1.0
        self.offset = float(a["add_offset"][0]) if "add_offset" in a else 0.0
        self.fill = [float(a[k][0]) for k in ("_FillValue", "missing_value") if k in a]

    def coord(self, name):
        v = self.h["vars"][name]
        n = v["dims"][0][1]
        dtype, size = NC_TYPES[v["type"]]
        return np.frombuffer(http_range(self.url, v["begin"], n * size), dtype)

    def rows(self, t, r0, r1):
        """Rows r0..r1 (inclusive) of time index t, unpacked to float32."""
        slab = self.ny * self.nx * self.size
        base = self.v["begin"] + (t * self.h["recsize"] if self.v["record"] else t * slab)
        raw = http_range(self.url, base + r0 * self.nx * self.size, (r1 - r0 + 1) * self.nx * self.size)
        q = np.frombuffer(raw, self.dtype).reshape(r1 - r0 + 1, self.nx)
        a = q.astype(np.float32) * self.scale + self.offset
        for f in self.fill:
            a[q == f] = np.nan
        return a


# ---------- main ----------
def encode(frames):
    """frames: (nt, ny, nx) float array, rows south to north."""
    finite = frames[np.isfinite(frames)]
    vmin, vmax = float(finite.min()), float(finite.max())
    if vmax <= vmin:
        vmax = vmin + 1.0
    q = np.round((frames - vmin) / (vmax - vmin) * 254)
    q = np.where(np.isfinite(frames), np.clip(q, 0, 254), 255).astype(np.uint8)
    return {"vmin": round(vmin, 4), "vmax": round(vmax, 4), "data": base64.b64encode(q.tobytes()).decode()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--event", default="helene", choices=sorted(EVENTS))
    ap.add_argument("--out", help="output folder (default data/<event>)")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    ev = EVENTS[args.event]
    out = args.out or os.path.join("data", args.event)
    os.makedirs(out, exist_ok=True)

    t0 = dt.datetime.fromisoformat(ev["start"]); t1 = dt.datetime.fromisoformat(ev["end"])
    step = dt.timedelta(hours=ev["step_hours"])
    frames = []
    t = t0
    while t <= t1:
        frames.append(t); t += step
    # hourly precipitation for the window ending at each frame
    rain_hours = sorted({f - dt.timedelta(hours=h) for f in frames for h in range(ev["step_hours"])})

    # ERA5 grid: latitude 90 -> -90, longitude 0 -> 359.75, 0.25 degrees
    r0 = round((90 - ev["lat1"]) / 0.25); r1 = round((90 - ev["lat0"]) / 0.25)
    c0 = round((ev["lon0"] % 360) / 0.25); c1 = round((ev["lon1"] % 360) / 0.25)
    ny, nx = r1 - r0 + 1, c1 - c0 + 1

    files = {}

    def remote(key, day):
        folder, var, level = SOURCES[key][:3]
        url = f"{BASE}/date-variable-{folder}/{day:%Y/%m/%d}/{var}/{level}.nc"
        if url not in files:
            files[url] = RemoteNC(url)
        return files[url]

    # open every needed file once (headers only) and check the grid on the first
    jobs = [(k, f) for k in SOURCES if k != "tp" for f in frames] + [("tp", h) for h in rain_hours]
    days = sorted({(k, when.date()) for k, when in jobs})
    print(f"Reading headers of {len(days)} files…", flush=True)
    with ThreadPoolExecutor(args.workers) as ex:
        opened = list(ex.map(lambda kd: (kd, RemoteNC(
            f"{BASE}/date-variable-{SOURCES[kd[0]][0]}/{kd[1]:%Y/%m/%d}/{SOURCES[kd[0]][1]}/{SOURCES[kd[0]][2]}.nc")), days))
    for (k, d), nc in opened:
        files[nc.url] = nc
    first = opened[0][1]
    lat = first.coord("latitude"); lon = first.coord("longitude")
    assert abs(lat[r0] - ev["lat1"]) < 1e-3 and abs(lat[r1] - ev["lat0"]) < 1e-3, "latitude grid mismatch"
    assert abs(lon[c0] - ev["lon0"] % 360) < 1e-3 and abs(lon[c1] - ev["lon1"] % 360) < 1e-3, "longitude grid mismatch"
    assert first.nt == 24, f"expected 24 hourly records, found {first.nt}"

    def fetch(job):
        k, when = job
        nc = remote(k, when.date())
        a = nc.rows(when.hour, r0, r1)[:, c0:c1 + 1]
        return job, SOURCES[k][3](a)[::-1]  # south to north

    print(f"Reading {len(jobs)} map slices for a {ny} × {nx} box…", flush=True)
    got = {}
    with ThreadPoolExecutor(args.workers) as ex:
        for n, (job, a) in enumerate(ex.map(fetch, jobs), 1):
            got[job] = a
            if n % 100 == 0 or n == len(jobs):
                print(f"  {n}/{len(jobs)}", flush=True)

    F = {k: np.stack([got[(k, f)] for f in frames]) for k in SOURCES if k != "tp"}
    F["precip3"] = np.stack([sum(got[("tp", f - dt.timedelta(hours=h))] for h in range(ev["step_hours"])) for f in frames])
    F["precip_acc"] = np.cumsum(F["precip3"], axis=0)
    F["wind850"] = np.hypot(F["u850"], F["v850"])
    F["wind250"] = np.hypot(F["u250"], F["v250"])
    F["ivt"] = np.hypot(F["ivtu"], F["ivtv"])

    layers = {}
    for k, a in F.items():
        name = f"{k}.json"
        with open(os.path.join(out, name), "w") as fh:
            json.dump(encode(a), fh, separators=(",", ":"))
        layers[k] = name

    meta = {
        "event": args.event, "title": ev["title"], "subtitle": ev["subtitle"],
        "source": "ERA5 reanalysis (Hersbach et al. 2020), Copernicus Climate Change Service, via ARCO-ERA5 on Google Cloud",
        "grid": {"lat0": ev["lat0"], "lon0": ev["lon0"], "dlat": 0.25, "dlon": 0.25, "ny": ny, "nx": nx},
        "times": [f"{f:%Y-%m-%dT%H:%M}" for f in frames],
        "layers": layers, "point": ev["point"], "markers": ev["markers"],
    }
    with open(os.path.join(out, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1)

    # quick sanity summary
    gx = round((ev["point"]["lon"] - ev["lon0"]) / 0.25); gy = round((ev["point"]["lat"] - ev["lat0"]) / 0.25)
    print(f"Done: {len(layers)} layers, {len(frames)} frames in {out}/")
    print(f"  storm-total rain at {ev['point']['name']}: {F['precip_acc'][-1, gy, gx]:.0f} mm; "
          f"box max {np.nanmax(F['precip_acc'][-1]):.0f} mm; max IVT {np.nanmax(F['ivt']):.0f} kg m-1 s-1")


if __name__ == "__main__":
    sys.exit(main())
