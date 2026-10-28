#!/usr/bin/env python3
"""What forecasts issued before Helene predicted for rain in the southern Appalachians.

For every 00 UTC run from 14 to 25 September 2024 this reads, from public
archives on AWS (no account needed):
  - GFS 0.25 deg: accumulated precipitation since the start of the run
  - GEFS 0.5 deg: 6-hourly precipitation from the control and 30 members
  - ECMWF ENS 0.25 deg: accumulated precipitation, control and 50 members
  - ECMWF HRES 0.25 deg: accumulated precipitation (runs reach 10 days)
and totals the rain each run predicted for the event window
(25 Sep 00 UTC to 28 Sep 00 UTC). What fell is taken from NCEP Stage IV
(radar estimates adjusted to rain gauges, 4 km, from the Iowa Environmental
Mesonet archive), with ERA5 rain from ARCO-ERA5 as a second reference.

Only the precipitation record is downloaded from each GRIB2 file, using the
.idx / .index byte offsets, so the whole run is about 3 GB of traffic.

Needs numpy and eccodes (pip install numpy eccodes).

    python prepare_forecasts.py

Writes data/forecasts/meta.json and uint8 map files (0..254 scaled, 255 = missing,
rows north to south).
"""
import argparse, datetime as dt, json, os, random, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import eccodes

from prepare_event import BASE as ERA5_BASE, RemoteNC, http_range

GFS = "https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.{d:%Y%m%d}/00/atmos/gfs.t00z.pgrb2.0p25.f{f:03d}"
GEFS = "https://noaa-gefs-pds.s3.amazonaws.com/gefs.{d:%Y%m%d}/00/atmos/pgrb2ap5/{m}.t00z.pgrb2a.0p50.f{f:03d}"
MEMBERS = ["gec00"] + [f"gep{i:02d}" for i in range(1, 31)]
ECMWF = "https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com/{d:%Y%m%d}/00z/ifs/0p25/{stream}/{d:%Y%m%d}000000-{f}h-{stream}-{kind}"
EC_MEMBERS = list(range(51))                      # 0 = control
HRES_MAX_STEP = 240
STAGE4 = "https://mesonet.agron.iastate.edu/archive/data/{t:%Y/%m/%d}/stage4/ST4.{t:%Y%m%d%H}.06h.grib"
ST4_D = 0.1                                       # map grid for Stage IV

W0, W1 = dt.datetime(2024, 9, 25, 0), dt.datetime(2024, 9, 28, 0)
INITS = [dt.datetime(2024, 9, d, 0) for d in range(14, 26)]
BOX = (34.5, 36.5, -84.0, -81.0)                # south, north, west, east: southern Appalachians
MAP = (24.0, 40.0, -92.0, -72.0)                 # map extent
POINT = {"name": "Asheville, NC", "lat": 35.60, "lon": -82.55}
THRESH = 200.0                                    # mm in 72 h, about 8 inches


def get(url, tries=5):
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read()
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(2 * (k + 1))  # back off when a server throttles (S3 answers 503 Slow Down)


IDX = {}


def acc(a, b):
    """.idx labels for an accumulation from hour a to b (NCEP writes whole days as "day")."""
    labels = [f":APCP:surface:{a}-{b} hour acc fcst:"]
    if a % 24 == 0 and b % 24 == 0:
        labels.append(f":APCP:surface:{a // 24}-{b // 24} day acc fcst:")
    return labels


def grib_record(url, labels):
    """Download and decode the one GRIB2 message whose .idx line matches one of `labels`."""
    if url not in IDX:
        IDX[url] = get(url + ".idx").decode().splitlines()
    lines = IDX[url]
    match = labels
    for i, line in enumerate(lines):
        if any(m in line for m in labels):
            start = int(line.split(":")[1])
            end = int(lines[i + 1].split(":")[1]) if i + 1 < len(lines) else None
            break
    else:
        raise KeyError(f"{match} not in {url}")
    data = http_range(url, start, end - start) if end else get(url)  # last record: whole file is rare
    return decode(data)


def decode(data):
    """GRIB2 message -> (values with rows from 90N, grid spacing, longitude of first column)."""
    h = eccodes.codes_new_from_message(data)
    try:
        ni, nj = eccodes.codes_get(h, "Ni"), eccodes.codes_get(h, "Nj")
        lat0 = eccodes.codes_get(h, "latitudeOfFirstGridPointInDegrees")
        lon0 = eccodes.codes_get(h, "longitudeOfFirstGridPointInDegrees")
        dlat = eccodes.codes_get(h, "jDirectionIncrementInDegrees")
        vals = eccodes.codes_get_values(h).reshape(nj, ni)
    finally:
        eccodes.codes_release(h)
    assert abs(lat0 - 90) < 1e-6, "expected grids starting at 90N"
    return vals.astype(np.float32), dlat, lon0


EC_INDEX = {}


def ec_index(init, stream, kind, f):
    url = ECMWF.format(d=init, stream=stream, f=f, kind=kind)
    if url not in EC_INDEX:
        recs = {}
        for line in get(url + ".index").decode().splitlines():
            r = json.loads(line)
            if r["param"] == "tp":
                recs[int(r.get("number", 0))] = (r["_offset"], r["_length"])
        EC_INDEX[url] = recs
    return url


def stage4_total():
    """Stage IV 6-hourly files summed over the window -> (values, lats, lons) at native 4 km points."""
    tot = None
    t = W0
    while t < W1:
        t += dt.timedelta(hours=6)
        h = eccodes.codes_new_from_message(get(STAGE4.format(t=t)))
        try:
            vals = eccodes.codes_get_values(h).astype(np.float64)
            miss = eccodes.codes_get(h, "missingValue")
            vals[vals == miss] = np.nan
            if tot is None:
                lats = eccodes.codes_get_array(h, "latitudes")
                lons = (eccodes.codes_get_array(h, "longitudes") + 180) % 360 - 180
        finally:
            eccodes.codes_release(h)
        tot = vals if tot is None else tot + vals
    return tot, lats, lons


def stage4_summary(tot, lats, lons):
    """Box mean and nearest-point value from native points, and a 0.1 degree map of cell means."""
    s, n, w, e = BOX
    inbox = (lats >= s) & (lats <= n) & (lons >= w) & (lons <= e) & np.isfinite(tot)
    box = float(tot[inbox].mean())
    k = np.nanargmin(np.where(np.isfinite(tot), (lats - POINT["lat"]) ** 2 + ((lons - POINT["lon"]) * np.cos(np.deg2rad(POINT["lat"]))) ** 2, np.inf))
    s, n, w, e = MAP
    ny, nx = round((n - s) / ST4_D) + 1, round((e - w) / ST4_D) + 1
    r = np.round((n - lats) / ST4_D).astype(int); c = np.round((lons - w) / ST4_D).astype(int)
    ok = (r >= 0) & (r < ny) & (c >= 0) & (c < nx) & np.isfinite(tot)
    idx = r[ok] * nx + c[ok]
    sums = np.bincount(idx, tot[ok], ny * nx); cnt = np.bincount(idx, minlength=ny * nx)
    grid = np.where(cnt > 0, sums / np.maximum(cnt, 1), np.nan).reshape(ny, nx)
    return box, float(tot[k]), grid


def polite_range(url, start, length, tries=10):
    """Range request with exponential backoff; the ECMWF bucket throttles bursts with 503 Slow Down."""
    for k in range(tries):
        try:
            return http_range(url, start, length, tries=1)
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(min(30, 2 ** k) * (0.5 + random.random()))


def ec_record(init, stream, kind, f, member):
    """ECMWF open data: total precipitation (mm) since the start of the run, for one member."""
    url = ec_index(init, stream, kind, f)
    off, length = EC_INDEX[url][member]
    vals, d, lon0 = decode(polite_range(url + ".grib2", off, length))
    return vals * 1000.0, d, lon0  # m -> mm


def crop(field, ext):
    """(global values from 90N, spacing, first longitude) -> ext (south, north, west, east)."""
    a, d, lon0 = field
    s, n, w, e = ext
    r0, r1 = round((90 - n) / d), round((90 - s) / d)
    c0 = round(((w - lon0) % 360) / d); c1 = c0 + round((e - w) / d)
    return a[r0:r1 + 1, c0:c1 + 1]


def box_mean(a, d, ext, box=BOX):
    s, n, w, e = ext
    lat = n - np.arange(a.shape[0]) * d
    lon = w + np.arange(a.shape[1]) * d
    rows = (lat >= box[0]) & (lat <= box[1]); cols = (lon >= box[2]) & (lon <= box[3])
    return float(np.nanmean(a[np.ix_(rows, cols)]))


def point_val(a, d, ext, p=POINT):
    s, n, w, e = ext
    return float(a[round((n - p["lat"]) / d), round((p["lon"] - w) / d)])


def quantize(frames, lo, hi):
    q = np.round((frames - lo) / (hi - lo) * 254)
    return np.where(np.isfinite(frames), np.clip(q, 0, 254), 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join("data", "forecasts"))
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    hrs = lambda t, i: int((t - i).total_seconds() // 3600)

    # ---- GFS: accumulated since the start of the run, so two records per run
    def gfs(init):
        fa, fb = hrs(W0, init), hrs(W1, init)
        b = crop(grib_record(GFS.format(d=init, f=fb), acc(0, fb)), MAP)
        a = crop(grib_record(GFS.format(d=init, f=fa), acc(0, fa)), MAP) if fa > 0 else 0
        return init, b - a

    # ---- GEFS: 6-hourly buckets
    jobs = [(init, m, f) for init in INITS for m in MEMBERS for f in range(hrs(W0, init) + 6, hrs(W1, init) + 1, 6)]

    def gefs(job):
        init, m, f = job
        return job, crop(grib_record(GEFS.format(d=init, m=m, f=f), acc(f - 6, f)), MAP)

    # ---- ECMWF: accumulated since the start of the run
    ec_jobs = [(i, "enfo", "ef", f, m) for i in INITS for f in {hrs(W0, i), hrs(W1, i)} if f > 0 for m in EC_MEMBERS]
    ec_jobs += [(i, "oper", "fc", f, 0) for i in INITS if hrs(W1, i) <= HRES_MAX_STEP for f in {hrs(W0, i), hrs(W1, i)} if f > 0]

    def ecmwf(job):
        i, stream, kind, f, m = job
        return job, crop(ec_record(i, stream, kind, f, m), MAP)

    # NOAA results are cached so that adding or changing another source does not download them again
    cache = os.path.join(".cache", "forecasts_noaa.npz")
    with ThreadPoolExecutor(args.workers) as ex:
        if os.path.exists(cache):
            z = np.load(cache)
            G = {i: z[f"gfs_{i:%Y%m%d}"] for i in INITS}
            gefs_tot = {i: z[f"gefs_{i:%Y%m%d}"] for i in INITS}
            print(f"GFS and GEFS: loaded from {cache}", flush=True)
        else:
            print(f"GFS: {len(INITS)} runs; GEFS: {len(jobs)} six-hour fields…", flush=True)
            G = dict(ex.map(gfs, INITS))
            E = {}
            for n, (job, a) in enumerate(ex.map(gefs, jobs), 1):
                E[job] = a
                if n % 500 == 0 or n == len(jobs):
                    print(f"  GEFS {n}/{len(jobs)}", flush=True)
            gefs_tot = {i: np.stack([sum(E[(i, m, f)] for f in range(hrs(W0, i) + 6, hrs(W1, i) + 1, 6)) for m in MEMBERS]) for i in INITS}
            os.makedirs(".cache", exist_ok=True)
            np.savez_compressed(cache, **{f"gfs_{i:%Y%m%d}": G[i] for i in INITS}, **{f"gefs_{i:%Y%m%d}": gefs_tot[i] for i in INITS})
    ec_cache = os.path.join(".cache", "forecasts_ecmwf.npz")
    if os.path.exists(ec_cache):
        z = np.load(ec_cache)
        ens_tot = {i: z[f"ens_{i:%Y%m%d}"] for i in INITS}
        hres_tot = {i: z[f"hres_{i:%Y%m%d}"] if f"hres_{i:%Y%m%d}" in z else None for i in INITS}
        print(f"ECMWF: loaded from {ec_cache}", flush=True)
    else:
        print(f"ECMWF: {len(ec_jobs)} fields…", flush=True)
        for i, stream, kind, f, m in sorted({j[:4] + (0,) for j in ec_jobs}):  # fetch each index once, serially
            ec_index(i, stream, kind, f)
        C = {}
        for n, (job, a) in enumerate(ThreadPoolExecutor(6).map(ecmwf, ec_jobs), 1):  # gentler on the ECMWF bucket
            C[job] = a
            if n % 250 == 0 or n == len(ec_jobs):
                print(f"  ECMWF {n}/{len(ec_jobs)}", flush=True)

        def ec_total(i, stream, kind, m):
            fa, fb = hrs(W0, i), hrs(W1, i)
            return np.maximum(C[(i, stream, kind, fb, m)] - (C[(i, stream, kind, fa, m)] if fa > 0 else 0), 0)
        ens_tot = {i: np.stack([ec_total(i, "enfo", "ef", m) for m in EC_MEMBERS]) for i in INITS}
        hres_tot = {i: ec_total(i, "oper", "fc", 0) if hrs(W1, i) <= HRES_MAX_STEP else None for i in INITS}
        os.makedirs(".cache", exist_ok=True)
        np.savez_compressed(ec_cache, **{f"ens_{i:%Y%m%d}": ens_tot[i] for i in INITS},
                            **{f"hres_{i:%Y%m%d}": hres_tot[i] for i in INITS if hres_tot[i] is not None})

    # ---- ERA5 hourly rain for the same window (hour ending at each time)
    hours = [W0 + dt.timedelta(hours=h) for h in range(1, hrs(W1, W0) + 1)]
    s, n, w, e = MAP
    r0, r1 = round((90 - n) / 0.25), round((90 - s) / 0.25)
    c0, c1 = round((w % 360) / 0.25), round((e % 360) / 0.25)
    ncs = {}

    def era(t):
        url = f"{ERA5_BASE}/date-variable-single_level/{t:%Y/%m/%d}/total_precipitation/surface.nc"
        nc = ncs.get(url) or ncs.setdefault(url, RemoteNC(url))
        return nc.rows(t.hour, r0, r1)[:, c0:c1 + 1] * 1000.0
    for t in hours[:1] + [h for h in hours if h.hour == 0]:  # open each daily file once before threading
        era(t)
    with ThreadPoolExecutor(16) as ex:
        era5 = np.maximum(sum(ex.map(era, hours)), 0)

    # ---- Stage IV: radar estimates adjusted to gauges
    print("Stage IV: 12 six-hour analyses…", flush=True)
    st4_box, st4_pt, st4_map = stage4_summary(*stage4_total())

    # ---- summarize
    runs = []
    for i in INITS:
        mem_box = [box_mean(a, 0.5, MAP) for a in gefs_tot[i]]
        mem_pt = [point_val(a, 0.5, MAP) for a in gefs_tot[i]]
        runs.append({
            "init": f"{i:%Y-%m-%dT%H:%M}", "leadDays": (W0 - i).days,
            "gfsBox": round(box_mean(G[i], 0.25, MAP), 1), "gfsPoint": round(point_val(G[i], 0.25, MAP), 1),
            "gefsBox": [round(v, 1) for v in mem_box], "gefsPoint": [round(v, 1) for v in mem_pt],
            "gefsMaxCell": round(float(np.max(crop_box(gefs_tot[i]))), 1),
            "probBoxCell": round(float(np.mean([np.any(crop_box(a[None])[0] >= THRESH) for a in gefs_tot[i]])), 3),
            "ecBox": [round(box_mean(a, 0.25, MAP), 1) for a in ens_tot[i]],
            "ecPoint": [round(point_val(a, 0.25, MAP), 1) for a in ens_tot[i]],
            "ecProbBoxCell": round(float(np.mean([np.any(crop_box(a[None], 0.25)[0] >= THRESH) for a in ens_tot[i]])), 3),
            "hresBox": None if hres_tot[i] is None else round(box_mean(hres_tot[i], 0.25, MAP), 1),
            "hresPoint": None if hres_tot[i] is None else round(point_val(hres_tot[i], 0.25, MAP), 1),
        })

    lo, hi = 0.0, 600.0
    gfs_frames = np.stack([G[i] for i in INITS])
    mean_frames = np.stack([gefs_tot[i].mean(0) for i in INITS])
    prob_frames = np.stack([(gefs_tot[i] >= THRESH).mean(0) for i in INITS])
    quantize(gfs_frames, lo, hi).tofile(os.path.join(args.out, "gfs.bin"))
    quantize(mean_frames, lo, hi).tofile(os.path.join(args.out, "gefs_mean.bin"))
    quantize(prob_frames, 0, 1).tofile(os.path.join(args.out, "gefs_prob.bin"))
    quantize(era5[None], lo, hi).tofile(os.path.join(args.out, "era5.bin"))
    ec_mean = np.stack([ens_tot[i].mean(0) for i in INITS])
    ec_prob = np.stack([(ens_tot[i] >= THRESH).mean(0) for i in INITS])
    hres = np.stack([hres_tot[i] if hres_tot[i] is not None else np.full_like(era5, np.nan) for i in INITS])
    quantize(ec_mean, lo, hi).tofile(os.path.join(args.out, "ec_mean.bin"))
    quantize(ec_prob, 0, 1).tofile(os.path.join(args.out, "ec_prob.bin"))
    quantize(hres, lo, hi).tofile(os.path.join(args.out, "hres.bin"))
    quantize(st4_map[None], lo, hi).tofile(os.path.join(args.out, "stage4.bin"))

    meta = {
        "window": [f"{W0:%Y-%m-%dT%H:%M}", f"{W1:%Y-%m-%dT%H:%M}"],
        "box": BOX, "point": POINT, "thresholdMm": THRESH, "extent": MAP,
        "era5Box": round(box_mean(era5, 0.25, MAP), 1), "era5Point": round(point_val(era5, 0.25, MAP), 1),
        "stage4Box": round(st4_box, 1), "stage4Point": round(st4_pt, 1),
        "runs": runs,
        "maps": {
            "gfs":       {"file": "gfs.bin", "d": 0.25, "shape": list(gfs_frames.shape[1:]), "vmin": lo, "vmax": hi, "frames": len(INITS)},
            "gefs_mean": {"file": "gefs_mean.bin", "d": 0.5, "shape": list(mean_frames.shape[1:]), "vmin": lo, "vmax": hi, "frames": len(INITS)},
            "gefs_prob": {"file": "gefs_prob.bin", "d": 0.5, "shape": list(prob_frames.shape[1:]), "vmin": 0, "vmax": 1, "frames": len(INITS)},
            "era5":      {"file": "era5.bin", "d": 0.25, "shape": list(era5.shape), "vmin": lo, "vmax": hi, "frames": 1},
            "stage4":    {"file": "stage4.bin", "d": ST4_D, "shape": list(st4_map.shape), "vmin": lo, "vmax": hi, "frames": 1},
            "hres":      {"file": "hres.bin", "d": 0.25, "shape": list(hres.shape[1:]), "vmin": lo, "vmax": hi, "frames": len(INITS)},
            "ec_mean":   {"file": "ec_mean.bin", "d": 0.25, "shape": list(ec_mean.shape[1:]), "vmin": lo, "vmax": hi, "frames": len(INITS)},
            "ec_prob":   {"file": "ec_prob.bin", "d": 0.25, "shape": list(ec_prob.shape[1:]), "vmin": 0, "vmax": 1, "frames": len(INITS)},
        },
        "source": "GFS and GEFS forecasts, NOAA/NCEP, via the NOAA Open Data Dissemination program on AWS; ECMWF IFS HRES and ENS, ECMWF open data on AWS (CC BY 4.0); Stage IV, NCEP, via the Iowa Environmental Mesonet; ERA5 via ARCO-ERA5",
    }
    with open(os.path.join(args.out, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1)

    print(f"Stage IV window rain: box {meta['stage4Box']} mm, {POINT['name']} {meta['stage4Point']} mm, map max {np.nanmax(st4_map):.0f} mm")
    print(f"ERA5 window rain: box {meta['era5Box']} mm, {POINT['name']} {meta['era5Point']} mm")
    print("init        lead  GFS box  GEFS box mean [min–max]   GFS pt  GEFS pt mean  P(cell≥200 in box)")
    for r in runs:
        b = np.array(r["gefsBox"]); p = np.array(r["gefsPoint"])
        print(f"{r['init'][:10]}  {r['leadDays']:>3}d  {r['gfsBox']:7.0f}  {b.mean():6.0f} [{b.min():4.0f}–{b.max():4.0f}]       "
              f"{r['gfsPoint']:6.0f}  {p.mean():8.0f}      {r['probBoxCell']:.2f}")
    print("init        lead  HRES box  ENS box mean [min–max]   HRES pt  ENS pt mean  P(cell≥200 in box)")
    for r in runs:
        b = np.array(r["ecBox"]); p = np.array(r["ecPoint"]); hb = r["hresBox"]; hp = r["hresPoint"]
        print(f"{r['init'][:10]}  {r['leadDays']:>3}d  {'   –' if hb is None else f'{hb:7.0f}'}   {b.mean():6.0f} [{b.min():4.0f}–{b.max():4.0f}]      "
              f"{'   –' if hp is None else f'{hp:6.0f}'}  {p.mean():8.0f}      {r['ecProbBoxCell']:.2f}")


def crop_box(stack, d=0.5):
    """Cells of a (members, lat, lon) stack on the MAP grid that fall inside BOX."""
    s, n, w, e = MAP
    lat = n - np.arange(stack.shape[1]) * d
    lon = w + np.arange(stack.shape[2]) * d
    rows = (lat >= BOX[0]) & (lat <= BOX[1]); cols = (lon >= BOX[2]) & (lon <= BOX[3])
    return stack[:, rows][:, :, cols]


if __name__ == "__main__":
    sys.exit(main())
