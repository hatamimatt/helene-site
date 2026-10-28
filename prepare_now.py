#!/usr/bin/env python3
"""Extreme-rain outlook for the next 5 days from the latest ECMWF ensemble (ENS).

Finds the newest ENS run in ECMWF's open data on AWS (no account needed),
reads total precipitation from all 50 members at 24-hour steps out to 120 h,
and computes for every grid point:
  - ensemble-mean rain for each of the next five 24-hour periods
  - the chance of at least 50 mm in each 24-hour period
  - the chance of at least 100 mm in any 72-hour window within the 5 days
  - the chance that the wettest 72-hour window is locally extreme: above the
    ERA5 99th percentile for that place and season (needs data/climate/ from
    prepare_climate.py), compared on the same 1 degree grid as the climate;
    if prepare_model_climate.py has run, the threshold is mapped to the same
    percentile of the model's own past forecasts, removing its rain bias
plus 850 hPa wind from one member for the globe's moving lines.

Writes data/now/ in the same binary format as data/globe/, on a 0.5 degree grid:
    meta.json    run time, valid times, grid, layers (file, vmin, vmax, frames)
    <layer>.bin  uint8, 0..254 scaled between vmin and vmax, 255 = missing;
                 frames in time order, rows from 90N to 90S, columns from 180W

Needs numpy and eccodes. ECMWF open data is CC BY 4.0: credit ECMWF.

    python prepare_now.py
"""
import argparse, datetime as dt, json, os, random, sys, threading, time, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from prepare_forecasts import decode, polite_range

BASE = "https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com"
URL = BASE + "/{r:%Y%m%d}/{r:%H}z/ifs/0p25/enfo/{r:%Y%m%d%H}0000-{f}h-enfo-ef"
MEMBERS = range(1, 51)              # perturbed members; the control run is not in the open-data ENS files
STEPS = [24, 48, 72, 96, 120]       # cumulative precipitation at the end of each day
DAY_MM, THREE_DAY_MM = 50.0, 100.0  # exceedance thresholds (about 2 and 4 inches)
FLOW_MEMBER = 1


def polite_get(url, tries=10):
    """GET with exponential backoff on throttling (503 Slow Down) and network errors; 403/404 raise at once."""
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (403, 404) or k == tries - 1:
                raise
        except OSError:
            if k == tries - 1:
                raise
        time.sleep(min(30, 2 ** k) * (0.5 + random.random()))


def latest_run(now):
    """Newest 00/12 UTC run whose 120 h file is published (S3 answers 403/404 for missing files)."""
    t = now.replace(minute=0, second=0, microsecond=0, hour=12 if now.hour >= 12 else 0)
    for _ in range(8):
        try:
            polite_get(URL.format(r=t, f=120) + ".index")
            return t
        except urllib.error.HTTPError as e:
            if e.code not in (403, 404):
                raise
        t -= dt.timedelta(hours=12)
    raise SystemExit("No ECMWF ENS run found in the last four days")


INDEX, LOCK = {}, threading.Lock()


def record(run, step, param, member, level=None):
    url = URL.format(r=run, f=step)
    with LOCK:
        if url not in INDEX:
            recs = {}
            for line in polite_get(url + ".index").decode().splitlines():
                r = json.loads(line)
                recs[(r["param"], r.get("levelist"), int(r.get("number", 0)))] = (r["_offset"], r["_length"])
            INDEX[url] = recs
    off, length = INDEX[url][(param, level, member)]
    vals, d, lon0 = decode(polite_range(url + ".grib2", off, length))
    assert abs(d - 0.25) < 1e-6 and abs(((lon0 + 180) % 360) - 0) < 1e-6, "expected 0.25 deg grid starting at 180W"
    return vals


def to_half_degree(a, how):
    """(721, 1440) at 0.25 deg -> (361, 720): 3x3 neighbourhood mean or max around every other point."""
    p = np.pad(a, ((1, 1), (0, 0)), mode="edge")
    p = np.concatenate([p[:, -1:], p, p[:, :1]], axis=1)
    views = [p[i:i + 721, j:j + 1440] for i in range(3) for j in range(3)]
    out = np.max(views, axis=0) if how == "max" else np.mean(views, axis=0)
    return out[::2, ::2]


# weights for a conservative 0.25 -> 1 degree average: the 1 degree cell spans +-0.5 degree,
# so the outermost of its five 0.25 degree points along each axis only half belongs to it
W5 = np.array([0.5, 1, 1, 1, 0.5])


def to_one_degree(a):
    """(721, 1440) at 0.25 deg, 90N->90S and 180W-> -> (181, 360) cell averages centred on whole degrees."""
    p = np.pad(a, ((2, 2), (0, 0)), mode="edge")
    p = np.concatenate([p[:, -2:], p, p[:, :2]], axis=1)
    out = np.zeros((181, 360))
    for i, wi in enumerate(W5):
        for j, wj in enumerate(W5):
            out += wi * wj * p[i:i + 721:4, j:j + 1440:4]
    return out / W5.sum() ** 2


def one_to_half(a):
    """(181, 360) at 1 deg -> (361, 720) at 0.5 deg by bilinear interpolation (longitude wraps)."""
    rows = np.empty((361, 360)); rows[0::2] = a; rows[1::2] = (a[:-1] + a[1:]) / 2
    out = np.empty((361, 720)); out[:, 0::2] = rows; out[:, 1::2] = (rows + np.roll(rows, -1, axis=1)) / 2
    return out


def load_climate(run, path=os.path.join("data", "climate")):
    """ERA5 thresholds from prepare_climate.py, if they were built for this time of year."""
    try:
        info = json.load(open(os.path.join(path, "meta.json")))
    except FileNotFoundError:
        print("No data/climate/: skipping the locally extreme layer (run prepare_climate.py)")
        return None
    m, d = map(int, info["center"].split("-"))
    gap = abs((run.date() - dt.date(run.year, m, d)).days)
    if min(gap, 365 - gap) > 20:
        print(f"data/climate/ is centred on {info['center']}, too far from the run date: skipping the locally extreme layer")
        return None
    q99, base = np.load(os.path.join(path, "q99_72h.npy")), np.load(os.path.join(path, "base_rate.npy"))
    threshold, info["calibrated"] = q99, False
    try:
        model = json.load(open(os.path.join(path, "model_meta.json")))
        if model["center"] == info["center"]:
            threshold = np.load(os.path.join(path, "model_threshold_72h.npy"))
            base = np.load(os.path.join(path, "model_base_rate.npy"))
            info["calibrated"], info["modelRuns"] = True, len(model["runs"])
    except FileNotFoundError:
        pass
    print("Locally extreme threshold:", "calibrated to the model climate" if info["calibrated"] else "ERA5 q99 (uncalibrated; run prepare_model_climate.py)")
    return info, q99, base, threshold


def quantize(frames, lo, hi):
    q = np.round((frames - lo) / (hi - lo) * 254)
    return np.where(np.isfinite(frames), np.clip(q, 0, 254), 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join("data", "now"))
    ap.add_argument("--run", help="run time as YYYYMMDDHH (default: latest available)")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    run = dt.datetime.strptime(args.run, "%Y%m%d%H") if args.run else latest_run(dt.datetime.utcnow())
    print(f"ECMWF ENS run {run:%Y-%m-%d %H} UTC, {len(MEMBERS)} members", flush=True)

    clim = load_climate(run)
    shape = (721, 1440)
    rain_sum = np.zeros((len(STEPS),) + shape)
    day_hits = np.zeros((len(STEPS),) + shape)
    three_day_hits = np.zeros(shape)
    local_hits = np.zeros((181, 360))
    acc_lock = threading.Lock()

    def member(m):
        cum = [np.zeros(shape, np.float32)] + [record(run, f, "tp", m) * 1000.0 for f in STEPS]  # m -> mm
        daily = np.stack([np.maximum(cum[k + 1] - cum[k], 0) for k in range(len(STEPS))])
        windows = [np.maximum(cum[k + 3] - cum[k], 0) for k in range(len(STEPS) - 2)]
        three = np.max(windows, axis=0)  # wettest 72 h window
        local = clim is not None and np.max([to_one_degree(w) for w in windows], axis=0) >= clim[3]
        with acc_lock:
            rain_sum[:] += daily
            day_hits[:] += daily >= DAY_MM
            three_day_hits[:] += three >= THREE_DAY_MM
            local_hits[:] += local
        return m

    with ThreadPoolExecutor(args.workers) as ex:
        for n, _ in enumerate(ex.map(member, MEMBERS), 1):
            if n % 10 == 0 or n == len(MEMBERS):
                print(f"  members {n}/{len(MEMBERS)}", flush=True)
        winds = list(ex.map(lambda job: record(run, job[0], job[1], FLOW_MEMBER, "850"),
                            [(f, p) for f in STEPS for p in ("u", "v")]))

    nm = len(MEMBERS)
    rain = np.stack([to_half_degree(a, "mean") for a in rain_sum / nm])
    p24 = np.stack([to_half_degree(a, "max") for a in day_hits / nm])
    p72 = to_half_degree(three_day_hits / nm, "max")[None]
    u850 = np.stack([to_half_degree(w, "mean") for w in winds[0::2]])
    v850 = np.stack([to_half_degree(w, "mean") for w in winds[1::2]])

    out = [("rain24", rain, 0, 300), ("p24", p24, 0, 1), ("p72", p72, 0, 1), ("u850", u850, -80, 80), ("v850", v850, -80, 80)]
    if clim is not None:
        plocal = local_hits / nm
        out += [("plocal", one_to_half(plocal)[None], 0, 1),
                ("q99", one_to_half(clim[1])[None], 0, 500),
                ("base", one_to_half(clim[2])[None], 0, 0.25)]
    layers = {}
    for name, frames, lo, hi in out:
        quantize(frames, lo, hi).tofile(os.path.join(args.out, f"{name}.bin"))
        layers[name] = {"file": f"{name}.bin", "vmin": lo, "vmax": hi, "frames": int(frames.shape[0])}

    meta = {
        "run": f"{run:%Y-%m-%dT%H:%M}",
        "created": dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M"),
        "members": nm, "flowMember": FLOW_MEMBER,
        "thresholds": {"dayMm": DAY_MM, "threeDayMm": THREE_DAY_MM},
        "climate": None if clim is None else {k: clim[0].get(k) for k in ("center", "halfWidthDays", "years", "quantile", "floorMm", "calibrated", "modelRuns")},
        "times": [f"{run + dt.timedelta(hours=f):%Y-%m-%dT%H:%M}" for f in STEPS],  # end of each 24 h period
        "grid": {"nlat": 361, "nlon": 720, "lat0": 90, "lon0": -180, "d": 0.5},
        "layers": layers,
        "source": "ECMWF IFS ensemble (ENS) open data, CC BY 4.0",
    }
    with open(os.path.join(args.out, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1)

    print(f"Done: data/now for run {meta['run']} UTC")
    for k, t in enumerate(meta["times"]):
        print(f"  day {k + 1} to {t}: max ensemble-mean rain {rain[k].max():.0f} mm, "
              f"area with ≥50% chance of {DAY_MM:.0f} mm: {(p24[k] >= 0.5).sum()} half-degree cells")
    print(f"  any 72 h ≥ {THREE_DAY_MM:.0f} mm: {(p72[0] >= 0.5).sum()} cells with ≥50% chance, max chance {p72.max():.2f}")
    if clim is not None:
        ratio = plocal / np.maximum(clim[2], 1 / 930)
        print(f"  locally extreme: {(plocal >= 0.5).sum()} one-degree cells with ≥50% chance; "
              f"{(ratio >= 10).sum()} cells at least 10× the usual chance")
        for k in np.argsort(-plocal.ravel())[:8]:
            r, c = divmod(k, 360)
            print(f"    {90 - r:+5.0f}°, {c - 180:+5.0f}°: chance {plocal[r, c] * 100:3.0f}% (usually {clim[2][r, c] * 100:.1f}%), "
                  f"ERA5 1-in-100 {clim[1][r, c]:.0f} mm, model threshold {clim[3][r, c]:.0f} mm")


if __name__ == "__main__":
    sys.exit(main())
