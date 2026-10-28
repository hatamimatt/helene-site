#!/usr/bin/env python3
"""Local climate of 3-day rain from ERA5, used to define "extreme" in prepare_now.py.

For every 1-degree cell, takes every 72-hour precipitation total ending
within +-15 days of a calendar date (default: today) in each year 1991-2020,
and computes:
  - q99: the 72-hour total exceeded 1% of the time (floored at 10 mm so deserts
    do not count a light shower as extreme)
  - base: how often the wettest 72-hour window in a 5-day period exceeds q99,
    using the same windows as the forecast (ending 72, 96 and 120 h after 12 UTC)

Reads ERA5 hourly precipitation, conservatively regridded to 1 degree, from the
public ARCO-ERA5 Zarr store on Google Cloud (no account needed), about 6 GB.

Writes data/climate/: meta.json, q99_72h.npy and base_rate.npy as float32 arrays
(rows 90N -> 90S, columns 180W -> 179E), shared with prepare_now.py.

Needs numpy and numcodecs.

    python prepare_climate.py               # season centred on today
    python prepare_climate.py --center 09-27
"""
import argparse, datetime as dt, json, math, os, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import numcodecs

from prepare_now import polite_get

ZARR = "https://storage.googleapis.com/gcp-public-data-arco-era5/ar/1959-2022-1h-360x181_equiangular_with_poles_conservative.zarr"
T0 = dt.datetime(1959, 1, 1)           # Zarr time axis: hours since this
CHUNK = 8                               # hours per chunk
YEARS = range(1991, 2021)
HALF = 15                               # days either side of the centre date
FLOOR_MM = 10.0
QUANTILE = 0.99
BLOSC = numcodecs.Blosc()
POOL = [None]


def array(name, shape, dtype):
    return np.frombuffer(BLOSC.decode(polite_get(f"{ZARR}/{name}")), dtype).reshape(shape)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--center", help="calendar date MM-DD (default: today)")
    ap.add_argument("--out", default=os.path.join("data", "climate"))
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    month, day = map(int, (args.center or dt.date.today().strftime("%m-%d")).split("-"))

    meta = json.load(urllib.request.urlopen(ZARR + "/.zmetadata"))["metadata"]
    nlon, nlat = meta["total_precipitation/.zarray"]["shape"][1:]
    lat = array("latitude/0", (nlat,), "<f8"); lon = array("longitude/0", (nlon,), "<f8")
    t = array("time/0", (meta["time/.zarray"]["chunks"][0],), "<i8")
    assert t[0] == 0 and np.all(np.diff(t[:1000]) == 1), "expected an hourly axis starting 1959-01-01"
    # reorder to rows 90N -> 90S, columns 180W -> 179E like the other globe data
    rows = np.argsort(-lat); cols = np.argsort(((lon + 180) % 360) - 180)

    def hours_for(year):
        c = dt.datetime(year, month, day, 12)
        start = c - dt.timedelta(days=HALF)                   # first pseudo-run at 12 UTC
        end = c + dt.timedelta(days=HALF + 5)                 # last run + 120 h
        return start, int((start - T0).total_seconds() // 3600) + 1, int((end - T0).total_seconds() // 3600)

    def read_year(year):
        fetch_pool = POOL[0]
        start, h0, h1 = hours_for(year)
        chunks = range(h0 // CHUNK, h1 // CHUNK + 1)
        data = np.concatenate(list(fetch_pool.map(lambda k: array(f"total_precipitation/{k}.0.0", (CHUNK, nlon, nlat), "<f4"), chunks)))
        rain = data[h0 - chunks[0] * CHUNK:h1 - chunks[0] * CHUNK + 1]  # hours (start, end]
        rain = np.maximum(rain, 0)[:, cols][:, :, rows].transpose(0, 2, 1) * 1000.0  # (hour, lat, lon) in mm
        cs = np.concatenate([np.zeros((1,) + rain.shape[1:], np.float32), np.cumsum(rain, axis=0, dtype=np.float32)])
        tot72 = lambda e: cs[e] - cs[e - 72]                  # 72 h ending e hours after start
        samples = np.stack([tot72(e) for e in range(72, len(rain) + 1, 6)])
        runs = np.stack([np.max([tot72(r + w) for w in (72, 96, 120)], axis=0) for r in range(0, 24 * (2 * HALF) + 1, 24)])
        return year, samples, runs

    print(f"ERA5 1991–2020, 72 h rain within ±{HALF} days of {month:02d}-{day:02d}…", flush=True)
    per_year = len(range(72, (2 * HALF + 5) * 24 + 1, 6))
    k = math.ceil((1 - QUANTILE) * per_year * len(YEARS))     # q99 = smallest of the k largest samples
    top = None; runs_all = []
    POOL[0] = ThreadPoolExecutor(args.workers)                 # chunk downloads, shared by the year workers
    with ThreadPoolExecutor(4) as years_pool:                  # four years at a time keeps memory near 2 GB
        for n, (year, samples, runs) in enumerate(years_pool.map(read_year, YEARS), 1):
            merged = samples if top is None else np.concatenate([top, samples])
            top = -np.partition(-merged, k - 1, axis=0)[:k]
            runs_all.append(runs)
            print(f"  {year} ({n}/{len(YEARS)})", flush=True)
    q99 = np.maximum(top.min(axis=0), FLOOR_MM).astype(np.float32)
    base = (np.concatenate(runs_all) >= q99).mean(axis=0).astype(np.float32)

    np.save(os.path.join(args.out, "q99_72h.npy"), q99)
    np.save(os.path.join(args.out, "base_rate.npy"), base)
    info = {
        "center": f"{month:02d}-{day:02d}", "halfWidthDays": HALF, "years": [YEARS[0], YEARS[-1]],
        "quantile": QUANTILE, "floorMm": FLOOR_MM, "samplesPerCell": per_year * len(YEARS),
        "grid": {"nlat": nlat, "nlon": nlon, "lat0": 90, "lon0": -180, "d": 1},
        "source": "ERA5 hourly precipitation, 1 degree conservative regrid, ARCO-ERA5 (Google Cloud)",
    }
    with open(os.path.join(args.out, "meta.json"), "w") as fh:
        json.dump(info, fh, indent=1)
    print(f"Done: q99 median {np.median(q99):.0f} mm, 90th pct {np.percentile(q99, 90):.0f} mm, max {q99.max():.0f} mm; "
          f"base rate median {np.median(base) * 100:.1f}%")
    for name, la, lo in [("Asheville, NC", 35.6, -82.55), ("Houston, TX", 29.76, -95.37), ("Phoenix, AZ", 33.45, -112.07), ("Mumbai", 19.08, 72.88)]:
        r, c = round(90 - la), round(lo + 180) % 360
        print(f"  {name:14s} q99 {q99[r, c]:6.0f} mm   base rate {base[r, c] * 100:4.1f}%")


if __name__ == "__main__":
    sys.exit(main())
