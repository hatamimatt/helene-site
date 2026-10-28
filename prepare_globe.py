#!/usr/bin/env python3
"""Build the global fields for the landing-page globe (index.html).

Reads whole-globe ERA5 maps from the public ARCO-ERA5 bucket (same source and
reader as prepare_event.py), averages them to 1 degree, and writes compact
binary files to data/globe/:

    meta.json    times, grid, and for each layer its file, vmin and vmax
    <layer>.bin  uint8, 0..254 scaled between vmin and vmax, 255 = missing;
                 frames in time order, rows from 90N to 90S, columns from 180W

    python prepare_globe.py
"""
import argparse, datetime as dt, json, os, sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from prepare_event import BASE, SOURCES, RemoteNC

START, END, STEP = "2024-09-23T00", "2024-09-29T18", 6
LAYERS = ["pw", "ivtu", "ivtv"]  # instantaneous; rain is built from hourly "tp"


def to_1deg(a):
    """(721, 1440) at 0.25 deg -> (181, 360) box means, columns starting at 180W."""
    p = np.pad(a, ((2, 2), (0, 0)), mode="edge")
    p = np.concatenate([p[:, -2:], p, p[:, :2]], axis=1)
    c = p[:724, :1440].reshape(181, 4, 360, 4).mean(axis=(1, 3))
    return np.roll(c, -180, axis=1)


def quantize(frames, symmetric=False):
    vmin, vmax = float(np.nanmin(frames)), float(np.nanmax(frames))
    if symmetric:
        m = max(abs(vmin), abs(vmax)); vmin, vmax = -m, m
    q = np.round((frames - vmin) / (vmax - vmin) * 254)
    q = np.where(np.isfinite(frames), np.clip(q, 0, 254), 255).astype(np.uint8)
    return q, round(vmin, 3), round(vmax, 3)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join("data", "globe"))
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    t, t1 = dt.datetime.fromisoformat(START), dt.datetime.fromisoformat(END)
    frames = []
    while t <= t1:
        frames.append(t); t += dt.timedelta(hours=STEP)
    hours = sorted({f - dt.timedelta(hours=h) for f in frames for h in range(STEP)})
    jobs = [(k, f) for k in LAYERS for f in frames] + [("tp", h) for h in hours]

    files = {}

    def url(k, day):
        folder, var, level = SOURCES[k][:3]
        return f"{BASE}/date-variable-{folder}/{day:%Y/%m/%d}/{var}/{level}.nc"

    needed = sorted({url(k, w.date()) for k, w in jobs})
    print(f"Reading headers of {len(needed)} files…", flush=True)
    with ThreadPoolExecutor(args.workers) as ex:
        for nc in ex.map(RemoteNC, needed):
            files[nc.url] = nc

    def fetch(job):
        k, when = job
        nc = files[url(k, when.date())]
        return job, to_1deg(SOURCES[k][3](nc.rows(when.hour, 0, nc.ny - 1))).astype(np.float32)

    print(f"Reading {len(jobs)} global maps…", flush=True)
    got = {}
    with ThreadPoolExecutor(args.workers) as ex:
        for n, (job, a) in enumerate(ex.map(fetch, jobs), 1):
            got[job] = a
            if n % 25 == 0 or n == len(jobs):
                print(f"  {n}/{len(jobs)}", flush=True)

    F = {k: np.stack([got[(k, f)] for f in frames]) for k in LAYERS}
    F["rain6"] = np.stack([sum(got[("tp", f - dt.timedelta(hours=h))] for h in range(STEP)) for f in frames])

    layers = {}
    for k, a in F.items():
        q, vmin, vmax = quantize(a, symmetric=k in ("ivtu", "ivtv"))
        q.tofile(os.path.join(args.out, f"{k}.bin"))
        layers[k] = {"file": f"{k}.bin", "vmin": vmin, "vmax": vmax}

    meta = {
        "source": "ERA5 reanalysis (Hersbach et al. 2020), Copernicus Climate Change Service, via ARCO-ERA5 on Google Cloud",
        "grid": {"nlat": 181, "nlon": 360, "lat0": 90, "lon0": -180, "d": 1},
        "times": [f"{f:%Y-%m-%dT%H:%M}" for f in frames],
        "layers": layers,
    }
    with open(os.path.join(args.out, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    print(f"Done: {len(layers)} layers × {len(frames)} frames in {args.out}/")


if __name__ == "__main__":
    sys.exit(main())
