#!/usr/bin/env python3
"""The last five days of the atmosphere behind extreme rain, from NOAA's GFS analyses.

For every 6-hour analysis time in the past five days (20 times, ending at the newest
complete GFS cycle), reads from the GFS 1 degree files on AWS (no account needed):
  - integrated vapor transport (IVT), east and north components: specific humidity
    times wind, integrated from the surface (or 1000 hPa) up to 300 hPa
  - precipitable water (total column water vapor)
  - 850 hPa wind
  - rain in the 6 hours up to that time (the first 6 hours of the previous cycle's run,
    so a model estimate, not a gauge or radar measurement)

Writes data/recent/ in the same format as data/globe/ (prepare_globe.py):
    meta.json    analysis times, grid, layers (file, vmin, vmax)
    <layer>.bin  uint8, 0..254 scaled between vmin and vmax, 255 = missing;
                 frames in time order, rows from 90N to 90S, columns from 180W
Decoded times are cached in .cache/recent/, so each 6-hourly run downloads one new time.
GFS data is from NOAA and in the public domain. Needs numpy and eccodes.

    python prepare_recent.py
"""
import argparse, datetime as dt, json, os, sys, urllib.error
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from prepare_forecasts import decode, get

BASE = "https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.{c:%Y%m%d}/{c:%H}/atmos/gfs.t{c:%H}z.pgrb2.1p00.f{f:03d}"
LEVELS = [1000, 975, 950, 925, 900, 850, 800, 750, 700, 650, 600, 550, 500, 450, 400, 350, 300]  # hPa
TIMES = 20                    # 6-hourly, five days
G = 9.80665
CACHE = os.path.join(".cache", "recent")
# fixed colour ranges, the same scale every day so frames and days compare
RANGES = {"ivtu": (-2000, 2000), "ivtv": (-2000, 2000), "pw": (0, 80), "rain6": (0, 100), "u850": (-60, 60), "v850": (-60, 60)}


def index(url):
    """GFS .idx -> {"VAR:level:time": (offset, length)}; the last record runs to the end of the file."""
    lines = get(url + ".idx").decode().splitlines()
    out = {}
    for k, line in enumerate(lines):
        parts = line.split(":")
        start = int(parts[1])
        end = int(lines[k + 1].split(":")[1]) - 1 if k + 1 < len(lines) else ""
        out.setdefault(":".join(parts[3:6]), (start, end))
    return out


def field(url, idx, key):
    start, end = idx[key]
    data = get_range(url, start, end)
    vals, d, lon0 = decode(data)
    assert abs(d - 1) < 1e-6 and abs(lon0) < 1e-6, "expected the GFS 1 degree grid starting at 0E"
    return np.roll(vals, 180, axis=1)  # columns from 180W like the other globe data


def get_range(url, start, end):
    import urllib.request
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    for k in range(5):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except Exception:
            if k == 4:
                raise


def latest_cycle(now):
    """Newest cycle whose 6 h file is published (GFS 1 degree files appear about 4 hours after the cycle)."""
    c = now.replace(minute=0, second=0, microsecond=0, hour=now.hour // 6 * 6)
    for _ in range(8):
        try:
            get(BASE.format(c=c, f=6) + ".idx", tries=1)
            return c
        except urllib.error.HTTPError:
            c -= dt.timedelta(hours=6)
    raise SystemExit("No GFS cycle found in the last two days")


def analysis(t):
    """All layers for one analysis time, cached as float32."""
    path = os.path.join(CACHE, f"{t:%Y%m%d%H}.npz")
    if os.path.exists(path):
        return dict(np.load(path))
    url0 = BASE.format(c=t, f=0)
    idx0 = index(url0)
    keys = [f"{v}:{p} mb:anl" for p in LEVELS for v in ("SPFH", "UGRD", "VGRD")]
    keys += ["PRES:surface:anl", "PWAT:entire atmosphere (considered as a single layer):anl"]
    with ThreadPoolExecutor(8) as ex:
        got = dict(zip(keys, ex.map(lambda k: field(url0, idx0, k), keys)))
    ps = got["PRES:surface:anl"] / 100.0  # Pa -> hPa
    ivtu = np.zeros_like(ps); ivtv = np.zeros_like(ps)
    for lo, hi in zip(LEVELS[:-1], LEVELS[1:]):  # trapezoid layers; skip layers below the ground
        q = lambda v, p: got[f"{v}:{p} mb:anl"]
        weight = (ps >= lo).astype(np.float32) * (lo - hi) * 100.0 / G / 2
        ivtu += weight * (q("SPFH", lo) * q("UGRD", lo) + q("SPFH", hi) * q("UGRD", hi))
        ivtv += weight * (q("SPFH", lo) * q("VGRD", lo) + q("SPFH", hi) * q("VGRD", hi))
    prev = t - dt.timedelta(hours=6)
    url6 = BASE.format(c=prev, f=6)
    rain = field(url6, index(url6), "APCP:surface:0-6 hour acc fcst")
    out = {"ivtu": ivtu, "ivtv": ivtv, "pw": got["PWAT:entire atmosphere (considered as a single layer):anl"],
           "rain6": np.maximum(rain, 0), "u850": got["UGRD:850 mb:anl"], "v850": got["VGRD:850 mb:anl"]}
    out = {k: v.astype(np.float32) for k, v in out.items()}
    np.savez_compressed(path, **out)
    return out


def quantize(frames, lo, hi):
    q = np.round((frames - lo) / (hi - lo) * 254)
    return np.where(np.isfinite(frames), np.clip(q, 0, 254), 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join("data", "recent"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True); os.makedirs(CACHE, exist_ok=True)

    last = latest_cycle(dt.datetime.utcnow())
    times = [last - dt.timedelta(hours=6 * k) for k in range(TIMES - 1, -1, -1)]
    print(f"GFS analyses {times[0]:%Y-%m-%d %H} to {times[-1]:%Y-%m-%d %H} UTC", flush=True)
    frames = []
    for n, t in enumerate(times, 1):
        frames.append(analysis(t))
        print(f"  {t:%Y-%m-%d %H} UTC ({n}/{len(times)})", flush=True)

    layers = {}
    for name, (lo, hi) in RANGES.items():
        quantize(np.stack([f[name] for f in frames]), lo, hi).tofile(os.path.join(args.out, f"{name}.bin"))
        layers[name] = {"file": f"{name}.bin", "vmin": lo, "vmax": hi}
    meta = {
        "source": "NOAA GFS analyses, 1 degree (NOAA Open Data Dissemination on AWS)",
        "created": dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M"),
        "grid": {"nlat": 181, "nlon": 360, "lat0": 90, "lon0": -180, "d": 1},
        "times": [f"{t:%Y-%m-%dT%H:%M}" for t in times],
        "layers": layers,
    }
    with open(os.path.join(args.out, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    # drop cached times older than the window
    keep = {f"{t:%Y%m%d%H}.npz" for t in times}
    for f in os.listdir(CACHE):
        if f not in keep:
            os.remove(os.path.join(CACHE, f))
    ivt = np.hypot(frames[-1]["ivtu"], frames[-1]["ivtv"])
    print(f"Done: data/recent, latest {meta['times'][-1]} UTC; max IVT {ivt.max():.0f} kg/m/s, "
          f"max 6 h rain {max(f['rain6'].max() for f in frames):.0f} mm, max PW {frames[-1]['pw'].max():.0f} mm")


if __name__ == "__main__":
    sys.exit(main())
