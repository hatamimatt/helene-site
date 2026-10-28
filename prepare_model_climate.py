#!/usr/bin/env python3
"""Model climate of 3-day rain from past ECMWF ensemble forecasts, to calibrate prepare_now.py.

The ensemble rains harder than ERA5 in the tropics, so comparing its forecasts with
ERA5's 99th percentile overstates the chance of locally extreme rain there. This
script finds the same percentile in the model's own past forecasts for the same
season and lead times (the idea behind ECMWF's Extreme Forecast Index), so that a
forecast is called extreme when it is extreme for the model.

For each 12 UTC run within +-15 days of a calendar date (default: today) in 2024,
2025 and earlier this year, it reads total precipitation from a rotating subset
of members at 24..120 h, forms the three 72-hour windows the outlook uses, averages
them onto the 1 degree grid of data/climate/, and computes per cell:
  - q99: the model's 99th percentile of 72-hour totals
  - threshold: the ERA5 1-in-100 total (q99_72h.npy) times the model/ERA5 ratio,
    smoothed over 3x3 degrees to damp sampling noise, floored at 10 mm
  - base: how often the wettest window of a 5-day forecast exceeds that threshold

Writes data/climate/model_threshold_72h.npy, model_base_rate.npy and model_meta.json.
Per-run downloads are cached in .cache/model_climate/ so the script can resume.
Needs numpy and eccodes, and data/climate/ from prepare_climate.py for the same date.

    python prepare_model_climate.py
"""
import argparse, datetime as dt, json, os, sys, urllib.error
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import prepare_now as now

YEARS = (2024, 2025, 2026)          # 0.25 degree ENS open data starts in February 2024
HALF = 15
PER_RUN = 6                         # members per run, rotated so every member is used
FLOOR_MM = 10.0
QUANTILE = 0.99
CACHE = os.path.join(".cache", "model_climate")


def smooth3(a):
    """3x3 mean on the 1 degree grid, longitude wrapping, edge rows repeated."""
    p = np.pad(a, ((1, 1), (0, 0)), mode="edge")
    p = np.concatenate([p[:, -1:], p, p[:, :1]], axis=1)
    return np.mean([p[i:i + a.shape[0], j:j + a.shape[1]] for i in range(3) for j in range(3)], axis=0)


def run_windows(run, members, workers):
    """(members, 3, 181, 360) float32: the three 72 h windows of each member at 1 degree, in mm."""
    path = os.path.join(CACHE, f"{run:%Y%m%d%H}.npy")
    if os.path.exists(path):
        return np.load(path)

    def member(m):
        cum = [np.zeros((721, 1440), np.float32)] + [now.record(run, f, "tp", m) * 1000.0 for f in now.STEPS]
        return np.stack([now.to_one_degree(np.maximum(cum[k + 3] - cum[k], 0)) for k in range(3)]).astype(np.float32)

    with ThreadPoolExecutor(workers) as ex:
        out = np.stack(list(ex.map(member, members)))
    now.INDEX.clear()
    np.save(path, out)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--climate", default=os.path.join("data", "climate"))
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    os.makedirs(CACHE, exist_ok=True)

    info = json.load(open(os.path.join(args.climate, "meta.json")))
    month, day = map(int, info["center"].split("-"))
    era5_q99 = np.load(os.path.join(args.climate, "q99_72h.npy"))
    today = dt.datetime.utcnow()

    runs = []
    for year in YEARS:
        c = dt.datetime(year, month, day, 12)
        runs += [r for r in (c + dt.timedelta(days=k) for k in range(-HALF, HALF + 1)) if r + dt.timedelta(days=2) < today]
    print(f"ECMWF ENS model climate around {info['center']}: {len(runs)} runs × {PER_RUN} members", flush=True)

    samples, used = [], []
    for n, run in enumerate(runs):
        members = [(n * PER_RUN + k) % 50 + 1 for k in range(PER_RUN)]
        try:
            samples.append(run_windows(run, members, args.workers))
            used.append(f"{run:%Y-%m-%dT%H}")
        except (urllib.error.HTTPError, KeyError) as e:
            print(f"  {run:%Y-%m-%d}: skipped ({e})", flush=True)
            continue
        print(f"  {run:%Y-%m-%d} ({n + 1}/{len(runs)})", flush=True)

    s = np.concatenate(samples)                                 # (runs*members, 3, 181, 360)
    windows = s.reshape(-1, *s.shape[2:])
    k = int(np.ceil((1 - QUANTILE) * len(windows)))
    model_q99 = -np.partition(-windows, k - 1, axis=0)[k - 1]
    ratio = smooth3(np.maximum(model_q99, FLOOR_MM)) / smooth3(era5_q99)
    threshold = np.maximum(era5_q99 * ratio, FLOOR_MM).astype(np.float32)
    base = (s.max(axis=1) >= threshold).mean(axis=0).astype(np.float32)

    np.save(os.path.join(args.climate, "model_threshold_72h.npy"), threshold)
    np.save(os.path.join(args.climate, "model_base_rate.npy"), base)
    meta = {
        "center": info["center"], "halfWidthDays": HALF, "runs": used, "membersPerRun": PER_RUN,
        "samplesPerCell": int(len(windows)), "quantile": QUANTILE, "floorMm": FLOOR_MM,
        "source": "ECMWF IFS ensemble (ENS) open data, CC BY 4.0; ratio to ERA5 q99 smoothed over 3x3 degrees",
    }
    with open(os.path.join(args.climate, "model_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1)

    lat = 90 - np.arange(181)
    w = np.cos(np.radians(lat))[:, None] * np.ones((1, 360))
    for name, sel in [("tropics 30S-30N", np.abs(lat) <= 30), ("extratropics", np.abs(lat) > 30)]:
        m = sel[:, None] & np.ones((1, 360), bool)
        print(f"  {name:16s} model/ERA5 q99 ratio, area-weighted mean {np.average(ratio[m], weights=w[m]):.2f}")
    print(f"Done: {len(used)} runs, {len(windows)} samples per cell; base rate median {np.median(base) * 100:.1f}%")
    for name, la, lo in [("Asheville, NC", 35.6, -82.55), ("Houston, TX", 29.76, -95.37), ("Phoenix, AZ", 33.45, -112.07), ("Mumbai", 19.08, 72.88)]:
        r, c = round(90 - la), round(lo + 180) % 360
        print(f"  {name:14s} ERA5 {era5_q99[r, c]:5.0f} mm  model-calibrated {threshold[r, c]:5.0f} mm  base {base[r, c] * 100:4.1f}%")


if __name__ == "__main__":
    sys.exit(main())
