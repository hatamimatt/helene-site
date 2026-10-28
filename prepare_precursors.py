#!/usr/bin/env python3
"""Precursor conditions in the weeks before Hurricane Helene, from ERA5.

For every day from 8 to 26 September (12 UTC) this reads the same calendar
day in every year from 1991 to 2024. 2024 is compared with a 1991-2020
baseline (mean and standard deviation for that date and grid cell), and with
every year 1991-2024 for ranks.

Writes data/precursors/:
    meta.json      grid, dates, map layers, regional index time series
    <layer>.bin    uint8 maps (0..254 scaled, 255 = missing), one per day,
                   rows from north to south, columns from west to east

Same public ARCO-ERA5 source and reader as prepare_event.py.

    python prepare_precursors.py
"""
import argparse, datetime as dt, json, os, sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from prepare_event import BASE, SOURCES as EVENT_SOURCES, RemoteNC

YEAR, BASELINE = 2024, (1991, 2020)
FIRST_YEAR = 1991
DAYS = [dt.date(YEAR, 9, d) for d in range(8, 27)]
HOUR = 12
LAT0, LAT1, LON0, LON1 = 5.0, 45.0, -110.0, -60.0   # south, north, west, east

SOURCES = dict(EVENT_SOURCES)
SOURCES.update({
    "sst":   ("single_level", "sea_surface_temperature", "surface", lambda a: a - 273.15),
    "swvl1": ("single_level", "volumetric_soil_water_layer_1", "surface", lambda a: a),
    "u200":  ("pressure_level", "u_component_of_wind", "200", lambda a: a),
    "v200":  ("pressure_level", "v_component_of_wind", "200", lambda a: a),
})
RAW = ["sst", "pw", "swvl1", "z500", "u200", "v200", "u850", "v850", "ivtu", "ivtv"]

# map layers derived from the raw variables
MAPS = {
    "sst":   dict(label="Sea surface temperature", unit="°C"),
    "pw":    dict(label="Precipitable water", unit="mm"),
    "swvl1": dict(label="Topsoil moisture (0–7 cm)", unit="m³ m⁻³"),
    "z500":  dict(label="500 hPa height", unit="m"),
    "shear": dict(label="Vertical wind shear, 200–850 hPa", unit="m s⁻¹"),
    "ivt":   dict(label="Integrated vapor transport", unit="kg m⁻¹ s⁻¹"),
}

# regional indices: box means of a map layer (south, north, west, east)
INDICES = {
    "gulf_sst":  dict(layer="sst", box=(18, 30, -98, -81), label="Gulf of Mexico sea surface temperature",
                      why="Ocean heat that fuels rapid intensification."),
    "carib_sst": dict(layer="sst", box=(15, 22, -88, -78), label="Northwestern Caribbean sea surface temperature",
                      why="Where Helene formed and first strengthened."),
    "gulf_shear": dict(layer="shear", box=(20, 28, -92, -82), label="Wind shear over the eastern Gulf",
                       why="Low shear lets a hurricane stay organized and strengthen."),
    "se_pw":     dict(layer="pw", box=(30, 37, -88, -78), label="Precipitable water over the Southeast",
                      why="How much water vapor the air holds before the storm arrives."),
    "app_soil":  dict(layer="swvl1", box=(34.5, 36.5, -84, -81), label="Topsoil moisture, southern Appalachians",
                      why="Wet soils turn more of the rain into runoff and floods."),
    "tv_z500":   dict(layer="z500", box=(33, 40, -92, -84), label="500 hPa height over the Tennessee and Ohio valleys",
                      why="Falling heights mark the trough that later steered Helene north."),
    "se_ivt":    dict(layer="ivt", box=(30, 37, -88, -78), label="Vapor transport into the Southeast",
                      why="The moisture being carried into the region."),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join("data", "precursors"))
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    r0, r1 = round((90 - LAT1) / 0.25), round((90 - LAT0) / 0.25)
    c0, c1 = round((LON0 % 360) / 0.25), round((LON1 % 360) / 0.25)
    ny, nx = r1 - r0 + 1, c1 - c0 + 1
    years = list(range(FIRST_YEAR, YEAR + 1))
    jobs = [(k, y, d) for k in RAW for y in years for d in DAYS]

    def fetch(job):
        k, y, d = job
        folder, var, level, conv = SOURCES[k]
        day = d.replace(year=y)
        nc = RemoteNC(f"{BASE}/date-variable-{folder}/{day:%Y/%m/%d}/{var}/{level}.nc")
        return job, conv(nc.rows(HOUR, r0, r1)[:, c0:c1 + 1]).astype(np.float32)

    print(f"Reading {len(jobs)} slices ({len(RAW)} variables × {len(years)} years × {len(DAYS)} days)…", flush=True)
    raw = {k: np.full((len(years), len(DAYS), ny, nx), np.nan, np.float32) for k in RAW}
    with ThreadPoolExecutor(args.workers) as ex:
        for n, ((k, y, d), a) in enumerate(ex.map(fetch, jobs), 1):
            raw[k][years.index(y), DAYS.index(d)] = a
            if n % 250 == 0 or n == len(jobs):
                print(f"  {n}/{len(jobs)}", flush=True)

    land = np.isnan(raw["sst"][-1, 0])
    F = {
        "sst": raw["sst"],
        "pw": raw["pw"],
        "swvl1": np.where(land, raw["swvl1"], np.nan),
        "z500": raw["z500"],
        "shear": np.hypot(raw["u200"] - raw["u850"], raw["v200"] - raw["v850"]),
        "ivt": np.hypot(raw["ivtu"], raw["ivtv"]),
    }
    base = [years.index(y) for y in range(BASELINE[0], BASELINE[1] + 1)]
    lat = LAT1 - np.arange(ny) * 0.25
    lon = LON0 + np.arange(nx) * 0.25

    def write(name, frames, symmetric=False, lo=None, hi=None):
        vmin = float(np.nanmin(frames)) if lo is None else lo
        vmax = float(np.nanmax(frames)) if hi is None else hi
        if symmetric:
            m = max(abs(vmin), abs(vmax)); vmin, vmax = -m, m
        q = np.round((frames - vmin) / (vmax - vmin) * 254)
        q = np.where(np.isfinite(frames), np.clip(q, 0, 254), 255).astype(np.uint8)
        q.tofile(os.path.join(args.out, f"{name}.bin"))
        return {"file": f"{name}.bin", "vmin": round(vmin, 4), "vmax": round(vmax, 4)}

    layers = {}
    with np.errstate(invalid="ignore", divide="ignore"):
        for k, info in MAPS.items():
            a = F[k]
            mean = np.nanmean(a[base], axis=0); std = np.nanstd(a[base], axis=0)
            anom = a[-1] - mean
            sigma = np.clip(anom / np.where(std > 0, std, np.nan), -6, 6)
            # 99th percentile limits keep a few extreme cells from flattening the colour scale
            lim = float(np.nanpercentile(np.abs(anom), 99))
            layers[k] = dict(info,
                             value=write(f"{k}_value", a[-1]),
                             anom=write(f"{k}_anom", np.clip(anom, -lim, lim), symmetric=True),
                             sigma=write(f"{k}_sigma", sigma, lo=-6, hi=6))

        indices = {}
        for key, ix in INDICES.items():
            s, n, w, e = ix["box"]
            rows = (lat >= s) & (lat <= n); cols = (lon >= w) & (lon <= e)
            box = F[ix["layer"]][:, :, rows][:, :, :, cols]
            wts = np.cos(np.deg2rad(lat[rows]))[:, None] * np.ones(cols.sum())
            valid = np.isfinite(box)
            series = np.nansum(np.where(valid, box, 0) * wts, axis=(2, 3)) / np.sum(valid * wts, axis=(2, 3))  # (years, days)
            b = series[base]
            this = series[-1]
            # rank 1 = highest of all years 1991-2024 for that date
            rank = [int((series[:, d] > this[d]).sum() + 1) for d in range(len(DAYS))]
            indices[key] = dict(
                label=ix["label"], why=ix["why"], box=ix["box"], layer=ix["layer"], unit=MAPS[ix["layer"]]["unit"],
                value=np.round(this, 3).tolist(), mean=np.round(b.mean(0), 3).tolist(), std=np.round(b.std(0), 3).tolist(),
                min=np.round(series[:-1].min(0), 3).tolist(), max=np.round(series[:-1].max(0), 3).tolist(),
                rank=rank, years=len(years))

    meta = {
        "source": "ERA5 reanalysis (Hersbach et al. 2020), Copernicus Climate Change Service, via ARCO-ERA5 on Google Cloud",
        "note": f"Daily maps at {HOUR:02d} UTC. Anomalies relative to {BASELINE[0]}–{BASELINE[1]} for the same date; ranks among {FIRST_YEAR}–{YEAR}.",
        "grid": {"lat0": LAT1, "lon0": LON0, "d": 0.25, "ny": ny, "nx": nx},
        "dates": [f"{d:%Y-%m-%d}" for d in DAYS],
        "baseline": list(BASELINE), "firstYear": FIRST_YEAR,
        "layers": layers, "indices": indices,
    }
    with open(os.path.join(args.out, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1)

    print(f"Done: {len(layers)} map layers, {len(indices)} indices, {len(DAYS)} days in {args.out}/")
    for key, ix in indices.items():
        v, m, sd, rk = ix["value"], ix["mean"], ix["std"], ix["rank"]
        print(f"  {key:11s} " + "  ".join(f"{DAYS[d]:%d}: {v[d]:.2f} ({(v[d]-m[d])/sd[d]:+.1f}σ, #{rk[d]})" for d in (0, 6, 12, 18)))


if __name__ == "__main__":
    sys.exit(main())
