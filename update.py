#!/usr/bin/env python3
"""Refresh the globe's outlook: what the scheduled GitHub Actions job runs.

1. If data/climate/ is missing or centred more than 10 days from today, rebuilds it
   for today (prepare_climate.py, then prepare_model_climate.py). prepare_now.py
   accepts climate files up to 20 days off, so this keeps a safe margin.
2. Runs prepare_now.py for the latest ECMWF ensemble run.

Writes climate_changed=true|false to $GITHUB_OUTPUT when run inside Actions, so the
workflow knows whether to commit data/climate/.

    python update.py
    python update.py --force-climate
"""
import argparse, datetime as dt, json, os, subprocess, sys

CLIMATE = os.path.join("data", "climate")
MAX_GAP_DAYS = 10


def climate_gap(today):
    """Days between today and the climate's centre date, or None if there is no usable climate."""
    try:
        center = json.load(open(os.path.join(CLIMATE, "meta.json")))["center"]
        model = json.load(open(os.path.join(CLIMATE, "model_meta.json")))["center"]
    except FileNotFoundError:
        return None
    if model != center:
        return None
    m, d = map(int, center.split("-"))
    gap = abs((today - dt.date(today.year, m, d)).days)
    return min(gap, 365 - gap)


def run(script, *args):
    print(f"\n=== {script} {' '.join(args)}", flush=True)
    subprocess.run([sys.executable, script, *args], check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force-climate", action="store_true", help="rebuild data/climate/ even if it is recent")
    args = ap.parse_args()

    today = dt.datetime.utcnow().date()
    gap = climate_gap(today)
    refresh = args.force_climate or gap is None or gap > MAX_GAP_DAYS
    print(f"Climate: {'missing or incomplete' if gap is None else f'centred {gap} days from today'}; "
          f"{'rebuilding for ' + today.strftime('%m-%d') if refresh else 'keeping it'}", flush=True)
    if refresh:
        run("prepare_climate.py", "--center", today.strftime("%m-%d"))
        run("prepare_model_climate.py")
    run("prepare_now.py")

    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as fh:
            fh.write(f"climate_changed={'true' if refresh else 'false'}\n")


if __name__ == "__main__":
    sys.exit(main())
