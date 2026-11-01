#!/usr/bin/env python3
"""Refresh the globe's past-five-days data: what the scheduled GitHub Actions job runs.

Runs prepare_recent.py, which adds the newest GFS analysis time and drops the oldest.
The ECMWF forecast scripts (prepare_now.py and its climate files) are kept in the
repository but are not part of the site for now.

    python update.py
"""
import subprocess, sys

if __name__ == "__main__":
    sys.exit(subprocess.run([sys.executable, "prepare_recent.py"]).returncode)
