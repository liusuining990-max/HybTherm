"""Read and write probe logs as CSV.

Format: a header row ``time_s,<label1>,<label2>,...`` followed by one row per
sample. Lines starting with ``#`` are comments (the firmware prints sensor
addresses that way). Missing readings may be written as empty fields or
``nan``; the DS18B20 error value -127 is converted to NaN on read.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

DS18B20_ERROR = -127.0


def read_log(path: str | Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (t, temps, labels) from a CSV log."""
    rows = []
    header = None
    with open(path, newline="") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = next(csv.reader([s]))
            if header is None:
                header = [p.strip() for p in parts]
                continue
            rows.append([float(p) if p.strip() not in ("", "nan", "NaN") else np.nan for p in parts])
    if header is None or header[0] != "time_s":
        raise ValueError(f"{path}: first non-comment row must start with 'time_s'")
    data = np.array(rows, dtype=float)
    data[data == DS18B20_ERROR] = np.nan
    return data[:, 0], data[:, 1:], header[1:]


def write_log(path: str | Path, t: np.ndarray, temps: np.ndarray, labels: list[str], comments: list[str] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        for c in comments or []:
            f.write(f"# {c}\n")
        w = csv.writer(f)
        w.writerow(["time_s", *labels])
        for ti, row in zip(t, temps):
            w.writerow([f"{ti:.1f}", *[f"{v:.4f}" for v in row]])


def select(temps: np.ndarray, labels: list[str], wanted: list[str]) -> np.ndarray:
    """Columns of ``temps`` for the labels in ``wanted``, in that order."""
    return np.column_stack([temps[:, labels.index(w)] for w in wanted])
