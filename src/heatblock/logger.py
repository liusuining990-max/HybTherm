"""Record the firmware's serial output to a CSV log.

Usage (real hardware):
    python -m heatblock.logger --port /dev/ttyACM0 --out data/raw/run1.csv \
        --labels A1,D6,A6,H12,ambient --duration 3600

The firmware prints comment lines starting with '#' (sensor addresses), then a
header ``time_s,T1,...``, then one line per second. This logger:
  * copies '#' lines into the CSV as comments,
  * replaces the firmware header with ``time_s`` + your probe labels,
  * writes a JSON sidecar with the port, start time and labels.

``--port loop://`` opens pyserial's loop-back device, used by the tests.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path


def record(ser, out_path: Path, labels: list[str], duration: float | None = None, max_lines: int | None = None) -> int:
    """Read lines from an open serial-like object and write the CSV. Returns data rows written."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_rows = 0
    t0 = time.monotonic()
    wrote_header = False
    with open(out_path, "w", newline="") as f:
        f.write("# heatblock-uniformity serial log\n")
        while True:
            if duration is not None and time.monotonic() - t0 > duration:
                break
            if max_lines is not None and n_rows >= max_lines:
                break
            raw = ser.readline()
            if not raw:
                continue
            line = raw.decode("ascii", errors="replace").strip()
            if not line:
                continue
            if line.startswith("#"):
                f.write(line + "\n")
                continue
            if line.startswith("time_s"):
                n_fw = len(line.split(",")) - 1
                if n_fw != len(labels):
                    raise ValueError(f"firmware reports {n_fw} probes but {len(labels)} labels were given")
                f.write("time_s," + ",".join(labels) + "\n")
                wrote_header = True
                continue
            if not wrote_header:
                continue  # wait for the header so columns are known
            f.write(line + "\n")
            f.flush()
            n_rows += 1
    return n_rows


def main(argv=None) -> None:
    import serial  # pyserial

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", required=True)
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--labels", required=True, help="comma-separated, in firmware probe order")
    p.add_argument("--duration", type=float, default=None, help="seconds; default runs until Ctrl-C")
    args = p.parse_args(argv)
    labels = [s.strip() for s in args.labels.split(",") if s.strip()]

    meta = {
        "port": args.port,
        "baud": args.baud,
        "labels": labels,
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    args.out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    with serial.serial_for_url(args.port, args.baud, timeout=2) as ser:
        try:
            n = record(ser, args.out, labels, duration=args.duration)
        except KeyboardInterrupt:
            n = None
    print(f"wrote {args.out}" + (f" ({n} rows)" if n is not None else ""))


if __name__ == "__main__":
    main()
