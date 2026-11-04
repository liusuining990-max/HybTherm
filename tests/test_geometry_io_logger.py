import numpy as np
import pytest

from heatblock.geometry import BlockGeometry
from heatblock.io import read_log, select, write_log
from heatblock.logger import record


def test_well_grid():
    g = BlockGeometry()
    names = g.well_names()
    assert len(names) == 96 and names[0] == "A1" and names[-1] == "H12"
    xy = g.well_xy()
    # 120 x 80 block, 9 mm pitch, grid centred
    assert xy[g.well_index("A1")] == pytest.approx([10.5e-3, 8.5e-3])
    assert xy[g.well_index("H12")] == pytest.approx([109.5e-3, 71.5e-3])
    assert np.diff(xy[:12, 0]) == pytest.approx(np.full(11, 9e-3))


def test_log_roundtrip_and_error_value(tmp_path):
    t = np.array([0.0, 1.0, 2.0])
    x = np.array([[22.0, 22.1], [22.5, -127.0], [23.0, 23.1]])
    p = tmp_path / "log.csv"
    write_log(p, t, x, ["A1", "H12"], comments=["synthetic"])
    t2, x2, labels = read_log(p)
    assert labels == ["A1", "H12"]
    assert np.isnan(x2[1, 1])  # -127 is the DS18B20 "disconnected" code
    assert select(x2, labels, ["H12"])[0, 0] == pytest.approx(22.1)


def test_serial_logger_with_loopback(tmp_path):
    serial = pytest.importorskip("serial")
    ser = serial.serial_for_url("loop://", timeout=0.2)
    lines = [
        "# n_probes=2",
        "# probe1=28FF000000000001",
        "# probe2=28FF000000000002",
        "time_s,T1,T2",
        "1.0,22.0625,22.1250",
        "2.0,22.1250,22.1875",
    ]
    ser.write(("\n".join(lines) + "\n").encode())
    out = tmp_path / "run.csv"
    n = record(ser, out, ["A1", "H12"], max_lines=2, duration=5)
    assert n == 2
    t, x, labels = read_log(out)
    assert labels == ["A1", "H12"]
    assert t.tolist() == [1.0, 2.0]
    assert "probe1=28FF000000000001" in out.read_text()


def test_logger_rejects_label_count_mismatch(tmp_path):
    serial = pytest.importorskip("serial")
    ser = serial.serial_for_url("loop://", timeout=0.2)
    ser.write(b"time_s,T1,T2,T3\n")
    with pytest.raises(ValueError):
        record(ser, tmp_path / "x.csv", ["A1", "H12"], duration=2)
