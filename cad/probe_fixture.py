"""Parametric probe-guide fixture for multi-point block measurements (CadQuery).

The fixture sits on top of the heating block and holds each temperature probe
vertical, at a fixed well position and a fixed depth, so that repeated runs put
the probes in the same place. Repeatability of probe placement is part of the
measurement uncertainty (see docs/REPORT.md).

Design choices
--------------
* Open frame, not a solid plate. A solid plate would act as a lid and lower
  the heat loss from the top face, i.e. the fixture would change the very
  temperature distribution it is meant to measure. The model's "insulated lid"
  variant shows how large that effect can be.
* A lip around the edge registers the fixture on the block's outer faces with
  a set clearance, so the probe positions follow the well grid.
* One chamfered corner marks the A1 corner (the block is symmetric, so the
  mark prevents mounting the fixture rotated by 180 degrees and swapping the
  probe labels).
* Probe bosses use a clearance hole; the probe is fixed with a drop of
  removable adhesive or a printed collar, and a depth line on the probe.

Run:  python cad/probe_fixture.py      -> cad/out/probe_fixture.{step,stl,svg}

All dimensions in millimetres.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cadquery as cq

ROWS = "ABCDEFGH"


@dataclass
class FixtureParams:
    block_x: float = 120.0
    block_y: float = 80.0
    pitch: float = 9.0  # well pitch (ANSI/SLAS 4-2004)
    probe_wells: list[str] = field(default_factory=lambda: ["A1", "A6", "D6", "H12"])
    probe_d: float = 6.0  # probe diameter (Adafruit 381 stainless tube: 6 mm)
    hole_clearance: float = 0.3  # diametral clearance for the probe
    boss_od: float = 12.0
    boss_h: float = 14.0  # guide length above the frame keeps the probe vertical
    frame_w: float = 8.0
    frame_t: float = 4.0
    strut_w: float = 6.0
    lip_h: float = 5.0  # how far the lip hangs down the block side
    lip_t: float = 2.0
    lip_clearance: float = 0.25  # per side, between lip and block
    corner_chamfer: float = 10.0  # marks the A1 corner


def well_xy(p: FixtureParams, name: str) -> tuple[float, float]:
    row = ROWS.index(name[0].upper())
    col = int(name[1:])
    x0 = (p.block_x - 11 * p.pitch) / 2
    y0 = (p.block_y - 7 * p.pitch) / 2
    return x0 + (col - 1) * p.pitch, y0 + row * p.pitch


def build(p: FixtureParams) -> cq.Workplane:
    # Work in block coordinates: origin at the A1 corner of the block, z = 0 at the block top.
    outer_x = p.block_x + 2 * (p.lip_clearance + p.lip_t)
    outer_y = p.block_y + 2 * (p.lip_clearance + p.lip_t)
    ox = -(p.lip_clearance + p.lip_t)
    oy = -(p.lip_clearance + p.lip_t)

    frame = (
        cq.Workplane("XY")
        .box(outer_x, outer_y, p.frame_t, centered=False)
        .translate((ox, oy, 0))
    )
    window = (
        cq.Workplane("XY")
        .box(p.block_x - 2 * p.frame_w, p.block_y - 2 * p.frame_w, p.frame_t + 2, centered=False)
        .translate((p.frame_w, p.frame_w, -1))
    )
    body = frame.cut(window)

    # Lip hanging down around the block
    lip_outer = cq.Workplane("XY").box(outer_x, outer_y, p.lip_h, centered=False).translate((ox, oy, -p.lip_h))
    lip_inner = (
        cq.Workplane("XY")
        .box(p.block_x + 2 * p.lip_clearance, p.block_y + 2 * p.lip_clearance, p.lip_h + 2, centered=False)
        .translate((-p.lip_clearance, -p.lip_clearance, -p.lip_h - 1))
    )
    body = body.union(lip_outer.cut(lip_inner))

    # Struts from each probe boss to the nearest long edge of the frame, and the bosses
    for w in p.probe_wells:
        x, y = well_xy(p, w)
        to_low = y  # distance to the y = 0 edge
        to_high = p.block_y - y
        if to_low <= to_high:
            y_a, y_b = 0.0, y
        else:
            y_a, y_b = y, p.block_y
        strut = (
            cq.Workplane("XY")
            .box(p.strut_w, (y_b - y_a), p.frame_t, centered=False)
            .translate((x - p.strut_w / 2, y_a, 0))
        )
        boss = cq.Workplane("XY").circle(p.boss_od / 2).extrude(p.frame_t + p.boss_h).translate((x, y, 0))
        body = body.union(strut).union(boss)

    # Probe holes
    for w in p.probe_wells:
        x, y = well_xy(p, w)
        hole = (
            cq.Workplane("XY")
            .circle((p.probe_d + p.hole_clearance) / 2)
            .extrude(p.frame_t + p.boss_h + 2)
            .translate((x, y, -1))
        )
        body = body.cut(hole)

    # A1-corner chamfer (vertical edge at the block's (0, 0) corner)
    c = p.corner_chamfer
    wedge = (
        cq.Workplane("XY")
        .polyline([(ox - 1, oy - 1), (ox + c, oy - 1), (ox - 1, oy + c)])
        .close()
        .extrude(p.frame_t + p.lip_h + 2)
        .translate((0, 0, -p.lip_h - 1))
    )
    body = body.cut(wedge)
    return body


def main() -> None:
    p = FixtureParams()
    part = build(p)
    out = Path(__file__).parent / "out"
    out.mkdir(exist_ok=True)
    cq.exporters.export(part, str(out / "probe_fixture.step"))
    cq.exporters.export(part, str(out / "probe_fixture.stl"), tolerance=0.05, angularTolerance=0.2)
    cq.exporters.export(
        part,
        str(out / "probe_fixture.svg"),
        opt={"projectionDir": (0.6, -0.9, 0.7), "width": 800, "height": 500, "showHidden": False, "strokeWidth": 0.4},
    )
    bb = part.val().BoundingBox()
    info = {
        "params_mm": asdict(p),
        "bounding_box_mm": [round(bb.xlen, 2), round(bb.ylen, 2), round(bb.zlen, 2)],
        "volume_mm3": round(part.val().Volume(), 1),
    }
    (out / "probe_fixture.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
