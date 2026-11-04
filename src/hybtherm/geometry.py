"""Block geometry and 96-well layout.

All lengths are in metres unless a name ends in ``_mm``.

The default block is a 120 x 80 x 20 mm aluminium block carrying a standard
96-well grid (8 rows x 12 columns, 9 mm pitch), centred on the block.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ROWS = "ABCDEFGH"
N_ROWS = 8
N_COLS = 12
WELL_PITCH = 9.0e-3  # m, ANSI/SLAS 4-2004 well-to-well spacing for 96-well plates


@dataclass(frozen=True)
class BlockGeometry:
    """Rectangular block with a centred 96-well grid."""

    length_x: float = 0.120  # m
    width_y: float = 0.080  # m
    height_z: float = 0.020  # m
    well_depth: float = 0.015  # m, measured down from the top face
    probe_depth: float = 0.010  # m below top face where well temperature is read

    def well_names(self) -> list[str]:
        """Return names A1..H12 in row-major order (A1, A2, ... H12)."""
        return [f"{r}{c}" for r in ROWS for c in range(1, N_COLS + 1)]

    def well_xy(self) -> np.ndarray:
        """Return an (96, 2) array of well centre coordinates (x, y) in metres.

        Row A is at the smallest y, column 1 at the smallest x.
        """
        span_x = (N_COLS - 1) * WELL_PITCH
        span_y = (N_ROWS - 1) * WELL_PITCH
        x0 = (self.length_x - span_x) / 2.0
        y0 = (self.width_y - span_y) / 2.0
        xs = x0 + WELL_PITCH * np.arange(N_COLS)
        ys = y0 + WELL_PITCH * np.arange(N_ROWS)
        xx, yy = np.meshgrid(xs, ys)  # shape (8, 12)
        return np.column_stack([xx.ravel(), yy.ravel()])

    def well_xyz(self) -> np.ndarray:
        """Return (96, 3) coordinates of the temperature read-out point of each well."""
        xy = self.well_xy()
        z = np.full((xy.shape[0], 1), self.height_z - self.probe_depth)
        return np.hstack([xy, z])

    def well_index(self, name: str) -> int:
        """Index of a well name such as 'A1' or 'H12' in the row-major order."""
        name = name.strip().upper()
        row = ROWS.index(name[0])
        col = int(name[1:])
        if not 1 <= col <= N_COLS:
            raise ValueError(f"bad well name {name!r}")
        return row * N_COLS + (col - 1)

    @property
    def volume(self) -> float:
        return self.length_x * self.width_y * self.height_z

    @property
    def area_top(self) -> float:
        return self.length_x * self.width_y

    @property
    def area_sides(self) -> float:
        return 2.0 * (self.length_x + self.width_y) * self.height_z
