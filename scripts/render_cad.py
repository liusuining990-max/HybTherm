"""Render a CadQuery part to PNG with matplotlib (no OpenGL needed)."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def render(shape, out_png, title, elev=28, azim=-60, color="#9ab7d3"):
    verts, tris = shape.tessellate(0.2, 0.3)
    v = np.array([(p.x, p.y, p.z) for p in verts])
    faces = [v[list(t)] for t in tris]
    fig = plt.figure(figsize=(7, 4.6), dpi=150)
    ax = fig.add_subplot(111, projection="3d")
    # simple Lambert shading
    n = np.cross(v[[t[1] for t in tris]] - v[[t[0] for t in tris]], v[[t[2] for t in tris]] - v[[t[0] for t in tris]])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    light = np.array([0.4, -0.5, 0.8]); light /= np.linalg.norm(light)
    shade = 0.35 + 0.65 * np.abs(n @ light)  # two-sided: triangle winding is not consistent
    base = np.array(matplotlib.colors.to_rgb(color))
    cols = np.clip(base[None, :] * shade[:, None], 0, 1)
    pc = Poly3DCollection(faces, facecolors=cols, edgecolors=cols, linewidths=0.15)
    ax.add_collection3d(pc)
    mins, maxs = v.min(0), v.max(0)
    ctr = (mins + maxs) / 2; r = (maxs - mins).max() / 2
    ax.set_xlim(ctr[0] - r, ctr[0] + r); ax.set_ylim(ctr[1] - r, ctr[1] + r); ax.set_zlim(ctr[2] - r * 0.5, ctr[2] + r * 0.5)
    ax.set_box_aspect((1, 1, 0.5))
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cad"))
    import probe_fixture as pf
    part = pf.build(pf.FixtureParams())
    render(part.val(), Path(__file__).resolve().parents[1] / "docs/figures/probe_fixture.png",
           "Probe-guide fixture (open frame, A1 corner chamfered)")
    print("ok")
