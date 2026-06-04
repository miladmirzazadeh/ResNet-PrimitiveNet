"""On-the-fly flaw injection — the robustness mechanism.

ArchCAD is "highly standardized" (drawings with >5% layer deviation are
discarded), so it has NO gaps/overshoots/floating lines. Real drafts and PDFs
do. We perturb ONLY the geometry the model sees; the LABEL stays perfect. That
gap between messy input and clean target is the supervision signal for
"a flawed wall is still a wall".

Re-randomized every epoch (no stored twins). Operates on the record list in
place of a copy, BEFORE features/raster/edges are derived.
"""
from __future__ import annotations
import math
import random

# classes we treat as "wall-like" linework worth flaw-injecting (ids from classes.py)
WALL_LIKE = {20, 21, 22, 23, 24}     # wall, columns, beams


def inject(prims: list[dict], rng: random.Random, scale: float,
           gap_frac=0.06, over_frac=0.05, jitter_frac=0.05, float_frac=0.03):
    """Return a new perturbed copy of `prims` (labels untouched).
    `scale` ~ drawing size, used to size perturbations in drawing units."""
    eps = 0.01 * scale                       # base perturbation magnitude
    out = []
    for p in prims:
        q = dict(p)
        wall = q["sem"] in WALL_LIKE and q["t"] in ("line",)
        if wall and rng.random() < gap_frac:           # open a gap: pull an endpoint inward
            _shift_end(q, rng, -rng.uniform(0.3, 1.0) * eps)
        elif wall and rng.random() < over_frac:        # overshoot: push an endpoint outward
            _shift_end(q, rng, rng.uniform(0.3, 1.0) * eps)
        if rng.random() < jitter_frac:                 # global small jitter (any primitive)
            jx, jy = rng.uniform(-eps, eps), rng.uniform(-eps, eps)
            q["x0"] += jx; q["y0"] += jy; q["x1"] += jx; q["y1"] += jy
            q["cx"] += jx; q["cy"] += jy
        out.append(q)
        if wall and rng.random() < float_frac:         # add a detached floating fragment (same label)
            out.append(_floating_fragment(q, rng, eps))
    return out


def _shift_end(q, rng, d):
    """Move one endpoint along the line direction by d (drawing units)."""
    dx, dy = q["x1"] - q["x0"], q["y1"] - q["y0"]
    L = math.hypot(dx, dy) or 1.0
    ux, uy = dx / L, dy / L
    if rng.random() < 0.5:
        q["x1"] += ux * d; q["y1"] += uy * d
    else:
        q["x0"] -= ux * d; q["y0"] -= uy * d
    q["cx"] = (q["x0"] + q["x1"]) / 2; q["cy"] = (q["y0"] + q["y1"]) / 2


def _floating_fragment(q, rng, eps):
    """A short, slightly offset copy that does not connect — labeled same as source."""
    off = rng.uniform(0.5, 2.0) * eps
    ox, oy = rng.uniform(-off, off), rng.uniform(-off, off)
    f = dict(q)
    f["x0"] += ox; f["y0"] += oy; f["x1"] += ox; f["y1"] += oy
    f["cx"] += ox; f["cy"] += oy
    return f
