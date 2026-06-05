"""Split line primitives at junctions (noding).

A single drawn line may run through several regions — e.g. wall | window-opening |
wall — yet be ONE primitive, so it can only get ONE label. We split each line at:
  - T-junctions: points where another line's endpoint lies on it,
  - X-crossings: points where another line crosses it,
so every segment between joints becomes its own primitive and can be labeled (by the
model, and by GPT) independently. Only 'line' primitives are split; arcs/circles/
regions pass through unchanged.

This also matches ArchCAD's training distribution, where walls are drawn as separate
segments at each junction.
"""
from __future__ import annotations
import numpy as np
from scipy.spatial import cKDTree


def _intersection_t(a, b, c, d):
    """Param t in (0,1) on segment a->b where segment c->d crosses it, else None."""
    r = b - a; s = d - c
    rxs = r[0] * s[1] - r[1] * s[0]
    if abs(rxs) < 1e-12:
        return None
    qp = c - a
    t = (qp[0] * s[1] - qp[1] * s[0]) / rxs
    u = (qp[0] * r[1] - qp[1] * r[0]) / rxs
    return t if (0.0 < t < 1.0 and 0.0 < u < 1.0) else None


def split_lines(prims, tol_frac=0.0025, crossings=True):
    """Return a new primitive list with line segments split at junctions."""
    lines = [p for p in prims if p["t"] == "line"]
    rest = [p for p in prims if p["t"] != "line"]
    if len(lines) < 2:
        return prims
    A = np.array([[p["x0"], p["y0"]] for p in lines], float)
    B = np.array([[p["x1"], p["y1"]] for p in lines], float)
    allpts = np.vstack([A, B])
    diag = float(np.hypot(np.ptp(allpts[:, 0]), np.ptp(allpts[:, 1]))) or 1.0
    tol = tol_frac * diag
    tree = cKDTree(allpts)
    lo = np.minimum(A, B) - tol; hi = np.maximum(A, B) + tol      # per-segment bbox

    out = []
    for k, p in enumerate(lines):
        a, b = A[k], B[k]; d = b - a; L = float(np.hypot(d[0], d[1]))
        if L < 2 * tol:
            out.append(p); continue
        u = d / L; nrm = np.array([-u[1], u[0]])
        ts = set()
        # T-junctions: any endpoint lying on this segment (interior, near the line)
        for ci in tree.query_ball_point((a + b) / 2.0, r=L / 2.0 + tol):
            P = allpts[ci]
            t = float(np.dot(P - a, u))
            if tol < t < L - tol and abs(float(np.dot(P - a, nrm))) < tol:
                ts.add(round(t, 2))
        # X-crossings: other segments that actually cross this one (bbox-pruned)
        if crossings:
            cand = np.where((lo[:, 0] <= hi[k, 0]) & (hi[:, 0] >= lo[k, 0]) &
                            (lo[:, 1] <= hi[k, 1]) & (hi[:, 1] >= lo[k, 1]))[0]
            for j in cand:
                if j == k:
                    continue
                t = _intersection_t(a, b, A[j], B[j])
                if t is not None and tol < t * L < L - tol:
                    ts.add(round(t * L, 2))
        if not ts:
            out.append(p); continue
        cuts = [0.0] + sorted(ts) + [L]
        for t0, t1 in zip(cuts, cuts[1:]):
            if t1 - t0 < tol:
                continue
            p0 = a + u * t0; p1 = a + u * t1
            q = dict(p)
            q["x0"], q["y0"] = float(p0[0]), float(p0[1])
            q["x1"], q["y1"] = float(p1[0]), float(p1[1])
            q["cx"], q["cy"] = float((p0[0] + p1[0]) / 2), float((p0[1] + p1[1]) / 2)
            out.append(q)
    return out + rest
