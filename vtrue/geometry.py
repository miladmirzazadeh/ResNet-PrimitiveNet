"""Primitive records -> normalized arrays + per-primitive (node) feature vectors.

A 'prim array' is a structured numpy view of the records, all coords normalized
to the chunk bbox so the model is scale/translation invariant at the node level
(relative geometry is added on top by edges.py).
"""
from __future__ import annotations
import numpy as np

TYPE_ID = {"line": 0, "arc": 1, "circle": 2, "other": 3}
NUM_TYPES = 4
NODE_FEAT_DIM = NUM_TYPES + 12   # one-hot(4) + [x0,y0,x1,y1,cx,cy,len,sin,cos,r,bw,bh]


def to_arrays(prims: list[dict]):
    """Records -> dict of float arrays (RAW coords). Endpoints/centers/radius/angle."""
    n = len(prims)
    A = np.zeros((n, 2), np.float64); B = np.zeros((n, 2), np.float64)
    C = np.zeros((n, 2), np.float64); R = np.zeros(n, np.float64)
    T = np.zeros(n, np.int64); SEM = np.zeros(n, np.int64)
    INS = [""] * n
    for i, p in enumerate(prims):
        A[i] = (p["x0"], p["y0"]); B[i] = (p["x1"], p["y1"])
        C[i] = (p["cx"], p["cy"]); R[i] = p["r"]
        T[i] = TYPE_ID.get(p["t"], 3); SEM[i] = p["sem"]; INS[i] = p["ins"]
    return {"A": A, "B": B, "C": C, "R": R, "T": T, "SEM": SEM, "INS": INS, "n": n}


def bbox(arr):
    """Robust bbox: percentile-based so a few stray primitives crossing the slice
    boundary (real ArchCAD coords range far outside the 0..980 canvas) don't blow
    up normalization and squash everything else into a dot."""
    pts = np.vstack([arr["A"], arr["B"], arr["C"]])
    lo = np.percentile(pts, 1.0, axis=0)
    hi = np.percentile(pts, 99.0, axis=0)
    w = max(float(hi[0] - lo[0]), 1e-6); h = max(float(hi[1] - lo[1]), 1e-6)
    return lo, w, h


def node_features(arr, lo, w, h) -> np.ndarray:
    """[N, NODE_FEAT_DIM] normalized features."""
    n = arr["n"]
    diag = float(np.hypot(w, h)) or 1.0
    A = (arr["A"] - lo); B = (arr["B"] - lo); C = (arr["C"] - lo)
    dx = arr["B"][:, 0] - arr["A"][:, 0]; dy = arr["B"][:, 1] - arr["A"][:, 1]
    length = np.hypot(dx, dy)
    ang = np.arctan2(dy, dx)
    feat = np.zeros((n, NODE_FEAT_DIM), np.float32)
    feat[np.arange(n), arr["T"]] = 1.0                       # one-hot type
    o = NUM_TYPES
    pos = np.stack([A[:, 0] / w, A[:, 1] / h, B[:, 0] / w, B[:, 1] / h,
                    C[:, 0] / w, C[:, 1] / h], axis=1)
    feat[:, o:o + 6] = np.clip(pos, -0.5, 1.5)               # bound out-of-canvas outliers
    feat[:, o + 6] = np.clip(length / diag, 0, 3)
    feat[:, o + 7] = np.sin(ang); feat[:, o + 8] = np.cos(ang)
    feat[:, o + 9] = np.clip(arr["R"] / diag, 0, 3)
    feat[:, o + 10] = np.clip(np.abs(dx) / w, 0, 2); feat[:, o + 11] = np.clip(np.abs(dy) / h, 0, 2)
    return feat
