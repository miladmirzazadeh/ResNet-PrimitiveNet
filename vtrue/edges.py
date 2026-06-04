"""Relational edge features over k-nearest-neighbour primitives — the graph that
makes the transformer graph-aware (the relations that define a wall: parallel,
collinear, connected, fixed perpendicular offset).

Computed per chunk with a cKDTree (O(N·k)), fully vectorized. Output feeds the
model as a relational message-passing stage (edges.py here only builds features;
the aggregation lives in model.RelationalLayer).

  build_edges(arr, lo, w, h, k) -> (nbr_idx [N,k] int64,
                                     edge_feat [N,k,EDGE_FEAT_DIM] float32,
                                     nbr_mask [N,k] bool)
"""
from __future__ import annotations
import numpy as np
from scipy.spatial import cKDTree

EDGE_FEAT_DIM = 13
# layout: [dist, gap, bear_sin, bear_cos, perp_offset, proj_overlap,
#          parallel, perpendicular, collinear, near_endpoint, bbox_iou,
#          concentric, same_type]


def build_edges(arr, lo, w, h, k=16):
    n = arr["n"]
    diag = float(np.hypot(w, h)) or 1.0
    A = (arr["A"] - lo) / diag                  # normalized endpoints/centers
    B = (arr["B"] - lo) / diag
    C = (arr["C"] - lo) / diag
    R = arr["R"] / diag
    T = arr["T"]
    d = B - A
    length = np.hypot(d[:, 0], d[:, 1]) + 1e-9
    ux, uy = d[:, 0] / length, d[:, 1] / length   # unit direction
    ang = np.arctan2(d[:, 1], d[:, 0])

    if n == 1:
        z = np.zeros((1, k), np.int64)
        return z, np.zeros((1, k, EDGE_FEAT_DIM), np.float32), np.zeros((1, k), bool)

    kk = min(k, n - 1)
    tree = cKDTree(C)
    _, idx = tree.query(C, k=kk + 1)              # includes self at col 0
    idx = np.asarray(idx)
    if idx.ndim == 1:
        idx = idx[:, None]
    nbr = idx[:, 1:kk + 1]                         # [N,kk] drop self
    # pad to k
    nbr_idx = np.zeros((n, k), np.int64)
    nbr_mask = np.zeros((n, k), bool)
    nbr_idx[:, :kk] = nbr; nbr_mask[:, :kk] = True

    ai = A[:, None, :]; bi = B[:, None, :]; ci = C[:, None, :]
    uxi = ux[:, None]; uyi = uy[:, None]; angi = ang[:, None]; leni = length[:, None]; ri = R[:, None]; ti = T[:, None]
    j = nbr_idx
    Aj = A[j]; Bj = B[j]; Cj = C[j]; angj = ang[j]; lenj = length[j]; Rj = R[j]; Tj = T[j]

    dvec = Cj - ci                                  # [N,k,2]
    dist = np.hypot(dvec[..., 0], dvec[..., 1])
    # bearing of neighbour relative to own direction
    bear = np.arctan2(dvec[..., 1], dvec[..., 0]) - angi
    bear_sin, bear_cos = np.sin(bear), np.cos(bear)
    # orientation
    dtheta = angj - angi
    parallel = np.abs(np.sin(dtheta))               # ~0 => parallel
    perpend = np.abs(np.cos(dtheta))                # ~0 => perpendicular
    parallel_s = 1.0 - parallel                     # 1 => parallel (nicer signal)
    perpend_s = 1.0 - perpend
    # perpendicular offset of neighbour center from own infinite line
    perp_offset = np.abs(dvec[..., 0] * uyi - dvec[..., 1] * uxi)
    collinear = parallel_s * np.exp(-perp_offset / 0.02)   # parallel AND on the line
    # projected overlap of neighbour onto own axis (origin = Ai)
    tj0 = (Aj[..., 0] - ai[..., 0]) * uxi + (Aj[..., 1] - ai[..., 1]) * uyi
    tj1 = (Bj[..., 0] - ai[..., 0]) * uxi + (Bj[..., 1] - ai[..., 1]) * uyi
    lo_t = np.minimum(tj0, tj1); hi_t = np.maximum(tj0, tj1)
    overlap = np.clip(np.minimum(hi_t, leni) - np.maximum(lo_t, 0.0), 0.0, None)
    proj_overlap = overlap / (leni + 1e-9)
    # endpoint gap = min distance among the 4 endpoint pairs
    def dpair(p, q):
        return np.hypot(p[..., 0] - q[..., 0], p[..., 1] - q[..., 1])
    gap = np.minimum.reduce([dpair(ai, Aj), dpair(ai, Bj), dpair(bi, Aj), dpair(bi, Bj)])
    near_endpoint = np.exp(-gap / 0.02)             # 1 => touching
    # bbox IoU
    def bbox_iou():
        ix0 = np.minimum(ai[..., 0], bi[..., 0]); ix1 = np.maximum(ai[..., 0], bi[..., 0])
        iy0 = np.minimum(ai[..., 1], bi[..., 1]); iy1 = np.maximum(ai[..., 1], bi[..., 1])
        jx0 = np.minimum(Aj[..., 0], Bj[..., 0]); jx1 = np.maximum(Aj[..., 0], Bj[..., 0])
        jy0 = np.minimum(Aj[..., 1], Bj[..., 1]); jy1 = np.maximum(Aj[..., 1], Bj[..., 1])
        ox = np.clip(np.minimum(ix1, jx1) - np.maximum(ix0, jx0), 0, None)
        oy = np.clip(np.minimum(iy1, jy1) - np.maximum(iy0, jy0), 0, None)
        inter = ox * oy
        area_i = (ix1 - ix0) * (iy1 - iy0); area_j = (jx1 - jx0) * (jy1 - jy0)
        return inter / (area_i + area_j - inter + 1e-9)
    iou = bbox_iou()
    # concentric (both curved, shared center)
    concentric = np.where((ri > 0) & (Rj > 0), np.exp(-dist / 0.02), 0.0)
    same_type = (Tj == ti).astype(np.float32)

    feat = np.stack([dist, gap, bear_sin, bear_cos, perp_offset, proj_overlap,
                     parallel_s, perpend_s, collinear, near_endpoint, iou,
                     concentric, same_type], axis=-1).astype(np.float32)
    feat[~nbr_mask] = 0.0
    return nbr_idx, feat, nbr_mask
