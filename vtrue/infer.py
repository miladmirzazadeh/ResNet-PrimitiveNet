"""Run the trained Vtrue model on a new plan (DXF or ArchCAD JSON).

  python -m vtrue.infer --weights runs/v1/best.pt --input plan.dxf --out pred.png

Produces: printed class histogram, a colored preview PNG, object groups
(connected components of same-class primitives) with bounding boxes — the
candidates you hand to GPT for final labeling.
"""
from __future__ import annotations
import argparse, math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from vtrue.model import VtrueModel
from vtrue.geometry import to_arrays, bbox, node_features, NODE_FEAT_DIM
from vtrue.edges import build_edges, EDGE_FEAT_DIM
from vtrue.raster import render
from vtrue.archcad import load_chunk
from vtrue.classes import ID2NAME, NUM_CLASSES


def dxf_to_prims(path):
    """Minimal DXF -> record list (same schema as archcad.load_chunk). sem=0 (unknown)."""
    import ezdxf
    doc = ezdxf.readfile(path); msp = doc.modelspace()
    prims = []
    def rec(t, x0, y0, x1, y1, cx, cy, r):
        prims.append({"t": t, "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                      "cx": cx, "cy": cy, "r": r, "sem": 0, "ins": ""})
    def walk(e, depth=0):
        dt = e.dxftype()
        if dt == "INSERT" and depth < 4:
            try:
                for ve in e.virtual_entities():
                    walk(ve, depth + 1)
            except Exception:
                pass
            return
        try:
            if dt == "LINE":
                a, b = e.dxf.start, e.dxf.end
                rec("line", a.x, a.y, b.x, b.y, (a.x + b.x) / 2, (a.y + b.y) / 2, 0.0)
            elif dt == "ARC":
                a, b = e.start_point, e.end_point; c = e.dxf.center
                rec("arc", a.x, a.y, b.x, b.y, c.x, c.y, e.dxf.radius)
            elif dt in ("CIRCLE", "ELLIPSE"):
                c = e.dxf.center; r = getattr(e.dxf, "radius", 0.0) or 1.0
                rec("circle", c.x - r, c.y, c.x + r, c.y, c.x, c.y, r)
            elif dt in ("LWPOLYLINE", "POLYLINE"):
                pts = [(p[0], p[1]) for p in e.get_points()] if dt == "LWPOLYLINE" \
                      else [(v.dxf.location.x, v.dxf.location.y) for v in e.vertices]
                if getattr(e, "closed", False) and len(pts) > 2:
                    pts = pts + [pts[0]]
                for a, b in zip(pts, pts[1:]):
                    if a != b:
                        rec("line", a[0], a[1], b[0], b[1], (a[0] + b[0]) / 2, (a[1] + b[1]) / 2, 0.0)
        except Exception:
            pass
    for e in msp:
        walk(e)
    return prims


def dense_filter(prims, k=6.0):
    """Drop far-flung outlier primitives (title blocks, borders, north arrows, stray
    marks) that would wreck per-plan normalization. Keeps the median +/- k*MAD spatial
    cluster. On clean 14m chunks this keeps everything; on real drawings it rescues the
    framing — the difference between '0 walls' and the real plan."""
    arr = to_arrays(prims)
    cx, cy = arr["C"][:, 0], arr["C"][:, 1]
    mx, my = np.median(cx), np.median(cy)
    madx = np.median(np.abs(cx - mx)) * 1.4826 + 1e-6
    mady = np.median(np.abs(cy - my)) * 1.4826 + 1e-6
    keep = (np.abs(cx - mx) < k * madx) & (np.abs(cy - my) < k * mady)
    return [p for p, kk in zip(prims, keep) if kk]


def load_model(weights, device="cpu"):
    ck = torch.load(weights, map_location=device)
    a = ck.get("args", {})
    m = VtrueModel(num_classes=int(a.get("num_classes", NUM_CLASSES)),
                   node_dim=int(a.get("node_dim", NODE_FEAT_DIM)),
                   edge_dim=int(a.get("edge_dim", EDGE_FEAT_DIM)),
                   dim=int(a.get("dim", 256)), depth=int(a.get("depth", 4)),
                   rel_layers=int(a.get("rel_layers", 2)), vis_dim=int(a.get("vis_dim", 128)),
                   use_vision=bool(a.get("vision", True)), use_edges=bool(a.get("edges", True)),
                   raster_size=int(a.get("raster_size", 256)),
                   backbone=a.get("backbone", "cnn3"), pretrained=False).to(device)
    m.load_state_dict(ck["model"]); m.eval()
    m._k = int(a.get("k", 16)); m._rs = int(a.get("raster_size", 256))
    return m


@torch.no_grad()
def predict(model, prims, device="cpu"):
    arr = to_arrays(prims); lo, w, h = bbox(arr)
    feats = torch.from_numpy(node_features(arr, lo, w, h))[None].to(device)
    raster = render(arr, lo, w, h, model._rs)[None].to(device) if model.use_vision else None
    ni = ef = nm = None
    if model.use_edges:
        a_ni, a_ef, a_nm = build_edges(arr, lo, w, h, model._k)
        ni = torch.from_numpy(a_ni)[None].to(device)
        ef = torch.from_numpy(a_ef)[None].to(device)
        nm = torch.from_numpy(a_nm)[None].to(device)
    logits = model(feats, raster, ni, ef, nm, None)
    return logits.argmax(-1)[0].cpu().numpy(), arr, lo, w, h


def group_objects(prims, pred, tol_frac=0.01, scale=1.0):
    """Connected components of same-class primitives joined by near endpoints."""
    n = len(prims); parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(a, b):
        parent[find(a)] = find(b)
    tol = tol_frac * scale
    cells = defaultdict(list)
    for i, p in enumerate(prims):
        for (x, y) in ((p["x0"], p["y0"]), (p["x1"], p["y1"])):
            cells[(round(x / tol), round(y / tol))].append(i)
    for ids in cells.values():
        for a in ids:
            for b in ids:
                if a < b and pred[a] == pred[b]:
                    union(a, b)
    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    return [g for g in groups.values()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--input", required=True, help=".dxf or ArchCAD .json")
    ap.add_argument("--out", default="pred.png")
    ap.add_argument("--no-clean", action="store_true", help="skip outlier-primitive removal")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(a.weights, device)
    prims = dxf_to_prims(a.input) if a.input.lower().endswith(".dxf") else load_chunk(a.input)
    if not prims:
        print("no primitives in", a.input); return
    if not a.no_clean:
        n0 = len(prims); prims = dense_filter(prims)
        if len(prims) < n0:
            print(f"cleaned {n0 - len(prims)} outlier primitives (title block / border / stray marks)")
    pred, arr, lo, w, h = predict(model, prims, device)
    print("classes:", {ID2NAME.get(int(k), k): v for k, v in sorted(Counter(pred).items())})
    groups = group_objects(prims, pred, scale=float(math.hypot(w, h)))
    countable = [g for g in groups if ID2NAME.get(int(pred[g[0]]), "") not in ("wall", "axis_grid", "glass", "others")]
    print(f"objects: {len(groups)} groups ({len(countable)} countable candidates for GPT)")

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    rng = np.random.default_rng(0)
    colors = rng.random((NUM_CLASSES, 3)) * 0.7 + 0.15
    fig, ax = plt.subplots(figsize=(13, 10))
    for i, p in enumerate(prims):
        c = colors[int(pred[i]) % NUM_CLASSES]
        ax.plot([p["x0"], p["x1"]], [p["y0"], p["y1"]], color=c, lw=1.1)
    ax.set_aspect("equal"); ax.axis("off"); ax.set_title("Vtrue prediction (color = class)")
    plt.tight_layout(); plt.savefig(a.out, dpi=120); print("preview ->", a.out)


if __name__ == "__main__":
    main()
