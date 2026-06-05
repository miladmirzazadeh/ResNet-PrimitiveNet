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


def dense_filter(prims, gap_ratio=5.0, core_frac=0.6):
    """Drop far-flung outlier primitives (title blocks, borders, north arrows) WITHOUT
    clipping the real plan. The plan is one spatially tight cluster; a title block sits
    far away across a big distance GAP. We sort primitives by distance from the median
    center and cut at the first large multiplicative jump in the outer tail — so a plan
    with content spread over a wide area is kept whole, and only truly separated outliers
    are removed. (The old MAD band wrongly clipped sparse walls beyond a dense fixture
    cluster.)"""
    if len(prims) < 8:
        return prims
    C = to_arrays(prims)["C"]
    mx, my = np.median(C[:, 0]), np.median(C[:, 1])
    d = np.hypot(C[:, 0] - mx, C[:, 1] - my)
    ds = np.sort(d); n = len(ds)
    start = int(n * core_frac)
    ratios = ds[start + 1:] / np.maximum(ds[start:-1], 1.0)
    if len(ratios) and ratios.max() > gap_ratio:
        cut = ds[start + int(np.argmax(ratios))]      # last distance before the jump
        return [p for p, kk in zip(prims, d <= cut) if kk]
    return prims


def deskew(prims):
    """Rotate the plan so its dominant wall direction is axis-aligned. The model was
    trained only on axis-aligned ArchCAD chunks, so a tilted plan reads as 'others'.
    Detects the dominant edge angle (length-weighted, folded to [0,90deg)) and rotates
    by -that. Returns (rotated_prims, degrees)."""
    arr = to_arrays(prims)
    dx = arr["B"][:, 0] - arr["A"][:, 0]; dy = arr["B"][:, 1] - arr["A"][:, 1]
    L = np.hypot(dx, dy); m = L > 1e-9
    if m.sum() < 5:
        return prims, 0.0
    ang = np.arctan2(dy[m], dx[m]) % (np.pi / 2)
    hist, edges = np.histogram(ang, bins=180, weights=L[m], range=(0, np.pi / 2))
    dom = (edges[hist.argmax()] + edges[hist.argmax() + 1]) / 2
    if dom > np.pi / 4:                       # rotate the short way
        dom -= np.pi / 2
    if abs(dom) < np.radians(0.8):
        return prims, 0.0
    c, s = np.cos(-dom), np.sin(-dom)
    out = []
    for p in prims:
        q = dict(p)
        for xk, yk in (("x0", "y0"), ("x1", "y1"), ("cx", "cy")):
            x, y = p[xk], p[yk]; q[xk] = x * c - y * s; q[yk] = x * s + y * c
        out.append(q)
    return out, np.degrees(dom)


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


def load_plan(path):
    return dxf_to_prims(path) if str(path).lower().endswith(".dxf") else load_chunk(path)


def label_plan(model, prims, device="cpu", clean=True, rotate=True, split=True, verbose=True):
    """Full inference: clean outliers -> split lines at junctions -> de-skew -> predict.
    Returns (pred, prims) where prims is the cleaned+split, ORIGINAL-orientation working
    set (pred[i] labels prims[i])."""
    if clean:
        n0 = len(prims); prims = dense_filter(prims)
        if verbose and len(prims) < n0:
            print(f"cleaned {n0 - len(prims)} outlier primitives (title block / border / stray marks)")
    if split:
        from vtrue.split import split_lines
        n0 = len(prims); prims = split_lines(prims)
        if verbose and len(prims) != n0:
            print(f"split lines at junctions: {n0} -> {len(prims)} primitives")
    prims_model = prims
    if rotate:
        prims_model, deg = deskew(prims)
        if verbose and abs(deg) > 0.8:
            print(f"de-skewed plan by {deg:.1f} deg for the model")
    pred, _arr, _lo, _w, _h = predict(model, prims_model, device)
    return pred, prims


def to_record(prims, pred, scale=None):
    """Structured prediction: primitives (id/type/coords/label) + object groups."""
    import math as _m
    if scale is None:
        arr = to_arrays(prims); _, w, h = bbox(arr); scale = _m.hypot(w, h)
    prim_recs = [{"id": i, "type": p["t"],
                  "coords": [round(p["x0"], 2), round(p["y0"], 2), round(p["x1"], 2), round(p["y1"], 2)],
                  "label": ID2NAME.get(int(pred[i]), str(int(pred[i])))} for i, p in enumerate(prims)]
    groups = group_objects(prims, pred, scale=float(scale))
    obj_recs = []
    for j, g in enumerate(groups):
        xs = [c for i in g for c in (prims[i]["x0"], prims[i]["x1"])]
        ys = [c for i in g for c in (prims[i]["y0"], prims[i]["y1"])]
        obj_recs.append({"id": j, "label": ID2NAME.get(int(pred[g[0]]), str(int(pred[g[0]]))),
                         "n": len(g), "members": g,
                         "bbox": [round(min(xs), 1), round(min(ys), 1), round(max(xs), 1), round(max(ys), 1)]})
    return {"primitives": prim_recs, "objects": obj_recs}


def render_preview(prims, pred, out, title="Vtrue prediction (color = class)", label_objs=None):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    rng = np.random.default_rng(0)
    colors = rng.random((NUM_CLASSES, 3)) * 0.7 + 0.15
    fig, ax = plt.subplots(figsize=(13, 10))
    for i, p in enumerate(prims):
        ax.plot([p["x0"], p["x1"]], [p["y0"], p["y1"]], color=colors[int(pred[i]) % NUM_CLASSES], lw=1.1)
    if label_objs:                       # overlay object-id numbers (for GPT review)
        for oid, (cx, cy) in label_objs.items():
            ax.text(cx, cy, str(oid), fontsize=7, color="red", ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="red", alpha=0.7))
    ax.set_aspect("equal"); ax.axis("off"); ax.set_title(title)
    plt.tight_layout(); plt.savefig(out, dpi=120); plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--input", required=True, help=".dxf or ArchCAD .json")
    ap.add_argument("--out", default="pred.png")
    ap.add_argument("--json-out", help="write structured prediction JSON (primitives + objects)")
    ap.add_argument("--no-clean", action="store_true", help="skip outlier-primitive removal")
    ap.add_argument("--no-rotate", action="store_true", help="skip auto de-skew")
    ap.add_argument("--no-split", action="store_true", help="skip splitting lines at junctions")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(a.weights, device)
    prims = load_plan(a.input)
    if not prims:
        print("no primitives in", a.input); return
    pred, prims = label_plan(model, prims, device, clean=not a.no_clean,
                             rotate=not a.no_rotate, split=not a.no_split)
    print("classes:", {ID2NAME.get(int(k), k): v for k, v in sorted(Counter(pred).items())})
    rec = to_record(prims, pred)
    countable = [o for o in rec["objects"] if o["label"] not in ("wall", "axis_grid", "glass", "others")]
    print(f"objects: {len(rec['objects'])} groups ({len(countable)} countable candidates for GPT)")
    if a.json_out:
        import json
        json.dump(rec, open(a.json_out, "w")); print("json ->", a.json_out)
    render_preview(prims, pred, a.out); print("preview ->", a.out)


if __name__ == "__main__":
    main()
