"""Parse an ArchCAD-400K JSON chunk into a flat list of primitive records.

ArchCAD JSON modality (per dataset card) is a list of objects like:
  {"type":"LINE","start":[x,y],"end":[x,y],"rgb":[..],"semantic":"wall","instance":"wall_3"}
  {"type":"CIRCLE","center":[x,y],"radius":r,"semantic":"...","instance":"..."}

We normalize every entity to a common record and EXPLODE polylines into segments,
so each record is one drawable stroke:

  rec = dict(t, x0, y0, x1, y1, cx, cy, r, sem, ins)
    t   : 'line' | 'arc' | 'circle' | 'other'
    x0..y1 : endpoints (raw coords)         r: radius (0 for lines)
    sem : contiguous class id (vtrue.classes)   ins: instance string ('' if none)

>>> ASSUMPTION CHECK: the exact field names ('type','start','end','center',
'radius','semantic','instance') and the entity 'type' values are confirmed by
scripts/kaggle_inspect.py. Anything unexpected falls back to 'other' / Others,
so a schema surprise degrades gracefully instead of crashing.
"""
from __future__ import annotations
import json
import math
from pathlib import Path

from vtrue.classes import semantic_to_id


def _xy(v):
    if isinstance(v, (list, tuple)) and len(v) >= 2:
        return float(v[0]), float(v[1])
    return None


def _rec(t, x0, y0, x1, y1, cx, cy, r, sem, ins):
    return {"t": t, "x0": x0, "y0": y0, "x1": x1, "y1": y1,
            "cx": cx, "cy": cy, "r": r, "sem": sem, "ins": ins or ""}


def _arc_endpoints(cx, cy, r, a0_deg, a1_deg):
    a0, a1 = math.radians(a0_deg), math.radians(a1_deg)
    return (cx + r * math.cos(a0), cy + r * math.sin(a0),
            cx + r * math.cos(a1), cy + r * math.sin(a1))


def parse_entity(e):
    """One ArchCAD entity -> list of primitive records (polylines explode)."""
    t = str(e.get("type", "")).upper()
    sem = semantic_to_id(e.get("semantic", e.get("semantic_id", 100)))
    ins = e.get("instance", e.get("instance_id", "")) or ""
    out = []
    try:
        if t in ("LINE", "SEGMENT"):
            a, b = _xy(e.get("start")), _xy(e.get("end"))
            if a and b:
                out.append(_rec("line", a[0], a[1], b[0], b[1],
                                (a[0] + b[0]) / 2, (a[1] + b[1]) / 2, 0.0, sem, ins))
        elif t == "CIRCLE":
            c, r = _xy(e.get("center")), float(e.get("radius", 0) or 0)
            if c and r > 0:
                out.append(_rec("circle", c[0] - r, c[1], c[0] + r, c[1], c[0], c[1], r, sem, ins))
        elif t == "ARC":
            c = _xy(e.get("center")); r = float(e.get("radius", 0) or 0)
            a, b = _xy(e.get("start")), _xy(e.get("end"))       # ArchCAD provides chord endpoints
            if a and b:
                cc = c or ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
                out.append(_rec("arc", a[0], a[1], b[0], b[1], cc[0], cc[1], r, sem, ins))
            elif c and r > 0 and e.get("start_angle") is not None and e.get("end_angle") is not None:
                x0, y0, x1, y1 = _arc_endpoints(c[0], c[1], r, float(e["start_angle"]), float(e["end_angle"]))
                out.append(_rec("arc", x0, y0, x1, y1, c[0], c[1], r, sem, ins))
            elif c and r > 0:
                out.append(_rec("circle", c[0] - r, c[1], c[0] + r, c[1], c[0], c[1], r, sem, ins))
        elif t == "ELLIPSE":
            a, b = _xy(e.get("start_point")), _xy(e.get("end_point"))   # ellipse uses *_point keys
            c = _xy(e.get("center")); r = float(e.get("radius_a", e.get("radius", 0)) or 0)
            if a and b:
                cc = c or ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
                out.append(_rec("arc", a[0], a[1], b[0], b[1], cc[0], cc[1], r, sem, ins))
            elif c:
                rr = r or 1.0
                out.append(_rec("circle", c[0] - rr, c[1], c[0] + rr, c[1], c[0], c[1], rr, sem, ins))
        elif t in ("POLYLINE", "LWPOLYLINE", "PATH", "POLYGON"):
            pts = e.get("points") or e.get("vertices") or e.get("path") or []
            pts = [_xy(p) for p in pts]
            pts = [p for p in pts if p]
            if str(e.get("closed", "")).lower() in ("true", "1") and len(pts) > 2:
                pts = pts + [pts[0]]
            for a, b in zip(pts, pts[1:]):
                if a == b:
                    continue
                out.append(_rec("line", a[0], a[1], b[0], b[1],
                                (a[0] + b[0]) / 2, (a[1] + b[1]) / 2, 0.0, sem, ins))
        else:                                                   # unknown geometry -> best effort
            a, b = _xy(e.get("start")), _xy(e.get("end"))
            if a and b:
                out.append(_rec("other", a[0], a[1], b[0], b[1],
                                (a[0] + b[0]) / 2, (a[1] + b[1]) / 2, 0.0, sem, ins))
    except Exception:
        pass
    return out


def load_chunk(path) -> list[dict]:
    """Load one ArchCAD JSON file -> list of primitive records."""
    data = json.loads(Path(path).read_text(errors="ignore"))
    if isinstance(data, dict):                                  # tolerate {"primitives":[...]} wrapper
        data = data.get("primitives") or data.get("entities") or data.get("elements") or []
    prims = []
    for e in data:
        if isinstance(e, dict):
            prims.extend(parse_entity(e))
    return prims
