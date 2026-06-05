"""GPT supervision at the LINE level — the wall_finder pattern.

GPT gets, for the same plan:
  - CLEAN image:  every line colored by its current class (legend in the prompt).
  - MARKED image: the same lines, each tagged with its id (L0, L1, ...).
  - LINES TABLE (JSON): {id, current label, coords [x0,y0,x1,y1]} per line.

GPT reads the drawing like an architect, decides what each line really is, and
returns ONLY the lines whose label should change ({id, from, to, reason}). We apply
those per-line, re-render, and iterate. The model does the geometry; GPT renames.

  OPENAI_API_KEY=sk-... python -m vtrue.gpt_label \
      --weights runs/v1/best.pt --input plan.dxf --out-dir gpt_out --iters 2 --model gpt-5.5
"""
from __future__ import annotations
import argparse, base64, json, os
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import torch

from vtrue.infer import load_model, load_plan, label_plan
from vtrue.classes import ID2NAME, NUM_CLASSES

CLASS_LIST = [ID2NAME[i] for i in sorted(ID2NAME)]
NAME2ID = {v: k for k, v in ID2NAME.items()}
_BASE = list(cm.tab20.colors) + list(cm.tab20b.colors)
CLASS_COLOR = {i: _BASE[i % len(_BASE)] for i in range(NUM_CLASSES)}


def _hex(c):
    return "#%02x%02x%02x" % tuple(int(x * 255) for x in c[:3])


def legend():
    return {ID2NAME[i]: _hex(CLASS_COLOR[i]) for i in sorted(ID2NAME)}


def lines_table(prims, pred, only_loose=False):
    return [{"id": f"L{i}", "model_guess": ID2NAME.get(int(pred[i]), str(int(pred[i]))),
             "layer": p.get("layer", ""),
             "coords": [round(p["x0"], 1), round(p["y0"], 1), round(p["x1"], 1), round(p["y1"], 1)]}
            for i, p in enumerate(prims) if not (only_loose and p.get("group") is not None)]


def render_lines(prims, pred, out, figsize=(18, 14), dpi=170):
    """Class-colored 'highlighted' plan with L# tags on the LOOSE lines only (block lines
    are UFO objects, handled separately). Each tag is placed BESIDE its line (perpendicular
    offset) with a faint background, so labels don't sit on top of the linework."""
    fig, ax = plt.subplots(figsize=figsize)
    for i, p in enumerate(prims):
        ax.plot([p["x0"], p["x1"]], [p["y0"], p["y1"]],
                color=CLASS_COLOR[int(pred[i]) % NUM_CLASSES], lw=1.1)
    ax.set_aspect("equal"); ax.autoscale()
    span = max(ax.get_xlim()[1] - ax.get_xlim()[0], ax.get_ylim()[1] - ax.get_ylim()[0]) or 1.0
    off = span * 0.006
    for i, p in enumerate(prims):
        if p.get("group") is not None:
            continue
        mx, my = (p["x0"] + p["x1"]) / 2, (p["y0"] + p["y1"]) / 2
        dx, dy = p["x1"] - p["x0"], p["y1"] - p["y0"]; L = (dx * dx + dy * dy) ** 0.5 or 1.0
        nx, ny = -dy / L, dx / L                       # unit normal -> label sits beside the line
        ax.text(mx + nx * off, my + ny * off, f"L{i}", fontsize=4, color="black",
                ha="center", va="center",
                bbox=dict(boxstyle="square,pad=0.04", fc="white", ec="none", alpha=0.55))
    ax.axis("off"); plt.tight_layout(); plt.savefig(out, dpi=dpi); plt.close(fig)
    return str(out)


def render(prims, pred, out, marked=False, figsize=(18, 14), dpi=160):
    """Colored-by-class plan. marked=True overlays each line's id (L#) at its midpoint."""
    fig, ax = plt.subplots(figsize=figsize)
    for i, p in enumerate(prims):
        ax.plot([p["x0"], p["x1"]], [p["y0"], p["y1"]],
                color=CLASS_COLOR[int(pred[i]) % NUM_CLASSES], lw=1.2)
        if marked:
            ax.text((p["x0"] + p["x1"]) / 2, (p["y0"] + p["y1"]) / 2, f"L{i}",
                    fontsize=4, color="black", ha="center", va="center")
    ax.set_aspect("equal"); ax.axis("off")
    plt.tight_layout(); plt.savefig(out, dpi=dpi); plt.close(fig)
    return str(out)


def _visible(rgb, floor=170):
    """Brighten dark DXF colors (preserving hue) so they're visible on a black bg — many
    CAD layers use dark ACI colors that a viewer auto-brightens but raw RGB does not."""
    m = max(rgb) or 1
    if m < floor:
        s = floor / m
        rgb = tuple(min(255, int(v * s)) for v in rgb)
    return tuple(min(1.0, v / 255) for v in rgb)


def render_original(prims, out, texts=None, figsize=(18, 14), dpi=160):
    """Complete native CAD view: every line in its ORIGINAL DXF color (dark colors
    brightened) PLUS the drawing's text labels, on a dark background like a CAD viewer.
    Colors, layers, and text all carry intent GPT uses (red=separator; label 'toilet')."""
    fig, ax = plt.subplots(figsize=figsize, facecolor="black")
    ax.set_facecolor("black")
    xs, ys = [], []
    for p in prims:
        ax.plot([p["x0"], p["x1"]], [p["y0"], p["y1"]],
                color=_visible(p.get("rgb", (255, 255, 255))), lw=1.0)
        xs += [p["x0"], p["x1"]]; ys += [p["y0"], p["y1"]]
    if texts and xs:
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        mx, my = (x1 - x0) * 0.15 + 1, (y1 - y0) * 0.15 + 1
        for t in texts:                                  # only labels within/near the plan
            if x0 - mx <= t["x"] <= x1 + mx and y0 - my <= t["y"] <= y1 + my:
                ax.text(t["x"], t["y"], t["s"], color=_visible(t.get("rgb", (200, 200, 200))),
                        fontsize=7, ha="left", va="bottom", clip_on=True)
    ax.set_aspect("equal"); ax.axis("off")
    plt.tight_layout(); plt.savefig(out, dpi=dpi, facecolor="black"); plt.close(fig)
    return str(out)


def build_components(prims, pred, max_area_frac=0.22):
    """Group primitives by their source CAD block (the drafter's own grouping). Each
    SMALL discrete block (window/door/fixture/furniture) becomes one labelable object
    UFO#. Loose lines AND large/structural blocks (e.g. a wall drawn as a block, or a
    block on a wall/grid/column layer) are NOT components — they keep per-line labels,
    so walls stay segment-level (a wall block can contain an opening)."""
    from collections import defaultdict, Counter
    g2idx = defaultdict(list)
    for i, p in enumerate(prims):
        if p.get("group") is not None:
            g2idx[p["group"]].append(i)
    if not g2idx:
        return []
    ax = [c for p in prims for c in (p["x0"], p["x1"])]; ay = [c for p in prims for c in (p["y0"], p["y1"])]
    plan_area = max((max(ax) - min(ax)) * (max(ay) - min(ay)), 1.0)
    struct = ("wall", "grid", "axis", "column", "beam")
    comps = []
    for g, idx in sorted(g2idx.items()):
        xs = [c for i in idx for c in (prims[i]["x0"], prims[i]["x1"])]
        ys = [c for i in idx for c in (prims[i]["y0"], prims[i]["y1"])]
        layer = Counter(prims[i]["layer"] for i in idx).most_common(1)[0][0]
        area = (max(xs) - min(xs)) * (max(ys) - min(ys))
        if area > max_area_frac * plan_area or any(s in layer.lower() for s in struct):
            continue                                  # structural block -> stay per-line (loose)
        comps.append({
            "ufo": "UFO%d" % (len(comps) + 1), "members": idx,
            "bbox": [round(min(xs), 1), round(min(ys), 1), round(max(xs), 1), round(max(ys), 1)],
            "layer": layer, "block": prims[idx[0]].get("block", ""),
            "model_guess": ID2NAME[int(Counter(int(pred[i]) for i in idx).most_common(1)[0][0])]})
    return comps


def render_ufo(prims, pred, comps, out, figsize=(18, 14), dpi=160):
    """Plan in light gray (so tags read clearly), each component's bbox in red, and its
    UFO# tag placed OUTSIDE the box with a leader line and an opaque background — tags no
    longer overlap the geometry. Tags are staggered to avoid colliding with each other."""
    import matplotlib.patches as mpatches
    fig, ax = plt.subplots(figsize=figsize)
    for p in prims:
        ax.plot([p["x0"], p["x1"]], [p["y0"], p["y1"]], color="0.6", lw=0.8)   # gray context
    ax.set_aspect("equal")
    ax.autoscale(); ymin, ymax = ax.get_ylim(); xmin, xmax = ax.get_xlim()
    off = (ymax - ymin) * 0.035
    for k, c in enumerate(comps):
        x0, y0, x1, y1 = c["bbox"]
        ax.add_patch(mpatches.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="red", lw=1.3))
        lx = (x0 + x1) / 2
        ly = y1 + off * (1 + (k % 3))                  # stagger vertically to reduce collisions
        ax.annotate(c["ufo"], xy=(lx, y1), xytext=(lx, ly), fontsize=9, color="red",
                    ha="center", va="bottom", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="red", alpha=0.95),
                    arrowprops=dict(arrowstyle="-", color="red", lw=0.7))
    ax.axis("off")
    plt.tight_layout(); plt.savefig(out, dpi=dpi); plt.close(fig)
    return str(out)


def _b64(path):
    return base64.b64encode(Path(path).read_bytes()).decode()


def _chat(client, model, content):
    """Robust call: newer models (gpt-5.x) reject custom temperature -> retry without it."""
    kw = dict(model=model, messages=[{"role": "user", "content": content}],
              response_format={"type": "json_object"})
    try:
        return client.chat.completions.create(temperature=0, **kw)
    except Exception:
        return client.chat.completions.create(**kw)


LINE_PROMPT = """You are correcting the per-line classification of the STRUCTURAL / LOOSE
lines of an architectural CAD floor plan — walls, separators, glazing, beams, columns.
(Furniture and fixtures are grouped objects handled separately; ignore them here.)

You get THREE images of the SAME plan, plus a JSON table of the loose lines:
- ORIGINAL: the plan in its native CAD colors, with text labels (the drafter's intent).
- HIGHLIGHTED: every line colored by its CURRENT model class (legend below).
- MARKED: the same, with each loose line's id (L0, L1, ...) printed BESIDE it.
- LINES TABLE: each loose line's id, current model class, CAD layer, and coords.

Your job: COMPARE the HIGHLIGHTED image to the ORIGINAL and find lines the model got WRONG,
then return the corrections. Reason from the drawing; the layer name and the model class are
HINTS ONLY — if they disagree with what you see, trust the drawing.

Common corrections:
- A wall colored as something else (or 'others') -> 'wall'.
- Closely-spaced parallel lines that run continuously along a long edge are a WALL, not
  'glass'; glazing ('glass') is only a SHORT section bridging an opening with wall on both
  sides. Fix 'glass' that is really wall, and 'wall' that is really a window.
- USE THE LAYER: a line whose layer is 'Beam' is a beam ('concrete_beam'/'steel_beam'); a
  'column' layer -> a column; a 'grid'/'axis' layer -> 'axis_grid' — even if the model
  called it 'wall'. The original image often draws these in a distinct color too.
- A red / distinctly-colored partition line in the original = a separator (often not a
  load-bearing wall) — classify per what it is, not automatically 'wall'.
- Lines wrongly in 'others' that clearly belong to a real structural class.

Color legend (current class -> color): {legend}
Valid classes (use EXACTLY one): {classes}

LOOSE LINES TABLE:
{table}

Respond with JSON ONLY, listing ONLY the lines to change:
{{"changes":[{{"id":"L12","to":"wall","reason":"continuous outer edge, model had glass"}}]}}
Every id MUST exist in the table."""


COMP_PROMPT = """You are labeling OBJECTS in an architectural CAD floor plan. Each object is
a CAD block (a window, door, fixture, or piece of furniture) that the drafter grouped; on the
TAGGED image it is boxed in red with a tag UFO1, UFO2, ... (tags sit just outside each box with
a leader line). You also get the ORIGINAL image in native CAD colors, with text labels.

Decide each object's TRUE class by READING THE DRAWING — the shape it makes, the native
colors, and any nearby room/text label in the original image are your primary evidence.

The JSON for each UFO gives its bounding box, its CAD layer, and its block name (NO prior
label — you decide). Treat the LAYER NAME as a HINT ONLY:
- Layers are often informative ('windows'->glass, 'doors'->a door, a fittings layer->toilet/
  sink/bathtub, a furniture layer->table/chair/bed/sofa) BUT layers can be wrong or generic.
Decide each object from its drawn SHAPE and any nearby text label in the original image; if
the layer disagrees with what you see, TRUST THE DRAWING.

Valid classes (use EXACTLY one): {classes}

Components to label:
{components}

Respond with JSON ONLY: {{"labels":[{{"ufo":"UFO1","class":"glass","reason":"shape = glazing bridging the wall; layer 'windows' agrees"}}]}}
Label every UFO."""


def review_components(comps, ufo_img, original_img, model="gpt-5.5"):
    from openai import OpenAI
    client = OpenAI()
    compact = [{"ufo": c["ufo"], "bbox": c["bbox"], "layer": c["layer"],
                "block": c["block"]} for c in comps]   # no model label -> GPT decides fresh
    prompt = COMP_PROMPT.format(classes=CLASS_LIST, components=json.dumps(compact))
    content = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_b64(original_img)}"}},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_b64(ufo_img)}"}},
    ]
    data = json.loads(_chat(client, model, content).choices[0].message.content)
    out = {}
    for x in data.get("labels", []):
        if x.get("ufo") and x.get("class") in NAME2ID:
            out[x["ufo"]] = x["class"]
            print(f"  {x['ufo']}: {x['class']}   ({str(x.get('reason',''))[:55]})")
    return out


def review_lines(prims, pred, clean_img, marked_img, original_img, model="gpt-5.5"):
    """GPT #2 — fix mislabeled LOOSE lines (walls/separators/glazing). Returns {idx: class}."""
    from openai import OpenAI
    client = OpenAI()
    table = lines_table(prims, pred, only_loose=True)
    prompt = LINE_PROMPT.format(legend=json.dumps(legend()), classes=CLASS_LIST, table=json.dumps(table))
    content = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_b64(original_img)}"}},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_b64(clean_img)}"}},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_b64(marked_img)}"}},
    ]
    data = json.loads(_chat(client, model, content).choices[0].message.content)
    changes = {}
    for c in data.get("changes", []):
        cid = str(c.get("id", "")); to = c.get("to")
        if cid.startswith("L") and cid[1:].isdigit() and to in NAME2ID:
            idx = int(cid[1:])
            if 0 <= idx < len(prims) and prims[idx].get("group") is None:   # loose only
                changes[idx] = NAME2ID[to]
                print(f"  {cid}: -> {to}   ({str(c.get('reason', ''))[:55]})")
    return changes


def supervise(model, prims, pred, out_dir, iters=3, gpt_model="gpt-5.5", device="cpu",
              input_path=None):
    """Two-stage GPT supervision:
      GPT #1 (objects): label each CAD block (UFO#) from the original + UFO-boxed image.
      GPT #2 (lines):   fix mislabeled LOOSE lines (walls/separators/glazing) by comparing
                        the class-highlighted image to the original, using L#-marked image
                        + per-line layers. Both treat layer/model output as hints and decide
                        from the drawing."""
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    texts = []
    if input_path and str(input_path).lower().endswith(".dxf"):
        from vtrue.infer import dxf_texts
        try:
            texts = dxf_texts(input_path); print(f"text labels overlaid: {len(texts)}")
        except Exception:
            pass
    original = render_original(prims, out / "original.png", texts=texts)
    comps = build_components(prims, pred)
    n_loose = sum(1 for p in prims if p.get("group") is None)
    print(f"objects (CAD blocks): {len(comps)}  |  loose lines: {n_loose}")
    labels = {}

    # ---- GPT #1: label the block OBJECTS (original + UFO-boxed image) ----
    if comps:
        ufo = render_ufo(prims, pred, comps, out / "ufo.png")
        print(f"[GPT-1 objects] {gpt_model} labeling {len(comps)} objects …")
        labels = review_components(comps, ufo, original, gpt_model)
        for c in comps:
            if c["ufo"] in labels:
                cid = NAME2ID[labels[c["ufo"]]]
                for i in c["members"]:
                    pred[i] = cid

    # ---- GPT #2: fix the LOOSE LINES, ITERATING (change only what's wrong, re-render,
    #      look again) until GPT reports no more changes or we hit max iters ----
    if n_loose:
        for it in range(max(1, iters)):
            clean = render(prims, pred, out / "highlighted.png", marked=False)
            marked = render_lines(prims, pred, out / "marked_lines.png")
            print(f"[GPT-2 lines · round {it + 1}/{iters}] reviewing {n_loose} loose lines …")
            changes = review_lines(prims, pred, clean, marked, original, gpt_model)
            if not changes:
                print("  no more changes — converged."); break
            for i, cid in changes.items():
                pred[i] = cid

    render(prims, pred, out / "final.png", marked=False)
    json.dump({"objects": [{**{k: c[k] for k in ("ufo", "bbox", "layer", "block")},
                            "label": labels.get(c["ufo"], c["model_guess"])} for c in comps],
               "lines": lines_table(prims, pred, only_loose=True)}, open(out / "labeled.json", "w"))
    print("final line classes:", dict(Counter(ID2NAME[int(c)] for c in pred)))
    print("done ->", out)
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out-dir", default="gpt_out")
    ap.add_argument("--iters", type=int, default=3, help="max GPT-2 line-review rounds")
    ap.add_argument("--model", default="gpt-5.5")
    ap.add_argument("--no-rotate", action="store_true")
    a = ap.parse_args()
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("set OPENAI_API_KEY")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(a.weights, device)
    prims = load_plan(a.input)
    if not prims:
        raise SystemExit("no primitives")
    pred, prims = label_plan(model, prims, device, rotate=not a.no_rotate)
    print("model line classes:", dict(Counter(ID2NAME[int(c)] for c in pred)))
    supervise(model, prims, pred, a.out_dir, a.iters, a.model, device, input_path=a.input)


if __name__ == "__main__":
    main()
