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


def lines_table(prims, pred):
    return [{"id": f"L{i}", "label": ID2NAME.get(int(pred[i]), str(int(pred[i]))),
             "coords": [round(p["x0"], 1), round(p["y0"], 1), round(p["x1"], 1), round(p["y1"], 1)]}
            for i, p in enumerate(prims)]


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


def render_original(prims, out, figsize=(18, 14), dpi=160):
    """Native CAD view: each line in its ORIGINAL DXF color (dark colors brightened) on a
    dark background (like a CAD viewer). The drafter's colors carry intent GPT can use
    (red separator vs wall; doors/windows in their layer color)."""
    fig, ax = plt.subplots(figsize=figsize, facecolor="black")
    ax.set_facecolor("black")
    for p in prims:
        ax.plot([p["x0"], p["x1"]], [p["y0"], p["y1"]],
                color=_visible(p.get("rgb", (255, 255, 255))), lw=1.0)
    ax.set_aspect("equal"); ax.axis("off")
    plt.tight_layout(); plt.savefig(out, dpi=dpi, facecolor="black"); plt.close(fig)
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


PROMPT = """You are auditing an auto-classified architectural CAD floor plan.

You are given THREE images of the SAME plan plus a lines table:
- ORIGINAL image: the plan in its NATIVE CAD colors (as the drafter drew it, dark bg).
  These colors carry intent: e.g. a line drawn in RED is often a separator / partition /
  zone boundary, NOT a structural wall; dashed or oddly-colored lines may be annotations
  or fixtures. Use the original colors together with the geometry to decide the TRUE class.
- CLEAN image: each line drawn in the color of its CURRENT (model) class (legend below).
- MARKED image: the same lines, each tagged with its id (L0, L1, ...).
- LINES TABLE (JSON): every line's id, current label, and coords [x0,y0,x1,y1].

Color legend for the CLEAN/MARKED images (current class -> color): {legend}
Valid classes (use EXACTLY one of these strings): {classes}

Read the drawing like an architect and decide what each line REALLY is. Then report
ONLY the lines whose label should CHANGE. Use the MARKED image to translate your
visual judgment into line ids. Guidance:
- A wall is one line, or TWO/THREE parallel lines a small distance apart (its faces).
- A window ('glass') is a SHORT section of thin parallel lines that BRIDGES AN OPENING
  in a wall, with solid wall on BOTH sides of it. Closely-spaced parallel lines are
  NOT automatically glass: if they run continuously along a long edge / the building
  perimeter (no opening), they are a WALL, not a window. Relabel such 'glass' -> 'wall'.
- A door = a leaf line + its swing arc inside a wall opening.
- Fixtures (toilet/sink/urinal/bathtub/squat_toilet), furniture (bed/sofa/table/chair),
  stairs, columns, holes — name them by their drawn shape.
- Lines mislabeled 'others' that clearly belong to a real class are the main target.
- Leave a line out if it is already correct or you are unsure.

Respond with JSON ONLY:
{{"changes":[{{"id":"L12","from":"others","to":"toilet","reason":"rounded WC outline"}}]}}
Every id MUST exist in the lines table.

LINES TABLE:
{table}"""


def review_round(prims, pred, clean_img, marked_img, original_img, model="gpt-5.5"):
    from openai import OpenAI
    client = OpenAI()
    table = lines_table(prims, pred)
    prompt = PROMPT.format(legend=json.dumps(legend()), classes=CLASS_LIST, table=json.dumps(table))
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
            if 0 <= idx < len(prims):
                changes[idx] = NAME2ID[to]
                print(f"  {cid}: {c.get('from')} -> {to}   ({str(c.get('reason',''))[:55]})")
    return changes


def supervise(model, prims, pred, out_dir, iters=2, gpt_model="gpt-5.5", device="cpu"):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    original = render_original(prims, out / "original.png")     # native CAD colors (fixed)
    for it in range(iters):
        clean = render(prims, pred, out / f"clean_{it}.png", marked=False)
        marked = render(prims, pred, out / f"marked_{it}.png", marked=True)
        print(f"[round {it}] {len(prims)} lines -> {gpt_model} …")
        ch = review_round(prims, pred, clean, marked, original, gpt_model)
        if not ch:
            print("  no changes; converged."); break
        for i, c in ch.items():
            pred[i] = c
    render(prims, pred, out / "final.png", marked=False)
    json.dump({"lines": lines_table(prims, pred)}, open(out / "labeled.json", "w"))
    print("final line classes:", dict(Counter(ID2NAME[int(c)] for c in pred)))
    print("done ->", out)
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out-dir", default="gpt_out")
    ap.add_argument("--iters", type=int, default=2)
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
    supervise(model, prims, pred, a.out_dir, a.iters, a.model, device)


if __name__ == "__main__":
    main()
