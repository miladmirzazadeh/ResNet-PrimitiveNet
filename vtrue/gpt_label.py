"""GPT supervision layer — review the model's labels and correct them, iterating.

The perception model finds and groups objects with exact coordinates but often dumps
ambiguous things (fixtures, furniture) into 'others'. This stage hands GPT-4o the
rendered plan (with object-id numbers) + the JSON of objects (label + bbox), and GPT
returns label corrections. We apply them, re-render, and repeat for a few rounds —
the model does the geometry, GPT does the naming.

  OPENAI_API_KEY=sk-... python -m vtrue.gpt_label \
      --weights runs/v1/best.pt --input plan.dxf --out-dir gpt_out --iters 2

Outputs (in --out-dir): labeled JSON, a colored preview, and a per-round review log.
Needs the `openai` package and an OpenAI API key. The model is never asked for
coordinates — only to rename objects it can see.
"""
from __future__ import annotations
import argparse, base64, json, os
from pathlib import Path

import torch

from vtrue.infer import (load_model, load_plan, label_plan, to_record, render_preview,
                         group_objects)
from vtrue.classes import ID2NAME

CLASS_LIST = [ID2NAME[i] for i in sorted(ID2NAME)]
# classes the model is reliable on — GPT focuses on the rest (fixtures/furniture/others)
TRUSTED = {"wall", "axis_grid", "glass", "concrete_column", "steel_column",
           "concrete_beam", "steel_beam", "parking_space", "pile"}


def _b64(path):
    return base64.b64encode(Path(path).read_bytes()).decode()


def review_round(rec, image_path, model_name="gpt-4o"):
    """One GPT pass. Returns {object_id: new_label} for objects GPT chooses to relabel."""
    from openai import OpenAI
    client = OpenAI()
    # only send objects worth reviewing (untrusted labels) to keep it focused
    review = [o for o in rec["objects"] if o["label"] not in TRUSTED]
    if not review:
        return {}
    compact = [{"id": o["id"], "label": o["label"], "bbox": o["bbox"], "n": o["n"]} for o in review]
    prompt = (
        "You are auditing an auto-labeled architectural CAD floor plan. The image shows "
        "the plan; each reviewable object is marked with its red id number. Below is the "
        "JSON of those objects (current label, bounding box [xmin,ymin,xmax,ymax], and "
        "primitive count n).\n\n"
        f"Valid classes (use EXACTLY these strings): {CLASS_LIST}\n\n"
        f"Objects to review:\n{json.dumps(compact)}\n\n"
        "Many are labeled 'others' or a wrong fixture. Using the drawing, correct the "
        "label of any object you can identify (e.g. a toilet outline -> 'toilet', a bed "
        "-> 'bed', a stair run -> 'staircase'). Leave an object out if you are unsure or "
        "it is already correct. Respond with JSON only: "
        '{"corrections":[{"id":<int>,"label":"<class>","reason":"<short>"}]}'
    )
    resp = client.chat.completions.create(
        model=model_name, temperature=0,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_b64(image_path)}"}},
        ]}])
    data = json.loads(resp.choices[0].message.content)
    out = {}
    for c in data.get("corrections", []):
        if c.get("label") in CLASS_LIST and isinstance(c.get("id"), int):
            out[c["id"]] = c["label"]
            print(f"  obj {c['id']}: {c.get('label')}  ({c.get('reason','')[:60]})")
    return out


def apply_corrections(prims, pred, rec, corrections):
    """Map object-level corrections back onto per-primitive labels."""
    name2id = {v: k for k, v in ID2NAME.items()}
    by_id = {o["id"]: o for o in rec["objects"]}
    for oid, label in corrections.items():
        if oid in by_id and label in name2id:
            for i in by_id[oid]["members"]:
                pred[i] = name2id[label]
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out-dir", default="gpt_out")
    ap.add_argument("--iters", type=int, default=2, help="GPT review rounds")
    ap.add_argument("--model", default="gpt-4o")
    ap.add_argument("--no-rotate", action="store_true")
    a = ap.parse_args()
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("set OPENAI_API_KEY")
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_model(a.weights, device)
    prims = load_plan(a.input)
    if not prims:
        raise SystemExit("no primitives")
    pred, prims = label_plan(model, prims, device, rotate=not a.no_rotate)

    for it in range(a.iters):
        rec = to_record(prims, pred)
        # object-id label positions for the review image
        label_pos = {o["id"]: ((o["bbox"][0] + o["bbox"][2]) / 2, (o["bbox"][1] + o["bbox"][3]) / 2)
                     for o in rec["objects"] if o["label"] not in TRUSTED}
        img = render_preview(prims, pred, out / f"review_{it}.png",
                             title=f"GPT review round {it}", label_objs=label_pos)
        print(f"[round {it}] reviewing {len(label_pos)} objects -> GPT…")
        corr = review_round(rec, img, a.model)
        if not corr:
            print("  no changes; converged."); break
        pred = apply_corrections(prims, pred, rec, corr)

    rec = to_record(prims, pred)
    json.dump(rec, open(out / "labeled.json", "w"))
    render_preview(prims, pred, out / "final.png", title="GPT-supervised labels")
    from collections import Counter
    print("final object classes:", dict(Counter(o["label"] for o in rec["objects"])))
    print("done ->", out)


if __name__ == "__main__":
    main()
