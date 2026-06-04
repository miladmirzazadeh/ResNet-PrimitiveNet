"""Per-class evaluation of a trained Vtrue model.

Reports precision / recall / F1 / IoU per class, plus overall accuracy, macro-F1,
and weighted-F1 (wF1) — the same semantic-spotting metrics DPSS reports in its
FloorPlanCAD Table 3 (their semantic F1 ~92 / wF1 ~93). Use it to judge whether
41K was enough and to see exactly which classes are weak.

  python -m vtrue.eval --weights runs/v1/best.pt --data data/archcad --split val
  python -m vtrue.eval --weights runs/v1/best.pt --data data/archcad --split test --json metrics.json
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from vtrue.dataset import ArchCADDataset, collate, IGNORE
from vtrue.classes import ID2NAME, NUM_CLASSES
from vtrue.infer import load_model


@torch.no_grad()
def run(model, loader, device, num_classes):
    tp = np.zeros(num_classes); fp = np.zeros(num_classes); fn = np.zeros(num_classes)
    inter = np.zeros(num_classes); union = np.zeros(num_classes); support = np.zeros(num_classes)
    for feats, sem, raster, ni, ef, nm, pad in loader:
        feats, sem, pad = feats.to(device), sem.to(device), pad.to(device)
        raster = raster.to(device) if raster is not None else None
        if ni is not None:
            ni, ef, nm = ni.to(device), ef.to(device), nm.to(device)
        pred = model(feats, raster, ni, ef, nm, pad).argmax(-1)
        valid = sem != IGNORE
        p = pred[valid].cpu().numpy(); g = sem[valid].cpu().numpy()
        for c in range(num_classes):
            pc = p == c; gc = g == c
            tp[c] += (pc & gc).sum(); fp[c] += (pc & ~gc).sum(); fn[c] += (~pc & gc).sum()
            inter[c] += (pc & gc).sum(); union[c] += (pc | gc).sum(); support[c] += gc.sum()
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / np.maximum(tp + fn, 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    iou = inter / np.maximum(union, 1)
    return dict(precision=prec, recall=rec, f1=f1, iou=iou, support=support, tp=tp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--split", default="val")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--json", help="optional path to dump metrics as JSON")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(a.weights, map_location="cpu"); args = ck.get("args", {})
    model = load_model(a.weights, device)
    num_classes = int(args.get("num_classes", NUM_CLASSES))

    ds = ArchCADDataset(a.data, a.split, flaw=False,
                        max_prims=int(args.get("max_prims", 2048)),
                        k=int(args.get("k", 16)),
                        raster_size=int(args.get("raster_size", 256)),
                        use_vision=bool(args.get("vision", True)),
                        use_edges=bool(args.get("edges", True)))
    loader = DataLoader(ds, batch_size=a.batch, shuffle=False, num_workers=a.workers,
                        collate_fn=collate, pin_memory=True)
    print(f"eval on {a.split}: {len(ds)} chunks, {num_classes} classes, device={device}")

    m = run(model, loader, device, num_classes)
    sup = m["support"]; present = sup > 0
    total = sup.sum()
    overall_acc = m["tp"].sum() / max(total, 1)
    macro_f1 = m["f1"][present].mean() if present.any() else 0.0
    w_f1 = (m["f1"][present] * sup[present]).sum() / max(sup[present].sum(), 1)
    mean_iou = m["iou"][present].mean() if present.any() else 0.0

    print("\n  id  class                support   prec    rec     F1     IoU")
    print("  " + "-" * 62)
    order = np.argsort(-sup)                       # most frequent first
    for c in order:
        if sup[c] == 0:
            continue
        print("  %2d  %-20s %8d  %.3f  %.3f  %.3f  %.3f" %
              (c, ID2NAME.get(int(c), str(c)), int(sup[c]),
               m["precision"][c], m["recall"][c], m["f1"][c], m["iou"][c]))
    print("  " + "-" * 62)
    print("  OVERALL   acc %.4f   macro-F1 %.4f   wF1 %.4f   mIoU %.4f" %
          (overall_acc, macro_f1, w_f1, mean_iou))
    print("  (DPSS FloorPlanCAD reference: semantic F1 ~92, wF1 ~93 — different "
          "dataset/scale, so treat as a ballpark, not a strict target.)")

    if a.json:
        out = {"split": a.split, "overall_acc": float(overall_acc),
               "macro_f1": float(macro_f1), "weighted_f1": float(w_f1), "mIoU": float(mean_iou),
               "per_class": {ID2NAME.get(int(c), str(c)): {
                   "support": int(sup[c]), "precision": float(m["precision"][c]),
                   "recall": float(m["recall"][c]), "f1": float(m["f1"][c]),
                   "iou": float(m["iou"][c])} for c in range(num_classes) if sup[c] > 0}}
        Path(a.json).write_text(json.dumps(out, indent=2))
        print("  metrics ->", a.json)


if __name__ == "__main__":
    main()
