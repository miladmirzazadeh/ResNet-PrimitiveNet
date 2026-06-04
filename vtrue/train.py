"""Train the Vtrue model on ArchCAD.

  python -m vtrue.train --data data/archcad --epochs 100 --batch 16 --out runs/v1

Reports val accuracy + macro mIoU. Time-budget stop + checkpoint/resume.
Class weights down-weight the dominant background-ish classes (axis/wall) so the
small, hard classes (doors/fixtures) aren't drowned out.
"""
from __future__ import annotations
import argparse, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from vtrue.dataset import ArchCADDataset, collate, IGNORE
from vtrue.geometry import NODE_FEAT_DIM
from vtrue.edges import EDGE_FEAT_DIM
from vtrue.model import VtrueModel
from vtrue.classes import NUM_CLASSES
from vtrue.archcad import load_chunk


def compute_class_weights(ds, num_classes, sample=3000, clip=10.0):
    """Inverse-frequency class weights from a sample of train chunks, so rare
    classes (e.g. fire_hydrant) aren't drowned by wall/axis. Normalized to mean 1,
    clipped so a single ultra-rare class can't explode the loss. Unseen -> 1."""
    import numpy as np, random as _r
    counts = np.zeros(num_classes, dtype=np.float64)
    files = ds.files if len(ds.files) <= sample else _r.Random(0).sample(ds.files, sample)
    for f in files:
        for p in load_chunk(f):
            c = p["sem"]
            if 0 <= c < num_classes:
                counts[c] += 1
    seen = counts > 0
    w = np.ones(num_classes, dtype=np.float64)
    med = np.median(counts[seen]) if seen.any() else 1.0
    w[seen] = np.clip(med / counts[seen], 1.0 / clip, clip)
    w = w / w[seen].mean() if seen.any() else w           # mean ~1
    print("[auto-weight] sampled %d files; class counts(min/med/max)=%d/%d/%d; "
          "weight range %.2f..%.2f" % (len(files), counts[seen].min() if seen.any() else 0,
          med, counts.max(), w.min(), w.max()))
    return torch.tensor(w, dtype=torch.float32)


def move(batch, device):
    feats, sem, raster, ni, ef, nm, pad = batch
    feats, sem, pad = feats.to(device), sem.to(device), pad.to(device)
    raster = raster.to(device) if raster is not None else None
    if ni is not None:
        ni, ef, nm = ni.to(device), ef.to(device), nm.to(device)
    return feats, sem, raster, ni, ef, nm, pad


@torch.no_grad()
def evaluate(model, loader, device, num_classes):
    model.eval()
    inter = np.zeros(num_classes); union = np.zeros(num_classes)
    correct = total = 0
    for batch in loader:
        feats, sem, raster, ni, ef, nm, pad = move(batch, device)
        logits = model(feats, raster, ni, ef, nm, pad)
        pred = logits.argmax(-1)
        valid = sem != IGNORE
        p = pred[valid].cpu().numpy(); g = sem[valid].cpu().numpy()
        correct += (p == g).sum(); total += g.size
        for c in range(num_classes):
            pc, gc = p == c, g == c
            inter[c] += (pc & gc).sum(); union[c] += (pc | gc).sum()
    iou = inter / np.maximum(union, 1); present = union > 0
    return correct / max(total, 1), float(iou[present].mean() if present.any() else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--num-classes", type=int, default=NUM_CLASSES)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--preset", choices=["toy", "medium", "large"], default=None,
                    help="toy: cnn3@256 dim256 (free GPU). medium: resnet50@512 dim384 "
                         "(recommended, ~€10-18 on 400K). large: hrnet_w32@768 dim512.")
    ap.add_argument("--backbone", default="cnn3",
                    help="cnn3 | resnet34 | resnet50 | hrnet_w32 | any timm features_only model")
    ap.add_argument("--no-pretrained", dest="pretrained", action="store_false", default=True)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--rel-layers", type=int, default=2)
    ap.add_argument("--vis-dim", type=int, default=128)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--max-prims", type=int, default=2048)
    ap.add_argument("--raster-size", type=int, default=256)
    ap.add_argument("--no-vision", dest="vision", action="store_false", default=True)
    ap.add_argument("--no-edges", dest="edges", action="store_false", default=True)
    ap.add_argument("--no-flaw", dest="flaw", action="store_false", default=True)
    ap.add_argument("--flaw-frac", type=float, default=0.2,
                    help="fraction of plans flawed PER EPOCH (rest stay clean). "
                         "Re-randomized each epoch. 0.2 keeps the model mostly on "
                         "ArchCAD's clean distribution.")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-amp", dest="amp", action="store_false", default=True,
                    help="disable mixed-precision (AMP is ~2x faster on A100; on by default)")
    ap.add_argument("--bg-classes", type=int, nargs="*", default=[0, 20],
                    help="manually down-weighted dominant classes (axis_grid, wall)")
    ap.add_argument("--bg-weight", type=float, default=0.3)
    ap.add_argument("--auto-weight", action="store_true",
                    help="compute inverse-frequency class weights from the data "
                         "(recommended for the 31-class imbalance; overrides --bg-classes)")
    ap.add_argument("--dominant", type=int, nargs="*", default=[0, 19, 20, 30],
                    help="bulky classes subsampled first (axis/glass/wall/Others) so rare "
                         "objects are never dropped by --max-prims")
    ap.add_argument("--time-budget", type=float, default=12.0, help="max hours")
    ap.add_argument("--out", type=Path, default=Path("runs/v1"))
    a = ap.parse_args()

    PRESETS = {
        "toy":    dict(backbone="cnn3",      dim=256, depth=4, rel_layers=2, vis_dim=128, raster_size=256),
        "medium": dict(backbone="resnet50",  dim=384, depth=6, rel_layers=3, vis_dim=256, raster_size=512),
        "large":  dict(backbone="hrnet_w32", dim=512, depth=8, rel_layers=3, vis_dim=256, raster_size=768),
    }
    if a.preset:
        for key, val in PRESETS[a.preset].items():
            setattr(a, key, val)
        print(f"[preset={a.preset}] {PRESETS[a.preset]}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    a.out.mkdir(parents=True, exist_ok=True)
    last_pt, best_pt = a.out / "last.pt", a.out / "best.pt"

    ds_kw = dict(max_prims=a.max_prims, k=a.k, raster_size=a.raster_size,
                 use_vision=a.vision, use_edges=a.edges, dominant=tuple(a.dominant))
    tr_ds = ArchCADDataset(a.data, "train", flaw=a.flaw, flaw_frac=a.flaw_frac, **ds_kw)
    tr = DataLoader(tr_ds, batch_size=a.batch, shuffle=True, num_workers=a.workers,
                    collate_fn=collate, drop_last=True, pin_memory=True,
                    persistent_workers=False)   # False so set_epoch re-randomizes flaws each epoch
    va_split = "val" if (a.data / "val").exists() else "test"
    va = DataLoader(ArchCADDataset(a.data, va_split, flaw=False, **ds_kw),
                    batch_size=a.batch, shuffle=False, num_workers=a.workers,
                    collate_fn=collate, pin_memory=True, persistent_workers=a.workers > 0)

    model = VtrueModel(num_classes=a.num_classes, node_dim=NODE_FEAT_DIM, edge_dim=EDGE_FEAT_DIM,
                       dim=a.dim, depth=a.depth, rel_layers=a.rel_layers, vis_dim=a.vis_dim,
                       use_vision=a.vision, use_edges=a.edges, raster_size=a.raster_size,
                       backbone=a.backbone, pretrained=a.pretrained).to(device)
    n_param = sum(p.numel() for p in model.parameters())
    print(f"params={n_param/1e6:.2f}M  vision={a.vision} edges={a.edges} flaw={a.flaw} "
          f"train={len(tr.dataset)} {va_split}={len(va.dataset)} device={device}")

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    if a.auto_weight:
        w = compute_class_weights(tr_ds, a.num_classes).to(device)
    else:
        w = torch.ones(a.num_classes, device=device)
        for c in a.bg_classes:
            if 0 <= c < a.num_classes:
                w[c] = a.bg_weight
    crit = nn.CrossEntropyLoss(weight=w, ignore_index=IGNORE)

    amp_on = a.amp and device == "cuda"
    amp_dtype = torch.bfloat16 if (amp_on and torch.cuda.is_bf16_supported()) else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=amp_on and amp_dtype == torch.float16)
    if amp_on:
        print(f"[amp] mixed precision on ({amp_dtype})")

    start, best = 0, 0.0
    if last_pt.exists():
        ck = torch.load(last_pt, map_location=device)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"]); start = ck["epoch"] + 1; best = ck["best"]
        print(f"[resume] epoch {start}, best mIoU {best:.4f}")

    t0 = time.time()
    for epoch in range(start, a.epochs):
        tr_ds.set_epoch(epoch)              # re-randomize which 20% get flawed this epoch
        model.train(); run = 0.0
        for batch in tr:
            feats, sem, raster, ni, ef, nm, pad = move(batch, device)
            opt.zero_grad()
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_on):
                logits = model(feats, raster, ni, ef, nm, pad)
                loss = crit(logits.reshape(-1, a.num_classes), sem.reshape(-1))
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            run += loss.item()
        sched.step()
        acc, miou = evaluate(model, va, device, a.num_classes)
        print(f"epoch {epoch+1}/{a.epochs} loss {run/max(len(tr),1):.4f} "
              f"val_acc {acc:.4f} val_mIoU {miou:.4f} ({(time.time()-t0)/60:.1f}m)")
        ck = {"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
              "epoch": epoch, "best": best,
              "args": {**vars(a), "node_dim": NODE_FEAT_DIM, "edge_dim": EDGE_FEAT_DIM,
                       "data": str(a.data), "out": str(a.out)}}
        torch.save(ck, last_pt)
        if miou > best:
            best = miou; ck["best"] = best; torch.save(ck, best_pt)
            print(f"  new best val_mIoU {best:.4f}")
        if (time.time() - t0) / 3600 >= a.time_budget:
            print("[time] budget reached; stopping."); break
    print(f"[done] best val_mIoU {best:.4f} -> {best_pt}")


if __name__ == "__main__":
    main()
