# Vtrue-thePlane

A multimodal **(vector + vision + relational graph)** primitive labeler for the
Vitruvian drafting agent. Turns any vector floor plan into per-primitive semantic
labels grounded in exact coordinates, groups them into objects, and hands them to
GPT for final reasoning.

Design rationale and every decision behind this build: see
[`vtruvian_model_design.html`](../vtruvian_model_design.html) (the deck).

---

## What it is (and isn't)
- **Is:** a ~3M-param model — geometry transformer + small vision CNN + relational
  kNN graph + DPSS-style gated fusion → per-primitive class. Pure PyTorch.
- **Isn't:** the full DPSS stack. We deliberately drop HRNet, PointTransformer
  (`pointops` CUDA), and the 800-query panoptic decoder. Instance separation is
  done by cheap geometric grouping + GPT, not a heavy decoder.

```
node features ─► embed ─► RelationalLayer×2 (kNN msg-passing w/ edge feats)
                                   │
raster ─► VisionCNN ─► sample at each primitive ─► gated fusion σ(tanh(...))
                                   │
                          Transformer encoder (global attention)
                                   │
                          Linear ─► class per primitive ─► grouping ─► GPT
```

## Layout
```
vtrue/
  classes.py    ArchCAD 30-class schema (+Others) + robust string->id
  archcad.py    ArchCAD JSON chunk -> primitive records (polylines exploded)
  geometry.py   records -> normalized node feature vectors
  edges.py      kNN relational edge features (cKDTree, vectorized)
  raster.py     render raster from primitives (for the vision branch)
  flaws.py      on-the-fly flaw injection (gaps/overshoots/jitter/floating)
  dataset.py    torch Dataset + collate (nodes + raster + edge graph)
  model.py      the multimodal graph-transformer
  train.py      training loop (mIoU, resume, time budget, class weights)
  infer.py      run on a new DXF / ArchCAD JSON -> labels + groups + preview
scripts/
  download_archcad.py   fetch json.zip from HF (token via env)
  prepare_archcad.py    unzip + train/val/test split
  kaggle_inspect.py     >>> RUN FIRST: verify JSON structure on Kaggle <<<
setup_runpod.sh   one-shot pod install
RUNPOD.md         step-by-step server guide
```

## The one thing to do first
The ArchCAD parser is written against the **documented** JSON schema. Before a real
training run, confirm the actual field names / `semantic` strings:

1. Run `scripts/kaggle_inspect.py` on Kaggle (instructions in its header).
2. Paste the printed report back.
3. If any `semantic` name differs from `vtrue/classes.py`, we add it to `ALIASES`
   (one line each). The parser already degrades gracefully (unknown → Others), so
   nothing crashes meanwhile.

## Quickstart (after data is prepared — see RUNPOD.md)
```bash
pip install -r requirements.txt
python -m vtrue.train --data data/archcad --out runs/v1 --epochs 100 --batch 16
python -m vtrue.infer --weights runs/v1/best.pt --input plan.dxf --out pred.png
```

## Robustness to messy real plans
ArchCAD is highly standardized (no gaps/overshoots). `flaws.py` injects them at
load time — re-randomized each epoch, labels untouched — so a wall with a gap, an
overshoot, or a floating fragment still trains as `wall`. Raster + edges are
recomputed from the flawed coords so all signals stay aligned. Disable with
`--no-flaw` for an ablation.

## Notes
- **Data:** ArchCAD-400K is **CC-BY-NC (non-commercial)** and gated. Mind this for
  productizing; the synthetic generator is the commercial-safe complement.
- **Layers:** never relied on — works on unlayered/PDF-derived vectors. (Layer/
  color can be added later as an optional bonus feature.)
- **Hatches:** kept in place (no pre-removal); bounded by ArchCAD's 14×14 m chunking.
# ResNet-PrimitiveNet
