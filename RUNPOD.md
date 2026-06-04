# RunPod setup — step by step

Because we avoided DPSS's custom CUDA (`pointops`, old `mmcv`, detectron2), there
is **no environment hell**. Any modern PyTorch image works; setup is a pip install.

---

## 0. Before you start
- You need your **Hugging Face token** with gated access to `jackluoluo/ArchCAD`
  (huggingface.co/settings/tokens). **Never commit it or paste it to anyone** —
  you'll `export` it on the pod only.
- We download **only `json.zip` (~393 MB)** — the vision branch re-renders rasters
  from vectors, so we don't need `png.zip` (1.8 GB).

## 1. Create the pod
- **GPU:** 1× **A100 (40 GB is plenty)** — or even an A40 / 4090; our model is ~3M
  params and the GPU is *not* the bottleneck (CPU dataloading is). Prefer a pod
  with **many vCPUs (16–32)**.
- **Template:** `RunPod PyTorch 2.x` (CUDA 12). Nothing special required.
- **Volume:** attach a **20–30 GB** persistent volume mounted at `/workspace`
  (holds the dataset + checkpoints across restarts).

## 2. Upload the code
From your laptop (or `git clone` if you push this folder to a repo):
```bash
# option A: scp the folder to the pod
scp -r Vtrue-thePlane root@<pod-ip>:/workspace/
# option B: git clone <your-repo> /workspace/Vtrue-thePlane
cd /workspace/Vtrue-thePlane
```

## 3. Install
```bash
bash setup_runpod.sh
```
Expect `vtrue import OK`.

## 4. Get the data (paste your token on the pod)
```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxx          # your gated token, on the pod only
python scripts/download_archcad.py --out /workspace/data/raw
python scripts/prepare_archcad.py  --raw /workspace/data/raw --out /workspace/data/archcad
```
You should see `split: {'train': ~32000, 'val': ~4000, 'test': ~4000}` (numbers
depend on the 40K subset).

## 5. Train  (public set = ~41K samples → `medium` preset, ~40–50 epochs)
```bash
python -m vtrue.train \
  --data /workspace/data/archcad \
  --out  /workspace/runs/v1 \
  --preset medium \
  --epochs 45 --batch 24 --workers 24 \
  --auto-weight --flaw-frac 0.2 --time-budget 10
```
- The public HF release is the **~41,097-sample subset** (not the full 400K), so
  each epoch is ~10× cheaper — but with less data + from-scratch components + flaw
  augmentation you want **more epochs (~40–50), not 10.**
- `--preset medium` = pretrained **ResNet50 @512px**, dim384/depth6 (~17M).
- `--auto-weight` computes inverse-frequency class weights (essential here — class
  `100/Others` and `wall` dominate; `single_door`, `fire_hydrant` are rare).
- **AMP on by default** (~2× on A100); batch 24 fits 40GB.
- **Watch `val_mIoU` and stop when it plateaus** (best is auto-saved). The cosine
  horizon is `--epochs`; `--time-budget` is the hard stop. Resumes from `last.pt`.

### Cost (41K, not 400K → very comfortable)
~4–6 min/epoch on an A100 with AMP → **45 epochs ≈ 3.5–5h ≈ €6–12.** You have
headroom: you could run longer, or try `--preset large` (HRNet@768) on this
smaller set and still likely stay under €20.

### First, a smoke run (do this once, then launch the real one)
```bash
python -m vtrue.train --data /workspace/data/archcad --out /workspace/runs/smoke \
  --preset medium --epochs 1 --batch 16 --workers 24 --time-budget 0.3
```
Confirms data loads, the ResNet downloads, shapes line up, loss drops. Check the
first epoch's wall-clock time — multiply by 10 to project the full cost before
committing.

### Cost control (€20 budget on 400K)
- Watch the **first epoch time**. If >80 min, drop to `--raster-size 384` or
  `--backbone resnet34`, or train on a subset.
- The presets: `toy` (free-GPU baseline), `medium` (recommended), `large`
  (HRNet@768 — **exceeds €20 on 400K**, use only on a subset).

## 6. Test on your own plan
```bash
python -m vtrue.infer --weights /workspace/runs/v1/best.pt \
  --input your_plan.dxf --out pred.png
```

## 7. Pull results back
```bash
scp root@<pod-ip>:/workspace/runs/v1/best.pt .
```

---

## Tuning knobs (if needed)
| Symptom | Fix |
|---|---|
| GPU under-utilized, slow epochs | raise `--workers` (CPU-bound), raise `--batch` |
| Out of memory | lower `--batch`, lower `--max-prims`, `--raster-size 192` |
| Hatch chunks too big | lower `--max-prims` (subsamples), or `--k 12` |
| Ablate a pathway | `--no-vision` / `--no-edges` / `--no-flaw` |
| Class imbalance | adjust `--bg-classes` / `--bg-weight` |
