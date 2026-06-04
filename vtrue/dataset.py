"""ArchCAD dataset -> multimodal training tensors.

__getitem__ flow (the order matters):
  load JSON -> (train) inject flaws -> subsample if huge -> normalize ->
  node features + raster + kNN edge graph.  Label = clean semantic id.
"""
from __future__ import annotations
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from vtrue.archcad import load_chunk
from vtrue.geometry import to_arrays, bbox, node_features, NODE_FEAT_DIM
from vtrue.edges import build_edges, EDGE_FEAT_DIM
from vtrue.raster import render
from vtrue import flaws

IGNORE = -100


class ArchCADDataset(Dataset):
    def __init__(self, root, split, max_prims=2048, k=16, raster_size=256,
                 use_vision=True, use_edges=True, flaw=True, flaw_frac=0.2, seed=0,
                 dominant=(0, 19, 20, 30)):    # axis_grid, glass, wall, Others — the bulky linework
        self.files = sorted((Path(root) / split).glob("*.json"))
        if not self.files:
            raise FileNotFoundError(f"No JSON chunks under {root}/{split}")
        self.max_prims = max_prims; self.k = k; self.raster_size = raster_size
        self.use_vision = use_vision; self.use_edges = use_edges
        self.flaw = flaw and split == "train"
        self.flaw_frac = flaw_frac      # fraction of plans flawed PER EPOCH (rest stay clean)
        self.seed = seed; self.epoch = 0
        self.dominant = set(dominant)   # bulky classes (axis/glass/wall) subsampled FIRST,
        #                                 so rare/structural objects (doors/fixtures) are never dropped

    def set_epoch(self, e):
        """Call once per epoch (train.py) so flaw selection/perturbation re-randomizes.
        Requires persistent_workers=False so workers re-fork with the new epoch."""
        self.epoch = int(e)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        rng = random.Random((self.seed * 1_000_003 + self.epoch * 2_654_435_761 + i) & 0xFFFFFFFF)
        prims = load_chunk(self.files[i])
        if not prims:
            return self._empty()
        # ~flaw_frac of plans get flaws this epoch; the rest stay clean (matches ArchCAD's clean distribution)
        if self.flaw and rng.random() < self.flaw_frac:
            arr0 = to_arrays(prims); _, w0, h0 = bbox(arr0)
            prims = flaws.inject(prims, rng, scale=float(np.hypot(w0, h0)))
        if len(prims) > self.max_prims:                     # STRATIFIED subsample (hatch-safe)
            rare = [j for j, p in enumerate(prims) if p["sem"] not in self.dominant]
            dom = [j for j, p in enumerate(prims) if p["sem"] in self.dominant]
            if len(rare) >= self.max_prims:                 # even rare alone overflow -> sample rare
                sel = rng.sample(rare, self.max_prims)
            else:                                           # keep ALL rare, fill the rest with dominant
                budget = self.max_prims - len(rare)
                sel = rare + rng.sample(dom, min(budget, len(dom)))
            prims = [prims[j] for j in sorted(sel)]
        arr = to_arrays(prims)
        lo, w, h = bbox(arr)
        feat = torch.from_numpy(node_features(arr, lo, w, h))
        sem = torch.from_numpy(arr["SEM"]).long()
        raster = render(arr, lo, w, h, self.raster_size) if self.use_vision else None
        if self.use_edges:
            ni, ef, nm = build_edges(arr, lo, w, h, self.k)
            ni = torch.from_numpy(ni); ef = torch.from_numpy(ef); nm = torch.from_numpy(nm)
        else:
            ni = ef = nm = None
        return feat, sem, raster, ni, ef, nm

    def _empty(self):
        feat = torch.zeros(1, NODE_FEAT_DIM)
        sem = torch.full((1,), IGNORE, dtype=torch.long)
        raster = torch.zeros(1, self.raster_size, self.raster_size) if self.use_vision else None
        ni = torch.zeros(1, self.k, dtype=torch.long) if self.use_edges else None
        ef = torch.zeros(1, self.k, EDGE_FEAT_DIM) if self.use_edges else None
        nm = torch.zeros(1, self.k, dtype=torch.bool) if self.use_edges else None
        return feat, sem, raster, ni, ef, nm


def collate(batch):
    """Pad nodes to N_max. Returns feats[B,N,F], sem[B,N], raster[B,1,S,S] or None,
    nbr_idx[B,N,k], edge_feat[B,N,k,E], nbr_mask[B,N,k], pad_mask[B,N] (True=pad)."""
    B = len(batch)
    N = max(b[0].shape[0] for b in batch)
    F = batch[0][0].shape[1]
    has_vis = batch[0][2] is not None
    has_edge = batch[0][3] is not None
    feats = torch.zeros(B, N, F)
    sem = torch.full((B, N), IGNORE, dtype=torch.long)
    pad_mask = torch.ones(B, N, dtype=torch.bool)            # True where padded
    raster = torch.stack([b[2] for b in batch]) if has_vis else None
    if has_edge:
        k = batch[0][3].shape[1]; E = batch[0][4].shape[2]
        nbr_idx = torch.zeros(B, N, k, dtype=torch.long)
        edge_feat = torch.zeros(B, N, k, E)
        nbr_mask = torch.zeros(B, N, k, dtype=torch.bool)
    else:
        nbr_idx = edge_feat = nbr_mask = None
    for b, (f, s, _r, ni, ef, nm) in enumerate(batch):
        n = f.shape[0]
        feats[b, :n] = f; sem[b, :n] = s; pad_mask[b, :n] = False
        if has_edge:
            nbr_idx[b, :n] = ni; edge_feat[b, :n] = ef; nbr_mask[b, :n] = nm
    return feats, sem, raster, nbr_idx, edge_feat, nbr_mask, pad_mask
