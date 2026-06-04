"""Vtrue model — multimodal graph-aware transformer over CAD primitives.

  node features ─► embed ─► RelationalLayer×R (kNN message passing w/ edge feats)
                                      │
  raster ─► VisionCNN ─► grid_sample at each primitive ─► gated fusion (DPSS eq.2-3)
                                      │
                              TransformerEncoder (global self-attention)
                                      │
                              Linear ─► per-primitive class

All three signals are optional (use_vision / use_edges) for ablation. Pure
PyTorch — no custom CUDA. ~3M params.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class VisionBackbone(nn.Module):
    """Raster [B,1,S,S] -> feature map [B,out_dim,h,w] sampled per primitive.

    backbone="cnn3"  -> the tiny 3-conv net (no deps, fast, lower ceiling).
    backbone=<timm>  -> a pretrained ImageNet backbone (e.g. 'resnet50',
                        'resnet34', 'hrnet_w32') via timm — the accuracy lever.
                        Pure pip, no custom CUDA. in_chans=1 adapts pretrained
                        weights to the single-channel raster.
    """
    def __init__(self, backbone="cnn3", out_dim=128, in_ch=1, pretrained=True):
        super().__init__()
        self.backbone = backbone
        if backbone == "cnn3":
            self.net = nn.Sequential(
                nn.Conv2d(in_ch, 32, 5, 2, 2), nn.GELU(),
                nn.Conv2d(32, 64, 3, 2, 1), nn.GELU(),
                nn.Conv2d(64, out_dim, 3, 2, 1), nn.GELU())
            self.proj = nn.Identity()
        else:
            import timm
            self.net = timm.create_model(backbone, pretrained=pretrained, features_only=True,
                                         out_indices=(3,), in_chans=in_ch)
            ch = self.net.feature_info.channels()[-1]
            self.proj = nn.Conv2d(ch, out_dim, 1)

    def forward(self, x):
        if self.backbone == "cnn3":
            return self.net(x)
        return self.proj(self.net(x)[-1])


class RelationalLayer(nn.Module):
    """kNN message passing: each node aggregates neighbours' embeddings + edge feats.
    Injects graph structure (parallel/collinear/connected) the transformer would
    otherwise have to infer from absolute coords."""
    def __init__(self, dim, edge_dim, hidden=None):
        super().__init__()
        hidden = hidden or dim
        self.msg = nn.Sequential(
            nn.Linear(2 * dim + edge_dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, nbr_idx, edge_feat, nbr_mask):
        B, N, D = x.shape
        k = nbr_idx.shape[2]
        gather = nbr_idx.reshape(B, N * k, 1).expand(B, N * k, D)
        x_nbr = torch.gather(x, 1, gather).reshape(B, N, k, D)       # neighbour embeddings
        x_self = x.unsqueeze(2).expand(B, N, k, D)
        m = self.msg(torch.cat([x_self, x_nbr, edge_feat], dim=-1))  # [B,N,k,D]
        m = m * nbr_mask.unsqueeze(-1)
        denom = nbr_mask.sum(2, keepdim=True).clamp(min=1)           # mean over valid neighbours
        agg = m.sum(2) / denom
        return self.norm(x + agg)


class VtrueModel(nn.Module):
    def __init__(self, num_classes=31, node_dim=16, edge_dim=13, dim=256,
                 depth=4, heads=8, dim_ff=512, dropout=0.1,
                 use_vision=True, use_edges=True, rel_layers=2, vis_dim=128, raster_size=256,
                 backbone="cnn3", pretrained=True):
        super().__init__()
        self.use_vision = use_vision; self.use_edges = use_edges
        self.raster_size = raster_size
        self.embed = nn.Sequential(nn.Linear(node_dim, dim), nn.GELU(), nn.LayerNorm(dim))
        if use_edges:
            self.rel = nn.ModuleList([RelationalLayer(dim, edge_dim) for _ in range(rel_layers)])
        if use_vision:
            self.vision = VisionBackbone(backbone, vis_dim, pretrained=pretrained)
            self.fuse_geo = nn.Linear(dim, vis_dim)
            self.fuse_vis = nn.Linear(vis_dim, vis_dim)
            self.fuse_w = nn.Linear(vis_dim, vis_dim)
            self.fuse_out = nn.Sequential(nn.Linear(dim + vis_dim, dim), nn.GELU(), nn.LayerNorm(dim))
        layer = nn.TransformerEncoderLayer(dim, heads, dim_ff, dropout,
                                           batch_first=True, activation="gelu", norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, depth)
        self.head = nn.Linear(dim, num_classes)
        self._coord_lo = 4          # node feat layout: one-hot(4) then x0,y0,x1,y1,cx,cy...

    def _sample_grid(self, feats):
        cs = self._coord_lo
        x0, y0, x1, y1, cx, cy = feats[..., cs:cs + 6].unbind(-1)
        xs = torch.stack([x0, cx, x1], -1); ys = torch.stack([y0, cy, y1], -1)
        return (torch.stack([xs, ys], -1).clamp(0, 1) * 2 - 1)       # [B,N,3,2]

    def forward(self, feats, raster=None, nbr_idx=None, edge_feat=None,
                nbr_mask=None, pad_mask=None):
        x = self.embed(feats)
        if self.use_edges and nbr_idx is not None:
            for layer in self.rel:
                x = layer(x, nbr_idx, edge_feat, nbr_mask)
        if self.use_vision and raster is not None:
            fmap = self.vision(raster)
            B, N = feats.shape[0], feats.shape[1]
            grid = self._sample_grid(feats).reshape(B, N * 3, 1, 2)
            samp = F.grid_sample(fmap, grid, mode="bilinear", align_corners=True)
            vis = samp.squeeze(-1).reshape(B, -1, N, 3).mean(-1).transpose(1, 2)
            w = torch.sigmoid(self.fuse_w(torch.tanh(self.fuse_geo(x) + self.fuse_vis(vis))))
            x = self.fuse_out(torch.cat([x, w * vis], -1))
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        return self.head(x)
