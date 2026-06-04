"""Render primitives to a grayscale raster for the vision pathway.

Rendered from the SAME (possibly flaw-injected) coordinates that produce the
node/edge features, so image and tokens always align. We never use ArchCAD's
shipped PNG during augmentation — it would show clean geometry while the vectors
are flawed.
"""
from __future__ import annotations
import numpy as np
import torch
from PIL import Image, ImageDraw


def render(arr, lo, w, h, size=256) -> torch.Tensor:
    """arr: geometry arrays (RAW coords). Returns [1,size,size] float, strokes=1."""
    img = Image.new("L", (size, size), 0)
    n = arr["n"]
    if n:
        d = ImageDraw.Draw(img)
        A = (arr["A"] - lo); B = (arr["B"] - lo); C = (arr["C"] - lo); R = arr["R"]
        sx = (size - 1) / w; sy = (size - 1) / h
        ax = A[:, 0] * sx; ay = A[:, 1] * sy
        bx = B[:, 0] * sx; by = B[:, 1] * sy
        cx = C[:, 0] * sx; cy = C[:, 1] * sy
        rpx = R * ((sx + sy) / 2)
        T = arr["T"]
        for i in range(n):
            if T[i] == 2 and rpx[i] > 0.5:                       # circle
                d.ellipse([cx[i] - rpx[i], cy[i] - rpx[i], cx[i] + rpx[i], cy[i] + rpx[i]], outline=255)
            else:                                                # line / arc (chord) / other
                d.line([ax[i], ay[i], bx[i], by[i]], fill=255, width=1)
    a = np.asarray(img, np.float32) / 255.0
    return torch.from_numpy(a)[None]
