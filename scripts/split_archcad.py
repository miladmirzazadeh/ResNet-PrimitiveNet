"""Fast train/val/test split: move pre-extracted ArchCAD JSONs into split dirs.

Deterministic MD5 split (stable across machines). Uses shutil.move (instant on the
same filesystem, falls back to copy across devices).

  python scripts/split_archcad.py --src /root/ex --out /root/archcad
"""
from __future__ import annotations
import argparse, hashlib, shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path, help="dir containing extracted *.json")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--val", type=float, default=0.1)
    ap.add_argument("--test", type=float, default=0.1)
    a = ap.parse_args()
    files = list(a.src.rglob("*.json"))
    if not files:
        raise SystemExit(f"no .json files under {a.src}")
    for s in ("train", "val", "test"):
        (a.out / s).mkdir(parents=True, exist_ok=True)
    n = {"train": 0, "val": 0, "test": 0}
    for jf in files:
        h = int(hashlib.md5(jf.stem.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        s = "test" if h < a.test else ("val" if h < a.test + a.val else "train")
        shutil.move(str(jf), str(a.out / s / jf.name)); n[s] += 1
    print("split:", n)


if __name__ == "__main__":
    main()
