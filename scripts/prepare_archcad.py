"""Unzip json.zip and split chunks into train/val/test by a deterministic hash.

  python scripts/prepare_archcad.py --raw data/raw --out data/archcad

Result: data/archcad/{train,val,test}/*.json  (the layout train.py expects).
Split is by filename MD5 so it's stable across machines.
"""
from __future__ import annotations
import argparse, hashlib, shutil, zipfile
from pathlib import Path


def split_for(name, val=0.1, test=0.1):
    h = int(hashlib.md5(name.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    if h < test:
        return "test"
    if h < test + val:
        return "val"
    return "train"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--out", type=Path, default=Path("data/archcad"))
    ap.add_argument("--val", type=float, default=0.1)
    ap.add_argument("--test", type=float, default=0.1)
    a = ap.parse_args()

    work = a.raw / "_json_unzipped"
    zips = list(a.raw.rglob("json.zip"))
    if not zips:
        raise SystemExit(f"json.zip not found under {a.raw} — run download_archcad.py first")
    work.mkdir(parents=True, exist_ok=True)
    for z in zips:
        print("unzip", z)
        with zipfile.ZipFile(z) as zf:
            zf.extractall(work)

    for s in ("train", "val", "test"):
        (a.out / s).mkdir(parents=True, exist_ok=True)
    n = {"train": 0, "val": 0, "test": 0}
    for jf in work.rglob("*.json"):
        s = split_for(jf.stem, a.val, a.test)
        shutil.copy(jf, a.out / s / jf.name); n[s] += 1
    print("split:", n, "->", a.out)
    if sum(n.values()) == 0:
        raise SystemExit("no .json files found inside json.zip — check the archive layout")


if __name__ == "__main__":
    main()
