"""Download the ArchCAD JSON modality from Hugging Face (gated).

We only need json.zip (~393 MB) — NOT png.zip (1.8 GB), because the vision branch
re-renders the raster from the vectors itself. point/svg/caption are unused too.

Auth: set HF_TOKEN in the environment (NEVER hard-code it). The token is your own
gated-access token from huggingface.co/settings/tokens.

  HF_TOKEN=hf_xxx python scripts/download_archcad.py --out data/raw
  HF_TOKEN=hf_xxx python scripts/download_archcad.py --out data/raw --with-images   # also png.zip
"""
from __future__ import annotations
import argparse, os
from pathlib import Path
from huggingface_hub import hf_hub_download

REPO = "jackluoluo/ArchCAD"
# the dataset stores files under data/ — adjust if Kaggle inspection shows otherwise
CANDIDATE_PATHS = ["data/{}", "{}"]


def fetch(fname, out, token):
    last = None
    for tmpl in CANDIDATE_PATHS:
        try:
            p = hf_hub_download(repo_id=REPO, filename=tmpl.format(fname),
                                repo_type="dataset", token=token, local_dir=out)
            print("downloaded:", p)
            return p
        except Exception as e:
            last = e
    raise SystemExit(f"could not fetch {fname}: {last}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/raw"))
    ap.add_argument("--with-images", action="store_true", help="also download png.zip (1.8GB)")
    a = ap.parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("set HF_TOKEN env var (your gated-access HF token).")
    a.out.mkdir(parents=True, exist_ok=True)
    fetch("json.zip", a.out, token)
    if a.with_images:
        fetch("png.zip", a.out, token)
    print("done ->", a.out)


if __name__ == "__main__":
    main()
