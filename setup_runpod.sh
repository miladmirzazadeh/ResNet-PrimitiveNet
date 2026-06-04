#!/usr/bin/env bash
# One-shot RunPod setup. Because we use NO custom CUDA (no pointops/detectron2/old
# mmcv), this is just a pip install on any modern PyTorch image — no env hell.
#
#   bash setup_runpod.sh
#
# Then (with your token, which you paste on the pod — never commit it):
#   export HF_TOKEN=hf_xxxxx
#   python scripts/download_archcad.py --out /workspace/data/raw
#   python scripts/prepare_archcad.py  --raw /workspace/data/raw --out /workspace/data/archcad
#   python -m vtrue.train --data /workspace/data/archcad --out /workspace/runs/v1 \
#       --epochs 100 --batch 16 --workers 16 --time-budget 10
set -e

echo "== python / torch =="
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

echo "== installing deps (pure PyTorch stack) =="
pip install -q --upgrade pip
# torch is already in the RunPod image; install the rest (timm = pretrained vision backbones)
pip install -q numpy scipy Pillow ezdxf tqdm "huggingface_hub>=0.23" tensorboard "timm>=1.0"

echo "== sanity: import the package =="
python -c "import vtrue, vtrue.model, vtrue.dataset, vtrue.edges; print('vtrue import OK')"

mkdir -p /workspace/data /workspace/runs
echo "== ready =="
echo "next: export HF_TOKEN=...  then run the download / prepare / train commands in RUNPOD.md"
